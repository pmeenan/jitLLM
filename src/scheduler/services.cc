// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "scheduler/services.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "base/bounded_queue.h"
#include "base/check.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/direct_reader.h"
#include "providers/storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"

namespace jitllm::scheduler {
namespace {

Outcome OutcomeOf(providers::ReadOutcome outcome) {
  switch (outcome) {
    case providers::ReadOutcome::kComplete:
      return Outcome::kSucceeded;
    case providers::ReadOutcome::kCancelled:
      return Outcome::kCancelled;
    case providers::ReadOutcome::kEndOfFile:
    case providers::ReadOutcome::kFailed:
      return Outcome::kFailed;
  }
  return Outcome::kFailed;
}

}  // namespace

StorageService::StorageService(providers::Storage& storage, providers::ReaderSettings reader,
                               CompletionBoard& board, QueueSettings queue)
    : queue_(queue.capacity, queue.reserved),
      storage_(storage),
      reader_(storage, reader),
      board_(board),
      batch_(queue.batch) {
  base::Check(batch_ > 0, "a lane turn takes at least one command");
}

void StorageService::Handle(const StorageCommand& command) {
  if (const auto* read = std::get_if<ReadCommand>(&command)) {
    const std::uint64_t key = KeyOf(read->operation);
    // The key is the operation's own, so the reader never joins it to
    // another read; a repeated command is a duplicate observation.
    const auto started = reader_.Read(key, read->spec, key);
    (void)board_.Accept(read->operation, started ? Acceptance::kAccepted : Acceptance::kNotStarted);
    return;
  }
  const std::uint64_t key = KeyOf(std::get<CancelRead>(command).operation);
  // Best effort: a read that already ended has nothing left to cancel.
  std::ignore = reader_.Withdraw(key, key);
  // Nothing may be published for it, but the queue has room: the owner may
  // hold a command or cancellation this queue refused.
  board_.Nudge();
}

bool StorageService::Turn(bool wait) {
  bool progress = false;
  for (std::size_t i = 0; i < batch_; ++i) {
    std::optional<StorageCommand> command = queue_.TryPop();
    if (!command) {
      break;
    }
    Handle(*command);
    progress = true;
  }
  // A read ends only once every request it started has completed, so its
  // result always proves no further access.
  for (const providers::FinishedRead& read : reader_.Poll(wait && !progress && reads() > 0)) {
    (void)board_.Complete(OperationOf(read.key), Terminal{.outcome = OutcomeOf(read.outcome),
                                                          .bytes = read.bytes,
                                                          .no_further_access = true});
    progress = true;
  }
  return progress;
}

void StorageService::Run() {
  const std::stop_token never;
  while (true) {
    if (reads() == 0) {
      std::optional<StorageCommand> command = queue_.Pop(never);
      if (!command) {
        return;  // closed and drained, with nothing in flight
      }
      Handle(*command);
    }
    if (!Turn(true)) {
      std::this_thread::yield();
    }
  }
}

DeviceService::DeviceService(providers::DeviceExecution& execution,
                             std::span<const providers::StreamId> streams, CompletionBoard& board,
                             DeviceSettings settings, providers::DeviceMemory* memory)
    : execution_(execution),
      memory_(memory),
      streams_(streams.begin(), streams.end()),
      board_(board),
      settings_(settings),
      queue_(settings.queue.capacity, settings.queue.reserved),
      handoff_(settings.handoff, 0) {
  base::Check(settings_.queue.batch > 0 && !streams_.empty() && settings_.refusals > 0,
              "a device service needs streams, a turn of at least one command and a refusal bound");
  watches_.reserve(settings_.handoff);
  releases_.reserve(settings_.handoff);
}

void DeviceService::Launch(DeviceCommand& command) {
  std::visit(
      [&]<typename Work>(Work& work) {
        if constexpr (std::is_same_v<Work, DeviceWork>) {
          Copy(command.operation, work);
        } else if constexpr (std::is_same_v<Work, LaunchWork>) {
          Run(command.operation, work);
        } else {
          Back(command.operation, work);
        }
      },
      command.work);
}

void DeviceService::Copy(OperationId operation, const DeviceWork& work) {
  if (work.stream >= streams_.size() || work.count == 0 || work.count > kMaxDeviceCopies) {
    (void)board_.Accept(operation, Acceptance::kNotStarted);
    return;
  }
  const providers::StreamId stream = streams_[work.stream];
  bool queued = false;
  bool refused = false;
  bool unknown = false;
  std::uint64_t bytes = 0;
  for (std::size_t i = 0; i < work.count; ++i) {
    const DeviceCopy& copy = work.copies.at(i);
    const auto copied = execution_.Copy(stream, copy.destination, copy.source, copy.size);
    if (copied) {
      queued = true;
      bytes += copy.size.value();
      continue;
    }
    // A known refusal queued nothing; an unknown outcome may have.
    refused = true;
    unknown = copied.error().error == providers::ProviderError::kUnknown;
    break;
  }
  Fence(operation, stream, queued, refused, unknown, bytes);
}

void DeviceService::Run(OperationId operation, LaunchWork& work) {
  if (work.stream >= streams_.size() || !work.job) {
    (void)board_.Accept(operation, Acceptance::kNotStarted);
    return;
  }
  const providers::StreamId stream = streams_[work.stream];
  const auto native = execution_.Submission(stream);
  JobResult result = JobResult::kNotStarted;
  if (native) {
    result = work.job(*native);
  } else if (native.error().error == providers::ProviderError::kUnknown) {
    result = JobResult::kUnknown;
  }
  Fence(operation, stream, result == JobResult::kQueued || result == JobResult::kFailed,
        result == JobResult::kFailed || result == JobResult::kUnknown,
        result == JobResult::kUnknown, 0);
}

void DeviceService::Fence(OperationId operation, providers::StreamId stream, bool queued,
                          bool refused, bool unknown, std::uint64_t bytes) {
  // Recorded even when nothing started: the provider counts an attempted
  // copy or launch as queued work, and only a fence seen complete balances
  // it.
  const auto fence = execution_.Record(stream);
  if (!queued && !unknown) {
    (void)board_.Accept(operation, Acceptance::kNotStarted);
    if (fence) {
      Hand(Watch{.operation = {}, .fence = *fence, .outcome = Outcome::kFailed, .bytes = 0});
    }
    return;
  }
  (void)board_.Accept(operation, unknown ? Acceptance::kUnknown : Acceptance::kAccepted);
  if (!fence) {
    // Nothing can show when the queued work ends.
    (void)board_.Complete(
        operation,
        Terminal{.outcome = Outcome::kFailed, .bytes = bytes, .no_further_access = false});
    return;
  }
  Hand(Watch{.operation = operation,
             .fence = *fence,
             .outcome = refused ? Outcome::kFailed : Outcome::kSucceeded,
             .bytes = bytes});
}

void DeviceService::Back(OperationId operation, const BackingWork& work) {
  if (memory_ == nullptr) {
    (void)board_.Accept(operation, Acceptance::kNotStarted);
    return;
  }
  const auto not_started = [&] { (void)board_.Accept(operation, Acceptance::kNotStarted); };
  // The provider's state is not what either outcome needs: the backing,
  // or the place, is left in a state nobody may reuse.
  const auto unproven = [&](Acceptance acceptance) {
    (void)board_.Accept(operation, acceptance);
    (void)board_.Complete(
        operation, Terminal{.outcome = Outcome::kFailed, .bytes = 0, .no_further_access = false});
  };
  const auto unknown = [](const providers::Failure& failure) {
    return failure.error == providers::ProviderError::kUnknown;
  };
  if (work.kind == BackingWork::Kind::kMap) {
    const auto created = memory_->Create(work.allocation_class, work.size);
    if (!created) {
      unknown(created.error()) ? unproven(Acceptance::kUnknown) : not_started();
      return;
    }
    const auto mapped = memory_->Map(work.reservation, work.offset, *created);
    if (!mapped) {
      if (unknown(mapped.error())) {
        unproven(Acceptance::kUnknown);
      } else {
        memory_->Release(*created) ? not_started() : unproven(Acceptance::kAccepted);
      }
      return;
    }
    const auto access =
        memory_->SetAccess(work.reservation, work.offset, work.size, providers::Access::kReadWrite);
    if (!access) {
      if (unknown(access.error())) {
        unproven(Acceptance::kUnknown);
      } else if (memory_->Unmap(work.reservation, work.offset, work.size) &&
                 memory_->Release(*created)) {
        not_started();
      } else {
        unproven(Acceptance::kAccepted);
      }
      return;
    }
  } else {
    const std::optional<providers::BackingId> backing =
        memory_->MappedAt(work.reservation, work.offset);
    if (!backing) {
      not_started();  // nothing is mapped there: nothing changed
      return;
    }
    const auto unmapped = memory_->Unmap(work.reservation, work.offset, work.size);
    if (!unmapped) {
      if (unknown(unmapped.error())) {
        unproven(Acceptance::kUnknown);
      } else if (unmapped.error().error == providers::ProviderError::kUndetermined) {
        // Refused because an earlier unknown outcome left the place
        // undetermined: nothing changed, but nothing about the place is
        // proven either, so it must never be handed back as resident.
        unproven(Acceptance::kAccepted);
      } else {
        not_started();
      }
      return;
    }
    // Unmapped, but the backing still exists until it is released: a
    // refusal here leaves it charged.
    if (const auto released = memory_->Release(*backing); !released) {
      unproven(unknown(released.error()) ? Acceptance::kUnknown : Acceptance::kAccepted);
      return;
    }
  }
  (void)board_.Accept(operation, Acceptance::kAccepted);
  (void)board_.Complete(operation, Terminal{.outcome = Outcome::kSucceeded,
                                            .bytes = work.size.value(),
                                            .no_further_access = true});
}

void DeviceService::Hand(const Watch& watch) {
  base::Check(!unhanded_, "a launch while a fence waits for the completion lane");
  const base::PushResult pushed = handoff_.TryPush(Watch{watch});
  base::Check(pushed != base::PushResult::kClosed, "a fence handed over after the lane finished");
  if (pushed != base::PushResult::kAccepted) {
    unhanded_ = watch;  // the completion lane is full: launch nothing more until it takes it
  }
}

bool DeviceService::HandPending() {
  if (!unhanded_) {
    return false;
  }
  const base::PushResult pushed = handoff_.TryPush(Watch{*unhanded_});
  base::Check(pushed != base::PushResult::kClosed, "a fence handed over after the lane finished");
  if (pushed != base::PushResult::kAccepted) {
    return false;
  }
  unhanded_.reset();
  return true;
}

void DeviceService::Finish() {
  if (!finished_) {
    finished_ = true;
    handoff_.Close();  // the completion lane drains what it has, then returns
  }
}

bool DeviceService::SubmissionTurn() {
  bool progress = HandPending();
  for (std::size_t i = 0; i < settings_.queue.batch && !unhanded_; ++i) {
    std::optional<DeviceCommand> command = queue_.TryPop();
    if (!command) {
      break;
    }
    const std::scoped_lock lock(submitting_);
    Launch(*command);
    progress = true;
  }
  if (!unhanded_ && queue_.drained()) {
    Finish();
  }
  return progress;
}

void DeviceService::RunSubmission() {
  const std::stop_token never;
  while (true) {
    if (unhanded_) {
      if (!SubmissionTurn()) {
        std::this_thread::yield();  // the completion lane is making room
      }
      continue;
    }
    std::optional<DeviceCommand> command = queue_.Pop(never);
    if (!command) {
      break;
    }
    const std::scoped_lock lock(submitting_);
    Launch(*command);
  }
  Finish();
}

bool DeviceService::CompletionTurn() {
  bool progress = false;
  // Bounded: a device that never completes fills this, then the handoff,
  // and then submission stops taking commands.
  while (watches_.size() + releases_.size() < settings_.handoff) {
    std::optional<Watch> watch = handoff_.TryPop();
    if (!watch) {
      break;
    }
    watches_.push_back(*watch);
    progress = true;
  }
  for (auto it = watches_.begin(); it != watches_.end();) {
    const auto state = execution_.Query(it->fence);
    if (state && *state == providers::FenceState::kPending) {
      it->refusals = 0;
      ++it;
      continue;
    }
    if (state) {
      if (it->operation.valid()) {
        (void)board_.Complete(
            it->operation,
            Terminal{.outcome = it->outcome, .bytes = it->bytes, .no_further_access = true});
      }
      releases_.push_back(Release{.fence = it->fence, .refusals = 0});
    } else if (state.error().error == providers::ProviderError::kUnknown ||
               (state.error().error == providers::ProviderError::kFailed &&
                ++it->refusals >= settings_.refusals)) {
      // A device fault, or a refusal that persists: the work is unproven,
      // and its fence is never touched again.
      if (it->operation.valid()) {
        (void)board_.Complete(
            it->operation,
            Terminal{.outcome = Outcome::kFailed, .bytes = it->bytes, .no_further_access = false});
      }
    } else {
      base::Check(state.error().error == providers::ProviderError::kFailed,
                  "the device refused a fence its completion lane holds");
      ++it;  // a known failure changed nothing: ask again next turn
      continue;
    }
    it = watches_.erase(it);
    progress = true;
  }
  if (!releases_.empty()) {
    // Never wait for a submission call: try again next turn.
    const std::unique_lock lock(submitting_, std::try_to_lock);
    if (lock.owns_lock()) {
      for (auto it = releases_.begin(); it != releases_.end();) {
        const auto released = execution_.Release(it->fence);
        if (!released && released.error().error == providers::ProviderError::kFailed &&
            ++it->refusals < settings_.refusals) {
          ++it;  // changed nothing: try again
          continue;
        }
        base::Check(released || released.error().error == providers::ProviderError::kUnknown ||
                        released.error().error == providers::ProviderError::kFailed,
                    "the device refused to release a fence seen complete");
        // Released, undetermined, or refused too often: never touched again.
        it = releases_.erase(it);
        progress = true;
      }
    }
  }
  return progress;
}

void DeviceService::RunCompletion() {
  const std::stop_token never;
  while (true) {
    if (watches_.empty() && releases_.empty()) {
      std::optional<Watch> watch = handoff_.Pop(never);
      if (!watch) {
        return;  // the submission lane finished and every fence is settled
      }
      watches_.push_back(*watch);
    }
    if (!CompletionTurn()) {
      if (settings_.poll_sleep.count() > 0) {
        std::this_thread::sleep_for(settings_.poll_sleep);
      } else {
        std::this_thread::yield();
      }
    }
  }
}

Lane<CpuCommand>::Handler CpuHandler(CompletionBoard& board) {
  return [&board](CpuCommand&& command) {
    (void)board.Accept(command.operation, Acceptance::kAccepted);
    const CpuResult result =
        command.job ? command.job() : CpuResult{.outcome = Outcome::kFailed, .bytes = 0};
    (void)board.Complete(
        command.operation,
        Terminal{.outcome = result.outcome, .bytes = result.bytes, .no_further_access = true});
  };
}

}  // namespace jitllm::scheduler
