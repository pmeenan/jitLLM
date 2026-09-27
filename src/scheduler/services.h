// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The provider lanes (D-048, docs/async-model.md#execution-and-thread-ownership):
// the storage service over one Storage provider, the device service's
// submission and completion lanes over one DeviceExecution provider, and
// the handler of the CPU worker lane (a scheduler::Lane). Each takes owned
// commands from a bounded queue, carries them out on its own thread, and
// reports every operation it was handed through the completion board: its
// acceptance (not started, accepted or unknown) and later its terminal
// result, with the proof that the provider touches its memory no more.
// No lane touches the scheduler's records or runs a continuation.
//
// Each service is a set of non-blocking turns and a Run loop that repeats
// them on a thread. Deterministic tests call the turns themselves, in the
// order they choose; programs run the loops on threads (only programs start
// threads: docs/architecture.md#layers-and-dependency-rules). Close refuses
// new commands; the loops carry out what was queued, drain every operation
// they accepted, and return.

#ifndef JITLLM_SCHEDULER_SERVICES_H_
#define JITLLM_SCHEDULER_SERVICES_H_

#include <chrono>
#include <cstddef>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include "base/bounded_queue.h"
#include "providers/device_execution.h"
#include "providers/direct_reader.h"
#include "providers/storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/lane.h"

namespace jitllm::scheduler {

struct QueueSettings {
  std::size_t capacity = 64;  // queued commands, including the reserve
  std::size_t reserved = 8;   // held back for cleanup commands
  std::size_t batch = 16;     // commands a turn takes
};

// The storage lane: whole reads (providers::DirectReader) into memory the
// scheduler protects, one read per operation. A read is accepted once the
// reader takes it, and ends only when every request it started has
// completed, so its terminal result always proves no further access. A
// cancelled read withdraws the operation's interest: the reader asks the
// provider to cancel what is in flight and still drains.
//
// The reader must take as many reads as the scheduler can have operations
// (ReaderSettings::reads), or an operation may be refused as not started.
// While requests are in flight, the Run loop waits in the provider for a
// completion; a queued command wakes it (Storage::Wake), so a new read or a
// cancellation never waits for an unrelated completion, even one that never
// comes. A cancellation the kernel cannot act on (a request stuck in an
// uninterruptible wait) still leaves the read to drain: its memory stays
// protected until then. Taking a cancellation publishes nothing, so the
// lane nudges the owner (CompletionBoard::Nudge): a command the full queue
// refused is offered again at once, not at the owner's next timer tick.
class StorageService {
 public:
  StorageService(providers::Storage& storage, providers::ReaderSettings reader,
                 CompletionBoard& board, QueueSettings queue);

  // Any thread.
  base::PushResult Submit(const StorageCommand& command,
                          base::PushKind kind = base::PushKind::kOrdinary) {
    const base::PushResult pushed = queue_.TryPush(StorageCommand(command), kind);
    if (pushed == base::PushResult::kAccepted) {
      storage_.Wake();  // after the push: the lane looks again once woken
    }
    return pushed;
  }
  void Close() {
    queue_.Close();
    storage_.Wake();
  }

  // The lane's thread: one round of commands and completions. With
  // `wait`, and nothing else to do, waits in the provider for a completion
  // if a request is in flight. True if anything happened.
  bool Turn(bool wait);
  // Until closed, with every accepted read drained.
  void Run();

  std::size_t reads() const { return reader_.reads(); }

 private:
  void Handle(const StorageCommand& command);

  base::BoundedQueue<StorageCommand> queue_;
  providers::Storage& storage_;
  providers::DirectReader reader_;
  CompletionBoard& board_;
  std::size_t batch_;
};

struct DeviceSettings {
  QueueSettings queue;
  // Fences in transit from the submission lane to the completion lane, and
  // the most the completion lane holds at once (watched or awaiting
  // release): past that, submission waits for room (backpressure).
  std::size_t handoff = 64;
  // How the completion lane waits between queries while fences are
  // pending: zero yields (a whole core, the fastest to notice); otherwise
  // it sleeps this long. A setting to measure, not a tuned value (RE-017).
  std::chrono::microseconds poll_sleep{0};
  // Known failures (the call changed nothing) in a row, of a query or a
  // release of one fence, before the completion lane stops asking: a
  // refusal that persists (a lost context) must not keep the lane, and
  // shutdown, waiting forever. A bound, not a tuned value.
  std::uint32_t refusals = 8;
};

// The device service's two lanes over one provider. The submission lane
// queues each operation's copies on one of its streams and records a fence
// after them; the completion lane queries the fences without blocking and
// publishes each completion, independent of any submission call that may
// block (D-048). Releasing a fence is a submission-side call, so the
// completion lane releases only when the submission lane is between calls
// (never waiting for it) and otherwise tries again next turn.
//
// A copy the provider refused queued nothing; if an operation's first copy
// is refused it did not start. A later refusal leaves earlier copies
// queued: the operation is accepted and fails once its fence completes. An
// unknown outcome is accepted as unknown and fenced the same way. A fence
// that cannot be recorded, or a query whose outcome is unknown, leaves the
// work unproven: the terminal result carries no proof, and the scheduler
// quarantines what the operation holds. Such a fence is never released.
// A query the provider keeps refusing (DeviceSettings::refusals) is
// treated the same way; a release it keeps refusing is abandoned, since
// the work it fenced was already proven complete, and the fence stays
// recorded (its stream cannot be destroyed).
//
// The streams belong to the program, which destroys them once both lanes
// have returned.
class DeviceService {
 public:
  DeviceService(providers::DeviceExecution& execution, std::span<const providers::StreamId> streams,
                CompletionBoard& board, DeviceSettings settings);

  base::PushResult Submit(const DeviceCommand& command,
                          base::PushKind kind = base::PushKind::kOrdinary) {
    return queue_.TryPush(DeviceCommand(command), kind);
  }
  // No new commands. The submission lane returns once it has carried out
  // what was queued; the completion lane, once every fence it was handed
  // has been seen and released (or is unproven).
  void Close() { queue_.Close(); }

  // The submission lane's thread.
  bool SubmissionTurn();
  void RunSubmission();
  // The completion lane's thread.
  bool CompletionTurn();
  void RunCompletion();

 private:
  struct Watch {
    OperationId operation;  // invalid for a fence that only balances its stream
    providers::FenceId fence;
    Outcome outcome = Outcome::kSucceeded;
    std::uint64_t bytes = 0;
    std::uint32_t refusals = 0;  // known query failures in a row
  };
  struct Release {
    providers::FenceId fence;
    std::uint32_t refusals = 0;  // known release failures in a row
  };

  void Launch(const DeviceCommand& command);
  // Submission side: hands a fence to the completion lane, or keeps it
  // until there is room; nothing more launches meanwhile.
  void Hand(const Watch& watch);
  bool HandPending();
  // The submission lane has returned: nothing more will be handed over.
  void Finish();

  providers::DeviceExecution& execution_;
  std::vector<providers::StreamId> streams_;
  CompletionBoard& board_;
  DeviceSettings settings_;
  base::BoundedQueue<DeviceCommand> queue_;
  base::BoundedQueue<Watch> handoff_;
  // Held by the submission lane around its provider calls; the completion
  // lane only ever tries it.
  std::mutex submitting_;
  // Submission lane only: at most one fence, since nothing launches while
  // it waits.
  std::optional<Watch> unhanded_;
  bool finished_ = false;
  // Completion lane only, together at most `handoff` (allocated once).
  std::vector<Watch> watches_;
  std::vector<Release> releases_;
};

// The CPU worker lane (a Lane of CpuCommand, at most four workers per
// queue, RE-017): accepts each job, runs it and publishes its result, with
// the proof that it touches its memory no more. Nothing preempts a job: it
// must return, since until it does its operation keeps its lease and the
// task its hold, and shutdown and the lane's destruction wait for it.
Lane<CpuCommand>::Handler CpuHandler(CompletionBoard& board);

}  // namespace jitllm::scheduler

#endif  // JITLLM_SCHEDULER_SERVICES_H_
