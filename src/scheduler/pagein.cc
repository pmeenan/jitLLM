// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The scheduler's page-ins and evictions (scheduler.h; D-033, D-048,
// D-081; docs/architecture.md#page-in-and-eviction-lifecycles-8): a load
// runs in stages, each an operation of its own, and the scheduler moves it
// from one to the next only on what the previous stage's completion
// proved. Publishing a stage can roll it back at once (a closed lane), which
// runs the load's next step before Publish returns; so a function that
// enters a stage does so last, and code that needs a load again after
// entering another's stage looks it up again.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <optional>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/check.h"
#include "catalog/catalog.h"
#include "providers/direct_reader.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/scheduler.h"
#include "scheduler/tasks.h"

namespace jitllm::scheduler {
namespace {

void Remove(std::deque<catalog::ExtentId>& queue, catalog::ExtentId extent) {
  std::erase(queue, extent);
}

}  // namespace

std::size_t Scheduler::slots_busy() const {
  return static_cast<std::size_t>(std::ranges::count(slots_, SlotState::kBusy));
}

std::size_t Scheduler::slots_quarantined() const {
  return static_cast<std::size_t>(std::ranges::count(slots_, SlotState::kQuarantined));
}

std::size_t Scheduler::copying() const {
  return static_cast<std::size_t>(std::ranges::count_if(loads_, [this](const auto& entry) {
    const Load& load = entry.second;
    if (load.stage != Stage::kCopying || !load.operation.valid()) {
      return false;
    }
    const std::optional<Operation>& operation = operations_[load.operation.index()];
    return operation && operation->id == load.operation && operation->published;
  }));
}

bool Scheduler::NoMailboxEver() const {
  // Quarantine is sticky, so a mailbox it holds never frees.
  return board_.available() == 0 && (board_.exhausted() || board_.open() <= quarantined_);
}

// Sources ---------------------------------------------------------------------------

void Scheduler::SetSource(catalog::ExtentId extent, const providers::ReadSpec& source) {
  base::Check(
      SetSource(
          extent,
          PageSource{.read = source, .landed = false, .destination = 0, .backing = std::nullopt})
          .has_value(),
      "a direct source for an extent that can take one");
}

std::expected<void, WorkError> Scheduler::SetSource(catalog::ExtentId extent,
                                                    const PageSource& source) {
  if (source.read.length == 0 || source.read.kind != providers::IoKind::kRead ||
      (source.landed && (slots_.empty() || source.destination == 0 ||
                         source.read.length > settings_.landing.slot_bytes.value())) ||
      (!source.landed && source.read.memory == nullptr) ||
      (source.backing && source.backing->size == Bytes())) {
    return std::unexpected(WorkError::kInvalid);
  }
  if (loads_.contains(extent) || evictions_.contains(extent)) {
    return std::unexpected(WorkError::kBusy);
  }
  const auto view = catalog_.Describe(extent);
  if (!view) {
    return std::unexpected(WorkError::kUnavailable);
  }
  if (source.write_back && view->descriptor.recovery != catalog::Recovery::kPreserve) {
    return std::unexpected(WorkError::kInvalid);  // only live mutable contents are written back
  }
  const auto old = sources_.find(extent);
  if (view->state != catalog::ExtentState::kNonresident) {
    // Its backing is where its last source put it: an eviction must unmap
    // that place, so the place cannot change until then. With no source
    // yet, its owner mapped it and says where.
    if (old != sources_.end() && old->second.backing != source.backing) {
      return std::unexpected(WorkError::kBusy);
    }
    // A write-back copies the live contents from where they are: the
    // destination (or memory, for a direct place) cannot move either.
    if (old != sources_.end() && source.write_back &&
        (old->second.landed != source.landed ||
         (source.landed ? old->second.destination != source.destination
                        : old->second.read.memory != source.read.memory))) {
      return std::unexpected(WorkError::kBusy);
    }
  } else if (view->preserved) {
    // The preserved contents are at the write-back place that wrote them
    // and nowhere else: a source naming another range would restore other
    // bytes under their generation (invariant 4). The backing may move.
    if (old == sources_.end() || !source.write_back || !old->second.write_back ||
        old->second.read.fd != source.read.fd || old->second.read.offset != source.read.offset ||
        old->second.read.length != source.read.length) {
      return std::unexpected(WorkError::kBusy);
    }
  }
  sources_[extent] = source;
  return {};
}

// Materialization -------------------------------------------------------------------

std::expected<Readiness, WorkError> Scheduler::Materialize(TaskId task,
                                                           const catalog::Closure& closure) {
  TaskRecord* record = Record(task);
  const std::optional<TaskView> view = tasks_.Describe(task);
  if (record == nullptr || record->finished || !view || view->cancelled) {
    return std::unexpected(WorkError::kClosed);
  }
  bool waiting = false;
  for (const auto& [extent, generation] : closure.extents) {
    const auto found = catalog_.Describe(extent);
    if (!found) {
      return std::unexpected(WorkError::kUnavailable);
    }
    switch (found->state) {
      case catalog::ExtentState::kResident:
        if (found->content_generation != generation || found->discarded) {
          return std::unexpected(WorkError::kStale);
        }
        continue;
      case catalog::ExtentState::kLoading: {
        const auto load = loads_.find(extent);
        base::Check(load != loads_.end(), "a loading extent without its page-in");
        if (std::ranges::find(load->second.waiters, task) == load->second.waiters.end()) {
          if (load->second.waiters.size() >= settings_.waiters) {
            return std::unexpected(WorkError::kBusy);
          }
          Join(load->second, *record);
        }
        waiting = true;
        continue;
      }
      case catalog::ExtentState::kNonresident: {
        const auto source = sources_.find(extent);
        if (source == sources_.end() || lanes_.storage == nullptr ||
            (source->second.landed && lanes_.device == nullptr) ||
            (source->second.backing && lanes_.device == nullptr && lanes_.backing == nullptr)) {
          return std::unexpected(WorkError::kUnavailable);
        }
        if (found->content_generation != generation) {
          return std::unexpected(WorkError::kStale);  // a load restores current contents only
        }
        if (source->second.write_back && !found->preserved) {
          // Nothing was written back at this generation: the place holds
          // no contents the closure could use (invariant 4).
          return std::unexpected(WorkError::kUnavailable);
        }
        if (board_.available() == 0) {
          return std::unexpected(NoMailboxEver() ? WorkError::kUnavailable : WorkError::kBusy);
        }
        if (auto begun = BeginPageIn(*record, extent, source->second); !begun) {
          return std::unexpected(begun.error());
        }
        waiting = true;
        continue;
      }
      case catalog::ExtentState::kEvicting: {
        // An unmap in flight: wait for it, then materialize again.
        const auto eviction = evictions_.find(extent);
        if (eviction == evictions_.end()) {
          return std::unexpected(WorkError::kBusy);
        }
        std::vector<TaskId>& waiters = eviction->second.waiters;
        if (std::ranges::find(waiters, task) == waiters.end()) {
          if (waiters.size() >= settings_.waiters) {
            return std::unexpected(WorkError::kBusy);
          }
          waiters.push_back(task);
          ++record->waiting;
        }
        waiting = true;
        continue;
      }
      case catalog::ExtentState::kQuarantined:
        return std::unexpected(WorkError::kUnavailable);
    }
  }
  return waiting ? Readiness::kWaiting : Readiness::kReady;
}

std::expected<void, WorkError> Scheduler::BeginPageIn(TaskRecord& record, catalog::ExtentId extent,
                                                      const PageSource& source) {
  const auto ticket = catalog_.BeginLoad(extent, settings_.budget);
  if (!ticket) {
    return std::unexpected(ErrorOf(ticket.error()));
  }
  // Recorded before any lane can see it: the extent is LOADING and charged
  // until the load's last stage settles it.
  Load& load = loads_[extent];
  load = Load{};
  load.ticket = *ticket;
  load.source = source;
  load.waiters.reserve(settings_.waiters);
  Join(load, record);
  if (source.landed && started_ >= 2 * slots_.size()) {
    // As many landed loads are in flight as the window allows: wait, in
    // order, without backing.
    load.stage = Stage::kQueued;
    queued_.push_back(extent);
    return {};
  }
  Proceed(extent);
  return {};
}

void Scheduler::Join(Load& load, TaskRecord& record) {
  load.waiters.push_back(record.id);
  ++record.waiting;
  if (Operation* operation = Find(load.operation)) {
    SetCritical(*operation, true);
  }
}

// The stages ------------------------------------------------------------------------

void Scheduler::Proceed(catalog::ExtentId extent) {
  const auto found = loads_.find(extent);
  if (found == loads_.end()) {
    return;
  }
  Load& load = found->second;
  if (load.stage == Stage::kQueued) {
    // Starting: count it among the landed loads in flight, then map its
    // backing or go straight on.
    if (load.source.landed) {
      load.started = true;
      ++started_;
    }
    if (load.source.backing) {
      load.stage = Stage::kMapping;
      if (!OpenStage(extent, load)) {
        blocked_.push_back(extent);
      }
      return;
    }
    load.stage = Stage::kMapping;  // nothing to map: as if mapped
    OnStage(extent, Outcome::kSucceeded, 0);
    return;
  }
  base::Check(load.stage == Stage::kSlot, "proceeding from a stage that waits for a completion");
  // A slot, in order: the first free one, else wait for one.
  const auto slot = std::ranges::find(slots_, SlotState::kFree);
  if (slot == slots_.end()) {
    slot_waiters_.push_back(extent);
    return;
  }
  const auto index = static_cast<std::size_t>(slot - slots_.begin());
  *slot = SlotState::kBusy;
  load.slot = index;
  load.stage = Stage::kReading;
  if (!OpenStage(extent, load)) {
    blocked_.push_back(extent);
  }
}

bool Scheduler::OpenStage(catalog::ExtentId extent, Load& load) {
  if (board_.available() == 0) {
    if (NoMailboxEver()) {
      // No identity will ever come: the stage can neither run nor be
      // proven not to be needed. What the load holds stays charged.
      QuarantineLoad(extent, Fault::kExhausted);
      return true;
    }
    return false;
  }
  Operation& operation = Open(Kind::kLoad);
  operation.extent = extent;
  load.operation = operation.id;
  const PageSource& source = load.source;
  switch (load.stage) {
    case Stage::kMapping:
    case Stage::kUnmapping: {
      base::Check(source.backing.has_value(), "mapping backing a source does not manage");
      const BackingPlace place = source.backing.value_or(BackingPlace{});
      operation.route = BackingRoute();
      operation.device = DeviceCommand{
          .operation = operation.id,
          .work = BackingWork{.kind = load.stage == Stage::kMapping ? BackingWork::Kind::kMap
                                                                    : BackingWork::Kind::kUnmap,
                              .reservation = place.reservation,
                              .offset = place.offset,
                              .size = place.size,
                              .allocation_class = place.allocation_class}};
      break;
    }
    case Stage::kReading: {
      base::Check(!source.landed || load.slot.has_value(), "a landed read without a slot");
      providers::ReadSpec spec = source.read;
      if (source.landed) {
        // The slot is host VMM the CPU maps (LandingZone).
        const std::uint64_t slot = settings_.landing.slots.at(load.slot.value_or(0));
        spec.memory = reinterpret_cast<std::byte*>(slot);  // NOLINT(performance-no-int-to-ptr)
      }
      operation.route = Route::kStorage;
      operation.read = ReadCommand{.operation = operation.id, .spec = spec};
      if (settings_.observer != nullptr) {
        settings_.observer->Staged(extent, PageInEvent::kReading);
      }
      break;
    }
    case Stage::kCopying: {
      base::Check(load.slot.has_value(), "a copy without a slot");
      DeviceWork work;
      work.stream = settings_.landing.stream;
      work.copies.at(0) = DeviceCopy{.destination = source.destination,
                                     .source = settings_.landing.slots.at(load.slot.value_or(0)),
                                     .size = Bytes(source.read.length)};
      work.count = 1;
      operation.route = Route::kDevice;
      operation.device = DeviceCommand{.operation = operation.id, .work = work};
      break;
    }
    case Stage::kQueued:
    case Stage::kSlot:
      base::Check(false, "opening an operation for a stage that has none");
      break;
  }
  SetCritical(operation, !load.cancelling && !load.waiters.empty());
  Publish(operation);  // last: a rollback runs the load's next step at once
  return true;
}

void Scheduler::OnStage(catalog::ExtentId extent, Outcome outcome, std::uint64_t bytes) {
  const auto found = loads_.find(extent);
  base::Check(found != loads_.end(), "a page-in stage lost its load");
  Load& load = found->second;
  load.operation = OperationId{};
  const bool succeeded = outcome == Outcome::kSucceeded;
  switch (load.stage) {
    case Stage::kMapping:
      // Known outcomes only: a mapping that failed changed nothing.
      if (!succeeded) {
        load.failed = true;
        EndLoad(extent, false);
        return;
      }
      load.mapped = load.source.backing.has_value();
      if (load.mapped && settings_.observer != nullptr) {
        settings_.observer->Staged(extent, PageInEvent::kMapped);
      }
      if (load.cancelling) {
        Unwind(extent, load);
        return;
      }
      if (load.source.landed) {
        load.stage = Stage::kSlot;
        Proceed(extent);
        return;
      }
      load.stage = Stage::kReading;
      if (!OpenStage(extent, load)) {
        blocked_.push_back(extent);
      }
      return;
    case Stage::kReading: {
      // Published only if the read moved the whole range.
      const bool whole = succeeded && bytes == load.source.read.length;
      if (!load.source.landed) {
        // A direct read that completed publishes even if every waiter
        // left meanwhile: its contents are whole.
        if (whole) {
          EndLoad(extent, true);
        } else {
          load.failed = true;
          Unwind(extent, load);
        }
        return;
      }
      if (whole && !load.cancelling) {
        load.stage = Stage::kCopying;
        if (!OpenStage(extent, load)) {
          blocked_.push_back(extent);
        }
        return;
      }
      // The read touches the slot no more, and no copy will read it.
      base::Check(load.slot.has_value(), "a landed load past its read without a slot");
      const std::size_t slot = load.slot.value_or(0);
      load.slot.reset();
      load.failed = true;
      FreeSlot(slot);
      Unwind(extent, loads_.at(extent));
      return;
    }
    case Stage::kCopying: {
      // The copy's fence has completed (or it never started): the slot is
      // free, and the extent holds the whole range only if it succeeded.
      base::Check(load.slot.has_value(), "a landed load past its read without a slot");
      const std::size_t slot = load.slot.value_or(0);
      load.slot.reset();
      FreeSlot(slot);
      if (succeeded) {
        EndLoad(extent, true);
        return;
      }
      Load& again = loads_.at(extent);
      again.failed = true;
      Unwind(extent, again);
      return;
    }
    case Stage::kUnmapping:
      if (succeeded) {
        load.mapped = false;
        EndLoad(extent, false);
        return;
      }
      // Refused with nothing changed: the backing is still mapped, and the
      // catalog must not call it released. Keep it charged.
      QuarantineLoad(extent, Fault::kBacking);
      return;
    case Stage::kQueued:
    case Stage::kSlot:
      base::Check(false, "a completion for a stage that waits for none");
      return;
  }
}

void Scheduler::Unwind(catalog::ExtentId extent, Load& load) {
  load.failed = true;
  if (!load.mapped) {
    EndLoad(extent, false);
    return;
  }
  load.stage = Stage::kUnmapping;
  if (!OpenStage(extent, load)) {
    blocked_.push_back(extent);
  }
}

void Scheduler::EndLoad(catalog::ExtentId extent, bool loaded) {
  const auto found = loads_.find(extent);
  base::Check(found != loads_.end(), "ending a page-in that is not in flight");
  Load load = std::move(found->second);
  loads_.erase(found);
  base::Check(!load.slot && (loaded || !load.mapped),
              "a page-in ended holding a slot, or failed with backing mapped");
  base::Check(loaded ? catalog_.CompleteLoad(load.ticket).has_value()
                     : catalog_.FailLoad(load.ticket, true).has_value(),
              "settling a page-in's load");
  if (settings_.observer != nullptr) {
    settings_.observer->Staged(extent, loaded ? PageInEvent::kResident : PageInEvent::kFailed);
  }
  // A task that joined after every other waiter left was waiting for the
  // drain, not the contents: it is not told of a failure it did not cause.
  for (const TaskId waiter : load.waiters) {
    Wake(waiter, !loaded && !load.cancelling);
  }
  if (load.started) {
    --started_;
  }
  StartQueued();
}

void Scheduler::StartQueued() {
  // Room in the window: start the loads waiting for it, in order. A
  // withdrawn one starts nothing (its CancelPageIn, still to come in the
  // same Withdraw, ends it).
  while (started_ < 2 * slots_.size() && !queued_.empty()) {
    const catalog::ExtentId next = queued_.front();
    queued_.pop_front();
    const auto found = loads_.find(next);
    if (found != loads_.end() && !found->second.cancelling) {
      Proceed(next);
    }
  }
}

void Scheduler::QuarantineLoad(catalog::ExtentId extent, Fault fault) {
  Fail(fault);
  const auto found = loads_.find(extent);
  if (found == loads_.end()) {
    return;
  }
  Load load = std::move(found->second);
  loads_.erase(found);
  Remove(queued_, extent);
  Remove(slot_waiters_, extent);
  Remove(blocked_, extent);
  if (load.slot) {
    // A read or copy may still touch it: never reused.
    slots_.at(*load.slot) = SlotState::kQuarantined;
  }
  base::Check(catalog_.FailLoad(load.ticket, false).has_value(), "quarantining a page-in's load");
  for (const TaskId waiter : load.waiters) {
    Wake(waiter, true);
  }
  if (load.started) {
    --started_;
  }
  StartQueued();
}

void Scheduler::CancelPageIn(catalog::ExtentId extent, Load& load) {
  load.cancelling = true;
  Operation* operation = Find(load.operation);
  if (operation != nullptr) {
    SetCritical(*operation, false);
  }
  switch (load.stage) {
    case Stage::kQueued:
      // Never started: nothing mapped, nothing read.
      Remove(queued_, extent);
      EndLoad(extent, false);
      return;
    case Stage::kSlot:
      Remove(slot_waiters_, extent);
      Unwind(extent, load);
      return;
    case Stage::kMapping:
    case Stage::kReading:
    case Stage::kCopying:
      if (operation == nullptr) {
        // Waiting for a mailbox: the stage never started.
        Remove(blocked_, extent);
        OnStage(extent, Outcome::kFailed, 0);
        return;
      }
      if (!operation->published) {
        RollBack(*operation);  // no lane has it: nothing to drain
        return;
      }
      if (load.stage == Stage::kReading) {
        // Best effort, from the cleanup reserve; the read drains either
        // way. Mapping and copying cannot be cancelled: they complete.
        const base::PushResult pushed = lanes_.storage->Submit(
            CancelRead{.operation = operation->id}, base::PushKind::kCleanup);
        if (pushed == base::PushResult::kFull) {
          cancels_.push_back(operation->id);
        }
      }
      return;
    case Stage::kUnmapping:
      return;  // unwinding already
  }
}

void Scheduler::FreeSlot(std::size_t slot) {
  base::Check(slots_.at(slot) == SlotState::kBusy, "freeing a slot that is not in use");
  slots_.at(slot) = SlotState::kFree;
  while (!slot_waiters_.empty()) {
    const catalog::ExtentId next = slot_waiters_.front();
    slot_waiters_.pop_front();
    // A withdrawn load starts no read (its CancelPageIn unwinds it).
    const auto found = loads_.find(next);
    if (found != loads_.end() && !found->second.cancelling) {
      Proceed(next);  // takes this slot
      return;
    }
    // A write-back waits in the same order (an extent is never loading
    // and evicting at once).
    const auto eviction = evictions_.find(next);
    if (eviction != evictions_.end() && eviction->second.stage == EvictStage::kSlot) {
      ProceedEviction(next);  // takes this slot
      return;
    }
  }
}

bool Scheduler::RetryBlocked() {
  bool progress = false;
  for (std::size_t n = blocked_.size(); n > 0 && !blocked_.empty(); --n) {
    const catalog::ExtentId extent = blocked_.front();
    blocked_.pop_front();
    const auto found = loads_.find(extent);
    const auto eviction = evictions_.find(extent);
    bool opened = false;
    if (found != loads_.end()) {
      opened = OpenStage(extent, found->second);
    } else if (eviction != evictions_.end() && !eviction->second.operation.valid()) {
      opened = OpenEvictStage(extent, eviction->second);
    } else {
      continue;
    }
    if (!opened) {
      blocked_.push_front(extent);  // still no mailbox: the rest wait too
      break;
    }
    progress = true;
  }
  return progress;
}

// Evictions -------------------------------------------------------------------------

std::expected<Readiness, WorkError> Scheduler::Evict(TaskId task, catalog::ExtentId extent) {
  TaskRecord* record = Record(task);
  const std::optional<TaskView> view = tasks_.Describe(task);
  if (record == nullptr || record->finished || !view || view->cancelled) {
    return std::unexpected(WorkError::kClosed);
  }
  const auto source = sources_.find(extent);
  const auto extent_view = catalog_.Describe(extent);
  if (!extent_view) {
    return std::unexpected(WorkError::kUnavailable);
  }
  // Live mutable contents at a write-back place are written back first;
  // invalidated ones need nothing preserved.
  const bool write_back = source != sources_.end() && source->second.write_back &&
                          extent_view->descriptor.recovery == catalog::Recovery::kPreserve &&
                          !extent_view->discarded;
  if (!write_back && (source == sources_.end() || !source->second.backing)) {
    // Unmanaged backing: the catalog alone records it.
    const auto ticket = catalog_.BeginEvict(extent);
    if (!ticket) {
      return std::unexpected(ErrorOf(ticket.error()));
    }
    base::Check(catalog_.CompleteEvict(*ticket).has_value(), "completing a fresh eviction");
    return Readiness::kReady;
  }
  const PageSource& place = source->second;
  if (write_back && (lanes_.storage == nullptr || (place.landed && lanes_.device == nullptr))) {
    return std::unexpected(WorkError::kInvalid);
  }
  if (place.backing && lanes_.device == nullptr && lanes_.backing == nullptr) {
    return std::unexpected(WorkError::kInvalid);
  }
  if (board_.available() == 0) {
    return std::unexpected(NoMailboxEver() ? WorkError::kUnavailable : WorkError::kBusy);
  }
  // Excludes new leases at once; the contents are read, and the backing
  // released, only on the lanes, after every consumer has retired (the
  // lease rule).
  const auto ticket = catalog_.BeginEvict(extent, write_back);
  if (!ticket) {
    return std::unexpected(ErrorOf(ticket.error()));
  }
  Eviction& eviction = evictions_[extent];
  eviction = Eviction{.ticket = *ticket,
                      .operation = {},
                      .evictor = task,
                      .waiters = {},
                      .stage = EvictStage::kUnmapping,
                      .slot = std::nullopt,
                      .write_back = write_back};
  eviction.waiters.reserve(settings_.waiters);
  eviction.waiters.push_back(task);
  ++record->waiting;
  if (write_back && place.landed) {
    eviction.stage = EvictStage::kSlot;
    ProceedEviction(extent);  // last: it may settle the eviction at once
    return Readiness::kWaiting;
  }
  eviction.stage = write_back ? EvictStage::kWriting : EvictStage::kUnmapping;
  if (!OpenEvictStage(extent, eviction)) {
    blocked_.push_back(extent);
  }
  return Readiness::kWaiting;
}

void Scheduler::ProceedEviction(catalog::ExtentId extent) {
  const auto found = evictions_.find(extent);
  if (found == evictions_.end()) {
    return;
  }
  Eviction& eviction = found->second;
  base::Check(eviction.stage == EvictStage::kSlot, "a write-back proceeding past its slot");
  const auto slot = std::ranges::find(slots_, SlotState::kFree);
  if (slot == slots_.end()) {
    slot_waiters_.push_back(extent);
    return;
  }
  *slot = SlotState::kBusy;
  eviction.slot = static_cast<std::size_t>(slot - slots_.begin());
  eviction.stage = EvictStage::kCopyOut;
  if (!OpenEvictStage(extent, eviction)) {
    blocked_.push_back(extent);
  }
}

bool Scheduler::OpenEvictStage(catalog::ExtentId extent, Eviction& eviction) {
  if (board_.available() == 0) {
    if (!NoMailboxEver()) {
      return false;
    }
    if (eviction.stage == EvictStage::kUnmapping && !eviction.write_back) {
      // Plain evictions open their unmap when they begin; this one never
      // started, so nothing changed.
      EndEviction(extent, false);
      return true;
    }
    // Nothing a lane will run touched the backing: the extent keeps its
    // contents, resident again. A write already made is merely unused.
    if (eviction.slot) {
      const std::size_t slot = *eviction.slot;
      eviction.slot.reset();
      FreeSlot(slot);
    }
    EndEviction(extent, false);
    return true;
  }
  const PageSource& place = sources_.at(extent);
  Operation& operation = Open(Kind::kEvict);
  operation.extent = extent;
  eviction.operation = operation.id;
  switch (eviction.stage) {
    case EvictStage::kCopyOut: {
      base::Check(eviction.slot.has_value(), "a write-back's copy without a slot");
      DeviceWork work;
      work.stream = settings_.landing.stream;
      work.copies.at(0) =
          DeviceCopy{.destination = settings_.landing.slots.at(eviction.slot.value_or(0)),
                     .source = place.destination,
                     .size = Bytes(place.read.length)};
      work.count = 1;
      operation.route = Route::kDevice;
      operation.device = DeviceCommand{.operation = operation.id, .work = work};
      break;
    }
    case EvictStage::kWriting: {
      providers::ReadSpec spec = place.read;
      spec.kind = providers::IoKind::kWrite;
      if (place.landed) {
        base::Check(eviction.slot.has_value(), "a landed write-back without a slot");
        const std::uint64_t slot = settings_.landing.slots.at(eviction.slot.value_or(0));
        spec.memory = reinterpret_cast<std::byte*>(slot);  // NOLINT(performance-no-int-to-ptr)
      }
      operation.route = Route::kStorage;
      operation.read = ReadCommand{.operation = operation.id, .spec = spec};
      break;
    }
    case EvictStage::kUnmapping: {
      base::Check(place.backing.has_value(), "unmapping backing a source does not manage");
      const BackingPlace backing = place.backing.value_or(BackingPlace{});
      operation.route = BackingRoute();
      operation.device =
          DeviceCommand{.operation = operation.id,
                        .work = BackingWork{.kind = BackingWork::Kind::kUnmap,
                                            .reservation = backing.reservation,
                                            .offset = backing.offset,
                                            .size = backing.size,
                                            .allocation_class = backing.allocation_class}};
      break;
    }
    case EvictStage::kSlot:
      base::Check(false, "opening an operation for a write-back waiting for a slot");
      break;
  }
  SetCritical(operation, true);
  Publish(operation);  // last: a rollback settles the stage at once
  return true;
}

void Scheduler::OnEvicted(catalog::ExtentId extent, Outcome outcome, std::uint64_t bytes) {
  const auto found = evictions_.find(extent);
  base::Check(found != evictions_.end(), "an eviction stage lost its eviction");
  Eviction& eviction = found->second;
  eviction.operation = OperationId{};
  const bool succeeded = outcome == Outcome::kSucceeded;
  const PageSource& place = sources_.at(extent);
  switch (eviction.stage) {
    case EvictStage::kCopyOut:
      if (succeeded) {
        eviction.stage = EvictStage::kWriting;
        if (!OpenEvictStage(extent, eviction)) {
          blocked_.push_back(extent);
        }
        return;
      }
      // The copy's fence completed (or it never started): the slot is
      // untouched from here on, and the extent's contents were only read.
      [[fallthrough]];
    case EvictStage::kWriting: {
      // The write has completed: the slot (or the extent's memory) is
      // touched no more. The contents are preserved only if it moved
      // the whole range.
      const bool whole =
          eviction.stage == EvictStage::kWriting && succeeded && bytes == place.read.length;
      if (eviction.slot) {
        const std::size_t slot = *eviction.slot;
        eviction.slot.reset();
        FreeSlot(slot);  // may hand the slot on; this eviction stays
      }
      Eviction& again = evictions_.at(extent);
      if (!whole) {
        EndEviction(extent, false);  // abandoned: resident again, contents intact
        return;
      }
      if (!place.backing) {
        EndEviction(extent, true);  // unmanaged backing: the catalog alone records it
        return;
      }
      again.stage = EvictStage::kUnmapping;
      if (!OpenEvictStage(extent, again)) {
        blocked_.push_back(extent);
      }
      return;
    }
    case EvictStage::kUnmapping:
      // Released: the backing generation advances. Refused with nothing
      // changed: the eviction is abandoned and the extent is resident
      // again.
      EndEviction(extent, succeeded);
      return;
    case EvictStage::kSlot:
      base::Check(false, "a completion for a write-back waiting for a slot");
      return;
  }
}

void Scheduler::EndEviction(catalog::ExtentId extent, bool evicted) {
  const auto found = evictions_.find(extent);
  base::Check(found != evictions_.end(), "ending an eviction that is not in flight");
  Eviction eviction = std::move(found->second);
  evictions_.erase(found);
  base::Check(!eviction.slot, "an eviction ended holding a slot");
  base::Check(evicted ? catalog_.CompleteEvict(eviction.ticket).has_value()
                      : catalog_.CancelEvict(eviction.ticket).has_value(),
              "settling an eviction");
  // Only the evictor is told of a refusal: a task materializing the extent
  // finds it resident again.
  for (const TaskId waiter : eviction.waiters) {
    Wake(waiter, !evicted && waiter == eviction.evictor);
  }
}

void Scheduler::QuarantineEvicting(catalog::ExtentId extent, Fault fault) {
  Fail(fault);
  const auto found = evictions_.find(extent);
  if (found == evictions_.end()) {
    return;
  }
  Eviction eviction = std::move(found->second);
  evictions_.erase(found);
  Remove(slot_waiters_, extent);
  Remove(blocked_, extent);
  if (eviction.slot) {
    // A copy or write may still touch it: never reused.
    slots_.at(*eviction.slot) = SlotState::kQuarantined;
  }
  base::Check(catalog_.QuarantineEviction(eviction.ticket).has_value(), "quarantining an eviction");
  for (const TaskId waiter : eviction.waiters) {
    Wake(waiter, true);
  }
}

}  // namespace jitllm::scheduler
