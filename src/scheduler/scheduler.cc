// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "scheduler/scheduler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "base/bounded_queue.h"
#include "base/check.h"
#include "catalog/catalog.h"
#include "providers/direct_reader.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/tasks.h"

namespace jitllm::scheduler {
namespace {

WorkError WorkErrorOf(catalog::CatalogError error) {
  switch (error) {
    case catalog::CatalogError::kNotResident:
      return WorkError::kNotResident;
    case catalog::CatalogError::kStaleContent:
      return WorkError::kStale;
    case catalog::CatalogError::kOverBudget:
      return WorkError::kOverBudget;
    case catalog::CatalogError::kExhausted:
    case catalog::CatalogError::kHeld:
    case catalog::CatalogError::kWrongState:
      return WorkError::kBusy;
    default:
      return WorkError::kUnavailable;
  }
}

}  // namespace

std::string ToString(WorkError error) {
  switch (error) {
    case WorkError::kClosed:
      return "the task takes no new work";
    case WorkError::kBusy:
      return "no room now: try again later";
    case WorkError::kNotResident:
      return "an extent is not resident";
    case WorkError::kStale:
      return "the contents changed since the closure was taken";
    case WorkError::kOverBudget:
      return "loading would exceed the execution budget";
    case WorkError::kUnavailable:
      return "an extent is unavailable";
    case WorkError::kInvalid:
      return "malformed work, or no lane for it";
  }
  return "unknown work error";
}

std::string ToString(Fault fault) {
  switch (fault) {
    case Fault::kContradiction:
      return "a lane's observations contradict each other";
    case Fault::kUnproven:
      return "an operation ended without proof of no further access";
  }
  return "unknown fault";
}

Scheduler::Scheduler(catalog::Catalog& catalog, CompletionBoard& board, base::WakeFlag& wake,
                     Lanes lanes, SchedulerSettings settings)
    : catalog_(catalog),
      board_(board),
      wake_(wake),
      lanes_(lanes),
      settings_(settings),
      controls_(settings.controls, settings.controls_reserved),
      tasks_(settings.tasks, settings.priorities, settings.aging_limit),
      records_(settings.tasks),
      operations_(board.capacity()) {
  base::Check(settings_.controls_per_turn > 0 && settings_.observations_per_turn > 0 &&
                  settings_.steps_per_turn > 0 && settings_.waiters > 0,
              "a scheduler turn needs non-zero batches and waiter lists");
  // Both are compared or waited on in the clock's nanoseconds, where an
  // unbounded value overflows (milliseconds::max() in a timed wait); a tick
  // that is not positive would spin instead of sleeping.
  base::Check(settings_.tick > std::chrono::milliseconds::zero() &&
                  settings_.tick <= SchedulerSettings::kLongest,
              "a scheduler tick must be positive and at most kLongest");
  base::Check(settings_.poll_window >= std::chrono::microseconds::zero() &&
                  settings_.poll_window <= SchedulerSettings::kLongest,
              "a scheduler poll window must be non-negative and at most kLongest");
  cancel_intents_.reserve(settings_.controls);
}

base::PushResult Scheduler::Post(Control&& control) {
  base::PushResult pushed = base::PushResult::kAccepted;
  {
    const std::scoped_lock lock(cancel_mutex_);
    if (const auto* cancel = std::get_if<CancelRequest>(&control)) {
      const std::uint64_t request = cancel->request;
      if (std::ranges::contains(cancel_intents_, request)) {
        // Queued already, with no start since: the owner applies it after
        // this call (it drops the entry before applying), so this intent
        // is covered at no cost.
        return base::PushResult::kAccepted;
      }
      pushed = controls_.TryPush(std::move(control), base::PushKind::kCleanup);
      if (pushed == base::PushResult::kAccepted) {
        cancel_intents_.push_back(request);  // one per queued control: within capacity
      }
    } else {
      const std::uint64_t request = std::get<StartRequest>(control).request;
      pushed = controls_.TryPush(std::move(control), base::PushKind::kOrdinary);
      if (pushed == base::PushResult::kAccepted) {
        // A later cancellation must follow this start in the queue.
        std::erase(cancel_intents_, request);
      }
    }
  }
  if (pushed == base::PushResult::kAccepted) {
    wake_.Signal();
  }
  return pushed;
}

void Scheduler::RequestShutdown() {
  shutdown_.store(true, std::memory_order_release);
  wake_.Signal();
}

Scheduler::TaskRecord* Scheduler::Record(TaskId task) {
  if (!task.valid() || task.index() >= records_.size()) {
    return nullptr;
  }
  std::optional<TaskRecord>& record = records_[task.index()];
  return record && record->id == task ? &*record : nullptr;
}

TaskView Scheduler::ViewOf(TaskId task) const {
  const std::optional<TaskView> view = tasks_.Describe(task);
  base::Check(view.has_value(), "a view of a task that is not live");
  return view.value_or(TaskView{});
}

Scheduler::Operation* Scheduler::Find(OperationId operation) {
  if (!operation.valid() || operation.index() >= operations_.size()) {
    return nullptr;
  }
  std::optional<Operation>& record = operations_[operation.index()];
  return record && record->id == operation ? &*record : nullptr;
}

void Scheduler::SetSource(catalog::ExtentId extent, const providers::ReadSpec& source) {
  sources_[extent] = source;
}

// Tasks ------------------------------------------------------------------------------

std::expected<TaskId, StartError> Scheduler::Start(std::uint64_t request,
                                                   std::unique_ptr<TaskProgram> program,
                                                   std::size_t priority) {
  return StartTask(request, program, priority, std::nullopt);
}

std::expected<TaskId, StartError> Scheduler::StartTask(std::uint64_t request,
                                                       std::unique_ptr<TaskProgram>& program,
                                                       std::size_t priority,
                                                       std::optional<TaskId> parent) {
  if (program == nullptr || priority >= settings_.priorities) {
    return std::unexpected(StartError::kInvalid);
  }
  // A faulted node admits no new request; live ones may still unwind. Nor
  // does one whose task or operation identities are all exhausted: the ID
  // space ends by refusing, never by reuse.
  if (!parent && (stopping_ || fault_ || tasks_.exhausted() || board_.exhausted())) {
    return std::unexpected(StartError::kStopped);
  }
  const auto created = tasks_.Create(parent);
  if (!created) {
    return std::unexpected(created.error() == TaskError::kFull ? StartError::kFull
                                                               : StartError::kClosed);
  }
  const TaskId id = *created;
  records_[id.index()] = TaskRecord{.id = id,
                                    .request = request,
                                    .priority = priority,
                                    .program = std::move(program),
                                    .waiting = 0,
                                    .failed = false,
                                    .finished = false};
  if (parent) {
    ++Record(*parent)->waiting;  // the parent waits for its children
  }
  MakeReady(*Record(id));
  return id;
}

bool Scheduler::Cancel(std::uint64_t request) {
  // Every root of the request, even one that finished: its children may
  // still run, and a tag used again while an earlier start drains names
  // both starts.
  bool found = false;
  for (const std::optional<TaskRecord>& record : records_) {
    // CancelTask may retire records, never move them.
    if (record && record->request == request && !ViewOf(record->id).parent) {
      found = true;
      CancelTask(record->id);
    }
  }
  return found;
}

void Scheduler::CancelTask(TaskId task) {
  // Closes the subtree to new children and operations, then finishes every
  // task in it; what they published stays owned until it drains.
  base::Check(tasks_.Cancel(task).has_value(), "cancelling a live task");
  for (const std::optional<TaskRecord>& record : records_) {
    // FinishTask may retire this or other records, never move them.
    if (record && !record->finished && ViewOf(record->id).cancelled) {
      FinishTask(record->id, TaskOutcome::kCancelled);
    }
  }
}

void Scheduler::MakeReady(TaskRecord& record) {
  if (!record.finished) {
    base::Check(tasks_.MakeReady(record.id, record.priority).has_value(),
                "an unfinished task can be made ready");
  }
}

void Scheduler::Wake(TaskId task, bool failed) {
  TaskRecord* record = Record(task);
  if (record == nullptr || record->finished) {
    return;
  }
  base::Check(record->waiting > 0, "a task woken for a dependency it did not wait on");
  --record->waiting;
  record->failed = record->failed || failed;
  if (record->waiting == 0) {
    MakeReady(*record);
  }
}

void Scheduler::StepTask(TaskId task) {
  TaskRecord* record = Record(task);
  if (record == nullptr || record->finished) {
    return;
  }
  TaskContext context(*this, task);
  const Step step = record->program->Advance(context);
  record = Record(task);  // still live: only a finished task retires
  switch (step.kind) {
    case Step::Kind::kYield:
      MakeReady(*record);
      break;
    case Step::Kind::kWait:
      if (record->waiting == 0) {
        MakeReady(*record);  // everything it waited for has already settled
      }
      break;
    case Step::Kind::kFinish:
      FinishTask(task, step.outcome);
      break;
  }
}

void Scheduler::FinishTask(TaskId task, TaskOutcome outcome) {
  TaskRecord* record = Record(task);
  if (record == nullptr || record->finished) {
    return;
  }
  base::Check(tasks_.Finish(task, outcome).has_value(), "finishing a live task");
  const TaskView view = ViewOf(task);
  const TaskOutcome finished = view.outcome.value_or(outcome);
  record->finished = true;
  // Told first: a rollback below may retire it.
  record->program->Finished(finished);
  Withdraw(task);
  if (finished == TaskOutcome::kCancelled) {
    // Cancel queued work: commands no lane has taken never reach one.
    for (std::deque<OperationId>& pending : pending_) {
      for (const OperationId id : std::vector<OperationId>(pending.begin(), pending.end())) {
        Operation* operation = Find(id);
        if (operation != nullptr && !operation->published && operation->kind != Kind::kRead &&
            operation->task == task) {
          RollBack(*operation);
        }
      }
    }
  }
  if (view.parent) {
    Wake(*view.parent, false);  // a failure shows in its view (child_failed)
  }
  TryRetire(task);
}

void Scheduler::TryRetire(TaskId task) {
  std::optional<TaskId> next = task;
  while (next) {
    const std::optional<TaskView> view = tasks_.Describe(*next);
    TaskRecord* record = Record(*next);
    if (!view || record == nullptr || !tasks_.Retire(*next)) {
      return;  // still computing, or operations or children outstanding
    }
    record->program->Retired();
    records_[next->index()].reset();
    next = view->parent;  // the parent may have been waiting on this child
  }
}

void Scheduler::Apply(Control&& control) {
  if (auto* start = std::get_if<StartRequest>(&control)) {
    const auto started = StartTask(start->request, start->program, start->priority, std::nullopt);
    if (!started && start->program != nullptr) {
      // Never admitted: it hears so, and goes with the control.
      start->program->Finished(stopping_ ? TaskOutcome::kCancelled : TaskOutcome::kFailed);
    }
    return;
  }
  (void)Cancel(std::get<CancelRequest>(control).request);
}

void Scheduler::BeginStop() {
  if (stopping_) {
    return;
  }
  stopping_ = true;
  controls_.Close();  // what is queued is still applied, and refused
  // Every unfinished task, including children of a root that finished.
  for (const std::optional<TaskRecord>& record : records_) {
    if (record && !record->finished) {
      CancelTask(record->id);
    }
  }
}

// Work -------------------------------------------------------------------------------

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
        Operation& operation = *Find(load->second);
        if (std::ranges::find(operation.waiters, task) == operation.waiters.end()) {
          if (operation.waiters.size() >= settings_.waiters) {
            return std::unexpected(WorkError::kBusy);
          }
          operation.waiters.push_back(task);
          ++record->waiting;
          SetCritical(operation, true);
        }
        waiting = true;
        continue;
      }
      case catalog::ExtentState::kNonresident: {
        const auto source = sources_.find(extent);
        if (source == sources_.end() || lanes_.storage == nullptr) {
          return std::unexpected(WorkError::kUnavailable);
        }
        if (found->content_generation != generation) {
          return std::unexpected(WorkError::kStale);  // a load restores current contents only
        }
        if (board_.available() == 0) {
          return std::unexpected(board_.exhausted() ? WorkError::kUnavailable : WorkError::kBusy);
        }
        const auto ticket = catalog_.BeginLoad(extent, settings_.budget);
        if (!ticket) {
          return std::unexpected(WorkErrorOf(ticket.error()));
        }
        // Recorded before any lane can see it: the extent is LOADING and
        // charged until the read's completion settles it.
        Operation& operation = Open(Kind::kRead);
        operation.ticket = *ticket;
        operation.spec = source->second;
        operation.waiters.reserve(settings_.waiters);
        operation.waiters.push_back(task);
        ++record->waiting;
        SetCritical(operation, true);
        loads_[extent] = operation.id;
        Publish(operation);
        waiting = true;
        continue;
      }
      case catalog::ExtentState::kEvicting:
        return std::unexpected(WorkError::kBusy);
      case catalog::ExtentState::kQuarantined:
        return std::unexpected(WorkError::kUnavailable);
    }
  }
  return waiting ? Readiness::kWaiting : Readiness::kReady;
}

std::expected<OperationId, WorkError> Scheduler::Submit(TaskId task,
                                                        const catalog::Closure& closure, Kind kind,
                                                        std::optional<DeviceWork> work,
                                                        CpuJob job) {
  if ((kind == Kind::kDevice &&
       (lanes_.device == nullptr || !work || work->count == 0 || work->count > kMaxDeviceCopies)) ||
      (kind == Kind::kCpu && (lanes_.cpu == nullptr || !job))) {
    return std::unexpected(WorkError::kInvalid);
  }
  TaskRecord* record = Record(task);
  const std::optional<TaskView> view = tasks_.Describe(task);
  if (record == nullptr || record->finished || !view || view->cancelled) {
    return std::unexpected(WorkError::kClosed);
  }
  // Not open() against capacity(): a mailbox whose generation is exhausted
  // is retired and never issued again.
  if (board_.available() == 0) {
    return std::unexpected(board_.exhausted() ? WorkError::kUnavailable : WorkError::kBusy);
  }
  // Prepare under the owner: generations checked and leases taken, all or
  // none, then the task's lifetime hold and the record, before any lane
  // can touch the memory.
  const auto lease = catalog_.AcquireLease(closure);
  if (!lease) {
    return std::unexpected(WorkErrorOf(lease.error()));
  }
  base::Check(catalog_.RecordUse(*lease, turn_).has_value(), "recording a fresh lease's use");
  base::Check(tasks_.PrepareOperation(task).has_value(), "an open task takes an operation");
  Operation& operation = Open(kind);
  operation.task = task;
  operation.lease = *lease;
  operation.work = work.value_or(DeviceWork{});
  if (kind == Kind::kCpu) {
    operation.job = CpuCommand{.operation = operation.id, .job = std::move(job)};
  }
  ++record->waiting;
  SetCritical(operation, true);
  const OperationId id = operation.id;
  Publish(operation);
  return id;
}

std::expected<void, WorkError> Scheduler::Evict(catalog::ExtentId extent) {
  const auto ticket = catalog_.BeginEvict(extent);
  if (!ticket) {
    return std::unexpected(WorkErrorOf(ticket.error()));
  }
  base::Check(catalog_.CompleteEvict(*ticket).has_value(), "completing a fresh eviction");
  return {};
}

bool TaskContext::TakeFailure() {
  Scheduler::TaskRecord* record = scheduler_.Record(task_);
  const bool failed = record != nullptr && record->failed;
  if (record != nullptr) {
    record->failed = false;
  }
  return failed;
}

bool TaskContext::ChildFailed() const {
  const std::optional<TaskView> view = scheduler_.tasks_.Describe(task_);
  return view && view->child_failed;
}

std::expected<TaskId, WorkError> TaskContext::Spawn(std::unique_ptr<TaskProgram> program,
                                                    std::size_t priority) {
  auto child = scheduler_.StartTask(0, program, priority, task_);
  if (!child) {
    switch (child.error()) {
      case StartError::kFull:
        return std::unexpected(WorkError::kBusy);
      case StartError::kInvalid:
        return std::unexpected(WorkError::kInvalid);
      case StartError::kClosed:
      case StartError::kStopped:
        return std::unexpected(WorkError::kClosed);
    }
  }
  return *child;
}

// Operations -------------------------------------------------------------------------

Scheduler::Operation& Scheduler::Open(Kind kind) {
  const OperationId id = board_.Open();
  base::Check(id.valid(), "opening an operation past the board's capacity");
  std::optional<Operation>& slot = operations_[id.index()];
  slot.emplace();
  slot->id = id;
  slot->kind = kind;
  return *slot;
}

void Scheduler::SetCritical(Operation& operation, bool critical) {
  if (operation.critical != critical) {
    operation.critical = critical;
    critical ? ++critical_ : --critical_;
  }
}

base::PushResult Scheduler::Push(Operation& operation) const {
  switch (operation.kind) {
    case Kind::kRead:
      return lanes_.storage->Submit(ReadCommand{.operation = operation.id, .spec = operation.spec});
    case Kind::kDevice:
      return lanes_.device->Submit(
          DeviceCommand{.operation = operation.id, .work = operation.work});
    case Kind::kCpu:
      // Moves the job only if the lane takes it.
      return lanes_.cpu->Submit(std::move(operation.job));
  }
  return base::PushResult::kClosed;
}

void Scheduler::Publish(Operation& operation) {
  std::deque<OperationId>& pending = pending_[static_cast<std::size_t>(operation.kind)];
  if (!pending.empty()) {
    pending.push_back(operation.id);  // behind what the lane refused before
    return;
  }
  switch (Push(operation)) {
    case base::PushResult::kAccepted:
      operation.published = true;
      return;
    case base::PushResult::kFull:
      pending.push_back(operation.id);  // backpressure: the next turns retry
      return;
    case base::PushResult::kClosed:
      RollBack(operation);
      return;
  }
}

bool Scheduler::Flush() {
  bool progress = false;
  for (std::deque<OperationId>& pending : pending_) {
    while (!pending.empty()) {
      Operation* operation = Find(pending.front());
      if (operation == nullptr || operation->published) {
        pending.pop_front();  // rolled back meanwhile
        continue;
      }
      const base::PushResult pushed = Push(*operation);
      if (pushed == base::PushResult::kFull) {
        break;
      }
      pending.pop_front();
      progress = true;
      if (pushed == base::PushResult::kAccepted) {
        operation->published = true;
      } else {
        RollBack(*operation);
      }
    }
  }
  for (auto it = cancels_.begin(); it != cancels_.end();) {
    Operation* operation = Find(*it);
    if (operation != nullptr && !operation->quarantined) {
      const base::PushResult pushed =
          lanes_.storage->Submit(CancelRead{.operation = operation->id}, base::PushKind::kCleanup);
      if (pushed == base::PushResult::kFull) {
        ++it;  // the intent stays; the read drains meanwhile
        continue;
      }
    }
    it = cancels_.erase(it);
    progress = true;
  }
  return progress;
}

void Scheduler::Withdraw(TaskId task) {
  for (const auto& [extent, id] : std::vector(loads_.begin(), loads_.end())) {
    Operation* operation = Find(id);
    if (operation == nullptr) {
      continue;
    }
    const auto waiter = std::ranges::find(operation->waiters, task);
    if (waiter == operation->waiters.end()) {
      continue;
    }
    // Only this task's interest: other waiters and the read stay.
    operation->waiters.erase(waiter);
    if (operation->waiters.empty() && !operation->cancelling) {
      CancelLoad(*operation);
    }
  }
}

void Scheduler::CancelLoad(Operation& operation) {
  operation.cancelling = true;
  SetCritical(operation, false);
  if (!operation.published) {
    RollBack(operation);  // no lane has it: nothing to drain
    return;
  }
  // Best effort, from the cleanup reserve; the read drains either way.
  const base::PushResult pushed =
      lanes_.storage->Submit(CancelRead{.operation = operation.id}, base::PushKind::kCleanup);
  if (pushed == base::PushResult::kFull) {
    cancels_.push_back(operation.id);
  }
}

void Scheduler::RollBack(Operation& operation) {
  // No lane took the command, so the scheduler itself proves it never
  // started, through the same mailbox a lane would have used.
  base::Check(board_.Accept(operation.id, Acceptance::kNotStarted) == Published::kRecorded,
              "rolling back an operation a lane may have seen");
  Conclude(operation, Outcome::kFailed, 0);
}

void Scheduler::Observe(const Observation& seen) {
  Operation* operation = Find(seen.operation);
  if (operation == nullptr || operation->quarantined) {
    return;  // a stale identity, or news of an operation already quarantined
  }
  if (seen.contradictory) {
    Quarantine(*operation, Fault::kContradiction);
    return;
  }
  if (seen.acceptance == Acceptance::kNotStarted) {
    Conclude(*operation, Outcome::kFailed, 0);  // it never touched memory
    return;
  }
  // Both halves first: a completion that came before its acceptance waits.
  if (seen.acceptance == Acceptance::kNone || !seen.terminal) {
    return;
  }
  if (!seen.terminal->no_further_access) {
    Quarantine(*operation, Fault::kUnproven);
    return;
  }
  Conclude(*operation, seen.terminal->outcome, seen.terminal->bytes);
}

void Scheduler::Conclude(Operation& operation, Outcome outcome, std::uint64_t bytes) {
  // The board must agree the operation is reconciled before anything it
  // holds is released: a contradiction that landed since the harvest keeps
  // the mailbox open, and the operation is quarantined instead.
  const OperationId id = operation.id;
  if (!board_.Close(id)) {
    Quarantine(operation, Fault::kContradiction);
    return;
  }
  SetCritical(operation, false);
  if (operation.kind == Kind::kRead) {
    const auto load = loads_.find(operation.ticket.extent);
    base::Check(load != loads_.end() && load->second == operation.id, "a page-in lost its extent");
    loads_.erase(load);
    // Published only if the read moved the whole range; otherwise the
    // backing is released, now that the read touches it no more.
    const bool loaded = outcome == Outcome::kSucceeded && bytes == operation.spec.length;
    base::Check(loaded ? catalog_.CompleteLoad(operation.ticket).has_value()
                       : catalog_.FailLoad(operation.ticket, true).has_value(),
                "settling a page-in's load");
    // A task that joined after every other waiter left was waiting for the
    // drain, not the contents: it is not told of a failure it did not cause.
    for (const TaskId waiter : operation.waiters) {
      Wake(waiter, !loaded && !operation.cancelling);
    }
  } else {
    base::Check(catalog_.ReleaseLease(operation.lease).has_value(),
                "releasing an operation's lease");
    base::Check(tasks_.RetireOperation(operation.task).has_value(), "retiring a task's operation");
    const TaskId task = operation.task;
    operations_[id.index()].reset();
    Wake(task, outcome != Outcome::kSucceeded);
    TryRetire(task);
    return;
  }
  operations_[id.index()].reset();
}

void Scheduler::Quarantine(Operation& operation, Fault fault) {
  operation.quarantined = true;
  ++quarantined_;
  fault_ = fault_.value_or(fault);
  SetCritical(operation, false);
  if (operation.kind == Kind::kRead) {
    // Its extent stays charged as quarantined; waiters stop waiting.
    loads_.erase(operation.ticket.extent);
    base::Check(catalog_.FailLoad(operation.ticket, false).has_value(),
                "quarantining a page-in's load");
    for (const TaskId waiter : operation.waiters) {
      Wake(waiter, true);
    }
    operation.waiters.clear();
    return;
  }
  // The lease and the task's hold stay: nothing it touched is reused.
  Wake(operation.task, true);
}

// The turn loop ----------------------------------------------------------------------

bool Scheduler::Turn() {
  ++turn_;
  ++stats_.turns;
  if (shutdown_.load(std::memory_order_acquire)) {
    BeginStop();
  }
  bool progress = false;
  for (std::size_t i = 0; i < settings_.controls_per_turn; ++i) {
    std::optional<Control> control = controls_.TryPop();
    if (!control) {
      break;
    }
    if (const auto* cancel = std::get_if<CancelRequest>(&*control)) {
      // Before it is applied: a cancellation posted from now on queues
      // again instead of joining this one.
      const std::scoped_lock lock(cancel_mutex_);
      std::erase(cancel_intents_, cancel->request);
    }
    Apply(std::move(*control));
    progress = true;
  }
  for (const Observation& seen : board_.Harvest(settings_.observations_per_turn)) {
    Observe(seen);
    progress = true;
  }
  progress = Flush() || progress;
  for (std::size_t i = 0; i < settings_.steps_per_turn; ++i) {
    const std::optional<TaskId> task = tasks_.NextReady();
    if (!task) {
      break;
    }
    StepTask(*task);
    progress = true;
  }
  return progress;
}

std::optional<std::expected<void, Fault>> Scheduler::Stopped() const {
  if (!stopping_ || board_.open() > quarantined_ || !cancels_.empty() || !controls_.drained()) {
    return std::nullopt;
  }
  // Every task finished at the stop; what remains is held by quarantined
  // work, whose capacity is not reclaimed.
  if (quarantined_ > 0) {
    return std::unexpected(fault_.value_or(Fault::kUnproven));
  }
  base::Check(tasks_.size() == 0, "a task outlived its operations after the stop");
  return std::expected<void, Fault>();
}

std::expected<void, Fault> Scheduler::Run() {
  auto progressed = std::chrono::steady_clock::now();
  while (true) {
    // Consumed before looking: a publication this turn misses signals again.
    (void)wake_.Consume();
    const bool progress = Turn();
    if (auto stopped = Stopped()) {
      return *stopped;
    }
    const auto now = std::chrono::steady_clock::now();
    if (progress) {
      progressed = now;
      continue;
    }
    if (critical_ > 0 && now - progressed < settings_.poll_window) {
      ++stats_.polls;
      std::this_thread::yield();
      continue;
    }
    ++stats_.sleeps;
    (void)wake_.WaitFor(settings_.tick);
  }
}

}  // namespace jitllm::scheduler
