// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The scheduler thread (D-048, docs/async-model.md): the node's single
// writer of task states, the catalog's loads and leases, operation records
// and retirement decisions. Other threads send it controls (start or cancel
// a request, shut down) and the lanes (services.h) publish what they
// observe on the completion board; neither changes its records.
//
// A turn drains a bounded batch of controls and of board observations,
// retries publications a full lane refused, and steps a bounded batch of
// ready tasks. Nothing in a turn blocks: a full lane leaves the command
// with its operation until a later turn, and a task waits on identified
// dependencies, never on a thread. Between turns the thread polls while a
// critical-path operation is in flight, for at most a window after its
// last progress (RE-017: a sleeping thread wakes slowly on the Spark), and
// otherwise sleeps on the wake flag. It consumes the flag before it looks
// at anything, so a publication it did not see re-signals it: no wakeup is
// lost (base/wake.h).
//
// A task is an explicit state machine (TaskProgram): each step does bounded
// work through its TaskContext and yields, waits, or finishes. Its work:
//
//   - Materialize a closure: resident extents with the recorded contents
//     are ready; a nonresident one starts a page-in from its registered
//     source; a load in flight is joined. Page-ins belong to the scheduler,
//     not the task: one per extent, with a bounded waiter list. A task that
//     leaves (cancelled or finished) removes only its own interest; the
//     last one to leave asks the storage lane to cancel, and the read still
//     drains. A read publishes resident contents only if it transferred the
//     whole range; otherwise the load fails, and the backing is released
//     only once the read's completion proves no further access.
//   - Submit device work or a CPU job over a closure: the operation leases
//     the closure (all or none, at the recorded contents), takes a lifetime
//     hold on the task and a mailbox, and is recorded before its command is
//     published. Its lease and hold are released only once its terminal
//     result proves no further access.
//   - Spawn children, whose finishing wakes the parent.
//
// Observations reconcile acceptance and the terminal result in either
// order. Not started (from the lane, or the scheduler's own rollback of a
// command no lane ever took) releases everything at once. A terminal result
// without the proof, or contradictory observations, quarantines the
// operation: a page-in's extent becomes QUARANTINED, device or CPU work
// keeps its lease and its task's hold, everything stays charged, and the
// node faults, which stops admitting requests. Quarantine is sticky here:
// recovery needs a validated quiescence procedure (docs/async-model.md).
//
// Cancelling a request cancels its task tree: no new children or
// operations, commands no lane has taken are rolled back, page-in interests
// are withdrawn, and every task finishes cancelled. Work already published
// stays owned until it drains; a task retires only when its operations and
// children have. Shutdown stops admission, cancels every task, drains
// every accepted operation and then returns; the program then closes the
// lanes and releases backing. Work that cannot be reconciled (quarantined)
// faults the shutdown instead of reporting its capacity reclaimed.
//
// Not yet here: VMM mapping and backing release on the device lane (the
// memory manager's), victim selection on a miss, admission's envelopes
// and the switching policy (admission.h) driving task starts, which
// reports a boundary only where the request's ProgramCursor::AtBoundary
// holds (execution/program.h).

#ifndef JITLLM_SCHEDULER_SCHEDULER_H_
#define JITLLM_SCHEDULER_SCHEDULER_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "base/bounded_queue.h"
#include "base/bytes.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "providers/direct_reader.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/lane.h"
#include "scheduler/services.h"
#include "scheduler/tasks.h"

namespace jitllm::scheduler {

// How a step ends.
struct Step {
  enum class Kind : std::uint8_t {
    kYield,   // ready again at once
    kWait,    // ready once every dependency it registered has settled
    kFinish,  // stop computing, with an outcome
  };
  Kind kind = Kind::kYield;
  TaskOutcome outcome = TaskOutcome::kSucceeded;

  static constexpr Step Yield() {
    return {.kind = Kind::kYield, .outcome = TaskOutcome::kSucceeded};
  }
  static constexpr Step Wait() { return {.kind = Kind::kWait, .outcome = TaskOutcome::kSucceeded}; }
  static constexpr Step Finish(TaskOutcome outcome) {
    return {.kind = Kind::kFinish, .outcome = outcome};
  }
};

enum class WorkError : std::uint8_t {
  kClosed,       // the task is cancelled or finished: no new work
  kBusy,         // no mailbox, waiter room or task slot now, or an eviction in progress
  kNotResident,  // a lease needs every extent resident
  kStale,        // the contents changed since the closure was taken
  kOverBudget,   // loading would exceed the execution budget B
  kUnavailable,  // quarantined, no registered source, unknown, or identities exhausted
  kInvalid,      // malformed work, or no lane for it
};

std::string ToString(WorkError error);

enum class Readiness : std::uint8_t {
  kReady,    // every extent is resident with the recorded contents
  kWaiting,  // page-ins are registered: return Step::Wait()
};

enum class Fault : std::uint8_t {
  kContradiction,  // a lane's observations contradict each other
  kUnproven,       // a terminal result without proof of no further access
};

std::string ToString(Fault fault);

class TaskContext;

// A task's continuation, stepped only on the scheduler thread. It owns its
// state; the scheduler owns it, from its start until the task retires.
class TaskProgram {
 public:
  TaskProgram() = default;
  TaskProgram(const TaskProgram&) = delete;
  TaskProgram& operator=(const TaskProgram&) = delete;
  TaskProgram(TaskProgram&&) = delete;
  TaskProgram& operator=(TaskProgram&&) = delete;
  virtual ~TaskProgram() = default;

  // One bounded step.
  virtual Step Advance(TaskContext& context) = 0;
  // The task's outcome, once it has one. A refused start is cancelled
  // (shutting down) or failed.
  virtual void Finished(TaskOutcome /*outcome*/) {}
  // Every operation and child it had has retired; it is destroyed next.
  virtual void Retired() {}
};

struct SchedulerSettings {
  std::size_t tasks = 64;  // the admitted task bound
  std::size_t priorities = 2;
  std::uint32_t aging_limit = 4;
  std::size_t controls = 64;           // queued controls, including the reserve ...
  std::size_t controls_reserved = 16;  // ... that only cancellations may use
  std::size_t controls_per_turn = 16;
  std::size_t observations_per_turn = 64;
  std::size_t steps_per_turn = 16;
  std::size_t waiters = 8;  // tasks waiting on one page-in
  // The execution budget B every load checks. Initialized so callers may
  // designate only the fields they change.
  base::Bytes budget = {};  // NOLINT(readability-redundant-member-init)
  // The bound on both durations below, checked when the scheduler is
  // built: the clock compares and waits in nanoseconds, where an unbounded
  // value overflows.
  static constexpr std::chrono::hours kLongest{1};
  // Polling after the last progress while a critical-path operation is in
  // flight: the window the task-lanes measurement used, not a tuned value
  // (RE-017; M3/M4 tune it against per-token latency). Zero never polls.
  std::chrono::microseconds poll_window{200};
  // The longest sleep: a timer tick, never needed for correctness. Positive.
  std::chrono::milliseconds tick{100};
};

// The lanes the scheduler publishes to; any may be absent (its work is
// then refused as invalid).
struct Lanes {
  StorageService* storage = nullptr;
  DeviceService* device = nullptr;
  Lane<CpuCommand>* cpu = nullptr;
};

// Controls from other threads.
struct StartRequest {
  std::uint64_t request = 0;  // the caller's tag, for cancelling it later
  std::size_t priority = 1;
  std::unique_ptr<TaskProgram> program;
};
struct CancelRequest {
  std::uint64_t request = 0;
};
using Control = std::variant<StartRequest, CancelRequest>;

enum class StartError : std::uint8_t {
  kStopped,  // shutting down, faulted, or identities exhausted: admission has stopped
  kFull,     // the task table is full
  kClosed,   // the parent takes no new children
  kInvalid,  // no program, or a priority out of range
};

struct SchedulerStats {
  std::uint64_t turns = 0;
  std::uint64_t polls = 0;   // idle turns spent polling for a critical completion
  std::uint64_t sleeps = 0;  // waits on the wake flag
};

class Scheduler {
 public:
  // The board's mailboxes bound the operations in flight. The catalog,
  // board, wake flag and lanes outlive the scheduler; the board signals
  // `wake`, and so do controls.
  Scheduler(catalog::Catalog& catalog, CompletionBoard& board, base::WakeFlag& wake, Lanes lanes,
            SchedulerSettings settings);
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler(Scheduler&&) = delete;
  Scheduler& operator=(Scheduler&&) = delete;
  ~Scheduler() = default;

  // Any thread. A full queue refuses the control and leaves it with the
  // caller, and so does a closed one (once shutting down). A cancellation
  // may use the reserve, and is an intent per request: while one for the
  // request is queued with no start of it posted since, another costs
  // nothing and is accepted as already queued.
  base::PushResult Post(Control&& control);
  // Any thread: a flag, however often it is set.
  void RequestShutdown();

  // The scheduler thread (or a deterministic test standing in for it).
  // Start applies at once, ahead of any control still queued: a
  // cancellation of the same request posted earlier and not yet applied
  // then reaches this start too. Post a StartRequest to order a start
  // after what was posted before it.
  std::expected<TaskId, StartError> Start(std::uint64_t request,
                                          std::unique_ptr<TaskProgram> program,
                                          std::size_t priority = 1);
  // Cancels the request's task tree, or trees if its tag was started again
  // while an earlier start drained; false if no such request is live.
  bool Cancel(std::uint64_t request);
  // Where a nonresident extent's contents are read from, into its backing.
  void SetSource(catalog::ExtentId extent, const providers::ReadSpec& source);
  // One turn; true if anything happened.
  bool Turn();
  // Turns until shutdown completes. Returns its result: a fault if work
  // could not be reconciled.
  std::expected<void, Fault> Run();
  // Once shutting down: nothing while work drains, then the result.
  std::optional<std::expected<void, Fault>> Stopped() const;

  const catalog::Catalog& catalog() const { return catalog_; }
  const TaskTable& tasks() const { return tasks_; }
  std::size_t operations() const { return board_.open(); }
  std::size_t quarantined() const { return quarantined_; }
  std::optional<Fault> fault() const { return fault_; }
  // The page-in in flight for an extent, if any.
  std::optional<OperationId> LoadOf(catalog::ExtentId extent) const {
    const auto found = loads_.find(extent);
    return found != loads_.end() ? std::optional(found->second) : std::nullopt;
  }
  const SchedulerStats& stats() const { return stats_; }

 private:
  friend class TaskContext;

  enum class Kind : std::uint8_t { kRead, kDevice, kCpu };
  static constexpr std::size_t kKinds = 3;

  struct TaskRecord {
    TaskId id;
    std::uint64_t request = 0;
    std::size_t priority = 0;
    std::unique_ptr<TaskProgram> program;
    std::uint32_t waiting = 0;  // page-ins, operations and children it waits for
    bool failed = false;        // a dependency failed since the program last looked
    bool finished = false;
  };

  struct Operation {
    OperationId id;
    Kind kind = Kind::kRead;
    bool published = false;  // a lane took its command
    bool quarantined = false;
    bool critical = false;
    // A page-in: its catalog load, source and waiters.
    catalog::Ticket ticket;
    providers::ReadSpec spec;
    std::vector<TaskId> waiters;
    bool cancelling = false;  // every waiter left
    // Device and CPU work: the task that owns it and the lease it holds.
    TaskId task;
    catalog::LeaseId lease;
    // The command, until a lane takes it.
    DeviceWork work;
    CpuCommand job;
  };

  TaskRecord* Record(TaskId task);
  Operation* Find(OperationId operation);
  // A live task's view; fatal for one that is not.
  TaskView ViewOf(TaskId task) const;

  // Moves from `program` only if the task is created.
  std::expected<TaskId, StartError> StartTask(std::uint64_t request,
                                              std::unique_ptr<TaskProgram>& program,
                                              std::size_t priority, std::optional<TaskId> parent);
  void Apply(Control&& control);
  void StepTask(TaskId task);
  void CancelTask(TaskId task);
  void FinishTask(TaskId task, TaskOutcome outcome);
  void TryRetire(TaskId task);
  void MakeReady(TaskRecord& record);
  void Wake(TaskId task, bool failed);
  void BeginStop();

  // TaskContext's work.
  std::expected<Readiness, WorkError> Materialize(TaskId task, const catalog::Closure& closure);
  std::expected<OperationId, WorkError> Submit(TaskId task, const catalog::Closure& closure,
                                               Kind kind, std::optional<DeviceWork> work,
                                               CpuJob job);
  std::expected<void, WorkError> Evict(catalog::ExtentId extent);

  // The operation lifecycle.
  Operation& Open(Kind kind);
  void Publish(Operation& operation);
  base::PushResult Push(Operation& operation) const;
  bool Flush();
  void Withdraw(TaskId task);
  void CancelLoad(Operation& operation);
  void RollBack(Operation& operation);
  void Observe(const Observation& seen);
  void Conclude(Operation& operation, Outcome outcome, std::uint64_t bytes);
  void Quarantine(Operation& operation, Fault fault);
  void SetCritical(Operation& operation, bool critical);

  catalog::Catalog& catalog_;
  CompletionBoard& board_;
  base::WakeFlag& wake_;
  Lanes lanes_;
  SchedulerSettings settings_;
  base::BoundedQueue<Control> controls_;
  // Requests with a cancellation queued and no start posted since: at most
  // one entry per queued control, allocated once. Held across a push, so
  // a push and its entry change together; taken after the queue's lock,
  // never inside it.
  std::mutex cancel_mutex_;
  std::vector<std::uint64_t> cancel_intents_;
  std::atomic<bool> shutdown_{false};

  // Everything below is the scheduler thread's.
  TaskTable tasks_;
  std::vector<std::optional<TaskRecord>> records_;    // by task index
  std::vector<std::optional<Operation>> operations_;  // by mailbox index
  std::map<catalog::ExtentId, OperationId> loads_;    // page-ins in flight
  std::map<catalog::ExtentId, providers::ReadSpec> sources_;
  // Commands a full lane refused, per lane, in publication order.
  std::array<std::deque<OperationId>, kKinds> pending_;
  // Page-in cancellations the storage lane refused, to ask again.
  std::vector<OperationId> cancels_;
  std::size_t critical_ = 0;
  std::size_t quarantined_ = 0;
  std::optional<Fault> fault_;
  bool stopping_ = false;
  std::uint64_t turn_ = 0;
  SchedulerStats stats_;
};

// What a task's step may do, on the scheduler thread.
class TaskContext {
 public:
  TaskContext(const TaskContext&) = delete;
  TaskContext& operator=(const TaskContext&) = delete;
  TaskContext(TaskContext&&) = delete;
  TaskContext& operator=(TaskContext&&) = delete;
  ~TaskContext() = default;

  TaskId task() const { return task_; }
  // The scheduler's turn counter, the logical tick of recorded uses.
  std::uint64_t tick() const { return scheduler_.turn_; }
  const catalog::Catalog& catalog() const { return scheduler_.catalog_; }
  // Whether a dependency (a page-in, an operation) failed since the last
  // call; clears it.
  bool TakeFailure();
  bool ChildFailed() const;

  std::expected<Readiness, WorkError> Materialize(const catalog::Closure& closure) {
    return scheduler_.Materialize(task_, closure);
  }
  // The operation's result wakes the task; a failure shows in TakeFailure.
  std::expected<OperationId, WorkError> SubmitDevice(const catalog::Closure& closure,
                                                     const DeviceWork& work) {
    return scheduler_.Submit(task_, closure, Scheduler::Kind::kDevice, work, {});
  }
  std::expected<OperationId, WorkError> SubmitCpu(const catalog::Closure& closure, CpuJob job) {
    return scheduler_.Submit(task_, closure, Scheduler::Kind::kCpu, std::nullopt, std::move(job));
  }
  // A child, ready at once; its finishing wakes this task.
  std::expected<TaskId, WorkError> Spawn(std::unique_ptr<TaskProgram> program,
                                         std::size_t priority = 1);
  // Evicts an unheld, evictable extent at once. Its backing stays mapped
  // until the memory manager releases backing through the device lane.
  std::expected<void, WorkError> Evict(catalog::ExtentId extent) {
    return scheduler_.Evict(extent);
  }

 private:
  friend class Scheduler;
  TaskContext(Scheduler& scheduler, TaskId task) : scheduler_(scheduler), task_(task) {}

  Scheduler& scheduler_;
  TaskId task_;
};

}  // namespace jitllm::scheduler

#endif  // JITLLM_SCHEDULER_SCHEDULER_H_
