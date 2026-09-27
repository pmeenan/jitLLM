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
//     source; a load in flight is joined, and so is an eviction in flight
//     (the task then materializes again). Page-ins belong to the
//     scheduler, not the task: one per extent, with a bounded waiter list.
//     A task that leaves (cancelled or finished) removes only its own
//     interest; when the last one leaves, the load starts no new stage,
//     asks the storage lane to cancel a read in flight, and still drains.
//     A page-in publishes resident contents only if its read transferred
//     the whole range and, when landed, its copy's fence has completed;
//     otherwise the load fails, and the backing is released only once
//     every stage's completion proves no further access.
//   - A page-in (docs/architecture.md#page-in-and-eviction-lifecycles-8)
//     runs in stages, each one operation with its own mailbox:
//       1. with managed backing (D-033), the device lane creates backing
//          in the source's allocation class, maps it at its place and sets
//          access;
//       2. a landed source (D-081) waits for a landing slot, in order;
//       3. the storage lane reads the file range with direct I/O, into the
//          slot, or for a direct source into the extent's own memory;
//       4. for a landed source, the device lane copies the slot into the
//          extent's device address on the zone's stream, and fences it;
//       5. the extent is published resident, and the slot freed, only once
//          that fence has completed.
//     A slot is reused only once the read into it and the copy out of it
//     have both been proven to touch it no more; one whose copy is
//     unproven is quarantined with the extent, never reused. At most twice
//     as many landed loads as the zone has slots are in flight at once
//     (the rest queue in order, unmapped), so backing is mapped at most
//     one zone ahead of the reads. A failed or withdrawn load unmaps and
//     releases the backing it mapped before the extent is nonresident
//     again. A stage that can never get a mailbox (every one retired or
//     held by quarantined work) quarantines its load instead of waiting.
//   - Evict an extent: with managed backing the device lane unmaps and
//     releases it (D-033) while the extent is EVICTING, and the task may
//     wait for it; otherwise the catalog alone records it. An unmap
//     refused with nothing changed leaves it resident; one of unknown
//     outcome, or refused because an earlier unknown outcome left its
//     place undetermined, quarantines it.
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
// Not yet here: write-back and spill of preserved contents (the reverse
// path through the zone), the D-033 handoff of a victim's backing to a
// load, victim selection on a miss, admission's envelopes
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
  kUnavailable,  // quarantined, no registered source, unknown, or no identity will ever come
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
  kBacking,        // backing could not be unmapped or released as the catalog needs
  kExhausted,      // a page-in stage can never get an identity: all retired or quarantined
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

// Managed backing (D-033): where the scheduler maps an extent's backing
// when it loads it, and unmaps and releases it when it evicts it.
struct BackingPlace {
  providers::ReservationId reservation;
  Bytes offset;
  Bytes size;  // the extent's backing, a multiple of the class's granularity
  std::size_t allocation_class = 0;
  bool operator==(const BackingPlace&) const = default;
};

// Where a nonresident extent's contents come from, and how they arrive.
struct PageSource {
  // The file range. A direct read lands at `read.memory`, which the CPU
  // and the storage provider reach (host backing, D-034); a landed read
  // ignores it.
  providers::ReadSpec read;
  // Landed (D-081): the read lands in a landing slot and the device lane
  // copies it to `destination`, the extent's device address.
  bool landed = false;
  std::uint64_t destination = 0;
  // Managed backing; without it the backing stays mapped at the
  // destination for as long as the source is registered.
  std::optional<BackingPlace> backing;
};

// The landing zone (D-081): a persistent pool of `slots` host-VMM
// addresses of `slot_bytes` each, mapped read-write for the CPU and the
// device. The program maps it and registers it in the catalog before the
// scheduler starts, and keeps it until the lanes have drained. Landed
// copies run on the device lane's stream `stream`.
struct LandingZone {
  std::vector<std::uint64_t> slots = {};  // NOLINT(readability-redundant-member-init)
  Bytes slot_bytes = {};                  // NOLINT(readability-redundant-member-init)
  std::uint32_t stream = 0;
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
  // Where landed page-ins land; none by default (a landed source is then
  // refused).
  LandingZone landing = {};  // NOLINT(readability-redundant-member-init)
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
            const SchedulerSettings& settings);
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
  // The scheduler thread, or before it runs (a task's step may call these).
  // Where a nonresident extent's contents are read from, directly into its
  // backing, which stays mapped: the PageSource below with neither a zone
  // nor managed backing, whose rules it must pass (fatal otherwise).
  void SetSource(catalog::ExtentId extent, const providers::ReadSpec& source);
  // Where an extent's contents come from and how they arrive (PageSource).
  // Refused (kBusy) while a load or eviction of it is in flight, or while
  // it is resident with backing mapped somewhere else, since its unmap
  // must find what its load mapped; kInvalid for a landed source with no
  // landing zone, a range that is empty or exceeds a slot, or no
  // destination. Registering a new place for a nonresident extent
  // relocates its next load (BP-P5).
  std::expected<void, WorkError> SetSource(catalog::ExtentId extent, const PageSource& source);
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
  // The operation of the page-in stage in flight for an extent, if any.
  std::optional<OperationId> LoadOf(catalog::ExtentId extent) const {
    const auto found = loads_.find(extent);
    return found != loads_.end() && found->second.operation.valid()
               ? std::optional(found->second.operation)
               : std::nullopt;
  }
  // Page-ins in flight, in any stage.
  std::size_t loads() const { return loads_.size(); }
  // Landing slots in use by a read or copy, and quarantined ones.
  std::size_t slots_busy() const;
  std::size_t slots_quarantined() const;
  // Landed loads whose copy into device memory a lane has taken and whose
  // fence has not yet been seen.
  std::size_t copying() const;
  const SchedulerStats& stats() const { return stats_; }

 private:
  friend class TaskContext;

  // What an operation is: a stage of a page-in, the unmap of an eviction,
  // or a task's device or CPU work.
  enum class Kind : std::uint8_t { kLoad, kEvict, kDevice, kCpu };
  // The lane an operation's command goes to.
  enum class Route : std::uint8_t { kStorage, kDevice, kCpu };
  static constexpr std::size_t kRoutes = 3;

  // A page-in's stages (the header's list).
  enum class Stage : std::uint8_t {
    kQueued,     // waiting for room among the landed loads in flight
    kMapping,    // 1: creating and mapping managed backing
    kSlot,       // 2: waiting for a landing slot
    kReading,    // 3
    kCopying,    // 4
    kUnmapping,  // unwinding a failed or withdrawn load's backing
  };
  enum class SlotState : std::uint8_t { kFree, kBusy, kQuarantined };

  // A page-in, the scheduler's: one per extent, with its waiters.
  struct Load {
    catalog::Ticket ticket;
    PageSource source;
    Stage stage = Stage::kQueued;
    OperationId operation;  // the stage's, while one is open
    std::optional<std::size_t> slot;
    std::vector<TaskId> waiters;
    bool cancelling = false;  // every waiter left
    bool started = false;     // counted among the landed loads in flight
    bool mapped = false;      // its managed backing is mapped
    bool failed = false;      // it will not publish
  };
  // An eviction whose backing the device lane unmaps and releases.
  struct Eviction {
    catalog::Ticket ticket;
    OperationId operation;
    TaskId evictor;               // the task that asked for it
    std::vector<TaskId> waiters;  // it, and tasks materializing the extent
  };

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
    Kind kind = Kind::kLoad;
    Route route = Route::kStorage;
    bool published = false;  // a lane took its command
    bool quarantined = false;
    bool critical = false;
    // A page-in stage or an eviction: the extent.
    catalog::ExtentId extent;
    // Device and CPU work: the task that owns it and the lease it holds.
    TaskId task;
    catalog::LeaseId lease;
    // The command, until a lane takes it.
    ReadCommand read;
    DeviceCommand device;
    CpuCommand job;
  };

  static WorkError ErrorOf(catalog::CatalogError error);
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
  // Moves from `device` or `job`, whichever `kind` names, only if the
  // operation is created.
  std::expected<OperationId, WorkError> Submit(TaskId task, const catalog::Closure& closure,
                                               Kind kind, DeviceCommand& device, CpuJob& job);
  std::expected<Readiness, WorkError> Evict(TaskId task, catalog::ExtentId extent);

  // Page-ins (pagein.cc).
  std::expected<void, WorkError> BeginPageIn(TaskRecord& record, catalog::ExtentId extent,
                                             const PageSource& source);
  void Join(Load& load, TaskRecord& record);
  // Runs the load's next stage, from where it stands.
  void Proceed(catalog::ExtentId extent);
  // Opens and publishes a stage's operation; false if no mailbox is free
  // (the load waits in blocked_ and is retried each turn).
  bool OpenStage(catalog::ExtentId extent, Load& load);
  void OnStage(catalog::ExtentId extent, Outcome outcome, std::uint64_t bytes);
  void Unwind(catalog::ExtentId extent, Load& load);
  void EndLoad(catalog::ExtentId extent, bool loaded);
  void QuarantineLoad(catalog::ExtentId extent, Fault fault);
  void CancelPageIn(catalog::ExtentId extent, Load& load);
  void FreeSlot(std::size_t slot);
  void StartQueued();
  bool RetryBlocked();
  // Every mailbox is retired or held by quarantined work: none will free.
  bool NoMailboxEver() const;
  // Evictions.
  void OnEvicted(catalog::ExtentId extent, Outcome outcome);
  void QuarantineEvicting(catalog::ExtentId extent, Fault fault);
  void Fail(Fault fault);

  // The operation lifecycle.
  Operation& Open(Kind kind);
  void Publish(Operation& operation);
  base::PushResult Push(Operation& operation) const;
  bool Flush();
  void Withdraw(TaskId task);
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
  std::map<catalog::ExtentId, Load> loads_;           // page-ins in flight
  std::map<catalog::ExtentId, Eviction> evictions_;
  std::map<catalog::ExtentId, PageSource> sources_;
  // The landing zone's slots, and the loads waiting: to start (kQueued),
  // for a slot (kSlot), or for a mailbox to open their next stage.
  std::vector<SlotState> slots_;
  std::size_t started_ = 0;  // landed loads past kQueued
  std::deque<catalog::ExtentId> queued_;
  std::deque<catalog::ExtentId> slot_waiters_;
  std::deque<catalog::ExtentId> blocked_;
  // Commands a full lane refused, per lane, in publication order.
  std::array<std::deque<OperationId>, kRoutes> pending_;
  // Page-in cancellations the storage lane refused, to ask again.
  std::vector<OperationId> cancels_;
  std::size_t critical_ = 0;
  std::size_t quarantined_ = 0;  // quarantined operations, each keeping its mailbox
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
    DeviceCommand device{.operation = {}, .work = work};
    CpuJob none;
    return scheduler_.Submit(task_, closure, Scheduler::Kind::kDevice, device, none);
  }
  // A job that queues kernel work on one of the device lane's streams
  // (commands.h): it runs on the submission lane, and the operation's
  // lease holds the closure until the fence after it completes.
  std::expected<OperationId, WorkError> SubmitLaunch(const catalog::Closure& closure,
                                                     LaunchWork work) {
    DeviceCommand device{.operation = {}, .work = std::move(work)};
    CpuJob none;
    return scheduler_.Submit(task_, closure, Scheduler::Kind::kDevice, device, none);
  }
  std::expected<OperationId, WorkError> SubmitCpu(const catalog::Closure& closure, CpuJob job) {
    DeviceCommand none;
    return scheduler_.Submit(task_, closure, Scheduler::Kind::kCpu, none, job);
  }
  // A child, ready at once; its finishing wakes this task.
  std::expected<TaskId, WorkError> Spawn(std::unique_ptr<TaskProgram> program,
                                         std::size_t priority = 1);
  // Evicts an unheld, evictable extent. With managed backing the device
  // lane unmaps and releases it first: kWaiting, and the task is woken
  // once the extent is nonresident (or, if the unmap failed, with a
  // failure). Otherwise at once, in the catalog only: its backing stays
  // mapped, as its source registered it.
  std::expected<Readiness, WorkError> Evict(catalog::ExtentId extent) {
    return scheduler_.Evict(task_, extent);
  }

 private:
  friend class Scheduler;
  TaskContext(Scheduler& scheduler, TaskId task) : scheduler_(scheduler), task_(task) {}

  Scheduler& scheduler_;
  TaskId task_;
};

}  // namespace jitllm::scheduler

#endif  // JITLLM_SCHEDULER_SCHEDULER_H_
