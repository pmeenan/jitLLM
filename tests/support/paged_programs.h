// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The task programs the paged harnesses post to the scheduler
// (paged_node.h; benchmarks/fp16_runner.cc, exl3_runner.cc and
// alternate_paged.cc) and the tests that check them. Each tells a Done
// what became of it; its destructor is its last touch of that Done, so the
// thread that posted it may end the Done's frame once `gone` is set.
//
// AcquireProgram is BP-S3's (docs/backend-proof.md): it makes room for a
// closure under the execution budget B with the memory module's planning
// (memory/materialize.h), evicting the victims it chooses, which are never
// the closure's own extents or the ones it protects, and then materializes
// the closure. With two models in one catalog, the victims are the other
// model's clean weights.

#ifndef JITLLM_TESTS_SUPPORT_PAGED_PROGRAMS_H_
#define JITLLM_TESTS_SUPPORT_PAGED_PROGRAMS_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"
#include "memory/materialize.h"
#include "scheduler/scheduler.h"

namespace jitllm::test_support {

// What a program tells the thread that posted it.
struct Done {
  std::atomic<int> outcome{-1};
  std::atomic<bool> retired{false};
  std::atomic<int> error{-1};  // a WorkError that ended it
  // The program is destroyed: retired, or never admitted (a refused start
  // goes with its control). Its last touch of `done`: a job it submitted
  // retired first (its lease held until its fence), and one never
  // submitted went with it.
  std::atomic<bool> gone{false};
};

// A program that tells `done` how it ended and when it is destroyed.
class HarnessProgram : public scheduler::TaskProgram {
 public:
  explicit HarnessProgram(Done& done) : done_(done) {}
  HarnessProgram(const HarnessProgram&) = delete;
  HarnessProgram& operator=(const HarnessProgram&) = delete;
  HarnessProgram(HarnessProgram&&) = delete;
  HarnessProgram& operator=(HarnessProgram&&) = delete;
  ~HarnessProgram() override { done_.gone.store(true); }

  void Finished(scheduler::TaskOutcome outcome) override {
    done_.outcome.store(static_cast<int>(outcome));
  }
  void Retired() override { done_.retired.store(true); }

 protected:
  scheduler::Step Fail(scheduler::WorkError error) {
    done_.error.store(static_cast<int>(error));
    return scheduler::Step::Finish(scheduler::TaskOutcome::kFailed);
  }

  Done& done_;
};

// Materializes a closure, then optionally runs one device job over it on
// the device lane's stream `stream`.
class RunProgram final : public HarnessProgram {
 public:
  RunProgram(Done& done, catalog::Closure closure, scheduler::DeviceJob job,
             std::uint32_t stream = 0)
      : HarnessProgram(done), closure_(std::move(closure)), job_(std::move(job)), stream_(stream) {}

  scheduler::Step Advance(scheduler::TaskContext& context) override {
    if (context.TakeFailure()) {
      return scheduler::Step::Finish(scheduler::TaskOutcome::kFailed);
    }
    if (submitted_) {
      return scheduler::Step::Finish(scheduler::TaskOutcome::kSucceeded);
    }
    const auto ready = context.Materialize(closure_);
    if (!ready) {
      if (ready.error() == scheduler::WorkError::kBusy) {
        return scheduler::Step::Yield();  // no mailbox now: again next turn
      }
      return Fail(ready.error());
    }
    if (*ready == scheduler::Readiness::kWaiting) {
      return scheduler::Step::Wait();
    }
    if (!job_) {
      return scheduler::Step::Finish(scheduler::TaskOutcome::kSucceeded);
    }
    // A refused launch is left in `work`.
    scheduler::LaunchWork work{.stream = stream_, .job = std::move(job_)};
    const auto submitted = context.SubmitLaunch(closure_, std::move(work));
    if (!submitted) {
      if (submitted.error() == scheduler::WorkError::kBusy) {
        // No mailbox now (other work holds them): the job, untouched,
        // again next turn.
        // NOLINTNEXTLINE(bugprone-use-after-move): SubmitLaunch leaves a refused launch in work
        job_ = std::move(work.job);
        return scheduler::Step::Yield();
      }
      return Fail(submitted.error());
    }
    submitted_ = true;
    return scheduler::Step::Wait();
  }

 private:
  catalog::Closure closure_;
  scheduler::DeviceJob job_;
  std::uint32_t stream_;
  bool submitted_ = false;
};

// Runs a call on the scheduler thread, which alone may change the
// scheduler's records (registering sources anew) or read the catalog once
// the scheduler runs.
class CallProgram final : public HarnessProgram {
 public:
  CallProgram(Done& done, std::function<std::expected<void, std::string>()> call)
      : HarnessProgram(done), call_(std::move(call)) {}
  scheduler::Step Advance(scheduler::TaskContext& /*context*/) override {
    status_ = call_();
    return scheduler::Step::Finish(status_ ? scheduler::TaskOutcome::kSucceeded
                                           : scheduler::TaskOutcome::kFailed);
  }
  const std::expected<void, std::string>& status() const { return status_; }

 private:
  std::function<std::expected<void, std::string>()> call_;
  std::expected<void, std::string> status_;
};

// Evicts extents in order, up to kWindow at once (each unmap, and
// write-back, in flight together), then waits for them all before the
// next; those not resident are passed over. With a handoff
// (scheduler::EvictOptions) each eviction keeps its managed backing
// parked, and a load this task starts may take it; whatever no load took
// is released when the task finishes.
class EvictingProgram : public HarnessProgram {
 public:
  static constexpr std::size_t kWindow = 256;

  EvictingProgram(Done& done, std::vector<catalog::ExtentId> extents,
                  scheduler::EvictOptions options)
      : HarnessProgram(done), extents_(std::move(extents)), options_(options) {}

 protected:
  // One round of evictions: nullopt once every one has ended, else the
  // step to return.
  std::optional<scheduler::Step> EvictRound(scheduler::TaskContext& context) {
    std::size_t issued = 0;
    while (next_ < extents_.size() && issued < kWindow) {
      const catalog::ExtentId extent = extents_[next_];
      if (context.catalog().Describe(extent).value().state != catalog::ExtentState::kResident) {
        ++next_;
        continue;
      }
      const auto evicted = context.Evict(extent, options_);
      if (!evicted) {
        if (evicted.error() == scheduler::WorkError::kBusy) {
          // No mailbox now: wait for those in flight, or try next turn.
          return issued > 0 ? scheduler::Step::Wait() : scheduler::Step::Yield();
        }
        return Fail(evicted.error());
      }
      ++next_;
      ++evicted_;
      if (*evicted == scheduler::Readiness::kWaiting) {
        ++issued;
      }
    }
    if (issued > 0) {
      return scheduler::Step::Wait();
    }
    return std::nullopt;
  }
  std::uint64_t evicted() const { return evicted_; }

 private:
  std::vector<catalog::ExtentId> extents_;
  scheduler::EvictOptions options_;
  std::size_t next_ = 0;
  std::uint64_t evicted_ = 0;
};

class EvictProgram final : public EvictingProgram {
 public:
  EvictProgram(Done& done, std::vector<catalog::ExtentId> extents,
               scheduler::EvictOptions options = {})
      : EvictingProgram(done, std::move(extents), options) {}
  scheduler::Step Advance(scheduler::TaskContext& context) override {
    if (context.TakeFailure()) {
      return scheduler::Step::Finish(scheduler::TaskOutcome::kFailed);
    }
    if (auto step = EvictRound(context)) {
      return *step;
    }
    return scheduler::Step::Finish(scheduler::TaskOutcome::kSucceeded);
  }
};

// What a swap did, and when (steady clock, on the scheduler thread): read
// by the poster once the Done is gone.
struct SwapReport {
  std::chrono::steady_clock::time_point started;
  std::chrono::steady_clock::time_point evicted;  // every eviction ended (or parked)
  std::chrono::steady_clock::time_point loaded;   // the incoming closure resident
  std::uint64_t evictions = 0;                    // extents resident when it began
  std::uint64_t loads = 0;                        // closure extents nonresident then
};

// A full swap (M3): evicts the outgoing extents (write-back first for live
// state at a write-back place), then materializes the incoming closure.
// With a handoff the evicted backing is parked, the incoming loads take it
// instead of creating their own (scheduler.h), and what they leave is
// released when this task finishes, after `loaded`. Nothing is leased.
class SwapProgram final : public EvictingProgram {
 public:
  SwapProgram(Done& done, std::vector<catalog::ExtentId> out, catalog::Closure in, bool handoff,
              SwapReport& report)
      : EvictingProgram(done, std::move(out), scheduler::EvictOptions{.handoff = handoff}),
        in_(std::move(in)),
        report_(report) {}

  scheduler::Step Advance(scheduler::TaskContext& context) override {
    if (context.TakeFailure()) {
      return scheduler::Step::Finish(scheduler::TaskOutcome::kFailed);
    }
    if (!started_) {
      started_ = true;
      report_.started = std::chrono::steady_clock::now();
    }
    if (!evicted_all_) {
      if (auto step = EvictRound(context)) {
        return *step;
      }
      evicted_all_ = true;
      report_.evicted = std::chrono::steady_clock::now();
      report_.evictions = evicted();
      for (const auto& [extent, generation] : in_.extents) {
        report_.loads +=
            context.catalog().Describe(extent).value().state == catalog::ExtentState::kResident ? 0
                                                                                                : 1;
      }
    }
    const auto ready = context.Materialize(in_);
    if (!ready) {
      if (ready.error() == scheduler::WorkError::kBusy) {
        return scheduler::Step::Yield();
      }
      return Fail(ready.error());
    }
    if (*ready == scheduler::Readiness::kWaiting) {
      return scheduler::Step::Wait();
    }
    report_.loaded = std::chrono::steady_clock::now();
    return scheduler::Step::Finish(scheduler::TaskOutcome::kSucceeded);
  }

 private:
  catalog::Closure in_;
  SwapReport& report_;
  bool started_ = false;
  bool evicted_all_ = false;
};

// Submits a job over a closure without materializing it first: with any
// extent nonresident the lease, and so the submission, must be refused
// before the job can run (BP-P2's incomplete closure).
class LaunchOnlyProgram final : public HarnessProgram {
 public:
  LaunchOnlyProgram(Done& done, catalog::Closure closure, std::atomic<bool>& ran,
                    std::uint32_t stream = 0)
      : HarnessProgram(done), closure_(std::move(closure)), ran_(ran), stream_(stream) {}
  scheduler::Step Advance(scheduler::TaskContext& context) override {
    if (submitted_) {
      return scheduler::Step::Finish(context.TakeFailure() ? scheduler::TaskOutcome::kFailed
                                                           : scheduler::TaskOutcome::kSucceeded);
    }
    std::atomic<bool>& ran = ran_;
    const auto submitted = context.SubmitLaunch(
        closure_, scheduler::LaunchWork{.stream = stream_, .job = [&ran](providers::NativeStream) {
                                          ran.store(true);
                                          return scheduler::JobResult::kQueued;
                                        }});
    if (!submitted) {
      return Fail(submitted.error());
    }
    submitted_ = true;
    return scheduler::Step::Wait();
  }

 private:
  catalog::Closure closure_;
  std::atomic<bool>& ran_;
  std::uint32_t stream_;
  bool submitted_ = false;
};

// What an acquisition did: the victims it evicted, in order, and how many
// of the closure's extents it found nonresident (and so loaded). Written on
// the scheduler thread; read by the poster once the Done is gone.
struct AcquireReport {
  std::vector<catalog::ExtentId> evicted;
  std::uint64_t loaded = 0;
  std::uint64_t plans = 0;  // materialization plans made
};

// Makes room for a closure under the budget B, then materializes it. Each
// round plans the closure's materialization in `domain`
// (memory::PlanMaterialization: the victims eligible for the shortfall,
// never the closure's own extents or `protect`), evicts every victim, each
// eviction waited for, and plans again, until the closure fits. Eligible
// means resident and unleased (Catalog::Evictable), and discardable
// scratch comes first: the node protects its shared workspace, which no
// source can restore, even from a closure that omits it. A closure that
// cannot fit (the eligible victims do not cover the shortfall), or whose
// extents are stale or quarantined, fails with kOverBudget or kUnavailable
// and evicts nothing more. Nothing is leased: a job that follows
// materializes the closure again and takes its own lease (RunProgram).
class AcquireProgram final : public HarnessProgram {
 public:
  // Rounds of planning before giving up: each round's victims cover its
  // shortfall, so one round suffices unless an eviction is abandoned.
  static constexpr std::uint64_t kRounds = 8;

  AcquireProgram(Done& done, catalog::Closure closure, catalog::DomainId domain, base::Bytes budget,
                 AcquireReport& report, std::vector<catalog::ExtentId> protect = {})
      : HarnessProgram(done),
        closure_(std::move(closure)),
        domain_(domain),
        budget_(budget),
        report_(report),
        protect_(std::move(protect)) {}

  scheduler::Step Advance(scheduler::TaskContext& context) override {
    if (context.TakeFailure()) {
      return scheduler::Step::Finish(scheduler::TaskOutcome::kFailed);
    }
    while (!materializing_) {
      while (next_ < victims_.size()) {
        const catalog::ExtentId extent = victims_[next_];
        const auto view = context.catalog().Describe(extent);
        if (!view || view->state != catalog::ExtentState::kResident) {
          ++next_;  // gone already (an eviction this program waited for)
          continue;
        }
        const auto evicted = context.Evict(extent);
        if (!evicted) {
          if (evicted.error() == scheduler::WorkError::kBusy) {
            return scheduler::Step::Yield();
          }
          return Fail(evicted.error());
        }
        report_.evicted.push_back(extent);
        ++next_;
        if (*evicted == scheduler::Readiness::kWaiting) {
          return scheduler::Step::Wait();
        }
      }
      const memory::MaterializationPlan plan =
          memory::PlanMaterialization(context.catalog(), domain_, budget_, closure_, protect_);
      ++report_.plans;
      if (!plan.feasible) {
        return Fail(plan.victims.sufficient || plan.shortfall.value() == 0
                        ? scheduler::WorkError::kUnavailable
                        : scheduler::WorkError::kOverBudget);
      }
      if (plan.shortfall.value() == 0) {
        materializing_ = true;
        report_.loaded = plan.load.size();
        break;
      }
      if (report_.plans > kRounds) {
        return Fail(scheduler::WorkError::kOverBudget);
      }
      victims_.clear();
      for (const memory::Victim& victim : plan.victims.victims) {
        victims_.push_back(victim.extent);
      }
      next_ = 0;
    }
    const auto ready = context.Materialize(closure_);
    if (!ready) {
      if (ready.error() == scheduler::WorkError::kBusy) {
        return scheduler::Step::Yield();
      }
      return Fail(ready.error());
    }
    if (*ready == scheduler::Readiness::kWaiting) {
      return scheduler::Step::Wait();
    }
    return scheduler::Step::Finish(scheduler::TaskOutcome::kSucceeded);
  }

 private:
  catalog::Closure closure_;
  catalog::DomainId domain_;
  base::Bytes budget_;
  AcquireReport& report_;
  std::vector<catalog::ExtentId> protect_;
  std::vector<catalog::ExtentId> victims_;
  std::size_t next_ = 0;
  bool materializing_ = false;
};

}  // namespace jitllm::test_support

#endif  // JITLLM_TESTS_SUPPORT_PAGED_PROGRAMS_H_
