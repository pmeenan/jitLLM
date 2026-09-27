// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The scheduler's D-081 page-in path and its evictions on the deterministic
// fakes (scheduler.h; D-033, D-048, D-081): managed backing created and
// mapped on the device lane, direct reads into a bounded landing zone, the
// copy into device backing, publication only after the copy's fence, and
// slots reused only after it. Failed and short reads, backing failures,
// cancellation in every stage, a full zone, unproven copies and unmaps,
// evictions that really release backing, relocation, and kernel jobs whose
// lease holds until their fence. The deterministic tests turn each lane
// themselves; the threaded one runs every lane on its own thread.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include "base/bounded_queue.h"
#include "base/bytes.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "expected_error.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/direct_reader.h"
#include "providers/fake/fake_device_execution.h"
#include "providers/fake/fake_device_memory.h"
#include "providers/fake/fake_storage.h"
#include "providers/storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/scheduler.h"
#include "scheduler/services.h"
#include "scheduler/tasks.h"

namespace {

using jitllm::base::Bytes;
using jitllm::base::PushResult;
using jitllm::catalog::Closure;
using jitllm::catalog::ExtentId;
using jitllm::catalog::ExtentState;
using jitllm::catalog::ExtentView;
using jitllm::catalog::Occupancy;
using jitllm::providers::Access;
using jitllm::providers::ProviderError;
using jitllm::providers::ReaderSettings;
using jitllm::providers::ReadSpec;
using jitllm::providers::ReservationId;
using jitllm::providers::StreamId;
using jitllm::providers::Submission;
using jitllm::providers::fake::FakeDeviceExecution;
using jitllm::providers::fake::FakeDeviceMemory;
using jitllm::providers::fake::FakeStorage;
using jitllm::providers::fake::kPoison;
using jitllm::scheduler::BackingPlace;
using jitllm::scheduler::CancelRequest;
using jitllm::scheduler::CompletionBoard;
using jitllm::scheduler::Control;
using jitllm::scheduler::DeviceService;
using jitllm::scheduler::DeviceSettings;
using jitllm::scheduler::Fault;
using jitllm::scheduler::JobResult;
using jitllm::scheduler::LandingZone;
using jitllm::scheduler::Lanes;
using jitllm::scheduler::LaunchWork;
using jitllm::scheduler::PageSource;
using jitllm::scheduler::QueueSettings;
using jitllm::scheduler::Readiness;
using jitllm::scheduler::Scheduler;
using jitllm::scheduler::SchedulerSettings;
using jitllm::scheduler::StartRequest;
using jitllm::scheduler::Step;
using jitllm::scheduler::StorageService;
using jitllm::scheduler::TaskContext;
using jitllm::scheduler::TaskOutcome;
using jitllm::scheduler::TaskProgram;
using jitllm::scheduler::WorkError;
using jitllm::test_support::Failed;

constexpr std::uint64_t kSize = 64ULL * 1024;  // one extent, one slot
constexpr std::size_t kExtents = 6;
constexpr std::size_t kSlots = 2;
// The last extent's range is shorter than its backing, as a group's last
// chunk is.
constexpr std::uint64_t kTail = kSize - 8192;
constexpr auto kPatience = std::chrono::seconds(120);

std::byte* At(std::uint64_t address) {
  return reinterpret_cast<std::byte*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t LengthOf(std::size_t extent) { return extent + 1 == kExtents ? kTail : kSize; }

// Materializes a closure, then optionally holds it with a kernel job.
class LoadProgram final : public TaskProgram {
 public:
  struct Report {
    std::optional<TaskOutcome> outcome;
    bool retired = false;
    std::optional<WorkError> error;
    int job_runs = 0;
  };
  LoadProgram(Report& report, Closure closure, std::optional<JobResult> job = std::nullopt)
      : report_(report), closure_(std::move(closure)), job_(job) {}

  Step Advance(TaskContext& context) override {
    if (context.TakeFailure()) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    if (!loaded_) {
      const auto ready = context.Materialize(closure_);
      if (!ready) {
        report_.error = ready.error();
        return Step::Finish(TaskOutcome::kFailed);
      }
      if (*ready == Readiness::kWaiting) {
        return Step::Wait();
      }
      loaded_ = true;
      if (job_) {
        const JobResult result = *job_;
        Report& report = report_;
        const auto submitted = context.SubmitLaunch(
            closure_, LaunchWork{.stream = 0, .job = [result, &report](auto /*stream*/) {
                                   ++report.job_runs;
                                   return result;
                                 }});
        if (!submitted) {
          report_.error = submitted.error();
          return Step::Finish(TaskOutcome::kFailed);
        }
        return Step::Wait();
      }
    }
    return Step::Finish(TaskOutcome::kSucceeded);
  }
  void Finished(TaskOutcome outcome) override { report_.outcome = outcome; }
  void Retired() override { report_.retired = true; }

 private:
  Report& report_;
  Closure closure_;
  std::optional<JobResult> job_;
  bool loaded_ = false;
};

// Evicts extents, waiting for each unmap.
class EvictProgram final : public TaskProgram {
 public:
  struct Report {
    std::optional<TaskOutcome> outcome;
    std::vector<std::expected<Readiness, WorkError>> results;
  };
  EvictProgram(Report& report, std::vector<ExtentId> extents)
      : report_(report), extents_(std::move(extents)) {}

  Step Advance(TaskContext& context) override {
    if (context.TakeFailure()) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    if (next_ < extents_.size()) {
      const auto result = context.Evict(extents_[next_++]);
      report_.results.push_back(result);
      if (!result) {
        return Step::Finish(TaskOutcome::kFailed);
      }
      return *result == Readiness::kWaiting ? Step::Wait() : Step::Yield();
    }
    return Step::Finish(TaskOutcome::kSucceeded);
  }
  void Finished(TaskOutcome outcome) override { report_.outcome = outcome; }

 private:
  Report& report_;
  std::vector<ExtentId> extents_;
  std::size_t next_ = 0;
};

class PageInTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // The zone: host backing the CPU and the device reach.
    zone_ = memory_.Reserve(Bytes(kSize * kSlots)).value();
    zone_backing_ = memory_.Create(kHostClass, Bytes(kSize * kSlots)).value();
    ASSERT_TRUE(memory_.Map(zone_, Bytes(0), zone_backing_).has_value());
    ASSERT_TRUE(
        memory_.SetAccess(zone_, Bytes(0), Bytes(kSize * kSlots), Access::kReadWrite).has_value());
    // Two places for the weights, the second for relocation: address
    // space only, until a load maps backing.
    weights_ = memory_.Reserve(Bytes(kSize * kExtents)).value();
    moved_ = memory_.Reserve(Bytes(kSize * kExtents)).value();
    file_.resize(kSize * kExtents);
    for (std::size_t i = 0; i < file_.size(); ++i) {
      file_[i] = static_cast<std::byte>((i * 131) + (i >> 16) + 1);
    }
    fd_ = storage_.AddFile(file_);
    domain_ = catalog_.AddDomain("node");
    for (std::size_t i = 0; i < kExtents; ++i) {
      extents_.push_back(
          catalog_
              .AddExtent(
                  {.domain = domain_,
                   .memory_class = jitllm::catalog::MemoryClass::kWeights,
                   .recovery = jitllm::catalog::Recovery::kFromArtifact,
                   .size = Bytes(kSize),
                   .content = {.artifact = {}, .group = 0, .chunk = static_cast<std::uint32_t>(i)}})
              .value());
    }
    stream_ = execution_.CreateStream().value();
    baseline_ = memory_.backings();
  }

  void Build(std::size_t slots = kSlots, std::size_t board = 32) {
    board_ = std::make_unique<CompletionBoard>(board, wake_);
    storage_lane_ = std::make_unique<StorageService>(
        storage_,
        ReaderSettings{
            .alignment = 4096, .request_bytes = 16 * 1024, .retries = 2, .reads = 32, .waiters = 8},
        *board_, QueueSettings{.capacity = 16, .reserved = 4, .batch = 16});
    device_lane_ = std::make_unique<DeviceService>(
        execution_, std::span<const StreamId>(&stream_, 1), *board_,
        DeviceSettings{.queue = {.capacity = 16, .reserved = 4, .batch = 16},
                       .handoff = 16,
                       .poll_sleep = std::chrono::microseconds(0)},
        &memory_);
    LandingZone landing{.slots = {}, .slot_bytes = Bytes(kSize), .stream = 0};
    for (std::size_t i = 0; i < slots; ++i) {
      landing.slots.push_back(Slot(i));
    }
    scheduler_ = std::make_unique<Scheduler>(
        catalog_, *board_, wake_,
        Lanes{.storage = storage_lane_.get(), .device = device_lane_.get(), .cpu = nullptr},
        SchedulerSettings{.tasks = 16,
                          .priorities = 2,
                          .aging_limit = 4,
                          .controls = 64,
                          .controls_reserved = 16,
                          .controls_per_turn = 16,
                          .observations_per_turn = 64,
                          .steps_per_turn = 16,
                          .waiters = 8,
                          .budget = Bytes(kSize * 64),
                          .poll_window = std::chrono::microseconds(200),
                          .tick = std::chrono::milliseconds(100),
                          .landing = landing});
    for (std::size_t i = 0; i < kExtents; ++i) {
      ASSERT_TRUE(scheduler_->SetSource(extents_[i], Source(i, weights_)).has_value());
    }
  }

  void TearDown() override {
    ReleaseReads();
    if (scheduler_ != nullptr) {
      // Evict what is resident, so every backing is released, then stop.
      EvictProgram::Report report;
      std::vector<ExtentId> resident;
      for (const ExtentId extent : extents_) {
        if (View(extent).state == ExtentState::kResident && View(extent).leases == 0) {
          resident.push_back(extent);
        }
      }
      if (!scheduler_->fault() &&
          scheduler_->Start(999, std::make_unique<EvictProgram>(report, resident)).has_value()) {
        Settle();
      }
      scheduler_->RequestShutdown();
      const auto give_up = std::chrono::steady_clock::now() + kPatience;
      while (!scheduler_->Stopped() && std::chrono::steady_clock::now() < give_up) {
        if (!Round()) {
          std::this_thread::yield();
        }
      }
      EXPECT_TRUE(scheduler_->Stopped().has_value());
      storage_lane_->Close();
      device_lane_->Close();
      for (int i = 0; i < 100; ++i) {
        (void)storage_lane_->Turn(false);
        (void)device_lane_->SubmissionTurn();
        execution_.Drain();
        (void)device_lane_->CompletionTurn();
      }
    }
  }

  std::uint64_t Slot(std::size_t i) const {
    return memory_.RangeOf(zone_).value().base + (i * kSize);
  }
  std::uint64_t Place(ReservationId reservation, std::size_t i) const {
    return memory_.RangeOf(reservation).value().base + (i * kSize);
  }
  PageSource Source(std::size_t i, ReservationId reservation) const {
    return PageSource{
        .read = ReadSpec{.fd = fd_, .offset = i * kSize, .memory = nullptr, .length = LengthOf(i)},
        .landed = true,
        .destination = Place(reservation, i),
        .backing = BackingPlace{.reservation = reservation,
                                .offset = Bytes(i * kSize),
                                .size = Bytes(kSize),
                                .allocation_class = kDeviceClass}};
  }

  // One turn of each lane and of the scheduler, in a fixed order. With
  // `device`, the fake device runs what is queued on its streams.
  bool Round(bool device = true) {
    bool progress = storage_lane_->Turn(false);
    progress = device_lane_->SubmissionTurn() || progress;
    if (device) {
      execution_.Drain();
    }
    progress = device_lane_->CompletionTurn() || progress;
    return scheduler_->Turn() || progress;
  }
  void Settle(bool device = true) {
    for (int i = 0; i < 1000 && Round(device); ++i) {
    }
  }
  void HoldNext(int requests) {
    for (int i = 0; i < requests; ++i) {
      storage_.ScriptNext(
          {.submission = Submission::kAccepted, .result = std::nullopt, .hold = true});
    }
  }
  void ReleaseReads() {
    std::vector<std::uint64_t> tokens;
    for (const auto& request : storage_.submitted()) {
      tokens.push_back(request.token);
    }
    for (const std::uint64_t token : tokens) {
      (void)storage_.Release(token);
    }
  }

  // The extent's device bytes equal its file range (the fake's device
  // memory is host memory underneath).
  bool Loaded(std::size_t i, ReservationId reservation) const {
    return std::memcmp(At(Place(reservation, i)), file_.data() + (i * kSize), LengthOf(i)) == 0;
  }
  ExtentView View(ExtentId extent) const { return catalog_.Describe(extent).value(); }
  Occupancy Occupied() const { return catalog_.OccupancyOf(domain_); }
  Closure Of(std::initializer_list<std::size_t> which) const {
    std::vector<ExtentId> extents;
    for (const std::size_t i : which) {
      extents.push_back(extents_.at(i));
    }
    return catalog_.ClosureOfExtents(extents).value();
  }
  Closure All() const { return catalog_.ClosureOfExtents(extents_).value(); }
  static std::unique_ptr<TaskProgram> Load(LoadProgram::Report& report, Closure closure,
                                           std::optional<JobResult> job = std::nullopt) {
    return std::make_unique<LoadProgram>(report, std::move(closure), job);
  }

  static constexpr std::size_t kDeviceClass = 0;
  static constexpr std::size_t kHostClass = 1;

  // Declared in dependency order: the scheduler goes first, the providers last.
  FakeDeviceMemory memory_{Bytes(kSize), Bytes(kSize * 64)};
  FakeStorage storage_{8, 4096};
  FakeDeviceExecution execution_;
  jitllm::catalog::Catalog catalog_;
  jitllm::base::WakeFlag wake_;
  std::unique_ptr<CompletionBoard> board_;
  std::unique_ptr<StorageService> storage_lane_;
  std::unique_ptr<DeviceService> device_lane_;
  std::unique_ptr<Scheduler> scheduler_;

  ReservationId zone_;
  jitllm::providers::BackingId zone_backing_;
  ReservationId weights_;
  ReservationId moved_;
  std::vector<std::byte> file_;
  int fd_ = -1;
  jitllm::catalog::DomainId domain_;
  std::vector<ExtentId> extents_;
  StreamId stream_;
  std::size_t baseline_ = 0;
};

TEST_F(PageInTest, ALandedLoadIsPublishedOnlyAfterItsCopysFence) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle(false);  // the device runs nothing queued on its stream
  // Backing mapped first, then the read landed in a slot, not the extent.
  EXPECT_EQ(memory_.backings(), baseline_ + 1);
  ASSERT_EQ(storage_.submitted().size(), 4U);
  for (const auto& request : storage_.submitted()) {
    EXPECT_GE(reinterpret_cast<std::uint64_t>(request.memory), Slot(0));
    EXPECT_LT(reinterpret_cast<std::uint64_t>(request.memory), Slot(0) + kSize);
  }
  EXPECT_EQ(std::memcmp(At(Slot(0)), file_.data(), kSize), 0);
  // The copy is queued but its fence has not completed: still LOADING,
  // the slot still busy, the extent's bytes untouched.
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kLoading);
  EXPECT_EQ(Occupied().loading, Bytes(kSize));
  EXPECT_EQ(scheduler_->slots_busy(), 1U);
  EXPECT_EQ(*At(Place(weights_, 0)), kPoison);
  EXPECT_FALSE(report.outcome.has_value());

  Settle();
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(report.retired);
  EXPECT_EQ(scheduler_->operations(), 0U);
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(Occupied().idle, Bytes(kSize));
  EXPECT_EQ(execution_.fences(), 0U);
}

// The zone bounds what is in flight: two slots, so at most two reads; the
// window lets two more loads map backing ahead and wait for a slot, and
// the rest wait unmapped. Slots are granted in the order loads asked.
TEST_F(PageInTest, AFullZoneHoldsLoadsBackInOrder) {
  Build();
  HoldNext(64);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, All())).has_value());
  Settle();
  EXPECT_EQ(scheduler_->slots_busy(), kSlots);
  EXPECT_EQ(scheduler_->loads(), kExtents);
  EXPECT_EQ(memory_.backings(), baseline_ + (2 * kSlots));  // mapped ahead: the window
  // The reader started only the two slots' reads.
  std::vector<std::uint64_t> offsets;
  for (const auto& request : storage_.submitted()) {
    if (request.offset % kSize == 0) {
      offsets.push_back(request.offset);
    }
  }
  EXPECT_EQ(offsets, (std::vector<std::uint64_t>{0, kSize}));

  // Released one read at a time, the loads finish in order.
  for (int round = 0; round < 40 && !report.outcome; ++round) {
    ReleaseReads();
    Settle();
  }
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  offsets.clear();
  for (const auto& request : storage_.submitted()) {
    if (request.offset % kSize == 0) {
      offsets.push_back(request.offset);
    }
  }
  std::vector<std::uint64_t> ordered;
  ordered.reserve(kExtents);
  for (std::size_t i = 0; i < kExtents; ++i) {
    ordered.push_back(i * kSize);
  }
  EXPECT_EQ(offsets, ordered);
  for (std::size_t i = 0; i < kExtents; ++i) {
    EXPECT_EQ(View(extents_[i]).state, ExtentState::kResident);
    EXPECT_TRUE(Loaded(i, weights_));
  }
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_ + kExtents);
}

// A slot is not read into again until the copy out of it has completed.
TEST_F(PageInTest, ASlotIsReusedOnlyAfterTheCopyOutOfItCompletes) {
  Build(1);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0, 1}))).has_value());
  Settle(false);
  // The first read is done and its copy queued; the second load waits.
  EXPECT_EQ(storage_.submitted().size(), 4U);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kLoading);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kLoading);
  EXPECT_EQ(scheduler_->slots_busy(), 1U);
  execution_.Drain();  // the copy runs, then its fence
  Settle(false);
  // Its fence completed: published, and the slot handed on.
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(storage_.submitted().size(), 8U);
  EXPECT_EQ(storage_.submitted().back().offset, kSize + (3ULL * 16 * 1024));
  Settle();
  EXPECT_TRUE(Loaded(1, weights_));
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
}

TEST_F(PageInTest, AFailedReadReleasesItsSlotAndItsBacking) {
  Build();
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = -EIO, .hold = false});
  LoadProgram::Report report;
  const std::uint64_t generation = View(extents_[0]).backing_generation;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kFailed);
  const ExtentView failed = View(extents_[0]);
  EXPECT_EQ(failed.state, ExtentState::kNonresident);
  EXPECT_EQ(failed.backing_generation, generation + 1);
  EXPECT_EQ(memory_.backings(), baseline_);  // unmapped and released
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(Occupied().Total(), Bytes());
  EXPECT_FALSE(scheduler_->fault().has_value());

  // A short read fails the same way, and the extent loads again later.
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = 4096, .hold = false});
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = 0, .hold = false});
  LoadProgram::Report again;
  ASSERT_TRUE(scheduler_->Start(2, Load(again, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(again.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kNonresident);
  EXPECT_EQ(memory_.backings(), baseline_);
  LoadProgram::Report third;
  ASSERT_TRUE(scheduler_->Start(3, Load(third, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(third.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(Loaded(0, weights_));
}

TEST_F(PageInTest, CancellingDuringTheReadDrainsItThenUnwinds) {
  Build();
  HoldNext(4);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  ASSERT_EQ(View(extents_[0]).state, ExtentState::kLoading);
  ASSERT_EQ(scheduler_->slots_busy(), 1U);
  ASSERT_TRUE(scheduler_->Cancel(1));
  EXPECT_EQ(report.outcome, TaskOutcome::kCancelled);
  EXPECT_TRUE(report.retired);  // the load is the scheduler's, not the task's
  // Nothing is released until the read has drained.
  EXPECT_EQ(scheduler_->slots_busy(), 1U);
  EXPECT_EQ(memory_.backings(), baseline_ + 1);
  Settle();  // the lane cancels the held requests; they complete
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kNonresident);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_EQ(storage_.in_flight(), 0U);
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(Occupied().Total(), Bytes());
}

// A copy cannot be cancelled: the load waits for its fence, then
// publishes the whole contents, and only then frees the slot.
TEST_F(PageInTest, CancellingMidCopyWaitsForTheFence) {
  Build(1);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle(false);
  ASSERT_EQ(View(extents_[0]).state, ExtentState::kLoading);
  ASSERT_TRUE(scheduler_->Cancel(1));
  EXPECT_TRUE(report.retired);
  // A second request waits for the one slot the copy still holds.
  LoadProgram::Report other;
  ASSERT_TRUE(scheduler_->Start(2, Load(other, Of({1}))).has_value());
  Settle(false);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kLoading);
  EXPECT_EQ(scheduler_->slots_busy(), 1U);
  EXPECT_EQ(storage_.submitted().size(), 4U);
  EXPECT_EQ(*At(Place(weights_, 0)), kPoison);
  Settle();
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(other.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(Loaded(1, weights_));
}

// A request cancelled while its loads wait for the window or a slot
// unwinds them at once: nothing was read, and backing mapped ahead is
// released.
TEST_F(PageInTest, CancellingWaitingLoadsReleasesWhatTheyMapped) {
  Build(1);
  HoldNext(64);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, All())).has_value());
  Settle();
  EXPECT_EQ(scheduler_->loads(), kExtents);
  EXPECT_EQ(memory_.backings(), baseline_ + 2);  // one reading, one mapped ahead
  ASSERT_TRUE(scheduler_->Cancel(1));
  // The queued ones ended at once; the read in flight and the backing
  // mapped ahead stay until the read drains and the unmap completes.
  EXPECT_EQ(scheduler_->loads(), 2U);
  EXPECT_EQ(memory_.backings(), baseline_ + 2);
  EXPECT_EQ(scheduler_->slots_busy(), 1U);
  Settle();
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_EQ(Occupied().Total(), Bytes());
  for (const ExtentId extent : extents_) {
    EXPECT_EQ(View(extent).state, ExtentState::kNonresident);
  }
}

// A copy whose fence cannot be proven leaves the slot it read and the
// extent it wrote undetermined: both are quarantined, never reused, and
// the node faults. Loads waiting for that slot can only be withdrawn.
TEST_F(PageInTest, AnUnprovenCopyQuarantinesItsSlotAndItsExtent) {
  Build(1);
  execution_.FailNextQuery(ProviderError::kUnknown, 1000);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(Occupied().quarantined, Bytes(kSize));
  EXPECT_EQ(scheduler_->slots_quarantined(), 1U);
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
  execution_.FailNextQuery(ProviderError::kUnknown, 0);
  EXPECT_EQ(Failed(scheduler_->Start(2, Load(report, Of({1})))),
            jitllm::scheduler::StartError::kStopped);  // admission has stopped
}

TEST_F(PageInTest, BackingThatCannotBeMadeChangesNothing) {
  Build();
  const std::uint64_t reads = storage_.submitted().size();
  memory_.FailNext(jitllm::providers::fake::Operation::kCreate, ProviderError::kOutOfMemory);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kNonresident);
  EXPECT_EQ(storage_.submitted().size(), reads);  // nothing read
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_);

  // A mapping refused after the backing was made: the backing is released.
  memory_.FailNext(jitllm::providers::fake::Operation::kMap, ProviderError::kFailed);
  LoadProgram::Report mapped;
  ASSERT_TRUE(scheduler_->Start(2, Load(mapped, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(mapped.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kNonresident);

  // Access refused: unmapped and released.
  memory_.FailNext(jitllm::providers::fake::Operation::kSetAccess, ProviderError::kFailed);
  LoadProgram::Report access;
  ASSERT_TRUE(scheduler_->Start(3, Load(access, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(access.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_FALSE(scheduler_->fault().has_value());
}

TEST_F(PageInTest, BackingOfUnknownOutcomeIsQuarantined) {
  Build();
  memory_.FailNext(jitllm::providers::fake::Operation::kCreate, ProviderError::kUnknown, true);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
}

// Eviction unmaps and releases the backing on the device lane (D-033); a
// later load maps fresh backing and restores the same bytes, at the same
// place or, registered again, at another (relocation, BP-P5).
TEST_F(PageInTest, EvictionReleasesBackingAndReloadsAreIdentical) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, All())).has_value());
  Settle();
  ASSERT_EQ(report.outcome, TaskOutcome::kSucceeded);
  ASSERT_EQ(memory_.backings(), baseline_ + kExtents);
  const std::uint64_t generation = View(extents_[2]).backing_generation;
  const std::uint64_t contents = View(extents_[2]).content_generation;

  EvictProgram::Report evicted;
  ASSERT_TRUE(scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, extents_)).has_value());
  ASSERT_TRUE(scheduler_->Turn());
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kEvicting);  // until its unmap completes
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
  ASSERT_EQ(evicted.results.size(), kExtents);
  EXPECT_EQ(evicted.results[0], Readiness::kWaiting);
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_EQ(Occupied().Total(), Bytes());
  EXPECT_EQ(View(extents_[2]).state, ExtentState::kNonresident);
  EXPECT_EQ(View(extents_[2]).backing_generation, generation + 1);
  EXPECT_EQ(View(extents_[2]).content_generation, contents);

  LoadProgram::Report reloaded;
  ASSERT_TRUE(scheduler_->Start(3, Load(reloaded, All())).has_value());
  Settle();
  ASSERT_EQ(reloaded.outcome, TaskOutcome::kSucceeded);
  for (std::size_t i = 0; i < kExtents; ++i) {
    EXPECT_TRUE(Loaded(i, weights_));
  }

  // Relocation: refused while resident at the old place, allowed once
  // evicted; the next load lands at the new one.
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[0], Source(0, moved_))), WorkError::kBusy);
  EvictProgram::Report again;
  ASSERT_TRUE(scheduler_->Start(4, std::make_unique<EvictProgram>(again, extents_)).has_value());
  Settle();
  ASSERT_EQ(again.outcome, TaskOutcome::kSucceeded);
  for (std::size_t i = 0; i < kExtents; ++i) {
    ASSERT_TRUE(scheduler_->SetSource(extents_[i], Source(i, moved_)).has_value());
  }
  LoadProgram::Report relocated;
  ASSERT_TRUE(scheduler_->Start(5, Load(relocated, All())).has_value());
  Settle();
  ASSERT_EQ(relocated.outcome, TaskOutcome::kSucceeded);
  for (std::size_t i = 0; i < kExtents; ++i) {
    EXPECT_TRUE(Loaded(i, moved_));
  }
  EXPECT_EQ(memory_.backings(), baseline_ + kExtents);
}

// A materialization that meets an eviction in flight waits for it, then
// loads again.
TEST_F(PageInTest, MaterializingDuringAnEvictionWaitsThenReloads) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[0]}))
          .has_value());
  LoadProgram::Report after;
  ASSERT_TRUE(scheduler_->Start(3, Load(after, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(after.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(0, weights_));
}

// An unmap refused with nothing changed abandons the eviction: the evictor
// hears of the failure, and a task that met the eviction while
// materializing finds the extent resident again, unharmed.
TEST_F(PageInTest, ARefusedUnmapLeavesTheExtentResidentForItsReaders) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  ASSERT_EQ(report.outcome, TaskOutcome::kSucceeded);
  memory_.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kFailed);
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[0]}))
          .has_value());
  LoadProgram::Report reader;
  ASSERT_TRUE(scheduler_->Start(3, Load(reader, Of({0}))).has_value());
  ASSERT_TRUE(scheduler_->Turn());  // both step: the eviction starts, the reader joins it
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kEvicting);
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(reader.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_EQ(memory_.backings(), baseline_ + 1);
  EXPECT_FALSE(scheduler_->fault().has_value());
}

TEST_F(PageInTest, AnUnmapOfUnknownOutcomeQuarantinesTheEviction) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  memory_.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kUnknown);
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[0]}))
          .has_value());
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(Occupied().quarantined, Bytes(kSize));
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
}

// An unknown outcome leaves the whole reservation undetermined, and the
// provider refuses every later call on it. Refusing an unmap there changed
// nothing, but it proves nothing about the place either: the extent is
// quarantined, still charged, never resident again for readers.
TEST_F(PageInTest, AnUnmapRefusedAsUndeterminedQuarantinesTheEviction) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0, 1}))).has_value());
  Settle();
  ASSERT_EQ(report.outcome, TaskOutcome::kSucceeded);
  memory_.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kUnknown);
  // Both unmaps queued in this order before the device lane runs either.
  EvictProgram::Report first;
  ASSERT_TRUE(scheduler_->Start(2, std::make_unique<EvictProgram>(first, std::vector{extents_[1]}))
                  .has_value());
  ASSERT_TRUE(scheduler_->Turn());
  EvictProgram::Report second;
  ASSERT_TRUE(scheduler_->Start(3, std::make_unique<EvictProgram>(second, std::vector{extents_[0]}))
                  .has_value());
  ASSERT_TRUE(scheduler_->Turn());
  Settle();
  EXPECT_TRUE(memory_.Undetermined(weights_));
  EXPECT_EQ(first.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(second.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kQuarantined);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(Occupied().quarantined, Bytes(2 * kSize));
  EXPECT_EQ(memory_.backings(), baseline_ + 2);
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
}

// A task that met an eviction in flight, and waits for it, hears of a
// failure when that eviction is quarantined, and the extent stays
// unavailable.
TEST_F(PageInTest, ATaskWaitingOnAnEvictionThatIsQuarantinedFails) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  memory_.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kUnknown);
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[0]}))
          .has_value());
  LoadProgram::Report reader;
  ASSERT_TRUE(scheduler_->Start(3, Load(reader, Of({0}))).has_value());
  ASSERT_TRUE(scheduler_->Turn());  // the eviction starts; the reader joins it
  ASSERT_EQ(View(extents_[0]).state, ExtentState::kEvicting);
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(reader.outcome, TaskOutcome::kFailed);
  EXPECT_TRUE(reader.retired);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(Occupied().quarantined, Bytes(kSize));
  EXPECT_EQ(scheduler_->loads(), 0U);
}

// A withdrawn load starts no new stage (docs/async-model.md). Here the
// device lane is full, so the first load's copy waits unpublished, while
// the storage lane has room. Withdrawing both loads rolls that copy back,
// which frees its slot: the second load, waiting for it and withdrawn in
// the same step, must unwind, not start a read into it.
TEST_F(PageInTest, AWithdrawnLoadStartsNoNewStage) {
  Build(1);
  HoldNext(4);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0, 1}))).has_value());
  Settle();
  ASSERT_EQ(scheduler_->slots_busy(), 1U);
  ASSERT_EQ(memory_.backings(), baseline_ + 2);  // the second is mapped, waiting for the slot
  ASSERT_EQ(storage_.submitted().size(), 4U);
  // Fill the device lane with commands that name no operation.
  int fillers = 0;
  while (device_lane_->Submit(jitllm::scheduler::DeviceCommand{
             .operation = {}, .work = jitllm::scheduler::DeviceWork{}}) == PushResult::kAccepted) {
    ++fillers;
  }
  ASSERT_GT(fillers, 0);
  // The first read completes; its copy cannot be published.
  ReleaseReads();
  for (int i = 0; i < 20; ++i) {
    (void)storage_lane_->Turn(false);
    (void)scheduler_->Turn();
  }
  ASSERT_EQ(View(extents_[0]).state, ExtentState::kLoading);
  ASSERT_EQ(scheduler_->slots_busy(), 1U);
  ASSERT_TRUE(scheduler_->Cancel(1));
  Settle();
  for (const auto& request : storage_.submitted()) {
    EXPECT_LT(request.offset, kSize) << "a read for a withdrawn load";
  }
  EXPECT_EQ(storage_.submitted().size(), 4U);
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kNonresident);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(Occupied().Total(), Bytes());
  EXPECT_FALSE(scheduler_->fault().has_value());
}

// Every mailbox held by quarantined work, with a withdrawn load that must
// still unmap what it mapped: no mailbox will ever come, so the load is
// quarantined (its backing stays charged) and the stop reports the fault
// instead of waiting forever.
TEST_F(PageInTest, AnUnwindThatCanNeverGetAMailboxFaultsTheStop) {
  Build(1, 2);
  LoadProgram::Report resident;
  ASSERT_TRUE(scheduler_->Start(1, Load(resident, Of({2}))).has_value());
  Settle();
  ASSERT_EQ(resident.outcome, TaskOutcome::kSucceeded);
  // The first load's copy is queued, its fence not complete; the second is
  // mapped and waits for the one slot.
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(2, Load(report, Of({0, 1}))).has_value());
  Settle(false);
  ASSERT_EQ(scheduler_->loads(), 2U);
  ASSERT_EQ(scheduler_->slots_busy(), 1U);
  ASSERT_EQ(scheduler_->operations(), 1U);
  // An unmap of unknown outcome keeps the other mailbox for good ...
  memory_.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kUnknown);
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(3, std::make_unique<EvictProgram>(evicted, std::vector{extents_[2]}))
          .has_value());
  Settle(false);
  ASSERT_EQ(View(extents_[2]).state, ExtentState::kQuarantined);
  // ... and so does the copy, once its fence cannot be proven. The second
  // load still waits for the slot, which is quarantined too; the stop
  // cancels its request, and it has no mailbox to unmap with.
  execution_.FailNextQuery(ProviderError::kUnknown, 1000);
  Settle();
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(scheduler_->slots_quarantined(), 1U);
  scheduler_->RequestShutdown();
  for (int i = 0; i < 1000 && !scheduler_->Stopped(); ++i) {
    (void)Round();
  }
  const auto stopped = scheduler_->Stopped();
  ASSERT_TRUE(stopped.has_value()) << "the stop waits forever for a mailbox";
  EXPECT_FALSE(stopped->has_value());
  EXPECT_EQ(report.outcome, TaskOutcome::kCancelled);
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kQuarantined);
  EXPECT_EQ(Occupied().quarantined, Bytes(3 * kSize));
  execution_.FailNextQuery(ProviderError::kUnknown, 0);
}

// The counterpart: one mailbox is held for good, but the other by a copy
// that will complete. The withdrawn load's unmap waits for that mailbox
// instead of being quarantined, and runs once the copy is concluded. (The
// unmap of unknown outcome is in another reservation, which it leaves
// undetermined; the withdrawn load's is unharmed.)
TEST_F(PageInTest, AnUnwindWaitsForAMailboxThatWillFree) {
  Build(1, 2);
  ASSERT_TRUE(scheduler_->SetSource(extents_[2], Source(2, moved_)).has_value());
  LoadProgram::Report resident;
  ASSERT_TRUE(scheduler_->Start(1, Load(resident, Of({2}))).has_value());
  Settle();
  ASSERT_EQ(resident.outcome, TaskOutcome::kSucceeded);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(2, Load(report, Of({0, 1}))).has_value());
  Settle(false);
  ASSERT_EQ(scheduler_->loads(), 2U);
  ASSERT_EQ(scheduler_->operations(), 1U);
  memory_.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kUnknown);
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(3, std::make_unique<EvictProgram>(evicted, std::vector{extents_[2]}))
          .has_value());
  Settle(false);
  ASSERT_EQ(View(extents_[2]).state, ExtentState::kQuarantined);
  ASSERT_TRUE(scheduler_->Cancel(2));
  for (int i = 0; i < 20; ++i) {
    (void)Round(false);  // the copy's fence stays pending
  }
  EXPECT_EQ(scheduler_->loads(), 2U);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kLoading);
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kCancelled);
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);  // its copy completed whole
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(Occupied().quarantined, Bytes(kSize));
  EXPECT_EQ(memory_.backings(), baseline_ + 2);  // the quarantined one, and extent 0's
  EXPECT_EQ(scheduler_->slots_quarantined(), 0U);
}

// A withdrawal that lands in the same turn as its copy's completion: the
// copy publishes whole bytes and frees the slot, the other withdrawn load
// unwinds without a read, and the next request's queued load starts.
TEST_F(PageInTest, AWithdrawalMeetingItsCopysCompletionHandsTheSlotOn) {
  Build(1);
  LoadProgram::Report withdrawn;
  ASSERT_TRUE(scheduler_->Start(1, Load(withdrawn, Of({0, 1}))).has_value());
  LoadProgram::Report next;
  ASSERT_TRUE(scheduler_->Start(2, Load(next, Of({2}))).has_value());
  Settle(false);
  ASSERT_EQ(scheduler_->loads(), 3U);
  ASSERT_EQ(scheduler_->copying(), 1U);
  ASSERT_EQ(View(extents_[2]).state, ExtentState::kLoading);
  ASSERT_EQ(memory_.backings(), baseline_ + 2);  // the third waits, unmapped
  // The fence completes and is posted, not yet harvested; then the cancel.
  execution_.Drain();
  (void)device_lane_->CompletionTurn();
  ASSERT_TRUE(scheduler_->Cancel(1));
  Settle();
  EXPECT_EQ(withdrawn.outcome, TaskOutcome::kCancelled);
  EXPECT_EQ(next.outcome, TaskOutcome::kSucceeded);
  for (const auto& request : storage_.submitted()) {
    EXPECT_TRUE(request.offset < kSize || request.offset >= 2 * kSize)
        << "a read for a withdrawn load";
  }
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(View(extents_[2]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(2, weights_));
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_ + 2);
  EXPECT_FALSE(scheduler_->fault().has_value());
}

// An unknown outcome on one map leaves the reservation undetermined, and
// the next map into it, for another extent, is refused (kUndetermined)
// before the driver is asked. That refusal changed nothing: its load fails
// cleanly with the new backing released, and its extent is not quarantined.
TEST_F(PageInTest, AMapRefusedAsUndeterminedFailsItsLoadCleanly) {
  Build();
  memory_.FailNext(jitllm::providers::fake::Operation::kMap, ProviderError::kUnknown);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0, 1}))).has_value());
  Settle();
  EXPECT_TRUE(memory_.Undetermined(weights_));
  EXPECT_EQ(report.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(Occupied().quarantined, Bytes(kSize));
  EXPECT_EQ(Occupied().Total(), Bytes(kSize));
  EXPECT_EQ(memory_.backings(), baseline_ + 1);  // the first load's, charged
  EXPECT_EQ(storage_.submitted().size(), 0U);    // neither load read
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
}

// A kernel job's lease holds its closure until the fence after it has
// completed, even once its request is cancelled; eviction is refused
// meanwhile (invariant 2). A job that queued nothing releases at once.
TEST_F(PageInTest, AJobsLeaseHoldsUntilItsFence) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}), JobResult::kQueued)).has_value());
  Settle(false);
  execution_.Drain();  // the load's copy and its fence
  Settle(false);
  ASSERT_EQ(report.job_runs, 1);
  EXPECT_EQ(View(extents_[0]).leases, 1U);
  ASSERT_TRUE(scheduler_->Cancel(1));
  EXPECT_EQ(report.outcome, TaskOutcome::kCancelled);
  EXPECT_FALSE(report.retired);  // its operation is still in flight
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[0]}))
          .has_value());
  Settle(false);
  EXPECT_EQ(evicted.outcome, TaskOutcome::kFailed);
  ASSERT_EQ(evicted.results.size(), 1U);
  EXPECT_EQ(Failed(evicted.results.front()), WorkError::kBusy);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  Settle();  // the fence completes
  EXPECT_TRUE(report.retired);
  EXPECT_EQ(View(extents_[0]).leases, 0U);

  LoadProgram::Report refused;
  ASSERT_TRUE(scheduler_->Start(3, Load(refused, Of({0}), JobResult::kNotStarted)).has_value());
  Settle();
  EXPECT_EQ(refused.job_runs, 1);
  EXPECT_EQ(refused.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).leases, 0U);
  EXPECT_EQ(execution_.fences(), 0U);
}

// A direct source reads into managed host backing the CPU maps (the token
// table the embedding lookup reads, D-081): no slot, no copy.
TEST_F(PageInTest, DirectSourcesReadIntoTheirOwnHostBacking) {
  Build();
  const ReservationId host = memory_.Reserve(Bytes(kSize)).value();
  const std::uint64_t address = memory_.RangeOf(host).value().base;
  ASSERT_TRUE(
      scheduler_
          ->SetSource(
              extents_[0],
              PageSource{
                  .read = ReadSpec{.fd = fd_, .offset = 0, .memory = At(address), .length = kSize},
                  .landed = false,
                  .destination = 0,
                  .backing = BackingPlace{.reservation = host,
                                          .offset = Bytes(0),
                                          .size = Bytes(kSize),
                                          .allocation_class = kHostClass}})
          .has_value());
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle(false);
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(std::memcmp(At(address), file_.data(), kSize), 0);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(execution_.fences(), 0U);  // nothing was copied
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[0]}))
          .has_value());
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(memory_.backings(), baseline_);
}

TEST_F(PageInTest, SourcesAreChecked) {
  Build();
  PageSource source = Source(0, weights_);
  source.read.length = kSize + 4096;  // larger than a slot
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[0], source)), WorkError::kInvalid);
  source = Source(0, weights_);
  source.destination = 0;
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[0], source)), WorkError::kInvalid);
  source = Source(0, weights_);
  source.landed = false;  // direct, with nowhere to land
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[0], source)), WorkError::kInvalid);
  // While a load is in flight, the source cannot move.
  HoldNext(4);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[0], Source(0, moved_))), WorkError::kBusy);
  ReleaseReads();
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
}

TEST(PageInZoneTest, ALandedSourceNeedsAZone) {
  FakeDeviceMemory memory{Bytes(kSize), Bytes(kSize * 4)};
  FakeDeviceExecution execution;
  FakeStorage storage{4, 4096};
  jitllm::catalog::Catalog catalog;
  jitllm::base::WakeFlag wake;
  CompletionBoard board{4, wake};
  const auto domain = catalog.AddDomain("node");
  const ExtentId extent = catalog
                              .AddExtent({.domain = domain,
                                          .memory_class = jitllm::catalog::MemoryClass::kWeights,
                                          .recovery = jitllm::catalog::Recovery::kFromArtifact,
                                          .size = Bytes(kSize),
                                          .content = {}})
                              .value();
  Scheduler scheduler(catalog, board, wake, Lanes{}, SchedulerSettings{});
  const PageSource landed{
      .read = ReadSpec{.fd = 3, .offset = 0, .memory = nullptr, .length = kSize},
      .landed = true,
      .destination = 4096,
      .backing = std::nullopt};
  EXPECT_EQ(Failed(scheduler.SetSource(extent, landed)), WorkError::kInvalid);
}

// The zone's copies run on one of the device lane's streams: a zone with
// no device lane, or a stream index the lane does not have, would fail
// every landed load. Refused when the scheduler is built.
TEST(PageInZoneDeathTest, TheLandingStreamMustBeADeviceLaneStream) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeDeviceExecution execution;
  const StreamId stream = execution.CreateStream().value();
  jitllm::catalog::Catalog catalog;
  jitllm::base::WakeFlag wake;
  CompletionBoard board{4, wake};
  DeviceService device(execution, std::span<const StreamId>(&stream, 1), board, DeviceSettings{});
  const auto build = [&](DeviceService* lane, std::uint32_t index) {
    const Scheduler scheduler(
        catalog, board, wake, Lanes{.storage = nullptr, .device = lane, .cpu = nullptr},
        SchedulerSettings{
            .landing = LandingZone{.slots = {4096}, .slot_bytes = Bytes(kSize), .stream = index}});
  };
  EXPECT_DEATH(build(&device, 1), "landing zone's stream");
  EXPECT_DEATH(build(nullptr, 0), "landing zone's stream");
  build(&device, 0);
}

// Every lane on its own thread and the fake device on another: requests
// load every extent through a one-slot zone, evict them, and load them
// again, some cancelled, while the owner sleeps on its wake flag. A lost
// wakeup hangs; a slot reused early or a publication before the copy's
// fence shows as a mismatch, and a race under ThreadSanitizer.
TEST_F(PageInTest, ThreadedLoadsAndEvictionsThroughTheZoneKeepEveryByte) {
  Build(1, 64);
  std::atomic<int> mismatches{0};
  std::atomic<int> done{0};
  std::atomic<int> evictions{0};
  constexpr int kTries = 1000;  // yields while other requests hold an extent
  constexpr int kRounds = 40;
  // Loads everything, checks the bytes while holding a job's lease, and
  // evicts what it can.
  class Cycle final : public TaskProgram {
   public:
    Cycle(Closure all, std::vector<ExtentId> extents, std::function<bool()> check,
          std::atomic<int>& mismatches, std::atomic<int>& evictions, std::atomic<int>& done)
        : all_(std::move(all)),
          extents_(std::move(extents)),
          check_(std::move(check)),
          mismatches_(mismatches),
          evictions_(evictions),
          done_(done) {}
    Step Advance(TaskContext& context) override {
      if (context.TakeFailure()) {
        return Step::Finish(TaskOutcome::kFailed);
      }
      if (!submitted_) {
        const auto ready = context.Materialize(all_);
        if (!ready) {
          return ready.error() == WorkError::kBusy ? Step::Yield()
                                                   : Step::Finish(TaskOutcome::kFailed);
        }
        if (*ready == Readiness::kWaiting) {
          return Step::Wait();
        }
        std::function<bool()> check = check_;
        std::atomic<int>& mismatches = mismatches_;
        const auto job = context.SubmitLaunch(
            all_, LaunchWork{.stream = 0, .job = [check, &mismatches](auto /*stream*/) {
                               if (!check()) {
                                 mismatches.fetch_add(1);
                               }
                               return JobResult::kQueued;
                             }});
        if (!job) {
          // Evicted meanwhile, or no mailbox: materialize again.
          return job.error() == WorkError::kBusy || job.error() == WorkError::kNotResident
                     ? Step::Yield()
                     : Step::Finish(TaskOutcome::kFailed);
        }
        submitted_ = true;
        return Step::Wait();
      }
      if (next_ < extents_.size()) {
        const auto evicted = context.Evict(extents_[next_]);
        if (evicted || evicted.error() != WorkError::kBusy || ++tries_ > kTries) {
          ++next_;  // evicted, or held by other requests too long: it stays
          tries_ = 0;
        }
        if (evicted && *evicted == Readiness::kWaiting) {
          evictions_.fetch_add(1);
          return Step::Wait();
        }
        return Step::Yield();
      }
      return Step::Finish(TaskOutcome::kSucceeded);
    }
    void Retired() override { done_.fetch_add(1); }

   private:
    Closure all_;
    std::vector<ExtentId> extents_;
    std::function<bool()> check_;
    std::atomic<int>& mismatches_;
    std::atomic<int>& evictions_;
    std::atomic<int>& done_;
    bool submitted_ = false;
    std::size_t next_ = 0;
    int tries_ = 0;
  };
  const std::function<bool()> check = [this] {
    for (std::size_t i = 0; i < kExtents; ++i) {
      if (!Loaded(i, weights_)) {
        return false;
      }
    }
    return true;
  };
  std::optional<std::expected<void, Fault>> result;
  {
    std::jthread owner([&] { result = scheduler_->Run(); });
    std::jthread storage([&] { storage_lane_->Run(); });
    std::jthread submission([&] { device_lane_->RunSubmission(); });
    std::jthread completion([&] { device_lane_->RunCompletion(); });
    std::jthread device([&](const std::stop_token& stop) {
      while (!stop.stop_requested()) {
        execution_.Drain();
        std::this_thread::yield();
      }
    });
    for (int n = 0; n < kRounds; ++n) {
      while (n - done.load() >= 4) {
        std::this_thread::yield();
      }
      Control start = StartRequest{
          .request = static_cast<std::uint64_t>(n + 1),
          .priority = 1,
          .program = std::make_unique<Cycle>(All(), extents_, check, mismatches, evictions, done)};
      // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
      while (scheduler_->Post(std::move(start)) == PushResult::kFull) {
        std::this_thread::yield();
      }
      if (n % 5 == 4) {
        Control cancel = CancelRequest{.request = static_cast<std::uint64_t>(n)};
        (void)scheduler_->Post(std::move(cancel));
      }
    }
    const auto give_up = std::chrono::steady_clock::now() + kPatience;
    while (done.load() < kRounds && std::chrono::steady_clock::now() < give_up) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(done.load(), kRounds);
    scheduler_->RequestShutdown();
    owner.join();
    storage_lane_->Close();
    device_lane_->Close();
    storage.join();
    submission.join();
    completion.join();
  }
  EXPECT_FALSE(result.has_value() && !result->has_value());
  EXPECT_EQ(mismatches.load(), 0);
  // Extents were evicted and read again, many times over.
  EXPECT_GT(evictions.load(), static_cast<int>(4 * kExtents));
  EXPECT_GT(storage_.submitted().size(), 4 * kExtents * 4);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(scheduler_->slots_quarantined(), 0U);
  EXPECT_EQ(scheduler_->loads(), 0U);
  const Occupancy occupied = Occupied();
  EXPECT_EQ(occupied.held, Bytes());
  EXPECT_EQ(occupied.loading, Bytes());
  EXPECT_EQ(occupied.evicting, Bytes());
  EXPECT_EQ(occupied.quarantined, Bytes());
  EXPECT_EQ(memory_.backings(), baseline_ + (occupied.idle.value() / kSize));
  scheduler_.reset();  // stopped: TearDown has nothing more to drain
}

}  // namespace
