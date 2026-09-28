// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The D-081 page-in path over the real providers on a Spark (label `gpu`)
// and on a discrete GPU the build targets (`gpu-discrete`, D-082):
// io_uring reads a direct-I/O file into a host-VMM landing zone, the device
// lane (or, in the second instantiation, a VMM lane of its own) creates and
// maps device-VMM backing for each extent, the device lane copies the
// landed bytes in on a CUDA stream, and the scheduler publishes each extent
// only once the copy's fence has completed. Device VMM is not CPU-mapped,
// so the test checks each extent by copying it back to host VMM under a
// lease. Evictions release the backing (D-033) and reloads, at the same
// place or relocated, restore the same bytes; a request cancelled with
// loads in flight drains them. Every lane runs on its own thread.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/bounded_queue.h"
#include "base/bytes.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/direct_reader.h"
#include "providers/uring_storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/scheduler.h"
#include "scheduler/services.h"

namespace {

using jitllm::base::Bytes;
using jitllm::base::PushResult;
using jitllm::catalog::ExtentId;
using jitllm::catalog::ExtentState;
using jitllm::providers::Access;
using jitllm::providers::BackingKind;
using jitllm::providers::ReservationId;
using jitllm::providers::StreamId;
using jitllm::scheduler::BackingPlace;
using jitllm::scheduler::CompletionBoard;
using jitllm::scheduler::Control;
using jitllm::scheduler::DeviceService;
using jitllm::scheduler::DeviceSettings;
using jitllm::scheduler::DeviceWork;
using jitllm::scheduler::Fault;
using jitllm::scheduler::LandingZone;
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

constexpr std::uint64_t kExtent = 2ULL << 20U;
constexpr std::size_t kExtents = 32;
constexpr std::size_t kSlots = 8;                          // 2 x depth 4 (D-081)
constexpr std::uint64_t kTail = kExtent - (64ULL * 1024);  // a group's last chunk is shorter
constexpr auto kPatience = std::chrono::seconds(120);

std::byte* At(std::uint64_t address) {
  return reinterpret_cast<std::byte*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t LengthOf(std::size_t i) { return i + 1 == kExtents ? kTail : kExtent; }

struct Signals {
  std::atomic<int> outcome{-1};
  std::atomic<bool> retired{false};
  std::atomic<bool> waiting{false};
  std::atomic<int> mismatches{0};
};

// Materializes the extents; with `check`, copies each back to the host
// result extent under a lease and compares it with the file: extent k
// with chunk `chunks[k]`, or chunk k if none are given.
class LoadAndCheck final : public TaskProgram {
 public:
  LoadAndCheck(Signals& signals, const jitllm::catalog::Catalog& catalog,
               std::vector<ExtentId> extents, std::vector<std::uint64_t> addresses, ExtentId result,
               std::uint64_t result_address, std::span<const std::byte> file, bool check,
               std::vector<std::size_t> chunks = {})
      : signals_(signals),
        catalog_(catalog),
        extents_(std::move(extents)),
        addresses_(std::move(addresses)),
        result_(result),
        result_address_(result_address),
        file_(file),
        check_(check),
        chunks_(std::move(chunks)) {}

  Step Advance(TaskContext& context) override {
    if (context.TakeFailure()) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    if (!loaded_) {
      const auto ready = context.Materialize(catalog_.ClosureOfExtents(extents_).value());
      if (!ready) {
        return Step::Finish(TaskOutcome::kFailed);
      }
      if (*ready == Readiness::kWaiting) {
        signals_.waiting.store(true);
        return Step::Wait();
      }
      loaded_ = true;
    }
    if (!check_) {
      return Step::Finish(TaskOutcome::kSucceeded);
    }
    if (next_ > 0) {
      // The previous copy's fence completed: compare what it brought back.
      const std::size_t i = chunks_.empty() ? next_ - 1 : chunks_.at(next_ - 1);
      if (std::memcmp(At(result_address_), file_.data() + (i * kExtent), LengthOf(i)) != 0) {
        signals_.mismatches.fetch_add(1);
      }
    }
    if (next_ == extents_.size()) {
      return Step::Finish(TaskOutcome::kSucceeded);
    }
    DeviceWork work;
    work.copies.at(0) = {.destination = result_address_,
                         .source = addresses_.at(next_),
                         .size = Bytes(LengthOf(chunks_.empty() ? next_ : chunks_.at(next_)))};
    work.count = 1;
    const std::array<ExtentId, 2> both = {extents_.at(next_), result_};
    if (!context.SubmitDevice(catalog_.ClosureOfExtents(both).value(), work)) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    ++next_;
    return Step::Wait();
  }
  void Finished(TaskOutcome outcome) override { signals_.outcome.store(static_cast<int>(outcome)); }
  void Retired() override { signals_.retired.store(true); }

 private:
  Signals& signals_;
  const jitllm::catalog::Catalog& catalog_;
  std::vector<ExtentId> extents_;
  std::vector<std::uint64_t> addresses_;
  ExtentId result_;
  std::uint64_t result_address_;
  std::span<const std::byte> file_;
  bool check_;
  std::vector<std::size_t> chunks_;
  bool loaded_ = false;
  std::size_t next_ = 0;
};

class EvictAll final : public TaskProgram {
 public:
  EvictAll(Signals& signals, std::vector<ExtentId> extents)
      : signals_(signals), extents_(std::move(extents)) {}
  Step Advance(TaskContext& context) override {
    if (context.TakeFailure()) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    while (next_ < extents_.size()) {
      const ExtentId extent = extents_[next_++];
      if (context.catalog().Describe(extent).value().state != ExtentState::kResident) {
        continue;
      }
      const auto evicted = context.Evict(extent);
      if (!evicted) {
        return Step::Finish(TaskOutcome::kFailed);
      }
      if (*evicted == Readiness::kWaiting) {
        return Step::Wait();
      }
    }
    return Step::Finish(TaskOutcome::kSucceeded);
  }
  void Finished(TaskOutcome outcome) override { signals_.outcome.store(static_cast<int>(outcome)); }
  void Retired() override { signals_.retired.store(true); }

 private:
  Signals& signals_;
  std::vector<ExtentId> extents_;
  std::size_t next_ = 0;
};

// Runs `probe` on the scheduler thread each turn until it returns true.
// The scheduler's and the catalog's records are that thread's while it
// runs: the test reads them only here, or once a task that ran after the
// change has retired.
class OnOwner final : public TaskProgram {
 public:
  OnOwner(Signals& signals, std::function<bool()> probe)
      : signals_(signals), probe_(std::move(probe)) {}
  Step Advance(TaskContext& /*context*/) override {
    return probe_() ? Step::Finish(TaskOutcome::kSucceeded) : Step::Yield();
  }
  void Finished(TaskOutcome outcome) override { signals_.outcome.store(static_cast<int>(outcome)); }
  void Retired() override { signals_.retired.store(true); }

 private:
  Signals& signals_;
  std::function<bool()> probe_;
};

bool WaitFor(const std::atomic<bool>& flag) {
  const auto give_up = std::chrono::steady_clock::now() + kPatience;
  while (!flag.load() && std::chrono::steady_clock::now() < give_up) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return flag.load();
}

// The parameter: whether VMM work runs on a VMM lane (BackingService).
class CudaPageIn : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    auto memory = jitllm::providers::cuda::OpenDeviceMemory(0);
    ASSERT_TRUE(memory.has_value()) << (memory ? "" : memory.error().detail);
    memory_ = std::move(*memory);
    auto execution = jitllm::providers::cuda::OpenDeviceExecution(0);
    ASSERT_TRUE(execution.has_value()) << (execution ? "" : execution.error().detail);
    execution_ = std::move(*execution);
    auto storage = jitllm::providers::UringStorage::Create(4);
    ASSERT_TRUE(storage.has_value()) << storage.error().message();
    storage_ = std::move(*storage);

    const char* scratch = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const std::filesystem::path directory = scratch != nullptr
                                                ? std::filesystem::path(scratch)
                                                : std::filesystem::path(::testing::TempDir());
    std::filesystem::create_directories(directory);
    fd_ = ::open(directory.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    ASSERT_GE(fd_, 0);
    file_.resize(kExtent * kExtents);
    for (std::uint64_t i = 0; i < file_.size(); ++i) {
      file_[i] = static_cast<std::byte>((i * 29) + (i >> 21) + 3);
    }
    auto* staging = static_cast<std::byte*>(
        std::aligned_alloc(4096, file_.size()));  // NOLINT(cppcoreguidelines-no-malloc)
    std::memcpy(staging, file_.data(), file_.size());
    const ssize_t written = ::pwrite(fd_, staging, file_.size(), 0);
    std::free(staging);  // NOLINT(cppcoreguidelines-no-malloc)
    ASSERT_EQ(written, static_cast<ssize_t>(file_.size()));

    // The zone and the check's result: host VMM, persistent.
    host_ = memory_->Reserve(Bytes(kExtent * (kSlots + 1))).value();
    host_backing_ =
        memory_->Create(ClassOf(BackingKind::kHost), Bytes(kExtent * (kSlots + 1))).value();
    ASSERT_TRUE(memory_->Map(host_, Bytes(0), host_backing_).has_value());
    ASSERT_TRUE(
        memory_->SetAccess(host_, Bytes(0), Bytes(kExtent * (kSlots + 1)), Access::kReadWrite)
            .has_value());
    host_base_ = memory_->RangeOf(host_).value().base;
    // Two places for the weights: address space until a load maps backing.
    for (ReservationId& place : places_) {
      place = memory_->Reserve(Bytes(kExtent * kExtents)).value();
    }
    baseline_ = memory_->backings();

    domain_ = catalog_.AddDomain("gb10");
    for (std::size_t i = 0; i < kExtents; ++i) {
      extents_.push_back(
          catalog_
              .AddExtent(
                  {.domain = domain_,
                   .memory_class = jitllm::catalog::MemoryClass::kWeights,
                   .recovery = jitllm::catalog::Recovery::kFromArtifact,
                   .size = Bytes(kExtent),
                   .content = {.artifact = {}, .group = 0, .chunk = static_cast<std::uint32_t>(i)}})
              .value());
    }
    // The zone and result, pinned: the catalog counts them (a declared pool).
    for (std::size_t i = 0; i <= kSlots; ++i) {
      const ExtentId pinned =
          catalog_
              .AddExtent({.domain = domain_,
                          .memory_class = jitllm::catalog::MemoryClass::kStaging,
                          .recovery = jitllm::catalog::Recovery::kPinned,
                          .size = Bytes(kExtent),
                          .content = {}},
                         true)
              .value();
      if (i == kSlots) {
        result_ = pinned;
      }
    }
    stream_ = execution_->CreateStream().value();

    storage_lane_ = std::make_unique<StorageService>(
        *storage_,
        jitllm::providers::ReaderSettings{
            .alignment = 4096, .request_bytes = 2U << 20U, .retries = 3, .reads = 64, .waiters = 8},
        board_, QueueSettings{.capacity = 64, .reserved = 8, .batch = 16});
    device_lane_ = std::make_unique<DeviceService>(
        *execution_, std::span<const StreamId>(&stream_, 1), board_,
        DeviceSettings{.queue = {.capacity = 64, .reserved = 8, .batch = 16},
                       .handoff = 64,
                       .poll_sleep = std::chrono::microseconds(0)},
        GetParam() ? nullptr : memory_.get());  // one lane calls the provider
    if (GetParam()) {
      backing_lane_ = std::make_unique<jitllm::scheduler::BackingService>(
          memory_.get(), board_, QueueSettings{.capacity = 64, .reserved = 8, .batch = 16});
    }
    LandingZone landing{.slots = {}, .slot_bytes = Bytes(kExtent), .stream = 0};
    for (std::size_t i = 0; i < kSlots; ++i) {
      landing.slots.push_back(host_base_ + (i * kExtent));
    }
    scheduler_ = std::make_unique<Scheduler>(
        catalog_, board_, wake_,
        jitllm::scheduler::Lanes{.storage = storage_lane_.get(),
                                 .device = device_lane_.get(),
                                 .cpu = nullptr,
                                 .backing = backing_lane_.get()},
        SchedulerSettings{
            .tasks = 8, .budget = Bytes(kExtent * (kExtents + kSlots + 1)), .landing = landing});
    Place(0);
    threads_.emplace_back([this] { result_status_ = scheduler_->Run(); });
    threads_.emplace_back([this] { storage_lane_->Run(); });
    threads_.emplace_back([this] { device_lane_->RunSubmission(); });
    threads_.emplace_back([this] { device_lane_->RunCompletion(); });
    if (backing_lane_ != nullptr) {
      threads_.emplace_back([this] { backing_lane_->Run(); });
    }
  }

  void TearDown() override {
    if (scheduler_ != nullptr && !threads_.empty()) {
      Signals evicted;
      Post(StartRequest{
          .request = 999, .priority = 1, .program = std::make_unique<EvictAll>(evicted, extents_)});
      EXPECT_TRUE(WaitFor(evicted.retired));
      Stop();
    }
    EXPECT_EQ(memory_->backings(), baseline_);  // every extent's backing released
    if (host_.valid()) {
      EXPECT_TRUE(memory_->Unmap(host_, Bytes(0), Bytes(kExtent * (kSlots + 1))).has_value());
      EXPECT_TRUE(memory_->Release(host_backing_).has_value());
      EXPECT_TRUE(memory_->Free(host_).has_value());
    }
    for (const ReservationId place : places_) {
      EXPECT_TRUE(memory_->Free(place).has_value());
    }
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
  }

  void Stop() {
    scheduler_->RequestShutdown();
    threads_.front().join();
    storage_lane_->Close();
    device_lane_->Close();
    if (backing_lane_ != nullptr) {
      backing_lane_->Close();
    }
    threads_.clear();
    EXPECT_EQ(storage_->in_flight(), 0U);
    EXPECT_TRUE(execution_->DestroyStream(stream_).has_value());
  }

  // Registers every extent's source at place `which`.
  void Place(std::size_t which) {
    addresses_.clear();
    for (std::size_t i = 0; i < kExtents; ++i) {
      const std::uint64_t address =
          memory_->RangeOf(places_.at(which)).value().base + (i * kExtent);
      addresses_.push_back(address);
      ASSERT_TRUE(
          scheduler_
              ->SetSource(extents_[i],
                          PageSource{.read = {.fd = fd_,
                                              .offset = i * kExtent,
                                              .memory = nullptr,
                                              .length = LengthOf(i)},
                                     .landed = true,
                                     .destination = address,
                                     .backing = BackingPlace{.reservation = places_.at(which),
                                                             .offset = Bytes(i * kExtent),
                                                             .size = Bytes(kExtent),
                                                             .allocation_class =
                                                                 ClassOf(BackingKind::kDevice)}})
              .has_value());
    }
  }

  std::size_t ClassOf(BackingKind kind) const {
    for (std::size_t i = 0; i < memory_->Classes().size(); ++i) {
      if (memory_->Classes()[i].kind == kind) {
        return i;
      }
    }
    ADD_FAILURE() << "no allocation class of that kind";
    return 0;
  }
  void Post(Control&& control) {
    // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
    while (scheduler_->Post(std::move(control)) == PushResult::kFull) {
      std::this_thread::yield();
    }
  }
  // Loads (and checks) every extent; returns the task's outcome.
  int Load(std::uint64_t request, bool check, Signals& signals) {
    Post(StartRequest{
        .request = request,
        .priority = 1,
        .program = std::make_unique<LoadAndCheck>(signals, catalog_, extents_, addresses_, result_,
                                                  host_base_ + (kSlots * kExtent), file_, check)});
    EXPECT_TRUE(WaitFor(signals.retired));
    return signals.outcome.load();
  }
  // Runs `probe` on the scheduler thread until it returns true.
  void Owner(std::uint64_t request, std::function<bool()> probe) {
    Signals signals;
    Post(StartRequest{.request = request,
                      .priority = 1,
                      .program = std::make_unique<OnOwner>(signals, std::move(probe))});
    ASSERT_TRUE(WaitFor(signals.retired));
  }
  int Evict(std::uint64_t request) {
    Signals signals;
    Post(StartRequest{.request = request,
                      .priority = 1,
                      .program = std::make_unique<EvictAll>(signals, extents_)});
    EXPECT_TRUE(WaitFor(signals.retired));
    return signals.outcome.load();
  }

  std::unique_ptr<jitllm::providers::VmmProvider> memory_;
  std::unique_ptr<jitllm::providers::DeviceExecution> execution_;
  std::unique_ptr<jitllm::providers::UringStorage> storage_;
  jitllm::catalog::Catalog catalog_;
  jitllm::base::WakeFlag wake_;
  CompletionBoard board_{128, wake_};
  std::unique_ptr<StorageService> storage_lane_;
  std::unique_ptr<DeviceService> device_lane_;
  std::unique_ptr<jitllm::scheduler::BackingService> backing_lane_;
  std::unique_ptr<Scheduler> scheduler_;
  std::optional<std::expected<void, Fault>> result_status_;
  std::vector<std::jthread> threads_;  // the scheduler first

  int fd_ = -1;
  std::vector<std::byte> file_;
  ReservationId host_;
  jitllm::providers::BackingId host_backing_;
  std::uint64_t host_base_ = 0;
  std::array<ReservationId, 2> places_;
  std::vector<std::uint64_t> addresses_;
  std::size_t baseline_ = 0;
  jitllm::catalog::DomainId domain_;
  std::vector<ExtentId> extents_;
  ExtentId result_;
  StreamId stream_;
};

// Loads land in the zone and reach device VMM intact; eviction releases
// every backing, and reloads, in place and relocated, restore the bytes.
TEST_P(CudaPageIn, LandedLoadsReachDeviceVmmAndReloadIdentically) {
  Signals first;
  ASSERT_EQ(Load(1, true, first), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(first.mismatches.load(), 0);
  EXPECT_EQ(memory_->backings(), baseline_ + kExtents);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);

  ASSERT_EQ(Evict(2), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(memory_->backings(), baseline_);  // released, not pooled (D-033)
  for (const ExtentId extent : extents_) {
    EXPECT_EQ(catalog_.Describe(extent).value().state, ExtentState::kNonresident);
  }
  Signals again;
  ASSERT_EQ(Load(3, true, again), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(again.mismatches.load(), 0);

  ASSERT_EQ(Evict(4), static_cast<int>(TaskOutcome::kSucceeded));
  Owner(6, [&] {  // relocated, on the thread that owns the sources
    Place(1);
    return true;
  });
  Signals moved;
  ASSERT_EQ(Load(5, true, moved), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(moved.mismatches.load(), 0);
  const auto occupied = catalog_.OccupancyOf(domain_);
  EXPECT_EQ(occupied.loading, Bytes());
  EXPECT_EQ(occupied.quarantined, Bytes());
  EXPECT_EQ(occupied.held, Bytes());
  EXPECT_EQ(occupied.idle, Bytes(kExtent * kExtents));
}

// A request cancelled while a read or copy of one of its loads holds a
// landing slot (the cancellation is applied on the scheduler thread the
// moment one does): every read drains, no slot stays busy, and each extent
// ends resident (its copy completed) or nonresident with its backing
// released. Whether a read the cancellation met was cancelled by io_uring
// or completed first is not observed here.
TEST_P(CudaPageIn, CancellingWithLoadsInFlightDrainsThem) {
  Signals signals;
  constexpr std::uint64_t kRequest = 10;
  Post(StartRequest{
      .request = kRequest,
      .priority = 1,
      .program = std::make_unique<LoadAndCheck>(signals, catalog_, extents_, addresses_, result_,
                                                host_base_ + (kSlots * kExtent), file_, false)});
  std::size_t busy = 0;
  bool cancelled = false;
  Owner(11, [&] {
    busy = scheduler_->slots_busy();
    if (busy == 0 && scheduler_->loads() > 0) {
      return false;  // mapping still: no read has started
    }
    cancelled = scheduler_->Cancel(kRequest);
    return true;
  });
  EXPECT_GT(busy, 0U) << "every load finished before a read was seen in flight";
  EXPECT_TRUE(cancelled);
  ASSERT_TRUE(WaitFor(signals.retired));
  EXPECT_EQ(signals.outcome.load(), static_cast<int>(TaskOutcome::kCancelled));
  // Once the withdrawn loads have drained, on the scheduler thread.
  std::size_t resident = 0;
  std::size_t settled = 0;
  std::size_t busy_after = 0;
  Owner(12, [&] {
    if (scheduler_->loads() > 0) {
      return false;
    }
    for (const ExtentId extent : extents_) {
      const auto state = catalog_.Describe(extent).value().state;
      settled += state == ExtentState::kResident || state == ExtentState::kNonresident ? 1 : 0;
      resident += state == ExtentState::kResident ? 1 : 0;
    }
    busy_after = scheduler_->slots_busy();
    return true;
  });
  EXPECT_EQ(settled, kExtents);
  EXPECT_LT(resident, kExtents);
  EXPECT_EQ(busy_after, 0U);
  EXPECT_EQ(memory_->backings(), baseline_ + resident);
  ASSERT_EQ(Evict(13), static_cast<int>(TaskOutcome::kSucceeded));
  // What stays loads intact afterwards.
  Signals after;
  ASSERT_EQ(Load(50, true, after), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(after.mismatches.load(), 0);
}

// The other timing: cancelled while a landed copy is in flight (the device
// lane has taken its operation, and may or may not have queued it on the
// GPU yet; its fence has not been seen). A copy cannot be recalled, so
// each such load waits for its fence and publishes the whole extent; its
// slot is freed only then. The extents it published hold exactly the
// file's bytes.
TEST_P(CudaPageIn, CancellingDuringACopyWaitsForItsFence) {
  Signals signals;
  constexpr std::uint64_t kRequest = 20;
  Post(StartRequest{
      .request = kRequest,
      .priority = 1,
      .program = std::make_unique<LoadAndCheck>(signals, catalog_, extents_, addresses_, result_,
                                                host_base_ + (kSlots * kExtent), file_, false)});
  std::size_t copying = 0;
  bool cancelled = false;
  Owner(21, [&] {
    copying = scheduler_->copying();
    if (copying == 0 && scheduler_->loads() > 0) {
      return false;  // no copy in flight yet
    }
    cancelled = scheduler_->Cancel(kRequest);
    return true;
  });
  EXPECT_GT(copying, 0U) << "every load finished before a copy was seen in flight";
  EXPECT_TRUE(cancelled);
  ASSERT_TRUE(WaitFor(signals.retired));
  EXPECT_EQ(signals.outcome.load(), static_cast<int>(TaskOutcome::kCancelled));
  std::size_t resident = 0;
  std::size_t settled = 0;
  std::size_t busy_after = 0;
  Owner(22, [&] {
    if (scheduler_->loads() > 0) {
      return false;
    }
    for (const ExtentId extent : extents_) {
      const auto state = catalog_.Describe(extent).value().state;
      settled += state == ExtentState::kResident || state == ExtentState::kNonresident ? 1 : 0;
      resident += state == ExtentState::kResident ? 1 : 0;
    }
    busy_after = scheduler_->slots_busy();
    return true;
  });
  EXPECT_EQ(settled, kExtents);
  EXPECT_GE(resident, copying);  // every copy in flight completed and published
  EXPECT_LT(resident, kExtents);
  EXPECT_EQ(busy_after, 0U);
  EXPECT_EQ(memory_->backings(), baseline_ + resident);
  // What the cancelled request published is whole: each extent copied back
  // under a lease and compared with its chunk of the file.
  std::vector<ExtentId> published;
  std::vector<std::uint64_t> places;
  std::vector<std::size_t> chunks;
  for (std::size_t i = 0; i < kExtents; ++i) {
    if (catalog_.Describe(extents_[i]).value().state == ExtentState::kResident) {
      published.push_back(extents_[i]);
      places.push_back(addresses_[i]);
      chunks.push_back(i);
    }
  }
  Signals checked;
  Post(StartRequest{.request = 30,
                    .priority = 1,
                    .program = std::make_unique<LoadAndCheck>(
                        checked, catalog_, published, places, result_,
                        host_base_ + (kSlots * kExtent), file_, true, chunks)});
  ASSERT_TRUE(WaitFor(checked.retired));
  ASSERT_EQ(checked.outcome.load(), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(checked.mismatches.load(), 0);
  ASSERT_EQ(Evict(90), static_cast<int>(TaskOutcome::kSucceeded));
  Signals after;
  ASSERT_EQ(Load(91, true, after), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(after.mismatches.load(), 0);
}

INSTANTIATE_TEST_SUITE_P(VmmWork, CudaPageIn, ::testing::Bool(), [](const auto& info) {
  return info.param ? std::string("OnAVmmLane") : std::string("OnTheDeviceLane");
});

}  // namespace
