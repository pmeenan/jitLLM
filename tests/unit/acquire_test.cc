// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// BP-S3's acquisition (tests/support/paged_programs.h AcquireProgram) on
// the deterministic fakes: two models' weights and a workspace both
// models' closures share, in one catalog under one scheduler and one
// landing zone, with an execution budget that never holds both models'
// weights. Acquiring one model evicts only the other's weights, as many as
// its shortfall needs, never the workspace (in the closure, or protected
// when it is not); a closure that cannot fit is refused and evicts
// nothing. The paged harnesses' node runs the same
// program on the real providers (cuda_paged_node_test.cc,
// benchmarks/alternate_paged.cc).

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "memory/materialize.h"
#include "paged_programs.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/direct_reader.h"
#include "providers/fake/fake_device_execution.h"
#include "providers/fake/fake_device_memory.h"
#include "providers/fake/fake_storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/scheduler.h"
#include "scheduler/services.h"

namespace {

namespace sc = jitllm::scheduler;
namespace ts = jitllm::test_support;
using jitllm::base::Bytes;
using jitllm::catalog::Closure;
using jitllm::catalog::ExtentId;
using jitllm::catalog::ExtentState;
using jitllm::providers::StreamId;

constexpr std::uint64_t kSize = 64ULL * 1024;  // one extent, one slot
constexpr std::size_t kPerModel = 3;
constexpr std::size_t kSlots = 2;

std::byte* At(std::uint64_t address) {
  return reinterpret_cast<std::byte*>(address);  // NOLINT(performance-no-int-to-ptr)
}

class AcquireTest : public ::testing::Test {
 protected:
  void SetUp() override {
    zone_ = memory_.Reserve(Bytes(kSize * kSlots)).value();
    const auto zone_backing = memory_.Create(kHostClass, Bytes(kSize * kSlots)).value();
    ASSERT_TRUE(memory_.Map(zone_, Bytes(0), zone_backing).has_value());
    ASSERT_TRUE(memory_
                    .SetAccess(zone_, Bytes(0), Bytes(kSize * kSlots),
                               jitllm::providers::Access::kReadWrite)
                    .has_value());
    domain_ = catalog_.AddDomain("node");
    // The workspace both closures share: resident scratch, mapped by hand.
    workspace_ = catalog_
                     .AddExtent({.domain = domain_,
                                 .memory_class = jitllm::catalog::MemoryClass::kScratch,
                                 .recovery = jitllm::catalog::Recovery::kDiscardable,
                                 .size = Bytes(kSize),
                                 .content = {}},
                                true)
                     .value();
    for (std::size_t m = 0; m < 2; ++m) {
      places_.at(m) = memory_.Reserve(Bytes(kSize * kPerModel)).value();
      bases_.at(m) = memory_.RangeOf(places_.at(m)).value().base;
      files_.at(m).resize(kSize * kPerModel);
      for (std::size_t i = 0; i < files_.at(m).size(); ++i) {
        files_.at(m)[i] = static_cast<std::byte>((i * 131) + (m * 71) + 1);
      }
      fds_.at(m) = storage_.AddFile(files_.at(m));
      std::array<std::uint8_t, 32> artifact{};
      artifact[0] = static_cast<std::uint8_t>(m + 1);
      for (std::size_t i = 0; i < kPerModel; ++i) {
        weights_.at(m).push_back(
            catalog_
                .AddExtent({.domain = domain_,
                            .memory_class = jitllm::catalog::MemoryClass::kWeights,
                            .recovery = jitllm::catalog::Recovery::kFromArtifact,
                            .size = Bytes(kSize),
                            .content = {.artifact = artifact,
                                        .group = 0,
                                        .chunk = static_cast<std::uint32_t>(i)}})
                .value());
      }
    }
    stream_ = execution_.CreateStream().value();
  }

  // The scheduler with budget B (in extents), and every weight's source.
  void Build(std::uint64_t budget_extents) {
    budget_ = Bytes(budget_extents * kSize);
    board_ = std::make_unique<sc::CompletionBoard>(32, wake_);
    storage_lane_ = std::make_unique<sc::StorageService>(
        storage_,
        jitllm::providers::ReaderSettings{
            .alignment = 4096, .request_bytes = 16 * 1024, .retries = 2, .reads = 32, .waiters = 8},
        *board_, sc::QueueSettings{.capacity = 16, .reserved = 4, .batch = 16});
    device_lane_ = std::make_unique<sc::DeviceService>(
        execution_, std::span<const StreamId>(&stream_, 1), *board_,
        sc::DeviceSettings{.queue = {.capacity = 16, .reserved = 4, .batch = 16},
                           .handoff = 16,
                           .poll_sleep = std::chrono::microseconds(0)},
        &memory_);
    sc::LandingZone landing{.slots = {}, .slot_bytes = Bytes(kSize), .stream = 0};
    const std::uint64_t zone_base = memory_.RangeOf(zone_).value().base;
    for (std::size_t i = 0; i < kSlots; ++i) {
      landing.slots.push_back(zone_base + (i * kSize));
    }
    scheduler_ = std::make_unique<sc::Scheduler>(
        catalog_, *board_, wake_,
        sc::Lanes{.storage = storage_lane_.get(), .device = device_lane_.get(), .cpu = nullptr},
        sc::SchedulerSettings{.tasks = 16, .budget = budget_, .landing = landing});
    for (std::size_t m = 0; m < 2; ++m) {
      for (std::size_t i = 0; i < kPerModel; ++i) {
        ASSERT_TRUE(
            scheduler_
                ->SetSource(
                    weights_.at(m)[i],
                    sc::PageSource{.read = {.fd = fds_.at(m),
                                            .offset = i * kSize,
                                            .memory = nullptr,
                                            .length = kSize},
                                   .landed = true,
                                   .destination = bases_.at(m) + (i * kSize),
                                   .backing = sc::BackingPlace{.reservation = places_.at(m),
                                                               .offset = Bytes(i * kSize),
                                                               .size = Bytes(kSize),
                                                               .allocation_class = kDeviceClass}})
                .has_value());
      }
    }
  }

  void TearDown() override {
    if (scheduler_ == nullptr) {
      return;
    }
    std::vector<ExtentId> all = weights_[0];
    all.insert(all.end(), weights_[1].begin(), weights_[1].end());
    ts::Done done;
    ASSERT_TRUE(Run(std::make_unique<ts::EvictProgram>(done, all), done));
    scheduler_->RequestShutdown();
    for (int i = 0; i < 1000 && !scheduler_->Stopped(); ++i) {
      Round();
    }
    EXPECT_TRUE(scheduler_->Stopped().has_value());
    storage_lane_->Close();
    device_lane_->Close();
    for (int i = 0; i < 100; ++i) {
      Round();
    }
  }

  bool Round() {
    bool progress = storage_lane_->Turn(false);
    progress = device_lane_->SubmissionTurn() || progress;
    execution_.Drain();
    progress = device_lane_->CompletionTurn() || progress;
    return scheduler_->Turn() || progress;
  }

  // Starts a program and turns everything until it is destroyed; whether
  // it succeeded.
  bool Run(std::unique_ptr<sc::TaskProgram> program, ts::Done& done) {
    if (!scheduler_->Start(++request_, std::move(program), 1)) {
      return false;
    }
    for (int i = 0; i < 10000 && !done.gone.load(); ++i) {
      Round();
    }
    EXPECT_TRUE(done.gone.load());
    return done.outcome.load() == static_cast<int>(sc::TaskOutcome::kSucceeded);
  }

  Closure ModelClosure(std::size_t m) const {
    std::vector<ExtentId> extents = weights_.at(m);
    extents.push_back(workspace_);
    return catalog_.ClosureOfExtents(extents).value();
  }
  std::size_t Resident(std::size_t m) const {
    std::size_t resident = 0;
    for (const ExtentId extent : weights_.at(m)) {
      resident += catalog_.Describe(extent).value().state == ExtentState::kResident ? 1 : 0;
    }
    return resident;
  }
  bool Loaded(std::size_t m) const {
    return std::memcmp(At(bases_.at(m)), files_.at(m).data(), kSize * kPerModel) == 0;
  }
  std::uint64_t Occupied() const { return catalog_.OccupancyOf(domain_).Total().value(); }
  std::set<ExtentId> Of(std::size_t m) const {
    return {weights_.at(m).begin(), weights_.at(m).end()};
  }

  static constexpr std::size_t kDeviceClass = 0;
  static constexpr std::size_t kHostClass = 1;

  jitllm::providers::fake::FakeDeviceMemory memory_{Bytes(kSize), Bytes(kSize * 64)};
  jitllm::providers::fake::FakeStorage storage_{8, 4096};
  jitllm::providers::fake::FakeDeviceExecution execution_;
  jitllm::catalog::Catalog catalog_;
  jitllm::base::WakeFlag wake_;
  std::unique_ptr<sc::CompletionBoard> board_;
  std::unique_ptr<sc::StorageService> storage_lane_;
  std::unique_ptr<sc::DeviceService> device_lane_;
  std::unique_ptr<sc::Scheduler> scheduler_;

  jitllm::providers::ReservationId zone_;
  jitllm::catalog::DomainId domain_;
  ExtentId workspace_;
  std::array<jitllm::providers::ReservationId, 2> places_{};
  std::array<std::uint64_t, 2> bases_{};
  std::array<std::vector<std::byte>, 2> files_;
  std::array<int, 2> fds_{};
  std::array<std::vector<ExtentId>, 2> weights_;
  StreamId stream_;
  Bytes budget_;
  std::uint64_t request_ = 0;
};

// B holds the workspace and four weight extents: one model and a third of
// the other.
TEST_F(AcquireTest, EachAcquisitionEvictsOnlyTheOtherModelsWeights) {
  Build(1 + kPerModel + 1);
  for (int round = 0; round < 3; ++round) {
    for (std::size_t m = 0; m < 2; ++m) {
      ts::Done done;
      ts::AcquireReport report;
      ASSERT_TRUE(
          Run(std::make_unique<ts::AcquireProgram>(done, ModelClosure(m), domain_, budget_, report),
              done))
          << "round " << round << " model " << m;
      EXPECT_EQ(Resident(m), kPerModel);
      EXPECT_TRUE(Loaded(m));
      EXPECT_EQ(catalog_.Describe(workspace_).value().state, ExtentState::kResident);
      EXPECT_LE(Occupied(), budget_.value());
      const std::set<ExtentId> other = Of(1 - m);
      for (const ExtentId extent : report.evicted) {
        EXPECT_TRUE(other.contains(extent));
      }
      if (round == 0 && m == 0) {
        EXPECT_TRUE(report.evicted.empty());  // room for the first
        EXPECT_EQ(report.loaded, kPerModel);
      } else {
        // The other model keeps what the budget leaves it; only what is
        // missing loads.
        EXPECT_EQ(report.evicted.size(), 2U);
        EXPECT_EQ(Resident(1 - m), 1U);
        EXPECT_EQ(report.loaded, round == 0 ? kPerModel : 2U);
      }
    }
  }
}

// A closure that omits the workspace: unprotected, the discardable
// workspace would be the first victim; protected (as the paged node
// protects it), only the other model's weights go.
TEST_F(AcquireTest, TheProtectedWorkspaceIsNeverAVictim) {
  Build(1 + kPerModel + 1);
  const auto weights_only = [&](std::size_t m) {
    return catalog_.ClosureOfExtents(weights_.at(m)).value();
  };
  {
    ts::Done done;
    ts::AcquireReport report;
    ASSERT_TRUE(Run(std::make_unique<ts::AcquireProgram>(done, weights_only(0), domain_, budget_,
                                                         report, std::vector{workspace_}),
                    done));
  }
  const auto unprotected =
      jitllm::memory::PlanMaterialization(catalog_, domain_, budget_, weights_only(1));
  ASSERT_FALSE(unprotected.victims.victims.empty());
  EXPECT_EQ(unprotected.victims.victims.front().extent, workspace_);

  ts::Done done;
  ts::AcquireReport report;
  ASSERT_TRUE(Run(std::make_unique<ts::AcquireProgram>(done, weights_only(1), domain_, budget_,
                                                       report, std::vector{workspace_}),
                  done));
  EXPECT_EQ(catalog_.Describe(workspace_).value().state, ExtentState::kResident);
  EXPECT_EQ(report.evicted.size(), 2U);
  const std::set<ExtentId> first = Of(0);
  for (const ExtentId extent : report.evicted) {
    EXPECT_TRUE(first.contains(extent));
  }
  EXPECT_EQ(Resident(1), kPerModel);
  EXPECT_TRUE(Loaded(1));
  EXPECT_LE(Occupied(), budget_.value());
}

// Even with every eligible victim gone, the closure exceeds B: refused as
// over budget before anything is evicted or loaded.
TEST_F(AcquireTest, AClosureThatCannotFitIsRefusedAndEvictsNothing) {
  Build(kPerModel);
  {
    ts::Done done;
    ts::AcquireReport report;
    std::vector<ExtentId> two = {weights_[1][0], weights_[1][1]};
    ASSERT_TRUE(Run(std::make_unique<ts::AcquireProgram>(
                        done, catalog_.ClosureOfExtents(two).value(), domain_, budget_, report),
                    done));
  }
  ts::Done done;
  ts::AcquireReport report;
  EXPECT_FALSE(Run(
      std::make_unique<ts::AcquireProgram>(done, ModelClosure(0), domain_, budget_, report), done));
  EXPECT_EQ(done.error.load(), static_cast<int>(sc::WorkError::kOverBudget));
  EXPECT_TRUE(report.evicted.empty());
  EXPECT_EQ(Resident(0), 0U);
  EXPECT_EQ(Resident(1), 2U);
}

}  // namespace
