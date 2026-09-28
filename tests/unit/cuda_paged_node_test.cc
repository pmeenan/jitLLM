// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// BP-S3's node on a real device (tests/support/paged_node.h): two models,
// each on its own stream, share one catalog domain, one scheduler with its
// lanes, one landing zone and one workspace, under an execution budget
// that holds the larger model's weights and half the smaller's. They
// alternate: each acquisition (AcquireProgram) evicts only the other
// model's weights, releasing their backing, and pages in what is missing
// through the zone into device VMM; each model's job then copies every
// weight extent through the shared workspace to pinned staging on its own
// stream, and the bytes must be the file's. The occupancy never exceeds
// the budget, and teardown leaves no backing. The real models alternate in
// benchmarks/alternate_paged.cc.

#include <cuda_runtime.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"
#include "paged_node.h"
#include "paged_programs.h"
#include "scheduler/commands.h"
#include "scheduler/scheduler.h"

namespace {

namespace sc = jitllm::scheduler;
namespace ts = jitllm::test_support;
using jitllm::base::Bytes;
using jitllm::catalog::ExtentId;
using jitllm::catalog::ExtentState;

constexpr std::uint64_t kExtent = ts::kPagedExtent;
constexpr std::array<std::size_t, 2> kExtents = {4, 3};  // model 0 is the larger

// A synthetic model: its weights in an unnamed direct-I/O file, read into
// managed device backing at one place, and pinned staging to read them
// back through the shared workspace.
class Model final : public ts::PagedModel {
 public:
  Model(ts::PagedNode& node, std::uint32_t index) : node_(node), index_(index) {}

  void Setup(const std::filesystem::path& directory) {
    const std::size_t extents = kExtents.at(index_);
    file_.resize(extents * kExtent);
    for (std::uint64_t i = 0; i < file_.size(); ++i) {
      file_[i] = static_cast<std::byte>((i * 29) + (i >> 21) + (std::uint64_t{index_} * 101) + 3);
    }
    fd_ = ::open(directory.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    ASSERT_GE(fd_, 0);
    auto* aligned = static_cast<std::byte*>(
        std::aligned_alloc(4096, file_.size()));  // NOLINT(cppcoreguidelines-no-malloc)
    std::memcpy(aligned, file_.data(), file_.size());
    const ssize_t written = ::pwrite(fd_, aligned, file_.size(), 0);
    std::free(aligned);  // NOLINT(cppcoreguidelines-no-malloc)
    ASSERT_EQ(written, static_cast<ssize_t>(file_.size()));
    place_ = node_.memory().Reserve(Bytes(file_.size())).value();
    base_ = node_.memory().RangeOf(place_).value().base;
    std::array<std::uint8_t, 32> artifact{};
    artifact[0] = static_cast<std::uint8_t>(index_ + 1);
    for (std::uint32_t i = 0; i < extents; ++i) {
      weights_.push_back(node_.catalog()
                             .AddExtent({.domain = node_.domain(),
                                         .memory_class = jitllm::catalog::MemoryClass::kWeights,
                                         .recovery = jitllm::catalog::Recovery::kFromArtifact,
                                         .size = Bytes(kExtent),
                                         .content = {.artifact = artifact, .group = 0, .chunk = i}})
                             .value());
    }
    auto staging = node_.Pinned(file_.size(), static_cast<int>(index_), staging_);
    ASSERT_TRUE(staging.has_value());
    staging_bytes_ = static_cast<std::byte*>(*staging);
  }

  void Register() {
    for (std::size_t i = 0; i < weights_.size(); ++i) {
      ASSERT_TRUE(
          node_.scheduler()
              .SetSource(weights_[i],
                         sc::PageSource{
                             .read = {.fd = fd_,
                                      .offset = i * kExtent,
                                      .memory = nullptr,
                                      .length = kExtent},
                             .landed = true,
                             .destination = base_ + (i * kExtent),
                             .backing = sc::BackingPlace{.reservation = place_,
                                                         .offset = Bytes(i * kExtent),
                                                         .size = Bytes(kExtent),
                                                         .allocation_class = node_.device_class()}})
              .has_value());
    }
    std::vector<ExtentId> all = weights_;
    all.insert(all.end(), node_.activations().extents.begin(), node_.activations().extents.end());
    all.insert(all.end(), node_.pool().extents.begin(), node_.pool().extents.end());
    all.insert(all.end(), staging_.begin(), staging_.end());
    closure_ = node_.catalog().ClosureOfExtents(all).value();
    fence_ = node_.catalog().ClosureOfExtents(staging_).value();
  }

  // Every weight extent through the workspace to the staging, on this
  // model's stream, under a lease on the whole closure.
  ts::Status ReadBack() {
    std::memset(staging_bytes_, 0, file_.size());
    const std::uint64_t workspace = node_.activations().base;
    const std::uint64_t base = base_;
    std::byte* staging = staging_bytes_;
    const std::size_t extents = weights_.size();
    return node_.Job(
        closure_,
        [=](jitllm::providers::NativeStream native) {
          auto* stream = static_cast<cudaStream_t>(native.handle);
          for (std::size_t i = 0; i < extents; ++i) {
            // NOLINTBEGIN(performance-no-int-to-ptr)
            if (cudaMemcpyAsync(reinterpret_cast<void*>(workspace),
                                reinterpret_cast<void*>(base + (i * kExtent)), kExtent,
                                cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
                cudaMemcpyAsync(staging + (i * kExtent), reinterpret_cast<void*>(workspace),
                                kExtent, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
              // NOLINTEND(performance-no-int-to-ptr)
              return sc::JobResult::kUnknown;  // a runtime error, even the first (AfterRefusal)
            }
          }
          return sc::JobResult::kQueued;
        },
        "reading the weights back", index_);
  }
  bool Intact() const { return std::memcmp(staging_bytes_, file_.data(), file_.size()) == 0; }

  std::size_t Resident() const {
    std::size_t resident = 0;
    for (const ExtentId extent : weights_) {
      resident += node_.catalog().Describe(extent).value().state == ExtentState::kResident ? 1 : 0;
    }
    return resident;
  }
  const jitllm::catalog::Closure& closure() const { return closure_; }
  const std::vector<ExtentId>& weights() const { return weights_; }

  std::uint32_t stream() const override { return index_; }
  const jitllm::catalog::Closure& fence_closure() const override { return fence_; }
  std::vector<ExtentId> managed_extents() const override { return weights_; }
  ts::Status Release() override {
    const bool freed = !place_.valid() || node_.memory().Free(place_).has_value();
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
    if (!freed) {
      return std::unexpected("a model's place still has mappings");
    }
    return {};
  }

 private:
  ts::PagedNode& node_;
  std::uint32_t index_;
  std::vector<std::byte> file_;
  int fd_ = -1;
  jitllm::providers::ReservationId place_;
  std::uint64_t base_ = 0;
  std::vector<ExtentId> weights_;
  std::vector<ExtentId> staging_;
  std::byte* staging_bytes_ = nullptr;
  jitllm::catalog::Closure closure_;
  jitllm::catalog::Closure fence_;  // the staging, always resident
};

TEST(CudaPagedNodeTest, TwoModelsAlternateAndEachEvictsOnlyTheOthersWeights) {
  const char* scratch = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  const std::filesystem::path directory = scratch != nullptr
                                              ? std::filesystem::path(scratch)
                                              : std::filesystem::path(::testing::TempDir());
  std::filesystem::create_directories(directory);

  ts::PagedNode node({.compute_streams = 2, .slots = 4, .inline_lanes = false, .coalesce = false});
  Model first(node, 0);
  Model second(node, 1);
  const std::array<Model*, 2> models = {&first, &second};
  const std::array<ts::PagedModel*, 2> teardown = {&first, &second};
  ts::Status ran = node.Open();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  for (Model* model : models) {
    model->Setup(directory);
  }
  ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent).has_value());
  // B: what is resident now, the larger model's weights and half the
  // smaller's (rounded up to an extent).
  const std::uint64_t fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  const std::uint64_t budget = fixed + ((kExtents[0] + 2) * kExtent);
  ASSERT_TRUE(node.Start(Bytes(budget)).has_value());
  for (Model* model : models) {
    model->Register();
  }
  node.Run();

  for (int round = 0; round < 3 && ran; ++round) {
    for (std::uint32_t m = 0; m < 2 && ran; ++m) {
      Model& model = *models.at(m);
      const Model& other = *models.at(1 - m);
      ts::AcquireReport report;
      ran = node.Acquire(model.closure(), report, "an acquisition");
      if (!ran) {
        break;
      }
      const std::set<ExtentId> others(other.weights().begin(), other.weights().end());
      for (const ExtentId extent : report.evicted) {
        EXPECT_TRUE(others.contains(extent)) << "round " << round << " model " << m;
      }
      if (round > 0 || m > 0) {
        EXPECT_FALSE(report.evicted.empty()) << "round " << round << " model " << m;
      }
      ran = model.ReadBack();
      if (!ran) {
        break;
      }
      EXPECT_TRUE(model.Intact()) << "round " << round << " model " << m;
      std::uint64_t occupied = 0;
      std::size_t resident = 0;
      std::size_t others_resident = 0;
      ran = node.Call(
          [&]() -> ts::Status {
            occupied = node.catalog().OccupancyOf(node.domain()).Total().value();
            resident = model.Resident();
            others_resident = other.Resident();
            return {};
          },
          "reading the occupancy");
      EXPECT_LE(occupied, budget);
      EXPECT_EQ(resident, kExtents.at(m));
      EXPECT_LT(others_resident, kExtents.at(1 - m));
    }
  }
  EXPECT_TRUE(ran.has_value()) << ran.error();
  const ts::Status finished = node.TearDown(teardown);
  EXPECT_TRUE(finished.has_value()) << finished.error();
}

}  // namespace
