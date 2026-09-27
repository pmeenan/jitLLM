// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// jitLLM's EXL3 launchers on a GB10 (label `gpu`; docs/backend-proof.md,
// P3), on a synthetic linear shaped like Qwen2.5-0.5B's q_proj (896 × 896,
// K = 4, mcg; random trellis words and ±1 side vectors, as P0's synthetic
// kernel cases):
// - every path of the linear (packed through the GEMM and the GEMV, the
//   fused gate/up multi-GEMM, the reconstruction path and its fused form)
//   gives identical bits with its operands in cudaMalloc memory and in
//   device VMM (BP-N3), and the paths agree with each other to within F16
//   rounding;
// - the over-read probe: with every operand ending flush against an
//   unmapped VMM granule, and then starting flush after one, every path
//   runs without a fault and gives the same bits, so no kernel reads or
//   writes outside its operands;
// - what does not fit is refused before launch: a cooperative grid larger
//   than the device holds at once, a lock area another live context uses,
//   and anything after a fault, which returns as a fault;
// - the registry binds each implementation to its own path only, and a
//   stale or foreign declaration binds nothing (BP-S2, BP-S4).
// The per-linear exactness against upstream is the sweep's
// (docs/experiments/backend-proof-p3/), not this test's.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <print>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "execution/registry.h"
#include "kernels/exl3/implementations.h"
#include "kernels/exl3/launch.h"
#include "kernels/exl3/linear.h"
#include "kernels/exl3/recon_gemm.h"
#include "kernels/exl3/validate.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"

namespace {

namespace exl3 = jitllm::kernels::exl3;
using exl3::Output;
using jitllm::base::Bytes;
using jitllm::providers::Access;
using jitllm::providers::BackingKind;
using jitllm::providers::DeviceExecution;
using jitllm::providers::FenceState;
using jitllm::providers::StreamId;

constexpr int kK = 896;
constexpr int kN = 896;
constexpr int kBits = 4;

// exl3-recon-pin.json's algorithms for 896 × 896 at 145 rows (F16 and F32
// outputs alike) and at 1,024 rows.
constexpr exl3::LtAlgorithm kPin145{{67, 316, 1, 0, 0, 66, 35, 0, 0}};
constexpr exl3::LtAlgorithm kPin1024{{67, 409, 1, 0, 0, 30, 35, 0, 0}};

std::uint64_t Mix(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

std::vector<__half> Halves(std::size_t count, std::uint64_t seed, float scale) {
  std::vector<__half> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto bits = static_cast<std::int64_t>(Mix(seed + i) >> 53U);
    values[i] = __float2half_rn(static_cast<float>(bits - 1024) / 1024.0F * scale);
  }
  return values;
}

std::vector<__half> Signs(std::size_t count, std::uint64_t seed, float scale) {
  std::vector<__half> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = __float2half_rn((Mix(seed + i) & 1U) != 0 ? scale : -scale);
  }
  return values;
}

std::vector<std::uint16_t> Words(std::size_t count, std::uint64_t seed) {
  std::vector<std::uint16_t> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = static_cast<std::uint16_t>(Mix(seed + i) >> 48U);
  }
  return values;
}

// Where operands go: cudaMalloc, device VMM, or device VMM with each operand
// flush against an unmapped granule, after it or before it.
enum class Placement : std::uint8_t { kCudaMalloc, kDeviceVmm, kFlushEnd, kFlushStart };

// The outputs of every path, as bytes.
struct Results {
  std::vector<std::byte> gemm;            // 16 rows, F16, with bias
  std::vector<std::byte> gemv;            // 1 row, F16, with bias
  std::vector<std::byte> gemm1;           // the same row through the GEMM
  std::vector<std::byte> multi;           // 8 rows, F32, two outputs
  std::vector<std::byte> recon;           // 145 rows, F32
  std::vector<std::byte> fused;           // 1,024 rows, F16, with bias
  std::vector<std::byte> fused_as_recon;  // the same 1,024 rows unfused (reported, not pinned)
};

class Exl3LinearTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    memory_ = std::move(jitllm::providers::cuda::OpenDeviceMemory(0).value());
    execution_ = std::move(jitllm::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    for (std::size_t i = 0; i < memory_->Classes().size(); ++i) {
      if (memory_->Classes()[i].kind == BackingKind::kDevice) {
        device_class_ = i;
      }
    }
    granule_ = memory_->Granularity().value();
  }

  void TearDown() override {
    Finish();
    ASSERT_TRUE(execution_->DestroyStream(stream_).has_value());
    FreeAll();
  }

  void FreeAll() {
    for (void* pointer : malloced_) {
      EXPECT_EQ(cudaFree(pointer), cudaSuccess);
    }
    malloced_.clear();
    for (const Mapped& m : mapped_) {
      ASSERT_TRUE(memory_->Unmap(m.reservation, Bytes(m.offset), Bytes(m.size)).has_value());
      ASSERT_TRUE(memory_->Release(m.backing).has_value());
      ASSERT_TRUE(memory_->Free(m.reservation).has_value());
    }
    mapped_.clear();
  }

  // `bytes` at an address aligned to 256 bytes (cudaMalloc, device VMM),
  // or, flush, ending exactly at the end of its mapping (the next granule
  // unmapped) or starting exactly at its start (the one before unmapped).
  std::uint64_t Allocate(Placement placement, std::uint64_t bytes) {
    if (placement == Placement::kCudaMalloc) {
      void* pointer = nullptr;
      EXPECT_EQ(cudaMalloc(&pointer, bytes), cudaSuccess);
      malloced_.push_back(pointer);
      return reinterpret_cast<std::uint64_t>(pointer);
    }
    const std::uint64_t mapped = (bytes + granule_ - 1) / granule_ * granule_;
    const bool guarded = placement == Placement::kFlushEnd || placement == Placement::kFlushStart;
    const auto reservation = memory_->Reserve(Bytes(mapped + (guarded ? granule_ : 0))).value();
    const auto backing = memory_->Create(device_class_, Bytes(mapped)).value();
    const std::uint64_t offset = placement == Placement::kFlushStart ? granule_ : 0;
    EXPECT_TRUE(memory_->Map(reservation, Bytes(offset), backing).has_value());
    EXPECT_TRUE(memory_->SetAccess(reservation, Bytes(offset), Bytes(mapped), Access::kReadWrite)
                    .has_value());
    mapped_.push_back({reservation, backing, offset, mapped});
    const std::uint64_t base = memory_->RangeOf(reservation).value().base + offset;
    return placement == Placement::kFlushEnd ? base + mapped - bytes : base;
  }

  void Finish() {
    const auto fence = execution_->Record(stream_).value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    FenceState state = FenceState::kPending;
    while ((state = execution_->Query(fence).value()) == FenceState::kPending &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ASSERT_EQ(state, FenceState::kComplete);
    ASSERT_TRUE(execution_->Release(fence).has_value());
  }

  // Waits for the device, since a pageable copy may return before its DMA
  // lands and runs on the legacy stream, not the provider's.
  template <typename T>
  std::uint64_t Upload(Placement placement, const std::vector<T>& values) {
    const std::uint64_t bytes = values.size() * sizeof(T);
    const std::uint64_t address = Allocate(placement, bytes);
    EXPECT_EQ(cudaMemcpy(reinterpret_cast<void*>(address), values.data(), bytes,  // NOLINT
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return address;
  }

  std::vector<std::byte> Download(std::uint64_t address, std::uint64_t bytes) {
    Finish();
    std::vector<std::byte> host(bytes);
    EXPECT_EQ(cudaMemcpy(host.data(), reinterpret_cast<const void*>(address), bytes,  // NOLINT
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return host;
  }

  std::unique_ptr<exl3::LaunchContext> Context(Placement placement) {
    const std::uint64_t locks = Allocate(
        placement == Placement::kCudaMalloc ? Placement::kCudaMalloc : Placement::kDeviceVmm,
        exl3::kLockBytes);
    auto launch = exl3::LaunchContext::Create(0, *execution_, stream_, locks);
    EXPECT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    return launch ? std::move(*launch) : nullptr;
  }

  exl3::Weights Linear(Placement placement, std::uint64_t seed) {
    exl3::Weights w{.trellis = 0, .suh = 0, .svh = 0, .k = kK, .n = kN, .bits = kBits};
    w.trellis = Upload(placement, Words(exl3::TrellisBytes(w) / 2, seed));
    w.suh = Upload(placement, Signs(kK, seed + 1, 1.0F / std::sqrt(static_cast<float>(kK))));
    w.svh = Upload(placement, Signs(kN, seed + 2, 1.0F));
    return w;
  }

  static std::uint64_t Bytes16(int rows, int columns) {
    return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns) * 2;
  }

  // Runs every path with every operand placed as asked.
  Results Run(Placement placement) {
    Results r;
    auto launch = Context(placement);
    auto gemm = exl3::ReconGemm::Create();
    EXPECT_TRUE(gemm.has_value()) << (gemm ? "" : gemm.error().detail);
    if (!launch || !gemm) {
      return r;
    }
    const exl3::Weights w = Linear(placement, 100);
    const std::uint64_t bias = Upload(placement, Halves(kN, 7, 0.5F));
    auto check = [](const std::expected<void, exl3::KernelFailure>& result) {
      EXPECT_TRUE(result.has_value()) << (result ? "" : result.error().detail);
    };

    // Packed through the GEMM: 16 rows at q_proj's tuned plan.
    {
      const std::uint64_t x = Upload(placement, Halves(Bytes16(16, kK) / 2, 1, 1.0F));
      const exl3::LinearOperands o{.weights = w,
                                   .x = x,
                                   .a_had = Allocate(placement, Bytes16(16, kK)),
                                   .y = Allocate(placement, Bytes16(16, kN)),
                                   .output = Output::kF16,
                                   .m = 16};
      check(exl3::PackedGemmLinear(*launch, o, {.shape = 2, .blocks = 14}, bias));
      r.gemm = Download(o.y, Bytes16(16, kN));
    }
    // One row through the GEMV, at upstream's choice, and through the GEMM.
    {
      const std::uint64_t x = Upload(placement, Halves(kK, 2, 1.0F));
      exl3::LinearOperands o{.weights = w,
                             .x = x,
                             .a_had = Allocate(placement, Bytes16(1, kK)),
                             .y = Allocate(placement, Bytes16(1, kN)),
                             .output = Output::kF16,
                             .m = 1};
      const auto plan = launch->UpstreamGemv(w, Output::kF16, 1);
      EXPECT_TRUE(plan.has_value() && plan->has_value());
      if (plan && *plan) {
        EXPECT_EQ((*plan)->config, 0);
        check(exl3::PackedGemvLinear(*launch, o, **plan, bias));
      }
      r.gemv = Download(o.y, Bytes16(1, kN));
      o.y = Allocate(placement, Bytes16(1, kN));
      check(exl3::PackedGemmLinear(*launch, o, {.shape = 2, .blocks = 20}, bias));
      r.gemm1 = Download(o.y, Bytes16(1, kN));
    }
    // The fused gate/up multi-GEMM: two linears of one input, F32 outputs.
    {
      const exl3::Weights second = Linear(placement, 200);
      const std::vector<std::uint64_t> table{w.trellis,  second.trellis, w.suh,
                                             second.suh, w.svh,          second.svh};
      const std::uint64_t tables = Upload(placement, table);
      const std::uint64_t x = Upload(placement, Halves(Bytes16(8, kK) / 2, 3, 1.0F));
      const exl3::MultiLinearOperands o{.first = w,
                                        .second = second,
                                        .trellis_table = tables,
                                        .suh_table = tables + 16,
                                        .svh_table = tables + 32,
                                        .x = x,
                                        .a_had = Allocate(placement, 2 * Bytes16(8, kK)),
                                        .y = Allocate(placement, 4 * Bytes16(8, kN)),
                                        .output = Output::kF32,
                                        .m = 8};
      check(exl3::MultiLinear(*launch, o, {.shape = 2, .blocks = 14, .concurrency = 2}));
      r.multi = Download(o.y, 4 * Bytes16(8, kN));
    }
    // The reconstruction path at 145 rows (F32 output) and the fused one at
    // 1,024 rows (F16 output, with bias), each with its pinned algorithm.
    {
      const std::uint64_t x = Upload(placement, Halves(Bytes16(145, kK) / 2, 4, 1.0F));
      const exl3::ReconstructedOperands o{
          .weights = w,
          .x = x,
          .xh = Allocate(placement, Bytes16(145, kK)),
          .w = Allocate(placement, exl3::ReconstructScratchBytes(w)),
          .y = Allocate(placement, 2 * Bytes16(145, kN)),
          .output = Output::kF32,
          .m = 145,
          .bias = 0};
      const std::vector<exl3::LtAlgorithm> pins{kPin145};
      check(exl3::ReconstructedLinear(*launch, **gemm, o, false, pins));
      r.recon = Download(o.y, 2 * Bytes16(145, kN));
    }
    {
      const std::uint64_t x = Upload(placement, Halves(Bytes16(1024, kK) / 2, 5, 1.0F));
      exl3::ReconstructedOperands o{.weights = w,
                                    .x = x,
                                    .xh = Allocate(placement, Bytes16(1024, kK)),
                                    .w = Allocate(placement, exl3::ReconstructScratchBytes(w)),
                                    .y = Allocate(placement, Bytes16(1024, kN)),
                                    .output = Output::kF16,
                                    .m = 1024,
                                    .bias = bias};
      const std::vector<exl3::LtAlgorithm> pins{kPin1024};
      check(exl3::ReconstructedLinear(*launch, **gemm, o, true, pins));
      r.fused = Download(o.y, Bytes16(1024, kN));
      o.y = Allocate(placement, Bytes16(1024, kN));
      check(exl3::ReconstructedLinear(*launch, **gemm, o, false, pins));
      r.fused_as_recon = Download(o.y, Bytes16(1024, kN));
    }
    Finish();
    EXPECT_FALSE(launch->faulted());
    return r;
  }

  std::unique_ptr<jitllm::providers::VmmProvider> memory_;
  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::size_t device_class_ = 0;
  std::uint64_t granule_ = 0;
  std::vector<void*> malloced_;
  struct Mapped {
    jitllm::providers::ReservationId reservation;
    jitllm::providers::BackingId backing;
    std::uint64_t offset;
    std::uint64_t size;
  };
  std::vector<Mapped> mapped_;
};

// Values of F16 or F32 bytes, for closeness checks.
std::vector<float> Floats(const std::vector<std::byte>& bytes, bool fp32) {
  std::vector<float> values;
  if (fp32) {
    values.resize(bytes.size() / 4);
    std::memcpy(values.data(), bytes.data(), bytes.size());
  } else {
    std::vector<__half> halves(bytes.size() / 2);
    std::memcpy(halves.data(), bytes.data(), bytes.size());
    for (const __half h : halves) {
      values.push_back(__half2float(h));
    }
  }
  return values;
}

// Root-mean-square difference relative to the reference's RMS.
double RelativeRms(const std::vector<float>& a, const std::vector<float>& reference) {
  double error = 0;
  double norm = 0;
  for (std::size_t i = 0; i < a.size() && i < reference.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(reference[i]);
    error += d * d;
    norm += static_cast<double>(reference[i]) * static_cast<double>(reference[i]);
  }
  return norm == 0 ? 1.0 : std::sqrt(error / norm);
}

TEST_F(Exl3LinearTest, EveryPathIsExactAcrossMemoryKindsAndThePathsAgree) {
  const Results malloced = Run(Placement::kCudaMalloc);
  const Results vmm = Run(Placement::kDeviceVmm);
  ASSERT_FALSE(malloced.gemm.empty());
  EXPECT_EQ(malloced.gemm, vmm.gemm);
  EXPECT_EQ(malloced.gemv, vmm.gemv);
  EXPECT_EQ(malloced.gemm1, vmm.gemm1);
  EXPECT_EQ(malloced.multi, vmm.multi);
  EXPECT_EQ(malloced.recon, vmm.recon);
  EXPECT_EQ(malloced.fused, vmm.fused);
  EXPECT_EQ(malloced.fused_as_recon, vmm.fused_as_recon);
  // Different kernels for the same product: close, not equal.
  const double gemv = RelativeRms(Floats(malloced.gemv, false), Floats(malloced.gemm1, false));
  const double fused =
      RelativeRms(Floats(malloced.fused, false), Floats(malloced.fused_as_recon, false));
  EXPECT_LT(gemv, 1e-2);
  EXPECT_LT(fused, 1e-2);
  std::println(
      "GEMV against GEMM at one row: relative RMS {:.3g}; fused against unfused "
      "reconstruction at 1,024 rows: {:.3g}",
      gemv, fused);
}

TEST_F(Exl3LinearTest, TheRegistryBindsEachPathToItsOwnCalls) {
  const auto declared = exl3::Implementations();
  ASSERT_EQ(declared.size(), 5U);
  auto registry = jitllm::execution::Registry::Create(declared);
  ASSERT_TRUE(registry.has_value());
  for (const auto& implementation : declared) {
    auto kernel = exl3::Kernel::Bind(implementation);
    ASSERT_TRUE(kernel.has_value()) << implementation.name;
    EXPECT_EQ(kernel->name(), implementation.name);
    EXPECT_EQ(kernel->operation(), implementation.operation);
    EXPECT_EQ(implementation.source, "exllamav3");
    EXPECT_EQ(implementation.revision.size(), 64U);
  }
  auto stale = declared[0];
  stale.variant += " (changed)";
  EXPECT_FALSE(exl3::Kernel::Bind(stale).has_value());
  auto foreign = declared[0];
  foreign.name = "exl3.linear.unknown";
  EXPECT_FALSE(exl3::Kernel::Bind(foreign).has_value());

  // A GEMV implementation refuses a GEMM plan, before anything is queued.
  auto launch = Context(Placement::kCudaMalloc);
  ASSERT_NE(launch, nullptr);
  const exl3::Kernel gemv = exl3::Kernel::Bind(declared[1]).value();
  ASSERT_EQ(gemv.name(), "exl3.linear.gemv");
  const exl3::LinearOperands o{.weights = Linear(Placement::kCudaMalloc, 1),
                               .x = Allocate(Placement::kCudaMalloc, Bytes16(1, kK)),
                               .a_had = Allocate(Placement::kCudaMalloc, Bytes16(1, kK)),
                               .y = Allocate(Placement::kCudaMalloc, Bytes16(1, kN)),
                               .output = Output::kF16,
                               .m = 1};
  const auto refused = gemv.Run(*launch, o, exl3::GemmPlan{.shape = 2, .blocks = 14}, 0);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().error, exl3::KernelError::kRejected);
  EXPECT_FALSE(launch->faulted());
}

TEST_F(Exl3LinearTest, RefusesGridsThatAreNotCoresident) {
  auto launch = Context(Placement::kCudaMalloc);
  ASSERT_NE(launch, nullptr);
  const auto coresident = launch->GemmCoresident(kBits, 2, Output::kF16);
  ASSERT_TRUE(coresident.has_value());
  EXPECT_GE(*coresident, launch->sm_count());
  const exl3::LinearOperands o{.weights = Linear(Placement::kCudaMalloc, 1),
                               .x = Allocate(Placement::kCudaMalloc, Bytes16(16, kK)),
                               .a_had = Allocate(Placement::kCudaMalloc, Bytes16(16, kK)),
                               .y = Allocate(Placement::kCudaMalloc, Bytes16(16, kN)),
                               .output = Output::kF16,
                               .m = 16};
  // q_proj at shape 2 has 196 split-K slices, more than fit at once.
  const auto refused = launch->Gemm(o, {.shape = 2, .blocks = *coresident + 1});
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().error, exl3::KernelError::kRejected);
  EXPECT_TRUE(launch->Gemm(o, {.shape = 2, .blocks = *coresident}).has_value());
  const auto multi = launch->MultiGemmCoresident(kBits, 2, Output::kF32);
  ASSERT_TRUE(multi.has_value());
  EXPECT_EQ(*multi, *coresident);
  Finish();
  EXPECT_FALSE(launch->faulted());
}

TEST_F(Exl3LinearTest, NoTwoLiveContextsShareLockSlots) {
  const std::uint64_t locks = Allocate(Placement::kCudaMalloc, exl3::kLockBytes + 4096);
  auto first = exl3::LaunchContext::Create(0, *execution_, stream_, locks);
  ASSERT_TRUE(first.has_value());
  EXPECT_FALSE(exl3::LaunchContext::Create(0, *execution_, stream_, locks).has_value());
  EXPECT_FALSE(exl3::LaunchContext::Create(0, *execution_, stream_, locks + 4096).has_value());
  EXPECT_FALSE(exl3::LaunchContext::Create(0, *execution_, stream_, locks + 8).has_value());
  first->reset();
  Finish();
  auto again = exl3::LaunchContext::Create(0, *execution_, stream_, locks);
  EXPECT_TRUE(again.has_value());
}

// Forwards to the CUDA provider, and reports its next Submission as an
// unknown outcome: a device fault.
class FaultingExecution final : public DeviceExecution {
 public:
  explicit FaultingExecution(DeviceExecution& inner) : inner_(inner) {}
  bool fault_next = false;

  std::expected<StreamId, jitllm::providers::Failure> CreateStream() override {
    return inner_.CreateStream();
  }
  std::expected<void, jitllm::providers::Failure> DestroyStream(StreamId stream) override {
    return inner_.DestroyStream(stream);
  }
  std::expected<void, jitllm::providers::Failure> Copy(StreamId stream, std::uint64_t destination,
                                                       std::uint64_t source, Bytes size) override {
    return inner_.Copy(stream, destination, source, size);
  }
  std::expected<jitllm::providers::NativeStream, jitllm::providers::Failure> Submission(
      StreamId stream) override {
    if (std::exchange(fault_next, false)) {
      return std::unexpected(jitllm::providers::Failure{
          .error = jitllm::providers::ProviderError::kUnknown, .detail = "a scripted fault"});
    }
    return inner_.Submission(stream);
  }
  std::expected<void, jitllm::providers::Failure> Wait(StreamId stream,
                                                       jitllm::providers::FenceId fence) override {
    return inner_.Wait(stream, fence);
  }
  std::expected<jitllm::providers::FenceId, jitllm::providers::Failure> Record(
      StreamId stream) override {
    return inner_.Record(stream);
  }
  std::expected<FenceState, jitllm::providers::Failure> Query(
      jitllm::providers::FenceId fence) override {
    return inner_.Query(fence);
  }
  std::expected<void, jitllm::providers::Failure> Release(
      jitllm::providers::FenceId fence) override {
    return inner_.Release(fence);
  }

 private:
  DeviceExecution& inner_;
};

TEST_F(Exl3LinearTest, AFaultReturnsAsAFaultAndStopsTheContext) {
  FaultingExecution faulting(*execution_);
  const std::uint64_t locks = Allocate(Placement::kCudaMalloc, exl3::kLockBytes);
  auto launch = exl3::LaunchContext::Create(0, faulting, stream_, locks);
  ASSERT_TRUE(launch.has_value());
  const exl3::LinearOperands o{.weights = Linear(Placement::kCudaMalloc, 1),
                               .x = Allocate(Placement::kCudaMalloc, Bytes16(1, kK)),
                               .a_had = Allocate(Placement::kCudaMalloc, Bytes16(1, kK)),
                               .y = Allocate(Placement::kCudaMalloc, Bytes16(1, kN)),
                               .output = Output::kF16,
                               .m = 1};
  faulting.fault_next = true;
  const auto faulted = (*launch)->Gemm(o, {.shape = 2, .blocks = 14});
  ASSERT_FALSE(faulted.has_value());
  EXPECT_EQ(faulted.error().error, exl3::KernelError::kUnknown);
  EXPECT_TRUE((*launch)->faulted());
  // Nothing more runs on the context: its lock slots are undetermined.
  const auto refused = (*launch)->Gemm(o, {.shape = 2, .blocks = 14});
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().error, exl3::KernelError::kRejected);
}

// The over-read probe (docs/artifact-format.md leaves EXL3's open): every
// operand ends flush against an unmapped granule, then starts flush after
// one. A read or write past either edge would fault; none does, and the
// results are the cudaMalloc run's bit for bit.
TEST_F(Exl3LinearTest, NoKernelReadsOrWritesOutsideItsOperands) {
  const Results malloced = Run(Placement::kCudaMalloc);
  const Results end = Run(Placement::kFlushEnd);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(malloced.gemm, end.gemm);
  EXPECT_EQ(malloced.gemv, end.gemv);
  EXPECT_EQ(malloced.gemm1, end.gemm1);
  EXPECT_EQ(malloced.multi, end.multi);
  EXPECT_EQ(malloced.recon, end.recon);
  EXPECT_EQ(malloced.fused, end.fused);
  EXPECT_EQ(malloced.fused_as_recon, end.fused_as_recon);
  FreeAll();
  const Results start = Run(Placement::kFlushStart);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(malloced.gemm, start.gemm);
  EXPECT_EQ(malloced.gemv, start.gemv);
  EXPECT_EQ(malloced.gemm1, start.gemm1);
  EXPECT_EQ(malloced.multi, start.multi);
  EXPECT_EQ(malloced.recon, start.recon);
  EXPECT_EQ(malloced.fused, start.fused);
  EXPECT_EQ(malloced.fused_as_recon, start.fused_as_recon);
}

}  // namespace
