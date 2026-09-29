// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8's fast path (kernels/ggml/jitllm_ops.h; D-085: speed before bit
// exactness) on a GB10 (label `gpu`), at the model's widths: the MXFP8
// quantization (every value within E4M3's rounding of its block's scale,
// the scale the smallest power of two that holds the block, the padding
// rows' scales zero) and the tensor-core product over it (the product of
// the quantized operands, from FP64); the fused hyper-connections, routing,
// Gated DeltaNet and QSA kernels against FP64 references; QSA's selection
// against an exact host selection (ties to the lower cell); and what their
// checks refuse.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/moe_cutlass.h"
#include "kernels/ggml/mxfp8_cutlass.h"
#include "kernels/ggml/tensors.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {

using jitllm::base::Bytes;
using jitllm::kernels::ggml::CublasHandle;
using jitllm::kernels::ggml::LaunchContext;
using jitllm::kernels::ggml::TensorArena;
using jitllm::providers::DeviceExecution;
using jitllm::providers::FenceState;
using jitllm::providers::StreamId;
namespace kg = jitllm::kernels::ggml;
namespace mx = jitllm::kernels::ggml::mxfp8;

constexpr std::uint64_t kWorkspace = 64ULL << 20;
constexpr std::int64_t kWidth = 2560;
constexpr std::int64_t kHc = 4;
constexpr float kEps = 1e-6f;

double Nmse(const std::vector<float>& got, const std::vector<double>& want) {
  EXPECT_EQ(got.size(), want.size());
  double error = 0.0;
  double norm = 0.0;
  for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    error += d * d;
    norm += want[i] * want[i];
  }
  return norm > 0.0 ? error / norm : error;
}

void ExpectNear(const std::vector<float>& got, const std::vector<double>& want, double bound,
                const std::string& what) {
  const double nmse = Nmse(got, want);
  EXPECT_LE(nmse, bound) << what;
  EXPECT_TRUE(std::ranges::all_of(got, [](float v) { return std::isfinite(v); })) << what;
  std::cout << what << ": NMSE " << nmse << " against FP64\n";
}

std::vector<float> Normal(std::uint64_t seed, std::size_t n, float scale = 1.0f) {
  std::mt19937_64 random(seed);
  std::normal_distribution<float> normal(0.0f, scale);
  std::vector<float> out(n);
  for (float& v : out) {
    v = normal(random);
  }
  return out;
}

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

std::uint16_t Bf16Bits(float v) {
  const auto bits = std::bit_cast<std::uint32_t>(v);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
}
float FromBf16(std::uint16_t b) { return std::bit_cast<float>(std::uint32_t{b} << 16U); }
std::vector<std::uint16_t> ToBf16(const std::vector<float>& x) {
  std::vector<std::uint16_t> out(x.size());
  std::ranges::transform(x, out.begin(), Bf16Bits);
  return out;
}

// An E4M3 code's value (bias 7, subnormals, 0x7F / 0xFF NaN).
double E4m3(std::uint8_t b) {
  const unsigned bits = b;
  const int e = static_cast<int>((bits >> 3U) & 15U);
  const int m = static_cast<int>(bits & 7U);
  if (e == 15 && m == 7) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double v =
      e == 0 ? (m / 8.0) * std::ldexp(1.0, -6) : (1.0 + (m / 8.0)) * std::ldexp(1.0, e - 7);
  return (bits & 0x80U) != 0 ? -v : v;
}
double E8m0(std::uint8_t e) { return std::ldexp(1.0, static_cast<int>(e) - 127); }

// MXFP8 rows (RowsLayout) back to doubles: [rows, k].
std::vector<double> Dequantize(const std::vector<std::uint8_t>& blob, std::int64_t k,
                               std::int64_t rows) {
  const mx::RowsLayout l{.k = static_cast<std::uint64_t>(k),
                         .rows = static_cast<std::uint64_t>(rows)};
  std::vector<double> out(static_cast<std::size_t>(k * rows));
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t i = 0; i < k; ++i) {
      const std::uint64_t at = jitllm::kernels::ggml::moe::SfOffset(
          static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(i / 32),
          static_cast<std::uint64_t>(k / 32));
      out[static_cast<std::size_t>((r * k) + i)] =
          E4m3(blob[mx::RowsLayout::codes() + static_cast<std::size_t>((r * k) + i)]) *
          E8m0(blob[l.scales() + at]);
    }
  }
  return out;
}

// Every value within E4M3's rounding of its block's scale (half a step of
// its binade, or of the subnormals'), each scale the smallest power of two
// that holds the block's largest magnitude at 448, and the padding rows'
// scales zero.
void ExpectQuantized(const std::vector<std::uint8_t>& blob, const std::vector<double>& x,
                     std::int64_t k, std::int64_t rows, const std::string& what) {
  const std::vector<double> q = Dequantize(blob, k, rows);
  const mx::RowsLayout l{.k = static_cast<std::uint64_t>(k),
                         .rows = static_cast<std::uint64_t>(rows)};
  std::size_t bad = 0;
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t b = 0; b < k / 32; ++b) {
      double amax = 0.0;
      for (std::int64_t i = 0; i < 32; ++i) {
        amax = std::max(amax, std::abs(x[static_cast<std::size_t>((r * k) + (b * 32) + i)]));
      }
      const double s =
          E8m0(blob[l.scales() + jitllm::kernels::ggml::moe::SfOffset(
                                     static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(b),
                                     static_cast<std::uint64_t>(k / 32))]);
      // (Margins for x computed in FP64 where the kernel quantized its F32.)
      if (amax > 0.0 && (amax > 448.0 * s * (1.0 + 1e-5) || amax <= 224.0 * s * (1.0 - 1e-5))) {
        ++bad;
      }
      for (std::int64_t i = 0; i < 32; ++i) {
        const auto at = static_cast<std::size_t>((r * k) + (b * 32) + i);
        const double v = std::abs(x[at]) / s;
        const double step =
            v < std::ldexp(1.0, -6) ? std::ldexp(1.0, -9) : std::ldexp(1.0, std::ilogb(v) - 3);
        if (std::abs(q[at] - x[at]) > (0.5 * step * s * (1.0 + 1e-4)) + (1e-6 * std::abs(x[at]))) {
          ++bad;
        }
      }
    }
  }
  for (std::uint64_t r = l.rows; r < mx::PaddedRows(l.rows); ++r) {
    for (std::int64_t b = 0; b < k / 32; ++b) {
      bad += blob[l.scales() +
                  jitllm::kernels::ggml::moe::SfOffset(r, static_cast<std::uint64_t>(b),
                                                       static_cast<std::uint64_t>(k / 32))] != 0;
    }
  }
  EXPECT_EQ(bad, 0U) << what;
}

class Qwen38FastTest : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(jitllm::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    int major = 0;
    int minor = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), cudaSuccess);
    const Bytes cublas_bytes = CublasHandle::UpstreamWorkspace((100 * major) + (10 * minor));
    auto handle = CublasHandle::Create(
        0, *execution_, stream_, {.base = Allocate(cublas_bytes.value()), .size = cublas_bytes});
    ASSERT_TRUE(handle.has_value()) << (handle ? "" : handle.error().detail);
    cublas_ = std::move(*handle);
    auto launch = LaunchContext::Create(0, *execution_, stream_,
                                        {.base = Allocate(kWorkspace), .size = Bytes(kWorkspace)},
                                        cublas_.get());
    ASSERT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    launch_ = std::move(*launch);
    arena_ = std::make_unique<TensorArena>(TensorArena::Create(1024).value());
    auto registry = jitllm::execution::Registry::Create(kg::Implementations());
    ASSERT_TRUE(registry.has_value());
    registry_ = std::make_unique<jitllm::execution::Registry>(std::move(*registry));
  }

  void TearDown() override {
    Finish();
    launch_.reset();
    cublas_.reset();
    ASSERT_TRUE(execution_->DestroyStream(stream_).has_value());
    for (void* pointer : device_) {
      EXPECT_EQ(cudaFree(pointer), cudaSuccess);
    }
  }

  void Finish() {
    const auto fence = execution_->Record(stream_).value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    FenceState state = FenceState::kPending;
    while ((state = execution_->Query(fence).value()) == FenceState::kPending &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ASSERT_EQ(state, FenceState::kComplete);
    ASSERT_TRUE(execution_->Release(fence).has_value());
  }

  std::uint64_t Allocate(std::size_t bytes) {
    void* pointer = nullptr;
    EXPECT_EQ(cudaMalloc(&pointer, std::max<std::size_t>(bytes, 256)), cudaSuccess);
    // Stale bytes would hide an output the kernels leave unwritten.
    EXPECT_EQ(cudaMemset(pointer, 0x7B, std::max<std::size_t>(bytes, 256)), cudaSuccess);
    // The memset runs on the legacy stream, which the provider's
    // non-blocking stream does not wait for: it must land before any kernel
    // writes the memory.
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    device_.push_back(pointer);
    return reinterpret_cast<std::uintptr_t>(pointer);
  }

  ggml_context* c() const { return arena_->context(); }
  LaunchContext& launch() { return *launch_; }

  template <typename T = float>
  ggml_tensor* Leaf(ggml_tensor* tensor, const std::vector<T>& data) {
    const std::uint64_t address = Allocate(ggml_nbytes(tensor));
    TensorArena::Bind(tensor, address);
    EXPECT_EQ(data.size() * sizeof(T), ggml_nbytes(tensor));
    EXPECT_EQ(cudaMemcpy(tensor->data, data.data(), ggml_nbytes(tensor), cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return tensor;
  }

  // A packed `type` [ne0, ne1] view of a blob's bytes from `offset` (as
  // the graph views its fusions' blobs).
  ggml_tensor* TypedView(ggml_tensor* blob, ggml_type type, std::int64_t ne0, std::int64_t ne1,
                         std::size_t offset) {
    const std::size_t row = ggml_row_size(type, ne0);
    ggml_tensor* v = ggml_view_1d(
        c(), blob,
        static_cast<std::int64_t>(row * static_cast<std::size_t>(ne1) / ggml_type_size(blob->type)),
        offset);
    v->type = type;
    v->ne[0] = ne0;
    v->ne[1] = ne1;
    v->nb[0] = ggml_type_size(type);
    v->nb[1] = row;
    v->nb[2] = row * static_cast<std::size_t>(ne1);
    v->nb[3] = v->nb[2];
    return v;
  }

  // Places every computed tensor the outputs need, binds the views, and
  // runs the plan the registry binds.
  void Run(std::vector<ggml_tensor*> outputs) {
    const std::vector<ggml_tensor*> nodes = kg::GraphOrder(outputs);
    for (ggml_tensor* node : nodes) {
      if (node->view_src == nullptr && node->data == nullptr) {
        TensorArena::Bind(node, Allocate(ggml_nbytes(node)));
      }
    }
    kg::BindViews(nodes);
    auto plan = kg::PlanGraph(nodes, false, kg::DeviceChoicesOf(launch()));
    ASSERT_TRUE(plan.has_value()) << plan.error().detail;
    auto bound = kg::BoundGraph::Bind(*registry_, *plan);
    ASSERT_TRUE(bound.has_value()) << bound.error().detail;
    ASSERT_TRUE(bound->Run(launch()).has_value());
  }

  std::string PlannedFor(ggml_tensor* node) {
    auto plan =
        kg::PlanGraph(std::vector<ggml_tensor*>{node}, false, kg::DeviceChoicesOf(launch()));
    return plan && plan->steps.size() == 1 ? std::string(plan->steps[0].implementation) : "";
  }

  template <typename T = float>
  std::vector<T> Download(const ggml_tensor* tensor) {
    Finish();
    std::vector<T> values(ggml_nbytes(tensor) / sizeof(T));
    EXPECT_EQ(cudaMemcpy(values.data(), tensor->data, ggml_nbytes(tensor), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return values;
  }

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::vector<void*> device_;
  std::unique_ptr<CublasHandle> cublas_;
  std::unique_ptr<LaunchContext> launch_;
  std::unique_ptr<TensorArena> arena_;
  std::unique_ptr<jitllm::execution::Registry> registry_;
};

// MXFP8 weights of real models' ranges: E4M3 codes (no NaN) and E8M0
// scales about 2^-8.
struct Weights {
  std::vector<std::uint8_t> codes;   // [n, k]
  std::vector<std::uint8_t> scales;  // [n, k / 32]
  double at(std::int64_t row, std::int64_t i, std::int64_t k) const {
    return E4m3(codes[static_cast<std::size_t>((row * k) + i)]) *
           E8m0(scales[static_cast<std::size_t>((row * (k / 32)) + (i / 32))]);
  }
};
Weights RandomWeights(std::uint64_t seed, std::int64_t k, std::int64_t n) {
  std::mt19937 random(static_cast<unsigned>(seed));
  Weights w;
  w.codes.resize(static_cast<std::size_t>(k * n));
  for (std::uint8_t& b : w.codes) {
    do {
      b = static_cast<std::uint8_t>(random() & 0xFFU);
    } while ((static_cast<unsigned>(b) & 0x7FU) == 0x7FU);
  }
  w.scales.resize(static_cast<std::size_t>((k / 32) * n));
  for (std::uint8_t& s : w.scales) {
    s = static_cast<std::uint8_t>(116 + (random() % 6));
  }
  return w;
}

TEST_F(Qwen38FastTest, TheMxfp8ProductIsTheProductOfItsQuantizedOperands) {
  // (Rows below one scale atom, odd tails across atoms, and past 4,096 rows,
  // where the tiles are swizzled.)
  for (const auto& [k, n, t] : {std::tuple<std::int64_t, std::int64_t, std::int64_t>{2560, 640, 9},
                                {2560, 48, 130},
                                {6144, 512, 257},
                                {640, 2560, 33},
                                {640, 256, 4133}}) {
    const std::string what =
        "k " + std::to_string(k) + " n " + std::to_string(n) + " t " + std::to_string(t);
    const std::vector<float> x_h =
        Normal(static_cast<std::uint64_t>(k + n + t), static_cast<std::size_t>(k * t), 2.0f);
    const Weights w = RandomWeights(static_cast<std::uint64_t>(n), k, n);
    ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, k, t), x_h);
    ggml_tensor* codes = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I8, k, n), w.codes);
    ggml_tensor* scales = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I8, k / 32, n), w.scales);
    ggml_tensor* q = kg::Mxfp8Quantize(c(), x);
    ggml_tensor* swizzled = kg::Mxfp8Swizzle(c(), scales);
    ggml_tensor* y = kg::Mxfp8Gemm(c(), q, codes, swizzled, GGML_TYPE_F32, t);
    ggml_tensor* y16 = kg::Mxfp8Gemm(c(), q, codes, swizzled, GGML_TYPE_BF16, t);
    Run({y, y16});
    EXPECT_EQ(PlannedFor(q), kg::kMxfp8QuantizeName);
    EXPECT_EQ(PlannedFor(swizzled), kg::kMxfp8SwizzleName);
    EXPECT_EQ(PlannedFor(y), kg::kMxfp8GemmName);
    const std::vector<double> x_d(x_h.begin(), x_h.end());
    const auto blob = Download<std::uint8_t>(q);
    ExpectQuantized(blob, x_d, k, t, "quantized " + what);
    // The swizzled scales are the artifact's, each in its place.
    const auto sw = Download<std::uint8_t>(swizzled);
    std::size_t misplaced = 0;
    const auto padded = static_cast<std::int64_t>(mx::PaddedRows(static_cast<std::uint64_t>(n)));
    for (std::int64_t r = 0; r < padded; ++r) {
      for (std::int64_t b = 0; b < k / 32; ++b) {
        const std::uint8_t want =
            r < n ? w.scales[static_cast<std::size_t>((r * (k / 32)) + b)] : 0;
        misplaced += sw[jitllm::kernels::ggml::moe::SfOffset(
                         static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(b),
                         static_cast<std::uint64_t>(k / 32))] != want;
      }
    }
    EXPECT_EQ(misplaced, 0U) << what;
    const std::vector<double> xq = Dequantize(blob, k, t);
    std::vector<double> want(static_cast<std::size_t>(n * t));
    std::vector<double> exact(static_cast<std::size_t>(n * t));
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t j = 0; j < n; ++j) {
        double sum = 0.0;
        double full = 0.0;
        for (std::int64_t i = 0; i < k; ++i) {
          const double wv = w.at(j, i, k);
          sum += wv * xq[static_cast<std::size_t>((r * k) + i)];
          full += wv * x_d[static_cast<std::size_t>((r * k) + i)];
        }
        want[static_cast<std::size_t>((r * n) + j)] = sum;
        exact[static_cast<std::size_t>((r * n) + j)] = full;
      }
    }
    const auto got = Download(y);
    ExpectNear(got, want, 1e-10, "product " + what);
    // And against the unquantized activations: MXFP8's rounding.
    ExpectNear(got, exact, 3e-3, "product (activations unquantized) " + what);
    const auto got16 = Download<std::uint16_t>(y16);
    std::vector<float> as_f32(got16.size());
    std::ranges::transform(got16, as_f32.begin(), FromBf16);
    ExpectNear(as_f32, want, 1e-5, "BF16 product " + what);
  }
}

TEST_F(Qwen38FastTest, Mxfp8QuantizationReadsBf16AndStridedRows) {
  constexpr std::int64_t k = 2560;
  constexpr std::int64_t t = 37;
  // Rows of a wider tensor (a stride of k + 64 values), and BF16 rows.
  const std::vector<float> wide_h = Normal(5, static_cast<std::size_t>((k + 64) * t), 30.0f);
  ggml_tensor* wide = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, k + 64, t), wide_h);
  ggml_tensor* rows = ggml_view_2d(c(), wide, k, t, wide->nb[1], 0);
  std::vector<float> bf_values(static_cast<std::size_t>(k * t));
  for (std::int64_t r = 0; r < t; ++r) {
    for (std::int64_t i = 0; i < k; ++i) {
      bf_values[static_cast<std::size_t>((r * k) + i)] =
          FromBf16(Bf16Bits(wide_h[static_cast<std::size_t>((r * (k + 64)) + i)]));
    }
  }
  ggml_tensor* bf = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, k, t), ToBf16(bf_values));
  ggml_tensor* from_rows = kg::Mxfp8Quantize(c(), rows);
  ggml_tensor* from_bf16 = kg::Mxfp8Quantize(c(), bf);
  Run({from_rows, from_bf16});
  std::vector<double> want(static_cast<std::size_t>(k * t));
  for (std::int64_t r = 0; r < t; ++r) {
    for (std::int64_t i = 0; i < k; ++i) {
      want[static_cast<std::size_t>((r * k) + i)] =
          wide_h[static_cast<std::size_t>((r * (k + 64)) + i)];
    }
  }
  ExpectQuantized(Download<std::uint8_t>(from_rows), want, k, t, "strided rows");
  const std::vector<double> bf_d(bf_values.begin(), bf_values.end());
  ExpectQuantized(Download<std::uint8_t>(from_bf16), bf_d, k, t, "BF16 rows");
}

// The hyper-connections' prep: [combine,] norm into BF16 and inject logits.
TEST_F(Qwen38FastTest, HcPrepCombinesNormalizesAndInjects) {
  // Up to 8 tokens the cluster form (HcPrepClusterKernel), past them a
  // block a token.
  for (const auto& [t, combine, inject] : {std::tuple<std::int64_t, bool, bool>{1, true, true},
                                           {8, true, false},
                                           {37, true, true},
                                           {9, false, true},
                                           {5, false, false}}) {
    const std::string what =
        "t " + std::to_string(t) + (combine ? " combine" : "") + (inject ? " inject" : "");
    const std::int64_t wide = kWidth * kHc;
    const auto seed = static_cast<std::uint64_t>(t);
    const auto res_h = Normal(11 + seed, static_cast<std::size_t>(wide * t), 3.0f);
    const auto out_h = Normal(12 + seed, static_cast<std::size_t>(kWidth * t));
    const auto logit_h = Normal(13 + seed, static_cast<std::size_t>(kHc * t), 2.0f);
    auto norm_h = Normal(14, static_cast<std::size_t>(wide), 0.3f);
    for (float& v : norm_h) {
      v += 1.0f;
    }
    const auto inj_f = Normal(15, static_cast<std::size_t>(wide * kHc), 0.02f);
    const auto inj_b = ToBf16(inj_f);
    ggml_tensor* res = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, kHc, t), res_h);
    ggml_tensor* norm = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, wide), norm_h);
    ggml_tensor* inj_w =
        inject ? Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, wide, kHc), inj_b) : nullptr;
    ggml_tensor* out =
        combine ? Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, t), out_h) : nullptr;
    ggml_tensor* logits =
        combine ? Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kHc, t), logit_h) : nullptr;
    ggml_tensor* blob = kg::HcPrep(c(), res, norm, inj_w, out, logits, kEps);
    Run({blob});
    EXPECT_EQ(PlannedFor(blob), kg::kHcPrepName);
    const kg::HcPrepLayout l{
        .width = kWidth, .hc = kHc, .t = t, .combine = combine, .inject = inject};
    const auto bytes = Download<std::uint8_t>(blob);
    // FP64: the combined streams, normalized, and the logits.
    std::vector<double> streams(static_cast<std::size_t>(wide * t));
    std::vector<double> normed(streams.size());
    std::vector<double> inj(static_cast<std::size_t>(kHc * t), 0.0);
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t k = 0; k < kHc; ++k) {
        const double g =
            combine ? 2.0 * Sigmoid(logit_h[static_cast<std::size_t>((r * kHc) + k)] / kHc) : 0.0;
        double sum = 0.0;
        for (std::int64_t i = 0; i < kWidth; ++i) {
          const auto at = static_cast<std::size_t>((((r * kHc) + k) * kWidth) + i);
          const double v =
              res_h[at] + (combine ? out_h[static_cast<std::size_t>((r * kWidth) + i)] * g : 0.0);
          streams[at] = v;
          sum += v * v;
        }
        const double s = 1.0 / std::sqrt((sum / kWidth) + kEps);
        for (std::int64_t i = 0; i < kWidth; ++i) {
          const auto at = static_cast<std::size_t>((((r * kHc) + k) * kWidth) + i);
          normed[at] = streams[at] * s * norm_h[static_cast<std::size_t>((k * kWidth) + i)];
        }
      }
      for (std::int64_t m = 0; m < kHc; ++m) {
        double dot = 0.0;
        for (std::int64_t i = 0; i < wide; ++i) {
          dot += static_cast<double>(FromBf16(inj_b[static_cast<std::size_t>((m * wide) + i)])) *
                 normed[static_cast<std::size_t>((r * wide) + i)];
        }
        inj[static_cast<std::size_t>((r * kHc) + m)] = dot;
      }
    }
    if (combine) {
      std::vector<float> got(streams.size());
      std::memcpy(got.data(), bytes.data() + kg::HcPrepLayout::streams(), got.size() * 4);
      ExpectNear(got, streams, 1e-12, "streams " + what);
    }
    std::vector<std::uint16_t> xn(normed.size());
    std::memcpy(xn.data(), bytes.data() + l.normed(), xn.size() * 2);
    std::vector<float> xn_f(xn.size());
    std::ranges::transform(xn, xn_f.begin(), FromBf16);
    ExpectNear(xn_f, normed, 2e-5, "normalized (BF16) " + what);
    if (inject) {
      std::vector<float> got(inj.size());
      std::memcpy(got.data(), bytes.data() + l.logits(), got.size() * 4);
      ExpectNear(got, inj, 1e-9, "inject logits " + what);
    }
  }
}

TEST_F(Qwen38FastTest, HcLoAndMixAreTheirFormulas) {
  constexpr std::int64_t t = 21;
  constexpr std::int64_t rank = 320;
  const std::int64_t wide = kWidth * kHc;
  const auto lo_h = Normal(21, static_cast<std::size_t>(rank * t), 4.0f);
  const auto xn_f = Normal(22, static_cast<std::size_t>(wide * t));
  const auto gate_f = Normal(23, static_cast<std::size_t>(wide * t), 3.0f);
  const auto xn_b = ToBf16(xn_f);
  const auto gate_b = ToBf16(gate_f);
  ggml_tensor* lo = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, rank, t), lo_h);
  ggml_tensor* xn = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, wide, t), xn_b);
  ggml_tensor* gate = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, wide, t), gate_b);
  ggml_tensor* act = kg::HcLo(c(), lo, kHc);
  ggml_tensor* mixed = kg::HcMixBf16(c(), xn, gate, kHc);
  ggml_tensor* blob = kg::HcMixBf16(c(), xn, gate, kHc, true, true);
  Run({act, mixed, blob});
  EXPECT_EQ(PlannedFor(act), kg::kHcLoName);
  EXPECT_EQ(PlannedFor(mixed), kg::kHcMixBf16Name);
  std::vector<double> want_lo(lo_h.size());
  for (std::size_t i = 0; i < lo_h.size(); ++i) {
    const double v = lo_h[i] / static_cast<double>(kHc);
    want_lo[i] = v * Sigmoid(v);
  }
  const auto act_b = Download<std::uint16_t>(act);
  std::vector<float> act_f(act_b.size());
  std::ranges::transform(act_b, act_f.begin(), FromBf16);
  ExpectNear(act_f, want_lo, 2e-5, "silu(lo / hc) in BF16");
  std::vector<double> want(static_cast<std::size_t>(kWidth * t), 0.0);
  for (std::int64_t r = 0; r < t; ++r) {
    for (std::int64_t i = 0; i < kWidth; ++i) {
      double sum = 0.0;
      for (std::int64_t k = 0; k < kHc; ++k) {
        const auto at = static_cast<std::size_t>((r * wide) + (k * kWidth) + i);
        sum += static_cast<double>(FromBf16(xn_b[at])) * Sigmoid(FromBf16(gate_b[at]));
      }
      want[static_cast<std::size_t>((r * kWidth) + i)] = sum / kHc;
    }
  }
  const auto got = Download(mixed);
  ExpectNear(got, want, 1e-11, "mix");
  // The blob: the same mix, its MXFP8 quantization and its BF16.
  const kg::HcMixLayout ml{.width = kWidth, .t = t, .bf16 = true};
  const auto bytes = Download<std::uint8_t>(blob);
  std::vector<float> blob_mixed(want.size());
  std::memcpy(blob_mixed.data(), bytes.data() + kg::HcMixLayout::mixed(), blob_mixed.size() * 4);
  EXPECT_EQ(blob_mixed, got);
  const std::vector<std::uint8_t> quantized(
      bytes.begin() + static_cast<std::ptrdiff_t>(ml.quantized()),
      bytes.begin() + static_cast<std::ptrdiff_t>(ml.quantized() + ml.quantized_bytes()));
  const std::vector<double> got_d(got.begin(), got.end());
  ExpectQuantized(quantized, got_d, kWidth, t, "the mix's MXFP8");
  std::vector<std::uint16_t> rounded(want.size());
  std::memcpy(rounded.data(), bytes.data() + ml.rounded(), rounded.size() * 2);
  EXPECT_EQ(rounded, ToBf16(got));
}

TEST_F(Qwen38FastTest, TheRouterPicksTheTopExpertsAndTheSharedGate) {
  constexpr std::int64_t experts = 512;
  constexpr std::int64_t used = 10;
  for (const std::int64_t t : {1, 11, 64}) {
    const auto seed = static_cast<std::uint64_t>(t);
    const auto logits_h = Normal(31 + seed, static_cast<std::size_t>(experts * t), 2.0f);
    const auto x_h = Normal(32 + seed, static_cast<std::size_t>(kWidth * t));
    const auto gate_b = ToBf16(Normal(33, kWidth, 0.05f));
    ggml_tensor* logits = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, experts, t), logits_h);
    ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, t), x_h);
    ggml_tensor* gate_row = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_BF16, kWidth), gate_b);
    ggml_tensor* routed = kg::MoeRouter(c(), logits, x, gate_row, used);
    Run({routed});
    EXPECT_EQ(PlannedFor(routed), kg::kMoeRouterName);
    const kg::MoeRouterLayout l{.used = used, .t = t};
    const auto blob = Download<std::int32_t>(routed);
    std::vector<float> weights(static_cast<std::size_t>(used * t));
    std::vector<float> gates(static_cast<std::size_t>(t));
    std::memcpy(weights.data(), blob.data() + (l.weights() / 4), weights.size() * 4);
    std::memcpy(gates.data(), blob.data() + (l.gate() / 4), gates.size() * 4);
    std::vector<double> want_w(weights.size());
    std::vector<double> want_g(gates.size());
    for (std::int64_t r = 0; r < t; ++r) {
      std::vector<double> p(static_cast<std::size_t>(experts));
      double most = -1e300;
      for (std::int64_t e = 0; e < experts; ++e) {
        most = std::max(most,
                        static_cast<double>(logits_h[static_cast<std::size_t>((r * experts) + e)]));
      }
      double sum = 0.0;
      for (std::int64_t e = 0; e < experts; ++e) {
        p[static_cast<std::size_t>(e)] =
            std::exp(logits_h[static_cast<std::size_t>((r * experts) + e)] - most);
        sum += p[static_cast<std::size_t>(e)];
      }
      std::vector<std::int32_t> order(static_cast<std::size_t>(experts));
      std::ranges::iota(order, 0);
      std::ranges::stable_sort(order, [&](std::int32_t a, std::int32_t b) {
        return p[static_cast<std::size_t>(a)] > p[static_cast<std::size_t>(b)];
      });
      double picked = 0.0;
      for (std::int64_t j = 0; j < used; ++j) {
        EXPECT_EQ(blob[static_cast<std::size_t>((r * used) + j)],
                  order[static_cast<std::size_t>(j)])
            << "token " << r << " pick " << j;
        picked += p[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])] / sum;
      }
      for (std::int64_t j = 0; j < used; ++j) {
        want_w[static_cast<std::size_t>((r * used) + j)] =
            p[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])] / sum /
            std::max(picked, 6.103515625e-5);
      }
      double dot = 0.0;
      for (std::int64_t i = 0; i < kWidth; ++i) {
        dot += static_cast<double>(FromBf16(gate_b[static_cast<std::size_t>(i)])) *
               x_h[static_cast<std::size_t>((r * kWidth) + i)];
      }
      want_g[static_cast<std::size_t>(r)] = dot;
    }
    ExpectNear(weights, want_w, 1e-10, "routing weights t " + std::to_string(t));
    ExpectNear(gates, want_g, 1e-10, "shared gate t " + std::to_string(t));
  }
  // NaN or all -inf logits still pick distinct experts within range (the
  // lowest, all probabilities counted 0), with zero weights.
  std::vector<float> bad_h(static_cast<std::size_t>(experts * 2));
  std::ranges::fill_n(bad_h.begin(), experts, std::numeric_limits<float>::quiet_NaN());
  std::ranges::fill_n(bad_h.begin() + experts, experts, -std::numeric_limits<float>::infinity());
  ggml_tensor* bad = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, experts, 2), bad_h);
  ggml_tensor* bad_x =
      Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, 2), Normal(34, 2 * kWidth));
  ggml_tensor* bad_gate =
      Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_BF16, kWidth), ToBf16(Normal(35, kWidth, 0.05f)));
  ggml_tensor* bad_routed = kg::MoeRouter(c(), bad, bad_x, bad_gate, used);
  Run({bad_routed});
  const auto bad_blob = Download<std::int32_t>(bad_routed);
  const kg::MoeRouterLayout bl{.used = used, .t = 2};
  for (std::int64_t r = 0; r < 2; ++r) {
    for (std::int64_t j = 0; j < used; ++j) {
      EXPECT_EQ(bad_blob[static_cast<std::size_t>((r * used) + j)], j) << "token " << r;
      EXPECT_EQ(std::bit_cast<float>(
                    bad_blob[(bl.weights() / 4) + static_cast<std::size_t>((r * used) + j)]),
                0.0f)
          << "token " << r;
    }
  }
}

TEST_F(Qwen38FastTest, GatedDeltaNetTakesBf16RowsAndQuantizesItsGate) {
  constexpr std::int64_t d = 128;
  constexpr std::int64_t heads = 48;
  constexpr std::int64_t channels = 10240;
  constexpr std::int64_t t = 19;
  // The convolution over BF16 rows is its F32 rows' of the same values.
  auto rows_f = Normal(41, static_cast<std::size_t>(channels * t));
  const auto rows_b = ToBf16(rows_f);
  std::ranges::transform(rows_b, rows_f.begin(), FromBf16);
  const auto history_h = Normal(42, static_cast<std::size_t>(3 * channels));
  const auto conv_w = Normal(43, static_cast<std::size_t>(4 * channels), 0.5f);
  ggml_tensor* rows32 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, channels, t), rows_f);
  ggml_tensor* rows16 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, channels, t), rows_b);
  ggml_tensor* history = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 3 * channels), history_h);
  ggml_tensor* weight = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 4, channels), conv_w);
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  const float eps = kEps / static_cast<float>(d);
  ggml_tensor* conv32 = kg::GdnConv(c(), rows32, history, weight, 4096, d, eps, scale);
  ggml_tensor* conv16 = kg::GdnConv(c(), rows16, history, weight, 4096, d, eps, scale);
  ggml_tensor* kept = kg::GdnHistory(c(), rows16, 3);
  // The gated norm into MXFP8, beside its F32 form.
  const auto o_h = Normal(44, static_cast<std::size_t>(d * heads * t));
  auto norm_h = Normal(45, d, 0.2f);
  for (float& v : norm_h) {
    v += 1.0f;
  }
  const auto z_b = ToBf16(Normal(46, static_cast<std::size_t>(d * heads * t), 2.0f));
  ggml_tensor* o = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, d, heads, t), o_h);
  ggml_tensor* norm = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, d), norm_h);
  ggml_tensor* z = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, d * heads, t), z_b);
  ggml_tensor* gated = kg::GdnNormGate(c(), o, norm, z, kEps, GGML_TYPE_F32);
  ggml_tensor* gated8 = kg::GdnNormGate(c(), o, norm, z, kEps, GGML_TYPE_I8);
  Run({conv32, conv16, kept, gated, gated8});
  EXPECT_EQ(PlannedFor(kept), kg::kGdnHistoryName);
  EXPECT_EQ(Download(conv16), Download(conv32));
  const auto kept_h = Download(kept);
  std::size_t wrong = 0;
  for (std::int64_t ch = 0; ch < channels; ++ch) {
    for (std::int64_t j = 0; j < 3; ++j) {
      wrong += kept_h[static_cast<std::size_t>((ch * 3) + j)] !=
               rows_f[static_cast<std::size_t>(((t - 3 + j) * channels) + ch)];
    }
  }
  EXPECT_EQ(wrong, 0U);
  std::vector<double> want(o_h.size());
  for (std::int64_t row = 0; row < heads * t; ++row) {
    double sum = 0.0;
    for (std::int64_t i = 0; i < d; ++i) {
      sum += static_cast<double>(o_h[static_cast<std::size_t>((row * d) + i)]) *
             o_h[static_cast<std::size_t>((row * d) + i)];
    }
    const double s = 1.0 / std::sqrt((sum / d) + kEps);
    for (std::int64_t i = 0; i < d; ++i) {
      const auto at = static_cast<std::size_t>((row * d) + i);
      want[at] = o_h[at] * s * norm_h[static_cast<std::size_t>(i)] * Sigmoid(FromBf16(z_b[at]));
    }
  }
  ExpectNear(Download(gated), want, 1e-11, "gated norm, BF16 gate");
  ExpectQuantized(Download<std::uint8_t>(gated8), want, d * heads, t, "gated norm into MXFP8");
}

// Decode's history (fewer rows than taps): the old history's last taps then
// the rows, from F32 and BF16 rows.
TEST_F(Qwen38FastTest, GdnHistoryShiftsTheOldHistoryForShortChunks) {
  constexpr std::int64_t channels = 10240;
  constexpr std::int64_t taps = 3;
  const auto history_h = Normal(51, static_cast<std::size_t>(taps * channels));
  ggml_tensor* history = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, taps * channels), history_h);
  for (const std::int64_t t : {std::int64_t{1}, std::int64_t{2}}) {
    auto rows_f =
        Normal(52 + static_cast<std::uint64_t>(t), static_cast<std::size_t>(channels * t));
    const auto rows_b = ToBf16(rows_f);
    std::ranges::transform(rows_b, rows_f.begin(), FromBf16);
    ggml_tensor* rows32 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, channels, t), rows_f);
    ggml_tensor* rows16 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, channels, t), rows_b);
    ggml_tensor* kept32 = kg::GdnHistory(c(), rows32, taps, history);
    ggml_tensor* kept16 = kg::GdnHistory(c(), rows16, taps, history);
    Run({kept32, kept16});
    EXPECT_EQ(PlannedFor(kept32), kg::kGdnHistoryName);
    for (ggml_tensor* kept : {kept32, kept16}) {
      const auto got = Download(kept);
      std::size_t wrong = 0;
      for (std::int64_t ch = 0; ch < channels; ++ch) {
        for (std::int64_t j = 0; j < taps; ++j) {
          const std::int64_t i = t + j;
          const float want = i < taps
                                 ? history_h[static_cast<std::size_t>((ch * taps) + i)]
                                 : rows_f[static_cast<std::size_t>(((i - taps) * channels) + ch)];
          wrong += got[static_cast<std::size_t>((ch * taps) + j)] != want;
        }
      }
      EXPECT_EQ(wrong, 0U) << "t " << t;
    }
  }
  // Without the old history, fewer rows than taps are refused.
  ggml_tensor* one = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, channels, 1),
                          std::vector<float>(static_cast<std::size_t>(channels), 1.0f));
  ggml_tensor* bare = kg::GdnHistory(c(), one, taps);
  TensorArena::Bind(bare, Allocate(ggml_nbytes(bare)));
  EXPECT_FALSE(kg::CheckGdnHistory(bare).has_value());
}

// jitllm.gdn.step is the columns kernel over the state in place: its
// attention output and the state it leaves equal gated_delta_net's (planned
// as jitllm.gated_delta_net.columns) output rows and state rows bit for bit.
TEST_F(Qwen38FastTest, GdnStepIsTheColumnsRecurrenceInPlace) {
  constexpr std::int64_t d = 128;
  constexpr std::int64_t qk_heads = 16;
  constexpr std::int64_t heads = 48;
  for (const std::int64_t t : {std::int64_t{1}, std::int64_t{4}, std::int64_t{16}}) {
    const std::string what = "t " + std::to_string(t);
    const auto seed = static_cast<std::uint64_t>(t) * 10;
    // Unit query and key heads, as the convolution's L2 norm leaves them.
    auto unit = [&](std::uint64_t s, std::int64_t n) {
      auto x = Normal(s, static_cast<std::size_t>(d * n * t));
      for (std::int64_t h = 0; h < n * t; ++h) {
        double sum = 0.0;
        for (std::int64_t i = 0; i < d; ++i) {
          sum += static_cast<double>(x[static_cast<std::size_t>((h * d) + i)]) *
                 x[static_cast<std::size_t>((h * d) + i)];
        }
        for (std::int64_t i = 0; i < d; ++i) {
          x[static_cast<std::size_t>((h * d) + i)] /= static_cast<float>(std::sqrt(sum));
        }
      }
      return x;
    };
    const auto q_h = unit(seed + 1, qk_heads);
    const auto k_h = unit(seed + 2, qk_heads);
    const auto v_h = Normal(seed + 3, static_cast<std::size_t>(d * heads * t));
    auto g_h = Normal(seed + 4, static_cast<std::size_t>(heads * t), 0.3f);
    for (float& g : g_h) {
      g = -std::abs(g);  // log of a decay in (0, 1]
    }
    auto beta_h = Normal(seed + 5, static_cast<std::size_t>(heads * t));
    for (float& b : beta_h) {
      b = static_cast<float>(Sigmoid(b));
    }
    const auto state_h = Normal(seed + 6, static_cast<std::size_t>(d * d * heads), 0.1f);
    auto leaf = [&](const std::vector<float>& data, std::int64_t n0, std::int64_t n1,
                    std::int64_t n2) {
      return Leaf(ggml_new_tensor_4d(c(), GGML_TYPE_F32, n0, n1, n2, 1), data);
    };
    ggml_tensor* q = leaf(q_h, d, qk_heads, t);
    ggml_tensor* k = leaf(k_h, d, qk_heads, t);
    ggml_tensor* v = leaf(v_h, d, heads, t);
    ggml_tensor* g = leaf(g_h, 1, heads, t);
    ggml_tensor* beta = leaf(beta_h, 1, heads, t);
    ggml_tensor* state_ref = leaf(state_h, d, d, heads);
    ggml_tensor* state = leaf(state_h, d, d, heads);
    ggml_tensor* state_read = leaf(state_h, d, d, heads);
    ggml_tensor* ref = ggml_gated_delta_net(c(), q, k, v, g, beta, state_ref, 1);
    ggml_tensor* step = kg::GdnStep(c(), q, k, v, g, beta, state);
    // A verify's: the same output, the state read and left as it was.
    ggml_tensor* read = kg::GdnStep(c(), q, k, v, g, beta, state_read, false);
    Run({ref, step, read});
    EXPECT_EQ(Download(read), Download(step)) << what;
    EXPECT_EQ(Download(state_read), state_h) << what;
    EXPECT_EQ(PlannedFor(ref), kg::kGatedDeltaNetColumnsName) << what;
    EXPECT_EQ(PlannedFor(step), kg::kGdnStepName) << what;
    const auto ref_h = Download(ref);
    const auto attn = Download(step);
    const auto state_after = Download(state);
    const auto rows = static_cast<std::size_t>(d * heads * t);
    ASSERT_EQ(ref_h.size(), rows + state_after.size()) << what;
    EXPECT_TRUE(std::equal(attn.begin(), attn.end(), ref_h.begin())) << what;
    EXPECT_TRUE(std::equal(state_after.begin(), state_after.end(),
                           ref_h.begin() + static_cast<std::ptrdiff_t>(rows)))
        << what;
    // The reference's state is untouched.
    EXPECT_EQ(Download(state_ref), state_h) << what;
  }
}

// jitllm.gemm.bf16's vector form (GemvBf16) at the hyper-connections' and
// the router's shapes, F32 and BF16 out, against FP64; past one column
// (kGemvBf16FastColumns) the same node runs cuBLAS. The one-column shapes
// reach each of the vector kernel's four forms (k 320, 2,560, 10,240 and
// 16,384).
TEST_F(Qwen38FastTest, GemvBf16IsTheProductAtDecodeWidths) {
  struct Shape {
    std::int64_t k, n, t;
    ggml_type out;
  };
  for (const Shape& s : {Shape{10240, 320, 1, GGML_TYPE_F32}, Shape{2560, 512, 1, GGML_TYPE_F32},
                         Shape{16384, 65, 1, GGML_TYPE_BF16}, Shape{10240, 324, 3, GGML_TYPE_F32},
                         Shape{320, 10240, 1, GGML_TYPE_BF16}, Shape{320, 10240, 8, GGML_TYPE_BF16},
                         Shape{2560, 512, 5, GGML_TYPE_F32}, Shape{16384, 64, 2, GGML_TYPE_F32},
                         Shape{320, 10240, 9, GGML_TYPE_BF16}}) {
    const std::string what = "k " + std::to_string(s.k) + " n " + std::to_string(s.n) + " t " +
                             std::to_string(s.t) + (s.out == GGML_TYPE_BF16 ? " BF16" : " F32");
    const auto w_b = ToBf16(
        Normal(static_cast<std::uint64_t>(s.k + s.n), static_cast<std::size_t>(s.k * s.n), 0.05f));
    const auto x_b =
        ToBf16(Normal(static_cast<std::uint64_t>(s.t + 7), static_cast<std::size_t>(s.k * s.t)));
    ggml_tensor* w = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, s.k, s.n), w_b);
    ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, s.k, s.t), x_b);
    ggml_tensor* y = kg::GemvBf16(c(), w, x, s.out);
    ASSERT_TRUE(kg::IsGemvBf16(y));
    Run({y});
    std::vector<double> want(static_cast<std::size_t>(s.n * s.t));
    for (std::int64_t col = 0; col < s.t; ++col) {
      for (std::int64_t row = 0; row < s.n; ++row) {
        double sum = 0.0;
        for (std::int64_t i = 0; i < s.k; ++i) {
          sum += static_cast<double>(FromBf16(w_b[static_cast<std::size_t>((row * s.k) + i)])) *
                 FromBf16(x_b[static_cast<std::size_t>((col * s.k) + i)]);
        }
        want[static_cast<std::size_t>((col * s.n) + row)] = sum;
      }
    }
    if (s.out == GGML_TYPE_BF16) {
      const auto got_b = Download<std::uint16_t>(y);
      std::vector<float> got(got_b.size());
      std::ranges::transform(got_b, got.begin(), FromBf16);
      ExpectNear(got, want, 2e-5, what);
    } else {
      ExpectNear(Download(y), want, 1e-10, what);
    }
  }
}

TEST_F(Qwen38FastTest, QsaPrepNormalizesAndRotates) {
  constexpr std::int64_t d = 256;
  constexpr std::int64_t heads = 24;
  constexpr std::int64_t t = 13;
  constexpr float base = 10000000.0f;
  const float theta_scale = std::pow(base, -2.0f / 64.0f);
  const auto q_full = Normal(51, static_cast<std::size_t>(2 * d * heads * t), 3.0f);
  auto norm_h = Normal(52, d, 0.2f);
  for (float& v : norm_h) {
    v += 1.0f;
  }
  std::vector<std::int32_t> pos(static_cast<std::size_t>(4 * t));
  for (std::int64_t r = 0; r < t; ++r) {
    for (std::int64_t s = 0; s < 4; ++s) {
      pos[static_cast<std::size_t>((s * t) + r)] = static_cast<std::int32_t>(1000 + (r * 311));
    }
  }
  ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 2 * d * heads, t), q_full);
  ggml_tensor* norm = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, d), norm_h);
  ggml_tensor* positions = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 4 * t), pos);
  ggml_tensor* q = kg::QsaPrep(c(), x, norm, positions, d, heads, 2 * d, kEps, theta_scale);
  // The gate's product quantization, from the same rows.
  const auto attn_h = Normal(53, static_cast<std::size_t>(d * heads * t));
  ggml_tensor* attn = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, d * heads, t), attn_h);
  ggml_tensor* gated = kg::QsaGateQuantize(c(), attn, x, d);
  Run({q, gated});
  EXPECT_EQ(PlannedFor(q), kg::kQsaPrepName);
  EXPECT_EQ(PlannedFor(gated), kg::kQsaGateQuantizeName);
  std::vector<double> want(static_cast<std::size_t>(d * heads * t));
  std::vector<double> want_gated(want.size());
  for (std::int64_t r = 0; r < t; ++r) {
    for (std::int64_t h = 0; h < heads; ++h) {
      const float* head = q_full.data() + (r * 2 * d * heads) + (h * 2 * d);
      double sum = 0.0;
      for (std::int64_t i = 0; i < d; ++i) {
        sum += static_cast<double>(head[i]) * head[i];
      }
      const double s = 1.0 / std::sqrt((sum / d) + kEps);
      std::vector<double> v(static_cast<std::size_t>(d));
      for (std::int64_t i = 0; i < d; ++i) {
        v[static_cast<std::size_t>(i)] = head[i] * s * norm_h[static_cast<std::size_t>(i)];
      }
      for (std::int64_t i = 0; i < 32; ++i) {
        const double theta = pos[static_cast<std::size_t>(r)] *
                             std::pow(static_cast<double>(theta_scale), static_cast<double>(i));
        const double x0 = v[static_cast<std::size_t>(i)];
        const double x1 = v[static_cast<std::size_t>(i + 32)];
        v[static_cast<std::size_t>(i)] = (x0 * std::cos(theta)) - (x1 * std::sin(theta));
        v[static_cast<std::size_t>(i + 32)] = (x0 * std::sin(theta)) + (x1 * std::cos(theta));
      }
      for (std::int64_t i = 0; i < d; ++i) {
        const auto at = static_cast<std::size_t>((((r * heads) + h) * d) + i);
        want[at] = v[static_cast<std::size_t>(i)];
        want_gated[at] = attn_h[at] * Sigmoid(head[d + i]);
      }
    }
  }
  // (Fast math's sine and cosine at positions in the thousands, as GGML's
  // rope_multi computes them.)
  ExpectNear(Download(q), want, 1e-6, "q norm and rotation");
  ExpectQuantized(Download<std::uint8_t>(gated), want_gated, d * heads, t,
                  "attention times its gate into MXFP8");
}

TEST_F(Qwen38FastTest, QsaSelectionKeepsTheBestCellsTiesToTheLowerCell) {
  constexpr std::int64_t heads = 4;
  constexpr std::int64_t ratio = 4;
  constexpr std::int64_t n_kv = 2304;
  constexpr std::int64_t blocks = n_kv / ratio;
  constexpr std::int64_t width = 2051;
  constexpr std::int64_t t = 37;
  // Scores on a coarse grid, so that many blocks tie; negative ones, which
  // relu zeroes; and a bias with blocks hidden (-inf) and forced (1e9).
  std::mt19937 random(61);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::vector<float> score(static_cast<std::size_t>(blocks * heads * t));
  for (float& s : score) {
    s = static_cast<float>(static_cast<int>(random() % 9) - 3) * 0.5f;
  }
  std::vector<float> bias(static_cast<std::size_t>(blocks * t), 0.0f);
  for (std::int64_t r = 0; r < t; ++r) {
    bias[static_cast<std::size_t>((r * blocks) + blocks - 1)] = -INFINITY;
    bias[static_cast<std::size_t>((r * blocks) + 3)] = 1e9f;
  }
  std::vector<std::int32_t> cell_block(static_cast<std::size_t>(n_kv));
  for (std::int64_t j = 0; j < n_kv; ++j) {
    cell_block[static_cast<std::size_t>(j)] = static_cast<std::int32_t>(j / ratio);
  }
  // Positions below the width (every earlier cell kept) and past it.
  std::vector<std::int32_t> pos(static_cast<std::size_t>(4 * t));
  for (std::int64_t r = 0; r < t; ++r) {
    pos[static_cast<std::size_t>(r)] = static_cast<std::int32_t>(1990 + (r * 8));
  }
  ggml_tensor* score_t = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, blocks, heads, t), score);
  ggml_tensor* bias_t = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, blocks, t), bias);
  ggml_tensor* cells_t = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_I32, n_kv), cell_block);
  ggml_tensor* pos_t = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 4 * t), pos);
  ggml_tensor* mask = kg::QsaSelect(c(), score_t, bias_t, cells_t, pos_t, width);
  Run({mask});
  EXPECT_EQ(PlannedFor(mask), kg::kQsaSelectName);
  const auto got = Download<std::uint16_t>(mask);
  std::size_t wrong = 0;
  for (std::int64_t r = 0; r < t; ++r) {
    std::vector<double> block(static_cast<std::size_t>(blocks));
    for (std::int64_t b = 0; b < blocks; ++b) {
      double s = 0.0;
      for (std::int64_t h = 0; h < heads; ++h) {
        s += std::max(0.0f, score[static_cast<std::size_t>((((r * heads) + h) * blocks) + b)]);
      }
      block[static_cast<std::size_t>(b)] = s + bias[static_cast<std::size_t>((r * blocks) + b)];
    }
    const std::int64_t p = pos[static_cast<std::size_t>(r)];
    std::vector<std::int64_t> order(static_cast<std::size_t>(n_kv));
    std::ranges::iota(order, std::int64_t{0});
    const auto value = [&](std::int64_t j) {
      return j <= p ? block[static_cast<std::size_t>(cell_block[static_cast<std::size_t>(j)])]
                    : -std::numeric_limits<double>::infinity();
    };
    std::ranges::stable_sort(order,
                             [&](std::int64_t a, std::int64_t b) { return value(a) > value(b); });
    std::vector<bool> keep(static_cast<std::size_t>(n_kv), false);
    for (std::int64_t i = 0; i < width; ++i) {
      keep[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])] = true;
    }
    for (std::int64_t j = 0; j < n_kv; ++j) {
      const bool visible = keep[static_cast<std::size_t>(j)] && j <= p;
      wrong += got[static_cast<std::size_t>((r * n_kv) + j)] != (visible ? 0x0000 : 0xFC00);
    }
  }
  EXPECT_EQ(wrong, 0U);
}

TEST_F(Qwen38FastTest, TheChecksRefuseWhatTheKernelsCannotRun) {
  auto reason = [](const auto& checked) {
    return checked.has_value() ? std::string("accepted") : checked.error().detail;
  };
  // Quantization rows not whole scale atoms along k.
  ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 96, 4),
                        std::vector<float>(static_cast<std::size_t>(96 * 4), 1.0f));
  ggml_tensor* q = kg::Mxfp8Quantize(c(), x);
  TensorArena::Bind(q, Allocate(ggml_nbytes(q)));
  EXPECT_NE(reason(kg::CheckMxfp8Quantize(q)), "accepted");
  // A product whose weight's scales are another weight's shape.
  ggml_tensor* x2 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 256, 9),
                         std::vector<float>(static_cast<std::size_t>(256 * 9), 1.0f));
  ggml_tensor* codes = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I8, 256, 64),
                            std::vector<std::int8_t>(static_cast<std::size_t>(256 * 64), 0x38));
  ggml_tensor* other = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I8, 8, 32),
                            std::vector<std::int8_t>(static_cast<std::size_t>(8 * 32), 127));
  ggml_tensor* q2 = kg::Mxfp8Quantize(c(), x2);
  ggml_tensor* sw = kg::Mxfp8Swizzle(c(), other);
  ggml_tensor* y = kg::Mxfp8Gemm(c(), q2, codes, sw, GGML_TYPE_F32, 9);
  for (ggml_tensor* t : {q2, sw, y}) {
    TensorArena::Bind(t, Allocate(ggml_nbytes(t)));
  }
  EXPECT_EQ(reason(kg::CheckMxfp8Quantize(q2)), "accepted");
  EXPECT_NE(reason(kg::CheckMxfp8Gemm(y)), "accepted");
  // A selection wider than the cells.
  ggml_tensor* score = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 64, 4, 2),
                            std::vector<float>(static_cast<std::size_t>(64 * 4 * 2), 0.0f));
  ggml_tensor* bias = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 64, 2),
                           std::vector<float>(static_cast<std::size_t>(64 * 2), 0.0f));
  ggml_tensor* cells =
      Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 256), std::vector<std::int32_t>(256, 0));
  ggml_tensor* pos =
      Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 8), std::vector<std::int32_t>(8, 0));
  ggml_tensor* mask = kg::QsaSelect(c(), score, bias, cells, pos, 257);
  TensorArena::Bind(mask, Allocate(ggml_nbytes(mask)));
  EXPECT_NE(reason(kg::CheckQsaSelect(mask)), "accepted");
  // More streams than the prep holds.
  ggml_tensor* streams = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1024, 9, 1),
                              std::vector<float>(static_cast<std::size_t>(1024 * 9), 1.0f));
  ggml_tensor* norm = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, std::int64_t{1024} * 9),
                           std::vector<float>(static_cast<std::size_t>(1024 * 9), 1.0f));
  ggml_tensor* prep = kg::HcPrep(c(), streams, norm, nullptr, nullptr, nullptr, kEps);
  TensorArena::Bind(prep, Allocate(ggml_nbytes(prep)));
  EXPECT_NE(reason(kg::CheckHcPrep(prep)), "accepted");
}

}  // namespace
