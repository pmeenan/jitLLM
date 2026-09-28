// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// jitLLM's Qwen-Image kernels (kernels/image/ops.h, gemm.h) on a GB10
// (label `gpu`): each against a host reference. The BF16 elementwise and row
// kernels against a host copy of the PyTorch expression they reproduce,
// rounding where it rounds: equal, or one BF16 step apart where the device's
// and the host's transcendental functions or summation orders differ; the
// products and attention against FP64 within a normalized squared error;
// and their refusals.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "kernels/image/gemm.h"
#include "kernels/image/ops.h"

namespace {

namespace ki = jitllm::kernels::image;
using Bf16 = std::uint16_t;

std::uint16_t ToBf16(float value) {
  auto u = std::bit_cast<std::uint32_t>(value);
  if ((u & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((u >> 16U) | 0x40U);
  }
  u += 0x7fffU + ((u >> 16U) & 1U);
  return static_cast<std::uint16_t>(u >> 16U);
}
float FromBf16(std::uint16_t bits) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}
float R(float v) { return FromBf16(ToBf16(v)); }

// BF16 steps between two finite values.
int Steps(std::uint16_t a, std::uint16_t b) {
  const auto ordered = [](std::uint16_t x) {
    return (x & 0x8000U) != 0 ? -static_cast<int>(x & 0x7fffU) : static_cast<int>(x);
  };
  return std::abs(ordered(a) - ordered(b));
}

std::vector<Bf16> Random(std::size_t n, std::uint32_t seed, float scale = 1.0f) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, scale);
  std::vector<Bf16> out(n);
  for (Bf16& v : out) {
    v = ToBf16(dist(gen));
  }
  return out;
}

template <typename T>
class Device {
 public:
  explicit Device(std::size_t n) : n_(n) {
    EXPECT_EQ(cudaMalloc(&p_, std::max<std::size_t>(n, 1) * sizeof(T)), cudaSuccess);
  }
  explicit Device(const std::vector<T>& host) : Device(host.size()) {
    EXPECT_EQ(cudaMemcpy(p_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;
  ~Device() { (void)cudaFree(p_); }
  T* get() const { return static_cast<T*>(p_); }
  std::vector<T> Get() const {
    std::vector<T> out(n_);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(out.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return out;
  }

 private:
  void* p_ = nullptr;
  std::size_t n_;
};

void Ok(const ki::Status& s, const std::string& what) {
  ASSERT_TRUE(s.has_value()) << what << ": " << s.error();
}

// Every element equal, or at most `steps` BF16 steps apart in at most
// `fraction` of them.
void ExpectClose(std::span<const Bf16> got, std::span<const Bf16> want, int steps, double fraction,
                 const std::string& what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  std::size_t off = 0;
  int worst = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const int d = Steps(got[i], want[i]);
    off += d != 0 ? 1 : 0;
    worst = std::max(worst, d);
  }
  EXPECT_LE(worst, steps) << what;
  EXPECT_LE(static_cast<double>(off), fraction * static_cast<double>(got.size())) << what;
}

double Nmse(std::span<const Bf16> got, std::span<const double> want) {
  double num = 0;
  double den = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = FromBf16(got[i]) - want[i];
    num += d * d;
    den += want[i] * want[i];
  }
  return num / std::max(den, 1e-300);
}

class ImageKernelsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    ASSERT_EQ(cublasCreate(&blas_), CUBLAS_STATUS_SUCCESS);
    ASSERT_EQ(cublasSetStream(blas_, stream_), CUBLAS_STATUS_SUCCESS);
  }
  void TearDown() override {
    (void)cublasDestroy(blas_);
    (void)cudaStreamDestroy(stream_);
  }
  void* s() const { return stream_; }
  cudaStream_t stream_ = nullptr;
  cublasHandle_t blas_ = nullptr;
};

// ---- attention

struct AttentionCase {
  std::int64_t q_rows, kv_rows, heads, kv_heads, pad;
  float q_scale = 2.0f;  // the queries' standard deviation
};

std::vector<double> AttentionReference(const std::vector<Bf16>& q, const std::vector<Bf16>& k,
                                       const std::vector<Bf16>& v, const AttentionCase& c,
                                       std::int64_t qs, std::int64_t ks, bool causal) {
  const float scale = 1.0f / std::sqrt(128.0f);
  std::vector<double> out(static_cast<std::size_t>(c.q_rows * c.heads * 128));
  const std::int64_t group = c.heads / c.kv_heads;
  for (std::int64_t h = 0; h < c.heads; ++h) {
    const std::int64_t kh = h / group;
    for (std::int64_t i = 0; i < c.q_rows; ++i) {
      const std::int64_t keys = causal ? i + 1 : c.kv_rows;
      std::vector<double> logit(static_cast<std::size_t>(keys));
      double mx = -std::numeric_limits<double>::infinity();
      for (std::int64_t j = 0; j < keys; ++j) {
        double dot = 0;
        for (std::int64_t d = 0; d < 128; ++d) {
          dot +=
              static_cast<double>(FromBf16(q[static_cast<std::size_t>((i * qs) + (h * 128) + d)])) *
              FromBf16(k[static_cast<std::size_t>((j * ks) + (kh * 128) + d)]);
        }
        logit[static_cast<std::size_t>(j)] = dot * scale;
        mx = std::max(mx, dot * scale);
      }
      double sum = 0;
      std::vector<double> acc(128, 0.0);
      for (std::int64_t j = 0; j < keys; ++j) {
        const double p = std::exp(logit[static_cast<std::size_t>(j)] - mx);
        sum += p;
        for (std::int64_t d = 0; d < 128; ++d) {
          acc[static_cast<std::size_t>(d)] +=
              p * FromBf16(v[static_cast<std::size_t>((j * ks) + (kh * 128) + d)]);
        }
      }
      for (std::int64_t d = 0; d < 128; ++d) {
        out[static_cast<std::size_t>((((i * c.heads) + h) * 128) + d)] =
            acc[static_cast<std::size_t>(d)] / sum;
      }
    }
  }
  return out;
}

TEST_F(ImageKernelsTest, FlashAttentionMatchesTheReference) {
  // Tiles partly past the rows and keys, one row and one key, a strided
  // layout, grouped heads, and the denoiser's key count; one past a tile in
  // each dimension, a 512² image's joint sequence (25 text + 1,024 image
  // rows, the image rows against all), a key count below one tile, and
  // queries large enough that the running maximum moves by tens of units
  // between tiles.
  const std::vector<AttentionCase> cases = {
      {200, 150, 2, 2, 0},   {128, 64, 1, 1, 0},   {1, 1, 1, 1, 0},
      {70, 90, 4, 2, 8},     {300, 4121, 2, 2, 0}, {129, 65, 2, 1, 0},
      {1024, 1049, 1, 1, 0}, {131, 7, 1, 1, 0},    {97, 333, 2, 2, 0, 16.0f}};
  for (const AttentionCase& c : cases) {
    const std::string name = std::format("q {} kv {} heads {}/{} pad {} q scale {}", c.q_rows,
                                         c.kv_rows, c.heads, c.kv_heads, c.pad, c.q_scale);
    const std::int64_t qs = (c.heads * 128) + c.pad;
    const std::int64_t ks = (c.kv_heads * 128) + c.pad;
    const auto q = Random(static_cast<std::size_t>(c.q_rows * qs), 11, c.q_scale);
    const auto k = Random(static_cast<std::size_t>(c.kv_rows * ks), 12, 2.0f);
    const auto v = Random(static_cast<std::size_t>(c.kv_rows * ks), 13);
    Device<Bf16> dq(q);
    Device<Bf16> dk(k);
    Device<Bf16> dv(v);
    Device<Bf16> out(static_cast<std::size_t>(c.q_rows * c.heads * 128));
    Ok(ki::FlashAttention(dq.get(), qs, dk.get(), ks, dv.get(), ks, out.get(), c.heads * 128,
                          c.q_rows, c.kv_rows, c.heads, c.kv_heads, 1.0f / std::sqrt(128.0f), s()),
       name);
    const double nmse = Nmse(out.Get(), AttentionReference(q, k, v, c, qs, ks, false));
    EXPECT_LE(nmse, 1e-4) << name;
  }
}

TEST_F(ImageKernelsTest, SmallAttentionIsCausalAndGrouped) {
  constexpr std::int64_t kRows = 37;
  constexpr std::int64_t kQ = std::int64_t{8} * 128;   // 8 query heads
  constexpr std::int64_t kKv = std::int64_t{2} * 128;  // 2 KV heads
  const AttentionCase c{kRows, kRows, 8, 2, 0};
  const auto q = Random(kRows * kQ, 21);
  const auto k = Random(kRows * kKv, 22);
  const auto v = Random(kRows * kKv, 23);
  Device<Bf16> dq(q);
  Device<Bf16> dk(k);
  Device<Bf16> dv(v);
  Device<Bf16> out(kRows * kQ);
  Ok(ki::SmallAttention(dq.get(), kQ, dk.get(), kKv, dv.get(), kKv, out.get(), kQ, kRows, 8, 2,
                        true, 1.0f / std::sqrt(128.0f), s()),
     "causal");
  EXPECT_LE(Nmse(out.Get(), AttentionReference(q, k, v, c, kQ, kKv, true)), 1e-4);
}

TEST_F(ImageKernelsTest, AttentionRefusesWhatItCannotIndex) {
  Device<Bf16> x(4096);
  // A stride that is not a multiple of 8, a misaligned base, a head count
  // that is not a multiple of the KV heads.
  EXPECT_FALSE(ki::FlashAttention(x.get(), 132, x.get(), 128, x.get(), 128, x.get(), 128, 4, 4, 1,
                                  1, 0.1f, s())
                   .has_value());
  EXPECT_FALSE(ki::FlashAttention(x.get() + 1, 128, x.get(), 128, x.get(), 128, x.get(), 128, 4, 4,
                                  1, 1, 0.1f, s())
                   .has_value());
  EXPECT_FALSE(ki::FlashAttention(x.get(), 384, x.get(), 256, x.get(), 256, x.get(), 384, 4, 4, 3,
                                  2, 0.1f, s())
                   .has_value());
  // A scale that is not positive and finite (masked keys would give NaN).
  for (const float scale : {0.0f, -0.1f, INFINITY, NAN}) {
    EXPECT_FALSE(ki::FlashAttention(x.get(), 128, x.get(), 128, x.get(), 128, x.get(), 128, 4, 4, 1,
                                    1, scale, s())
                     .has_value())
        << scale;
  }
  // Element counts whose product would overflow are refused, not wrapped.
  EXPECT_FALSE(
      ki::Upsample2x(x.get(), x.get(), std::int64_t{1} << 30, std::int64_t{1} << 30, 4, s())
          .has_value());
  EXPECT_FALSE(ki::Im2Col3x3(x.get(), std::int64_t{1} << 20, std::int64_t{1} << 20, 1 << 20, 0, 1,
                             x.get(), s())
                   .has_value());
  EXPECT_FALSE(ki::SmallAttention(x.get(), 128, x.get(), 128, x.get(), 128, x.get(), 128, 1025, 1,
                                  1, true, 0.1f, s())
                   .has_value());
}

// ---- rows and elements

TEST_F(ImageKernelsTest, LayerNormModulateSelectsTheTimestepRow) {
  constexpr std::int64_t kRows = 9;
  constexpr std::int64_t kWidth = 4096;
  constexpr std::int64_t kFirstTarget = 3;
  const auto x = Random(kRows * kWidth, 31, 3.0f);
  const auto mod = Random(kWidth * 2 * 2, 32, 0.5f);  // two rows of stride 2 x width
  Device<Bf16> dx(x);
  Device<Bf16> dm(mod);
  Device<Bf16> out(kRows * kWidth);
  Ok(ki::LayerNormModulate(dx.get(), out.get(), kRows, kWidth, 1e-6f, dm.get(), 2 * kWidth,
                           kFirstTarget, s()),
     "layer norm");
  std::vector<Bf16> want(kRows * kWidth);
  for (std::int64_t r = 0; r < kRows; ++r) {
    double mean = 0;
    for (std::int64_t c = 0; c < kWidth; ++c) {
      mean += FromBf16(x[static_cast<std::size_t>((r * kWidth) + c)]);
    }
    mean /= kWidth;
    double var = 0;
    for (std::int64_t c = 0; c < kWidth; ++c) {
      const double d = FromBf16(x[static_cast<std::size_t>((r * kWidth) + c)]) - mean;
      var += d * d;
    }
    const auto rstd = static_cast<float>(1.0 / std::sqrt((var / kWidth) + 1e-6));
    const Bf16* scale = mod.data() + (r >= kFirstTarget ? 0 : 2 * kWidth);
    for (std::int64_t c = 0; c < kWidth; ++c) {
      const float n =
          R((FromBf16(x[static_cast<std::size_t>((r * kWidth) + c)]) - static_cast<float>(mean)) *
            rstd);
      want[static_cast<std::size_t>((r * kWidth) + c)] = ToBf16(n * R(1.0f + FromBf16(scale[c])));
    }
  }
  ExpectClose(out.Get(), want, 1, 0.01, "LayerNormModulate");
}

TEST_F(ImageKernelsTest, GatedResidualAndEulerStepRoundAsPyTorch) {
  constexpr std::int64_t kRows = 5;
  constexpr std::int64_t kWidth = 64;
  const auto x = Random(kRows * kWidth, 41);
  const auto y = Random(kRows * kWidth * 2, 42);  // stride 2 x width
  const auto mod = Random(2 * kWidth, 43);
  Device<Bf16> dx(x);
  Device<Bf16> dy(y);
  Device<Bf16> dm(mod);
  Ok(ki::GatedResidual(dx.get(), dy.get(), 2 * kWidth, kRows, kWidth, dm.get(), kWidth, 2, s()),
     "gated residual");
  std::vector<Bf16> want(x.size());
  for (std::int64_t r = 0; r < kRows; ++r) {
    for (std::int64_t c = 0; c < kWidth; ++c) {
      const float g = FromBf16(mod[static_cast<std::size_t>((r >= 2 ? 0 : kWidth) + c)]);
      const float t =
          R(R(std::tanh(g)) * FromBf16(y[static_cast<std::size_t>((r * 2 * kWidth) + c)]));
      want[static_cast<std::size_t>((r * kWidth) + c)] =
          ToBf16(FromBf16(x[static_cast<std::size_t>((r * kWidth) + c)]) + t);
    }
  }
  ExpectClose(dx.Get(), want, 1, 0.01, "GatedResidual");
  // The scheduler's step, dt rounded to BF16 first.
  const auto sample = Random(1000, 44);
  const auto noise = Random(1000, 45);
  Device<Bf16> ds(sample);
  Device<Bf16> dn(noise);
  Device<Bf16> dout(1000);
  const float dt = -0.0130362511f;
  Ok(ki::EulerStep(ds.get(), dn.get(), dout.get(), dt, true, 1000, s()), "euler");
  std::vector<Bf16> step(1000);
  for (std::size_t i = 0; i < step.size(); ++i) {
    step[i] = ToBf16(FromBf16(sample[i]) + R(R(dt) * FromBf16(noise[i])));
  }
  ExpectClose(dout.Get(), step, 0, 0.0, "EulerStep");
}

TEST_F(ImageKernelsTest, NormsAndRotaryRoundAsPyTorch) {
  constexpr std::int64_t kRows = 6;
  constexpr std::int64_t kHeads = 3;
  const auto x = Random(kRows * kHeads * 128, 51, 4.0f);
  const auto w = Random(128, 52);
  std::vector<float> freqs(kRows * 128);
  for (std::size_t i = 0; i < freqs.size(); i += 2) {
    const float angle = static_cast<float>(i) * 0.37f;
    freqs[i] = std::cos(angle);
    freqs[i + 1] = std::sin(angle);
  }
  Device<Bf16> dx(x);
  Device<Bf16> dw(w);
  Device<float> df(freqs);
  Ok(ki::HeadNormRopeComplex(dx.get(), kHeads * 128, kRows, kHeads, dw.get(), df.get(), 1e-6f, s()),
     "rotary");
  std::vector<Bf16> want(x.size());
  for (std::int64_t r = 0; r < kRows; ++r) {
    for (std::int64_t h = 0; h < kHeads; ++h) {
      const Bf16* v = x.data() + (((r * kHeads) + h) * 128);
      double ss = 0;
      for (std::int64_t d = 0; d < 128; ++d) {
        ss += static_cast<double>(FromBf16(v[d])) * FromBf16(v[d]);
      }
      const auto rrms = static_cast<float>(1.0 / std::sqrt((ss / 128) + 1e-6));
      for (std::int64_t p = 0; p < 64; ++p) {
        const float a =
            R(R(FromBf16(v[2 * p]) * rrms) * FromBf16(w[static_cast<std::size_t>(2 * p)]));
        const float b = R(R(FromBf16(v[(2 * p) + 1]) * rrms) *
                          FromBf16(w[static_cast<std::size_t>((2 * p) + 1)]));
        const float c = freqs[static_cast<std::size_t>(((r * 64) + p) * 2)];
        const float sn = freqs[static_cast<std::size_t>((((r * 64) + p) * 2) + 1)];
        want[static_cast<std::size_t>((((r * kHeads) + h) * 128) + (2 * p))] =
            ToBf16((a * c) - (b * sn));
        want[static_cast<std::size_t>((((r * kHeads) + h) * 128) + (2 * p) + 1)] =
            ToBf16((a * sn) + (b * c));
      }
    }
  }
  ExpectClose(dx.Get(), want, 1, 0.01, "HeadNormRopeComplex");
  // Qwen3-VL's RMSNorm: w * bf16(x * rrms).
  constexpr std::int64_t kNormRows = 4;
  constexpr std::int64_t kNormWidth = 256;
  const auto rows = Random(kNormRows * kNormWidth, 53, 2.0f);
  const auto rw = Random(kNormWidth, 54);
  Device<Bf16> drows(rows);
  Device<Bf16> drw(rw);
  Device<Bf16> dn(kNormRows * kNormWidth);
  Ok(ki::RmsNorm(drows.get(), drw.get(), dn.get(), kNormRows, kNormWidth, 1e-6f, s()), "rms");
  std::vector<Bf16> rms(rows.size());
  for (std::int64_t r = 0; r < kNormRows; ++r) {
    double ss = 0;
    for (std::int64_t c = 0; c < kNormWidth; ++c) {
      ss += static_cast<double>(FromBf16(rows[static_cast<std::size_t>((r * 256) + c)])) *
            FromBf16(rows[static_cast<std::size_t>((r * 256) + c)]);
    }
    const auto rrms = static_cast<float>(1.0 / std::sqrt((ss / 256) + 1e-6));
    for (std::int64_t c = 0; c < kNormWidth; ++c) {
      rms[static_cast<std::size_t>((r * 256) + c)] =
          ToBf16(FromBf16(rw[static_cast<std::size_t>(c)]) *
                 R(FromBf16(rows[static_cast<std::size_t>((r * 256) + c)]) * rrms));
    }
  }
  ExpectClose(dn.Get(), rms, 1, 0.01, "RmsNorm");
}

TEST_F(ImageKernelsTest, SwiGluAddAndEmbeddings) {
  constexpr std::int64_t kRows = 3;
  const auto g = Random(kRows * 40, 61);
  const auto u = Random(kRows * 40, 62);
  Device<Bf16> dg(g);
  Device<Bf16> du(u);
  Device<Bf16> out(kRows * 20);
  // Rows of 20 at strides of 40.
  Ok(ki::SwiGlu(dg.get(), 40, du.get(), 40, out.get(), kRows, 20, s()), "swiglu");
  std::vector<Bf16> want(kRows * 20);
  for (std::int64_t r = 0; r < kRows; ++r) {
    for (std::int64_t c = 0; c < 20; ++c) {
      const float x = FromBf16(g[static_cast<std::size_t>((r * 40) + c)]);
      want[static_cast<std::size_t>((r * 20) + c)] = ToBf16(
          R(x / (1.0f + std::exp(-x))) * FromBf16(u[static_cast<std::size_t>((r * 40) + c)]));
    }
  }
  ExpectClose(out.Get(), want, 1, 0.05, "SwiGlu");
  // An id outside the table writes zeros and raises the flag.
  constexpr std::int64_t kTableWidth = 8;
  const auto table = Random(5 * kTableWidth, 63);
  Device<Bf16> dt(table);
  Device<Bf16> rows(3 * kTableWidth);
  Device<std::int32_t> ids(std::vector<std::int32_t>{4, 5, 0});
  Device<std::int32_t> bad(std::vector<std::int32_t>{0});
  Ok(ki::EmbedRows(dt.get(), 5, ids.get(), 3, 8, rows.get(), bad.get(), s()), "embed");
  const auto got = rows.Get();
  EXPECT_EQ(bad.Get()[0], 1);
  for (std::int64_t c = 0; c < 8; ++c) {
    EXPECT_EQ(got[static_cast<std::size_t>(c)], table[static_cast<std::size_t>(32 + c)]);
    EXPECT_EQ(got[static_cast<std::size_t>(8 + c)], 0);
    EXPECT_EQ(got[static_cast<std::size_t>(16 + c)], table[static_cast<std::size_t>(c)]);
  }
}

// ---- products and the VAE

TEST_F(ImageKernelsTest, LinearAndConvolutionMatchFp64) {
  // out[m, n] = x[m, k] . w[n, k]^T
  constexpr std::int64_t kM = 7;
  constexpr std::int64_t kN = 40;
  constexpr std::int64_t kK = 96;
  const auto x = Random(kM * kK, 71);
  const auto w = Random(kN * kK, 72);
  Device<Bf16> dx(x);
  Device<Bf16> dw(w);
  Device<Bf16> out(kM * kN);
  Ok(ki::Linear(blas_, dx.get(), kK, dw.get(), kK, out.get(), kN, kM, kN, kK), "linear");
  std::vector<double> want(kM * kN);
  for (std::int64_t m = 0; m < kM; ++m) {
    for (std::int64_t n = 0; n < kN; ++n) {
      for (std::int64_t k = 0; k < kK; ++k) {
        want[static_cast<std::size_t>((m * 40) + n)] +=
            static_cast<double>(FromBf16(x[static_cast<std::size_t>((m * 96) + k)])) *
            FromBf16(w[static_cast<std::size_t>((n * 96) + k)]);
      }
    }
  }
  EXPECT_LE(Nmse(out.Get(), want), 1e-5);
  // A 3x3 convolution through im2col in two pixel blocks, with a bias.
  constexpr std::int64_t kC = 5;
  constexpr std::int64_t kO = 6;
  constexpr std::int64_t kH = 7;
  constexpr std::int64_t kW = 9;
  const auto img = Random(kC * kH * kW, 73);
  const auto cw = Random(kO * kC * 9, 74);
  const auto bias = Random(kO, 75);
  Device<Bf16> di(img);
  Device<Bf16> dcw(cw);
  Device<Bf16> db(bias);
  Device<Bf16> col(kC * 9 * kH * kW);
  Device<Bf16> conv(kO * kH * kW);
  Ok(ki::FillBias(db.get(), conv.get(), kO, kH * kW, s()), "bias");
  for (std::int64_t p0 = 0; p0 < kH * kW; p0 += 4 * kW) {
    const std::int64_t count = std::min(4 * kW, (kH * kW) - p0);
    Ok(ki::Im2Col3x3(di.get(), kC, kH, kW, p0, count, col.get(), s()), "im2col");
    Ok(ki::ConvProduct(blas_, dcw.get(), kO, kC * 9, col.get(), count, conv.get() + p0, kH * kW,
                       count, true),
       "conv");
  }
  std::vector<double> cwant(kO * kH * kW);
  for (std::int64_t o = 0; o < kO; ++o) {
    for (std::int64_t y = 0; y < kH; ++y) {
      for (std::int64_t xx = 0; xx < kW; ++xx) {
        double acc = FromBf16(bias[static_cast<std::size_t>(o)]);
        for (std::int64_t c = 0; c < kC; ++c) {
          for (std::int64_t ky = 0; ky < 3; ++ky) {
            for (std::int64_t kx = 0; kx < 3; ++kx) {
              const std::int64_t sy = y + ky - 1;
              const std::int64_t sx = xx + kx - 1;
              if (sy < 0 || sy >= kH || sx < 0 || sx >= kW) {
                continue;
              }
              acc += static_cast<double>(
                         FromBf16(img[static_cast<std::size_t>((((c * kH) + sy) * kW) + sx)])) *
                     FromBf16(cw[static_cast<std::size_t>((((((o * kC) + c) * 3) + ky) * 3) + kx)]);
            }
          }
        }
        cwant[static_cast<std::size_t>((((o * kH) + y) * kW) + xx)] = acc;
      }
    }
  }
  EXPECT_LE(Nmse(conv.Get(), cwant), 1e-5);
}

TEST_F(ImageKernelsTest, VaeElementwiseStepsMatchTheirDefinitions) {
  // QwenImage21RMS_norm then SiLU, over 3 channels of 10 pixels.
  constexpr std::int64_t kChannels = 3;
  const auto x = Random(kChannels * 10, 81, 2.0f);
  const auto gamma = Random(kChannels, 82);
  Device<Bf16> dx(x);
  Device<Bf16> dg(gamma);
  Device<Bf16> out(kChannels * 10);
  Ok(ki::ChannelRmsNorm(dx.get(), dg.get(), out.get(), 3, 10, true, s()), "channel norm");
  std::vector<Bf16> want(30);
  for (std::int64_t p = 0; p < 10; ++p) {
    double ss = 0;
    for (std::int64_t c = 0; c < 3; ++c) {
      ss += static_cast<double>(FromBf16(x[static_cast<std::size_t>((c * 10) + p)])) *
            FromBf16(x[static_cast<std::size_t>((c * 10) + p)]);
    }
    const auto denom = static_cast<float>(std::max(std::sqrt(ss), 1e-12));
    for (std::int64_t c = 0; c < 3; ++c) {
      const float n = R(FromBf16(x[static_cast<std::size_t>((c * 10) + p)]) / denom);
      const float y =
          R(R(n * std::numbers::sqrt3_v<float>) * FromBf16(gamma[static_cast<std::size_t>(c)]));
      want[static_cast<std::size_t>((c * 10) + p)] = ToBf16(y / (1.0f + std::exp(-y)));
    }
  }
  ExpectClose(out.Get(), want, 1, 0.1, "ChannelRmsNorm");
  // Nearest 2x, then DupUp3D's shortcut at the last frame (2 channels to 1,
  // factor_t 2: repeats 4, output channel 0 reads the repeated channel
  // ((0 * 2 + 1) * 2 + b) * 2 + d, input channel (4 + 2b + d) / 4 = 1).
  const auto small = Random(std::size_t{2} * 2 * 3, 83);
  Device<Bf16> ds(small);
  Device<Bf16> up(std::size_t{2} * 4 * 6);
  Device<Bf16> sum(std::size_t{4} * 6);
  Ok(ki::Upsample2x(ds.get(), up.get(), 2, 2, 3, s()), "upsample");
  const auto u = up.Get();
  for (std::int64_t c = 0; c < 2; ++c) {
    for (std::int64_t y = 0; y < 4; ++y) {
      for (std::int64_t xx = 0; xx < 6; ++xx) {
        EXPECT_EQ(u[static_cast<std::size_t>((((c * 4) + y) * 6) + xx)],
                  small[static_cast<std::size_t>((((c * 2) + (y / 2)) * 3) + (xx / 2))]);
      }
    }
  }
  ASSERT_EQ(cudaMemset(sum.get(), 0, std::size_t{24} * 2), cudaSuccess);
  Ok(ki::AddDupUp(sum.get(), ds.get(), 2, 1, 2, 2, 3, s()), "dupup");
  const auto got = sum.Get();
  for (std::int64_t y = 0; y < 4; ++y) {
    for (std::int64_t xx = 0; xx < 6; ++xx) {
      EXPECT_EQ(got[static_cast<std::size_t>((y * 6) + xx)],
                small[static_cast<std::size_t>(((2 + (y / 2)) * 3) + (xx / 2))]);
    }
  }
  EXPECT_FALSE(ki::AddDupUp(sum.get(), ds.get(), 3, 1, 2, 2, 3, s()).has_value());
}

}  // namespace
