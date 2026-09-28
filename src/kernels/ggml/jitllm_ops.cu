// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// jitLLM's own kernels on GGML tensors (jitllm_ops.h): MXFP8 vector
// products, dequantization, quantization and the tensor-core product's
// launch (mxfp8_cutlass.h), and NVFP4 table rows.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <expected>
#include <format>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/moe_cutlass.h"
#include "kernels/ggml/mxfp8_cutlass.h"
#include "kernels/ggml/mxfp8_quant.cuh"

namespace jitllm::kernels::ggml {
namespace {

// An E8M0 scale: 2^(e - 127); e = 255 is NaN (OCP MX v1.0).
__device__ __forceinline__ float E8m0(std::uint8_t e) {
  if (e == 0) {
    return __uint_as_float(0x00400000u);  // 2^-127, a subnormal
  }
  return e == 255 ? __uint_as_float(0x7fc00000u) : __uint_as_float(std::uint32_t{e} << 23);
}

// Two E4M3 codes (low byte first) as floats.
__device__ __forceinline__ float2 E4m3x2(std::uint16_t pair) {
  const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
  return __half22float2(__half2(raw));
}

// Sixteen E4M3 codes as floats.
__device__ __forceinline__ void Decode16(const uint4 q, float out[16]) {
  const std::uint32_t words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const float2 lo = E4m3x2(static_cast<std::uint16_t>(words[w] & 0xffffu));
    const float2 hi = E4m3x2(static_cast<std::uint16_t>(words[w] >> 16));
    out[4 * w + 0] = lo.x;
    out[4 * w + 1] = lo.y;
    out[4 * w + 2] = hi.x;
    out[4 * w + 3] = hi.y;
  }
}

constexpr int kWarps = 8;  // rows per block

// One warp per output row. Lane l takes the row's 16-code vectors l, l + 32,
// ...; each vector is half of one 32-code block, so its dot product is
// scaled by its block's scale before it joins the lane's sum.
template <int kColumns>
__global__ void __launch_bounds__(kWarps * 32)
    Mxfp8Gemv(const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
              const float* __restrict__ x, float* __restrict__ y, int n, int k, int x_stride) {
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int row = static_cast<int>(blockIdx.x) * kWarps + static_cast<int>(threadIdx.x) / 32;
  if (row >= n) {
    return;
  }
  const std::uint8_t* w = codes + static_cast<std::int64_t>(row) * k;
  const std::uint8_t* s = scales + static_cast<std::int64_t>(row) * (k / 32);
  float sum[kColumns];
#pragma unroll
  for (int c = 0; c < kColumns; ++c) {
    sum[c] = 0.0f;
  }
  const int vectors = k / 16;
  for (int v = lane; v < vectors; v += 32) {
    float wf[16];
    Decode16(*reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(v) * 16), wf);
    const float scale = E8m0(s[v / 2]);
#pragma unroll
    for (int c = 0; c < kColumns; ++c) {
      const float4* xv =
          reinterpret_cast<const float4*>(x + static_cast<std::int64_t>(c) * x_stride + v * 16);
      float dot = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float4 a = xv[i];
        dot = fmaf(wf[4 * i + 0], a.x, dot);
        dot = fmaf(wf[4 * i + 1], a.y, dot);
        dot = fmaf(wf[4 * i + 2], a.z, dot);
        dot = fmaf(wf[4 * i + 3], a.w, dot);
      }
      sum[c] = fmaf(dot, scale, sum[c]);
    }
  }
#pragma unroll
  for (int c = 0; c < kColumns; ++c) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      sum[c] += __shfl_xor_sync(0xffffffffu, sum[c], offset);
    }
    if (lane == 0) {
      y[static_cast<std::int64_t>(c) * n + row] = sum[c];
    }
  }
}

// Sixteen codes a thread, into sixteen BF16 (exact: an E4M3 value times a
// power of two, within BF16's range for the scales real weights carry).
__global__ void Mxfp8ToBf16(const std::uint8_t* __restrict__ codes,
                            const std::uint8_t* __restrict__ scales,
                            __nv_bfloat16* __restrict__ out, std::int64_t vectors) {
  const std::int64_t v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= vectors) {
    return;
  }
  float wf[16];
  Decode16(*reinterpret_cast<const uint4*>(codes + v * 16), wf);
  const float scale = E8m0(scales[v / 2]);
  __align__(16) __nv_bfloat16 b[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    b[i] = __float2bfloat16_rn(wf[i] * scale);
  }
  uint4* dst = reinterpret_cast<uint4*>(out + v * 16);
  dst[0] = *reinterpret_cast<const uint4*>(&b[0]);
  dst[1] = *reinterpret_cast<const uint4*>(&b[8]);
}

// One block per id, one thread per value.
__global__ void Nvfp4RowsKernel(const std::uint8_t* __restrict__ table, std::int64_t rows,
                                const std::int32_t* __restrict__ ids,
                                const float* __restrict__ global, float* __restrict__ out,
                                int values) {
  constexpr float kE2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const int j = static_cast<int>(threadIdx.x);
  const std::int64_t i = blockIdx.x;
  const std::int32_t id = ids[i];
  float* dst = out + i * values + j;
  if (id < 0 || id >= rows) {
    *dst = __uint_as_float(0x7fc00000u);
    return;
  }
  const int row_bytes = values / 2 + values / 16;
  const std::uint8_t* row = table + static_cast<std::int64_t>(id) * row_bytes;
  const std::uint8_t byte = row[j / 2];
  const std::uint8_t code = (j % 2 == 0) ? (byte & 0x0f) : (byte >> 4);
  const float magnitude = kE2m1[code & 7];
  const std::uint8_t scale_code = row[values / 2 + j / 16];
  const float scale = __half2float(__half(__nv_cvt_fp8_to_halfraw(scale_code, __NV_E4M3)));
  *dst = ((code & 8) != 0 ? -magnitude : magnitude) * scale * global[0];
}

// The better of two (value, index) candidates: the higher value, the lower
// index among equals, a NaN never (index -1 is no candidate).
struct Best {
  float value;
  int index;
};

__device__ __forceinline__ Best Better(Best a, Best b) {
  if (b.index < 0) {
    return a;
  }
  if (a.index < 0) {
    return b;
  }
  if (b.value > a.value || (b.value == a.value && b.index < a.index)) {
    return b;
  }
  return a;
}

constexpr int kArgmaxThreads = 256;

// One block a row. Each thread scans a strided part of the row, then the
// block reduces; Better is commutative and associative over non-NaN
// candidates, so the order of either step never changes the answer.
__global__ void __launch_bounds__(kArgmaxThreads)
    ArgmaxKernel(const float* __restrict__ x, std::int32_t* __restrict__ out, int n) {
  const float* row = x + static_cast<std::int64_t>(blockIdx.x) * n;
  Best best{0.0f, -1};
  for (int i = static_cast<int>(threadIdx.x); i < n; i += kArgmaxThreads) {
    const float v = row[i];
    if (!isnan(v)) {
      best = Better(best, Best{v, i});
    }
  }
  __shared__ float values[kArgmaxThreads];
  __shared__ int indices[kArgmaxThreads];
  values[threadIdx.x] = best.value;
  indices[threadIdx.x] = best.index;
  __syncthreads();
  for (int stride = kArgmaxThreads / 2; stride > 0; stride /= 2) {
    if (static_cast<int>(threadIdx.x) < stride) {
      const Best other{values[threadIdx.x + stride], indices[threadIdx.x + stride]};
      best = Better(Best{values[threadIdx.x], indices[threadIdx.x]}, other);
      values[threadIdx.x] = best.value;
      indices[threadIdx.x] = best.index;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    // A row of NaN gives 0, never -1: the index feeds unchecked row
    // lookups (the Markov head's, the verify's embedding rows).
    out[blockIdx.x] = indices[0] < 0 ? 0 : indices[0];
  }
}

// One block a range, 16 bytes a thread per step.
__global__ void CopyRangesKernel(const RangeCopy* __restrict__ ranges) {
  const RangeCopy r = ranges[blockIdx.x];
  const std::uint64_t vectors = r.bytes / 16;
  const auto* from = reinterpret_cast<const uint4*>(r.from);  // NOLINT(performance-no-int-to-ptr)
  auto* to = reinterpret_cast<uint4*>(r.to);                  // NOLINT(performance-no-int-to-ptr)
  for (std::uint64_t v = threadIdx.x; v < vectors; v += blockDim.x) {
    to[v] = from[v];
  }
}

// Thirty-two values of row `x` from `at` as floats.
__device__ __forceinline__ void Load32(const float* x, float v[32]) {
  const auto* p = reinterpret_cast<const float4*>(x);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float4 f = p[i];
    v[(4 * i) + 0] = f.x;
    v[(4 * i) + 1] = f.y;
    v[(4 * i) + 2] = f.z;
    v[(4 * i) + 3] = f.w;
  }
}
__device__ __forceinline__ void Load32(const __nv_bfloat16* x, float v[32]) {
  const auto* p = reinterpret_cast<const uint4*>(x);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 q = p[i];
    const std::uint32_t words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      v[(8 * i) + (2 * w) + 0] = __uint_as_float(words[w] << 16U);
      v[(8 * i) + (2 * w) + 1] = __uint_as_float(words[w] & 0xffff0000U);
    }
  }
}

// One 32-value block a thread (rows past `rows`, up to the scale atoms'
// padding, write a zero scale): its E8M0 scale and 32 E4M3 codes.
template <typename T>
__global__ void Mxfp8QuantizeKernel(const T* __restrict__ x, std::int64_t x_stride,
                                    std::uint8_t* __restrict__ codes,
                                    std::uint8_t* __restrict__ scales, int k, int rows,
                                    std::int64_t items) {
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= items) {
    return;
  }
  const int blocks = k / 32;
  const auto r = static_cast<int>(i / blocks);
  const auto b = static_cast<int>(i % blocks);
  const std::uint64_t at =
      moe::SfOffset(static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(b),
                    static_cast<std::uint64_t>(blocks));
  if (r >= rows) {
    scales[at] = 0;
    return;
  }
  float v[32];
  Load32(x + (static_cast<std::int64_t>(r) * x_stride) + (b * 32), v);
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    amax = fmaxf(amax, fabsf(v[j]));
  }
  const std::uint32_t e = mxfp8::ScaleCode(amax);
  const float inverse = mxfp8::InverseScale(e);
  std::uint32_t packed[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    packed[j] =
        mxfp8::Pack4(v[(4 * j) + 0], v[(4 * j) + 1], v[(4 * j) + 2], v[(4 * j) + 3], inverse);
  }
  auto* out = reinterpret_cast<uint4*>(codes + (static_cast<std::int64_t>(r) * k) + (b * 32));
  out[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  out[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
  scales[at] = static_cast<std::uint8_t>(e);
}

// One scale a thread: row r's block b to its swizzled place (zero for the
// padding rows).
__global__ void Mxfp8SwizzleKernel(const std::uint8_t* __restrict__ in,
                                   std::uint8_t* __restrict__ out, int n, int blocks,
                                   std::int64_t items) {
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= items) {
    return;
  }
  const auto r = static_cast<int>(i / blocks);
  const auto b = static_cast<int>(i % blocks);
  out[moe::SfOffset(static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(b),
                    static_cast<std::uint64_t>(blocks))] = r < n ? in[i] : 0;
}

template <int kColumns>
void LaunchGemv(const ggml_tensor* node, cudaStream_t stream) {
  const ggml_tensor* codes = node->src[0];
  const ggml_tensor* x = node->src[2];
  const int n = static_cast<int>(codes->ne[1]);
  const int k = static_cast<int>(codes->ne[0]);
  const dim3 grid(static_cast<unsigned>((n + kWarps - 1) / kWarps));
  Mxfp8Gemv<kColumns><<<grid, kWarps * 32, 0, stream>>>(
      static_cast<const std::uint8_t*>(codes->data),
      static_cast<const std::uint8_t*>(node->src[1]->data), static_cast<const float*>(x->data),
      static_cast<float*>(node->data), n, k, static_cast<int>(x->nb[1] / sizeof(float)));
}

}  // namespace

std::expected<void, KernelFailure> RunMxfp8MulMatVec(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMxfp8MulMatVec(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    cudaStream_t stream = context.stream();
    switch (node->src[2]->ne[1]) {
      case 1:
        LaunchGemv<1>(node, stream);
        break;
      case 2:
        LaunchGemv<2>(node, stream);
        break;
      case 3:
        LaunchGemv<3>(node, stream);
        break;
      case 4:
        LaunchGemv<4>(node, stream);
        break;
      case 5:
        LaunchGemv<5>(node, stream);
        break;
      case 6:
        LaunchGemv<6>(node, stream);
        break;
      case 7:
        LaunchGemv<7>(node, stream);
        break;
      default:
        LaunchGemv<8>(node, stream);
        break;
    }
  });
}

std::expected<void, KernelFailure> RunMxfp8Dequant(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMxfp8Dequant(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const std::int64_t vectors = ggml_nelements(node->src[0]) / 16;
    constexpr int kThreads = 256;
    Mxfp8ToBf16<<<static_cast<unsigned>((vectors + kThreads - 1) / kThreads), kThreads, 0,
                  context.stream()>>>(static_cast<const std::uint8_t*>(node->src[0]->data),
                                      static_cast<const std::uint8_t*>(node->src[1]->data),
                                      static_cast<__nv_bfloat16*>(node->data), vectors);
  });
}

std::expected<void, KernelFailure> RunArgmax(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckArgmax(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* x = node->src[0];
    ArgmaxKernel<<<static_cast<unsigned>(x->ne[1]), kArgmaxThreads, 0, context.stream()>>>(
        static_cast<const float*>(x->data), static_cast<std::int32_t*>(node->data),
        static_cast<int>(x->ne[0]));
  });
}

std::expected<void, KernelFailure> CopyRanges(LaunchContext& launch, const RangeCopy* ranges,
                                              std::uint32_t count) {
  if (count == 0) {
    return {};
  }
  if (ranges == nullptr || count > kMaxRangeCopies) {
    return std::unexpected(KernelFailure{.error = KernelError::kRejected,
                                         .detail = "range copies: none given, or too many"});
  }
  for (std::uint32_t i = 0; i < count; ++i) {
    const RangeCopy& r = ranges[i];
    if (r.from == 0 || r.to == 0 || r.from % 16 != 0 || r.to % 16 != 0 || r.bytes % 16 != 0) {
      return std::unexpected(KernelFailure{
          .error = KernelError::kRejected,
          .detail = "range copies: every range 16-byte aligned, a multiple of 16 bytes"});
    }
  }
  return launch.Run(base::Bytes(0), [ranges, count](ggml_backend_cuda_context& context) {
    CopyRangesKernel<<<count, 256, 0, context.stream()>>>(ranges);
  });
}

std::expected<void, KernelFailure> RunNvfp4Rows(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckNvfp4Rows(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* table = node->src[0];
    const int values = static_cast<int>(node->ne[0]);
    const unsigned ids = static_cast<unsigned>(node->src[1]->ne[0]);
    Nvfp4RowsKernel<<<ids, static_cast<unsigned>(values), 0, context.stream()>>>(
        static_cast<const std::uint8_t*>(table->data), table->ne[1],
        static_cast<const std::int32_t*>(node->src[1]->data),
        static_cast<const float*>(node->src[2]->data), static_cast<float*>(node->data), values);
  });
}

std::expected<void, KernelFailure> RunMxfp8Quantize(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMxfp8Quantize(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* x = node->src[0];
    const int k = JitllmOpInt(node, 0);
    const int rows = JitllmOpInt(node, 1);
    const mxfp8::RowsLayout layout{.k = static_cast<std::uint64_t>(k),
                                   .rows = static_cast<std::uint64_t>(rows)};
    auto* base = static_cast<std::uint8_t*>(node->data);
    const std::int64_t items = static_cast<std::int64_t>(mxfp8::PaddedRows(layout.rows)) * (k / 32);
    constexpr int kThreads = 256;
    const auto grid = static_cast<unsigned>((items + kThreads - 1) / kThreads);
    const auto stride = static_cast<std::int64_t>(x->nb[1] / ggml_type_size(x->type));
    if (x->type == GGML_TYPE_BF16) {
      Mxfp8QuantizeKernel<<<grid, kThreads, 0, context.stream()>>>(
          static_cast<const __nv_bfloat16*>(x->data), stride, base + layout.codes(),
          base + layout.scales(), k, rows, items);
    } else {
      Mxfp8QuantizeKernel<<<grid, kThreads, 0, context.stream()>>>(
          static_cast<const float*>(x->data), stride, base + layout.codes(), base + layout.scales(),
          k, rows, items);
    }
  });
}

std::expected<void, KernelFailure> RunMxfp8Swizzle(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMxfp8Swizzle(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const int n = JitllmOpInt(node, 0);
    const int blocks = JitllmOpInt(node, 1) / 32;
    const std::int64_t items =
        static_cast<std::int64_t>(mxfp8::PaddedRows(static_cast<std::uint64_t>(n))) * blocks;
    constexpr int kThreads = 256;
    Mxfp8SwizzleKernel<<<static_cast<unsigned>((items + kThreads - 1) / kThreads), kThreads, 0,
                         context.stream()>>>(static_cast<const std::uint8_t*>(node->src[0]->data),
                                             static_cast<std::uint8_t*>(node->data), n, blocks,
                                             items);
  });
}

namespace {

mxfp8::Gemm Mxfp8GemmOf(const ggml_tensor* node) {
  const int k = JitllmOpInt(node, 0);
  const int t = JitllmOpInt(node, 2);
  const mxfp8::RowsLayout a{.k = static_cast<std::uint64_t>(k),
                            .rows = static_cast<std::uint64_t>(t)};
  const auto* base = static_cast<const std::uint8_t*>(node->src[0]->data);
  return {.m = t,
          .n = JitllmOpInt(node, 1),
          .k = k,
          .a = base + a.codes(),
          .a_scales = base + a.scales(),
          .b = node->src[1]->data,
          .b_scales = node->src[2]->data,
          .d = node->data,
          .bf16 = node->type == GGML_TYPE_BF16};
}

}  // namespace

std::expected<std::uint64_t, KernelFailure> PlanMxfp8Gemm(const LaunchContext& launch,
                                                          const ggml_tensor* node) {
  if (auto checked = CheckMxfp8Gemm(node); !checked) {
    return std::unexpected(checked.error());
  }
  const auto& device = ggml_cuda_info().devices[launch.device()];
  if (device.cc != 1210 || !mxfp8::Available()) {
    return std::unexpected(
        KernelFailure{.error = KernelError::kRejected,
                      .detail = "the MXFP8 product runs on a compute capability 12.1 device only"});
  }
  return static_cast<std::uint64_t>(mxfp8::Scratch(Mxfp8GemmOf(node), device.nsm));
}

std::expected<void, KernelFailure> RunMxfp8Gemm(LaunchContext& launch, ggml_tensor* node) {
  auto scratch = PlanMxfp8Gemm(launch, node);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  const int sms = ggml_cuda_info().devices[launch.device()].nsm;
  int status = 0;
  auto ran = launch.Run(base::Bytes(*scratch), [&](ggml_backend_cuda_context& context) {
    mxfp8::Gemm gemm = Mxfp8GemmOf(node);
    if (*scratch > 0) {
      ggml_cuda_pool_alloc<std::uint8_t> pool(context.pool(), *scratch);
      gemm.scratch = pool.get();
      status = mxfp8::Run(gemm, sms, context.stream());
    } else {
      status = mxfp8::Run(gemm, sms, context.stream());
    }
  });
  if (!ran) {
    return ran;
  }
  if (status != 0) {
    return std::unexpected(KernelFailure{
        .error = KernelError::kRejected,
        .detail = std::format("CUTLASS refused the MXFP8 product (status {})", status - 1)});
  }
  return {};
}

}  // namespace jitllm::kernels::ggml
