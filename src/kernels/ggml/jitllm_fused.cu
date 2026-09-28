// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// jitLLM's fusions of Qwen3.8's hyper-connection and MoE-output elementwise
// work, the BF16 conversion and the BF16 product (jitllm_ops.h). Each
// kernel repeats the arithmetic of the GGML nodes it replaces, operation by
// operation in their order, so that its result is theirs bit for bit: this
// file builds with GGML's device flags (-use_fast_math, as its own units,
// third_party/patches/ggml/0002), writes GGML's expressions for the
// functions (op_sigmoid, ggml_cuda_op_silu_single, scale_f32's
// scale · x + bias), reduces each norm as rms_norm_f32<1024> does, and
// writes a product that GGML computes in a node of its own and then adds
// with __fmul_rn, which is never contracted into a multiply-add.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/validate.h"

namespace jitllm::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Refused(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

constexpr int kThreads = 256;
constexpr int kGdnColumns = 4;      // value columns a warp of the recurrence keeps
constexpr int kNormThreads = 1024;  // rms_norm_f32's block for rows of 1,024 or more
constexpr int kMaxStreams = 8;      // the checks' bound on hc

// GGML's op_sigmoid and ggml_cuda_op_silu_single.
__device__ __forceinline__ float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float Silu(float x) { return x / (1.0f + expf(-x)); }
// scale_f32: scale · x + bias, the bias a kernel argument as there (ggml_scale's
// 0), so that it is not folded away.
__device__ __forceinline__ float Scale(float x, float scale, float bias) {
  return scale * x + bias;
}

// GGML's warp_reduce_sum over 32 lanes.
__device__ __forceinline__ float WarpSum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset, 32);
  }
  return x;
}

// rms_norm_f32<1024>'s scale of a row of n >= 1,024 floats: each thread sums
// the squares of its columns tid, tid + 1,024, ... in order, then GGML's
// block_reduce (a warp sum, each warp's sum through shared memory, a warp
// sum of those), then rsqrtf(mean + eps). GGML's `tmp / ncols + eps`
// compiles (fast math) to one multiply-add of the sum, the approximate
// reciprocal of ncols and eps; it is written so here, since the compiler
// would not contract it where only one thread uses the result. Every thread
// returns it; `shared` is free again when it returns.
__device__ float RowScale(const float* __restrict__ x, int n, float eps, float* shared) {
  const int tid = static_cast<int>(threadIdx.x);
  float tmp = 0.0f;
  for (int col = tid; col < n; col += kNormThreads) {
    const float xi = x[col];
    tmp += xi * xi;
  }
  tmp = WarpSum(tmp);
  if (tid % 32 == 0) {
    shared[tid / 32] = tmp;
  }
  __syncthreads();
  tmp = WarpSum(shared[tid % 32]);
  __syncthreads();
  const float inverse = 1.0f / static_cast<float>(n);
  return rsqrtf(fmaf(tmp, inverse, eps));
}

// res + out · (2 · sigmoid(inject / hc)), four columns a thread.
__global__ void HcCombineKernel(const float4* __restrict__ res, const float4* __restrict__ out,
                                const float* __restrict__ inject, float4* __restrict__ dst,
                                int width4, int hc, std::int64_t n4, float inv_hc, float bias) {
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= n4) {
    return;
  }
  const std::int64_t row = i / width4;  // t · hc + k
  const std::int64_t c4 = i % width4;
  const std::int64_t t = row / hc;
  const float w = Scale(Sigmoid(Scale(inject[row], inv_hc, bias)), 2.0f, bias);
  const float4 b = out[(t * width4) + c4];
  const float4 r = res[i];
  dst[i] = make_float4(r.x + __fmul_rn(b.x, w), r.y + __fmul_rn(b.y, w), r.z + __fmul_rn(b.z, w),
                       r.w + __fmul_rn(b.w, w));
}

template <typename T>
__device__ __forceinline__ T Store(float v);
template <>
__device__ __forceinline__ float Store<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ nv_bfloat16 Store<nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// One stream of one token a block: rms_norm, then times the weight.
template <typename T>
__global__ void __launch_bounds__(kNormThreads)
    HcNormKernel(const float* __restrict__ x, const float* __restrict__ weight, T* __restrict__ dst,
                 int width, int hc, float eps) {
  __shared__ float shared[32];
  const std::int64_t row = (static_cast<std::int64_t>(blockIdx.x) * hc) + blockIdx.y;
  const float* xr = x + (row * width);
  const float* wk = weight + (static_cast<std::int64_t>(blockIdx.y) * width);
  T* out = dst + (row * width);
  const float scale = RowScale(xr, width, eps, shared);
  for (int c = static_cast<int>(threadIdx.x); c < width; c += kNormThreads) {
    out[c] = Store<T>(scale * xr[c] * wk[c]);
  }
}

// One token a block: each stream's norm scale, then the gated streams
// folded in order and scaled by 1 / hc.
__global__ void __launch_bounds__(kNormThreads)
    HcMixKernel(const float* __restrict__ x, const float* __restrict__ weight,
                const float* __restrict__ gate, float* __restrict__ dst, int width, int hc,
                float eps, float inv_hc, float bias) {
  __shared__ float shared[32];
  __shared__ float scales[kMaxStreams];
  const std::int64_t t = blockIdx.x;
  const std::int64_t stride = static_cast<std::int64_t>(hc) * width;
  const float* xt = x + (t * stride);
  const float* gt = gate + (t * stride);
  for (int k = 0; k < hc; ++k) {
    const float s = RowScale(xt + (static_cast<std::int64_t>(k) * width), width, eps, shared);
    if (threadIdx.x == 0) {
      scales[k] = s;
    }
  }
  __syncthreads();
  for (int c = static_cast<int>(threadIdx.x); c < width; c += kNormThreads) {
    float acc = 0.0f;
    for (int k = 0; k < hc; ++k) {
      const std::int64_t at = (static_cast<std::int64_t>(k) * width) + c;
      const float xn = scales[k] * xt[at] * weight[at];
      const float gated = __fmul_rn(xn, Sigmoid(gt[at]));
      acc = k == 0 ? gated : acc + gated;
    }
    dst[(t * width) + c] = Scale(acc, inv_hc, bias);
  }
}

__device__ __forceinline__ float ExpertScale(const float* __restrict__ scales, std::int32_t id,
                                             int experts) {
  return id >= 0 && id < experts ? scales[id] : __uint_as_float(0x7fc00000u);
}

// silu(gate · s_gate[e]) · (up · s_up[e]), four columns a thread.
__global__ void MoeGluKernel(const float4* __restrict__ gate, const float4* __restrict__ up,
                             const std::int32_t* __restrict__ ids,
                             const float* __restrict__ gate_scale,
                             const float* __restrict__ up_scale, float4* __restrict__ dst, int n4,
                             int used, int ids_stride, int experts, std::int64_t total4) {
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= total4) {
    return;
  }
  const std::int64_t row = i / n4;  // t · used + j
  const std::int64_t t = row / used;
  const std::int64_t j = row % used;
  const std::int32_t id = ids[(t * ids_stride) + j];
  const float gs = ExpertScale(gate_scale, id, experts);
  const float us = ExpertScale(up_scale, id, experts);
  const float4 g = gate[i];
  const float4 u = up[i];
  dst[i] = make_float4(Silu(g.x * gs) * (u.x * us), Silu(g.y * gs) * (u.y * us),
                       Silu(g.z * gs) * (u.z * us), Silu(g.w * gs) * (u.w * us));
}

// Σ_j (down_j · s_down[e_j]) · w_j in expert order, plus shared ·
// sigmoid(shared gate), four columns a thread.
__global__ void MoeCombineKernel(const float4* __restrict__ down,
                                 const std::int32_t* __restrict__ ids,
                                 const float* __restrict__ down_scale,
                                 const float* __restrict__ weights,
                                 const float4* __restrict__ shared,
                                 const float* __restrict__ shared_gate, float4* __restrict__ dst,
                                 int width4, int used, int ids_stride, int experts,
                                 std::int64_t total4) {
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= total4) {
    return;
  }
  const std::int64_t t = i / width4;
  const std::int64_t c4 = i % width4;
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int j = 0; j < used; ++j) {
    const float ds = ExpertScale(down_scale, ids[(t * ids_stride) + j], experts);
    const float w = weights[(t * used) + j];
    const float4 d = down[(((t * used) + j) * width4) + c4];
    const float4 e = make_float4(__fmul_rn(d.x * ds, w), __fmul_rn(d.y * ds, w),
                                 __fmul_rn(d.z * ds, w), __fmul_rn(d.w * ds, w));
    if (j == 0) {
      acc = e;
    } else {
      acc = make_float4(acc.x + e.x, acc.y + e.y, acc.z + e.z, acc.w + e.w);
    }
  }
  const float g = Sigmoid(shared_gate[t]);
  const float4 s = shared[i];
  dst[i] = make_float4(acc.x + __fmul_rn(s.x, g), acc.y + __fmul_rn(s.y, g),
                       acc.z + __fmul_rn(s.z, g), acc.w + __fmul_rn(s.w, g));
}

// gated_delta_net_cuda<128, false, false>'s recurrence, each column's
// arithmetic as its SASS computes it (sm_121a, fast math): kv = k · S's
// column by a multiply-add chain over each lane's rows, a warp sum,
// delta = beta · (v - g · kv), S = g · S + k · delta, attn = q · S by the same
// chain and sum, times the scale. Upstream gives each warp one column, so
// 6,144 warps stream the whole sequence in several waves; here a warp keeps
// kColumns columns in registers and reads each token's k and q once for
// them, so one wave covers the heads.
template <int kColumns>
__global__ void __launch_bounds__(128)
    GdnColumnsKernel(const float* __restrict__ q, const float* __restrict__ k,
                     const float* __restrict__ v, const float* __restrict__ g,
                     const float* __restrict__ beta, const float* __restrict__ state_in,
                     float* __restrict__ dst, float* __restrict__ state_out, int heads,
                     int qk_heads, int tokens, std::int64_t sq1, std::int64_t sq2, std::int64_t sv1,
                     std::int64_t sv2, std::int64_t sb1, std::int64_t sb2, float scale) {
  constexpr int kS = 128;
  constexpr int kRows = kS / 32;
  const int lane = static_cast<int>(threadIdx.x);
  const int h = static_cast<int>(blockIdx.x);
  const int col0 =
      ((static_cast<int>(blockIdx.y) * blockDim.y) + static_cast<int>(threadIdx.y)) * kColumns;
  const int hq = h % qk_heads;
  float s[kColumns][kRows];
  const float* s0 = state_in + (static_cast<std::int64_t>(h) * kS * kS);
#pragma unroll
  for (int c = 0; c < kColumns; ++c) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      s[c][r] = s0[(static_cast<std::int64_t>(col0 + c) * kS) + (r * 32) + lane];
    }
  }
  float* attn = dst + (static_cast<std::int64_t>(h) * kS);
  // Each token's inputs are loaded a token ahead, so that the loads' latency
  // overlaps the previous token's arithmetic.
  float kn[kRows];
  float qn[kRows];
  float vn[kColumns];
  float beta_n = 0.0f;
  float g_n = 0.0f;
  const auto load = [&](int t) {
    const float* qt = q + (t * sq2) + (hq * sq1);
    const float* kt = k + (t * sq2) + (hq * sq1);
    const float* vt = v + (t * sv2) + (h * sv1);
    const std::int64_t gb = (t * sb2) + (h * sb1);
    beta_n = beta[gb];
    g_n = g[gb];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      kn[r] = kt[(r * 32) + lane];
      qn[r] = qt[(r * 32) + lane];
    }
#pragma unroll
    for (int c = 0; c < kColumns; ++c) {
      vn[c] = vt[col0 + c];
    }
  };
  if (tokens > 0) {
    load(0);
  }
  for (int t = 0; t < tokens; ++t) {
    float kr[kRows];
    float qr[kRows];
    float vr[kColumns];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      kr[r] = kn[r];
      qr[r] = qn[r];
    }
#pragma unroll
    for (int c = 0; c < kColumns; ++c) {
      vr[c] = vn[c];
    }
    const float beta_t = beta_n;
    const float g_t = expf(g_n);
    if (t + 1 < tokens) {
      load(t + 1);
    }
    float kv[kColumns];
#pragma unroll
    for (int c = 0; c < kColumns; ++c) {
      kv[c] = fmaf(kr[0], s[c][0], 0.0f);
#pragma unroll
      for (int r = 1; r < kRows; ++r) {
        kv[c] = fmaf(kr[r], s[c][r], kv[c]);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int c = 0; c < kColumns; ++c) {
        kv[c] = __fadd_rn(kv[c], __shfl_xor_sync(0xffffffffu, kv[c], offset, 32));
      }
    }
    float at[kColumns];
#pragma unroll
    for (int c = 0; c < kColumns; ++c) {
      const float delta = __fmul_rn(beta_t, fmaf(-g_t, kv[c], vr[c]));
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        s[c][r] = fmaf(g_t, s[c][r], __fmul_rn(kr[r], delta));
      }
      at[c] = fmaf(qr[0], s[c][0], 0.0f);
#pragma unroll
      for (int r = 1; r < kRows; ++r) {
        at[c] = fmaf(qr[r], s[c][r], at[c]);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int c = 0; c < kColumns; ++c) {
        at[c] = __fadd_rn(at[c], __shfl_xor_sync(0xffffffffu, at[c], offset, 32));
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int c = 0; c < kColumns; ++c) {
        attn[col0 + c] = __fmul_rn(at[c], scale);
      }
    }
    attn += static_cast<std::int64_t>(kS) * heads;
  }
  float* s1 = state_out + (static_cast<std::int64_t>(h) * kS * kS);
#pragma unroll
  for (int c = 0; c < kColumns; ++c) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      s1[(static_cast<std::int64_t>(col0 + c) * kS) + (r * 32) + lane] = s[c][r];
    }
  }
}

// The same recurrence with each value column split over 8 lanes of 16
// rows (4 columns a warp, 16 warps a block: 64 columns of one head; lane
// part p holds rows 32j + 4p .. 32j + 4p + 3, so that the 8 parts' float4
// reads of a staged row are 128 contiguous bytes, free of bank conflicts). The
// recurrence is sequential in tokens, and each token's k, q and v come from
// memory: read one token at a time, every step waits a memory latency. So
// the block stages chunks of kGdnChunk tokens' k, q, v, gate and beta in
// shared memory with asynchronous copies, the next chunk's while it runs
// this one's. The column sums take three shuffle steps. The F32 sums run in
// another order than upstream's, so the results differ from its in the
// last bits (tests/unit/qwen38_fused_test.cc bounds them).
constexpr int kGdnChunk = 16;
constexpr int kGdnBlockColumns = 64;
struct GdnStage {
  float k[kGdnChunk][128];
  float q[kGdnChunk][128];
  float v[kGdnChunk][kGdnBlockColumns];
  float g[kGdnChunk];
  float beta[kGdnChunk];
};

__global__ void __launch_bounds__(512, 2)
    GdnLanesKernel(const float* __restrict__ q, const float* __restrict__ k,
                   const float* __restrict__ v, const float* __restrict__ g,
                   const float* __restrict__ beta, const float* __restrict__ state_in,
                   float* __restrict__ dst, float* __restrict__ state_out, int heads, int qk_heads,
                   int tokens, std::int64_t sq1, std::int64_t sq2, std::int64_t sv1,
                   std::int64_t sv2, std::int64_t sb1, std::int64_t sb2, float scale) {
  constexpr int kS = 128;
  constexpr int kRows = 16;  // rows a lane holds
  __shared__ __align__(16) GdnStage stage[2];
  const int lane = static_cast<int>(threadIdx.x);
  const int tid = (static_cast<int>(threadIdx.y) * 32) + lane;
  const int part = lane % 8;  // which 16 rows
  const int h = static_cast<int>(blockIdx.x);
  const int col0 = static_cast<int>(blockIdx.y) * kGdnBlockColumns;
  const int local = (static_cast<int>(threadIdx.y) * 4) + (lane / 8);  // column in the block
  const int col = col0 + local;
  const int hq = h % qk_heads;
  const auto load = [&](int chunk, GdnStage& to) {
    const int t0 = chunk * kGdnChunk;
    const int n = min(kGdnChunk, tokens - t0);
    for (int i = tid; i < n * 32; i += 512) {
      const int tok = i / 32;
      const int ch = i % 32;
      const std::int64_t at = ((t0 + tok) * sq2) + (hq * sq1) + (ch * 4);
      __pipeline_memcpy_async(&to.k[tok][ch * 4], k + at, 16);
      __pipeline_memcpy_async(&to.q[tok][ch * 4], q + at, 16);
    }
    for (int i = tid; i < n * (kGdnBlockColumns / 4); i += 512) {
      const int tok = i / (kGdnBlockColumns / 4);
      const int ch = i % (kGdnBlockColumns / 4);
      __pipeline_memcpy_async(&to.v[tok][ch * 4],
                              v + ((t0 + tok) * sv2) + (h * sv1) + col0 + (ch * 4), 16);
    }
    if (tid < n) {
      const std::int64_t gb = ((t0 + tid) * sb2) + (h * sb1);
      __pipeline_memcpy_async(&to.g[tid], g + gb, 4);
      __pipeline_memcpy_async(&to.beta[tid], beta + gb, 4);
    }
    __pipeline_commit();
  };
  // Element i of a lane's rows is row Row(i).
  const auto row = [part](int i) { return (32 * (i / 4)) + (4 * part) + (i % 4); };
  float s[kRows];
  const float* s0 =
      state_in + (static_cast<std::int64_t>(h) * kS * kS) + (static_cast<std::int64_t>(col) * kS);
#pragma unroll
  for (int i = 0; i < kRows; i += 4) {
    const float4 f = *reinterpret_cast<const float4*>(s0 + row(i));
    s[i] = f.x;
    s[i + 1] = f.y;
    s[i + 2] = f.z;
    s[i + 3] = f.w;
  }
  float* attn = dst + (static_cast<std::int64_t>(h) * kS) + col;
  const int chunks = (tokens + kGdnChunk - 1) / kGdnChunk;
  if (chunks > 0) {
    load(0, stage[0]);
  }
  for (int c = 0; c < chunks; ++c) {
    if (c + 1 < chunks) {
      load(c + 1, stage[(c + 1) % 2]);
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();
    const GdnStage& st = stage[c % 2];
    const int n = min(kGdnChunk, tokens - (c * kGdnChunk));
    for (int tok = 0; tok < n; ++tok) {
      const float beta_t = st.beta[tok];
      const float g_t = expf(st.g[tok]);
      const float v_t = st.v[tok][local];
      float kr[kRows];
#pragma unroll
      for (int i = 0; i < kRows; i += 4) {
        const float4 kf = *reinterpret_cast<const float4*>(&st.k[tok][row(i)]);
        kr[i] = kf.x;
        kr[i + 1] = kf.y;
        kr[i + 2] = kf.z;
        kr[i + 3] = kf.w;
      }
      float kv = 0.0f;
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        kv = fmaf(kr[i], s[i], kv);
      }
#pragma unroll
      for (int offset = 4; offset > 0; offset >>= 1) {
        kv += __shfl_xor_sync(0xffffffffu, kv, offset, 32);
      }
      const float delta = beta_t * fmaf(-g_t, kv, v_t);
      float at = 0.0f;
#pragma unroll
      for (int i = 0; i < kRows; i += 4) {
        const float4 qf = *reinterpret_cast<const float4*>(&st.q[tok][row(i)]);
        const float qr[4] = {qf.x, qf.y, qf.z, qf.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          s[i + j] = fmaf(g_t, s[i + j], kr[i + j] * delta);
          at = fmaf(qr[j], s[i + j], at);
        }
      }
#pragma unroll
      for (int offset = 4; offset > 0; offset >>= 1) {
        at += __shfl_xor_sync(0xffffffffu, at, offset, 32);
      }
      if (part == 0) {
        *attn = at * scale;
      }
      attn += static_cast<std::int64_t>(kS) * heads;
    }
    // The buffer is loaded again two chunks on.
    __syncthreads();
  }
  float* s1 =
      state_out + (static_cast<std::int64_t>(h) * kS * kS) + (static_cast<std::int64_t>(col) * kS);
#pragma unroll
  for (int i = 0; i < kRows; i += 4) {
    *reinterpret_cast<float4*>(s1 + row(i)) = make_float4(s[i], s[i + 1], s[i + 2], s[i + 3]);
  }
}

// rms_norm_f32<256>'s scale of a 128-value row held one value a thread by
// the 128 threads of a block: its four warps' sums, then the sum over the
// eight warps GGML's 256-thread block has (the last four hold zeros).
__device__ float RowScale128(float v, float eps, float* shared) {
  const int tid = static_cast<int>(threadIdx.x);
  float tmp = WarpSum(fmaf(v, v, 0.0f));
  if (tid % 32 == 0) {
    shared[tid / 32] = tmp;
  }
  __syncthreads();
  const int lane = tid % 32;
  tmp = WarpSum(lane < 4 ? shared[lane] : 0.0f);
  __syncthreads();
  return rsqrtf(fmaf(tmp, 1.0f / 128.0f, eps));
}

// One head of one token a block, a channel a thread: the 4-tap causal
// convolution over the history then the rows (ssm_conv's multiply-add chain
// and its zero bias), silu, and for the query and key heads the L2 norm
// (rms_norm with eps, then scale_f32's scale).
__global__ void __launch_bounds__(128)
    GdnConvKernel(const float* __restrict__ x, const float* __restrict__ history,
                  const float* __restrict__ weight, float* __restrict__ out, int channels,
                  int qk_channels, float eps, float scale, float bias) {
  __shared__ float shared[4];
  const int c = (static_cast<int>(blockIdx.x) * 128) + static_cast<int>(threadIdx.x);
  const int t = static_cast<int>(blockIdx.y);
  float in[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int tau = t + j;  // time in the history-then-rows sequence
    in[j] =
        tau < 3 ? history[(c * 3) + tau] : x[(static_cast<std::int64_t>(tau - 3) * channels) + c];
  }
  const float* w = weight + (c * 4);
  float sum = fmaf(in[0], w[0], 0.0f);
  sum = fmaf(in[1], w[1], sum);
  sum = fmaf(in[2], w[2], sum);
  sum = fmaf(in[3], w[3], sum);
  sum = sum + bias;
  const float v = sum / (1.0f + expf(-sum));  // ggml_cuda_op_silu_single
  float y = v;
  if (static_cast<int>(blockIdx.x) * 128 < qk_channels) {  // a whole head: uniform
    const float s = RowScale128(v, eps, shared);
    y = Scale(s * v, scale, bias);
  }
  out[(static_cast<std::int64_t>(t) * channels) + c] = y;
}

// One 128-value head a warp: rms_norm_f32<256>'s scale (each warp-sized
// quarter summed as one of its warps would, then the eight warps' sums),
// times the weight, times sigmoid(z).
template <typename T>
__global__ void __launch_bounds__(256)
    GdnNormGateKernel(const float* __restrict__ o, const float* __restrict__ weight,
                      const float* __restrict__ z, T* __restrict__ out, int rows, float eps) {
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int row = (static_cast<int>(blockIdx.x) * 8) + (static_cast<int>(threadIdx.x) / 32);
  if (row >= rows) {
    return;
  }
  const float* x = o + (static_cast<std::int64_t>(row) * 128);
  const float* g = z + (static_cast<std::int64_t>(row) * 128);
  float v[4];
  float quarter[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    v[k] = x[(32 * k) + lane];
    quarter[k] = WarpSum(fmaf(v[k], v[k], 0.0f));
  }
  const float mine = lane == 0   ? quarter[0]
                     : lane == 1 ? quarter[1]
                     : lane == 2 ? quarter[2]
                     : lane == 3 ? quarter[3]
                                 : 0.0f;
  const float tmp = WarpSum(mine);
  const float s = rsqrtf(fmaf(tmp, 1.0f / 128.0f, eps));
  T* dst = out + (static_cast<std::int64_t>(row) * 128);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int d = (32 * k) + lane;
    dst[d] = Store<T>(s * v[k] * weight[d] * Sigmoid(g[d]));
  }
}

__global__ void Bf16Kernel(const float* __restrict__ x, nv_bfloat16* __restrict__ dst,
                           std::int64_t n) {
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i < n) {
    dst[i] = __float2bfloat16(x[i]);
  }
}

unsigned Blocks(std::int64_t items, int threads) {
  return static_cast<unsigned>((items + threads - 1) / threads);
}

}  // namespace

std::expected<void, KernelFailure> RunHcCombine(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckHcCombine(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* res = node->src[0];
    const std::int64_t n4 = ggml_nelements(node) / 4;
    const int hc = static_cast<int>(res->ne[1]);
    HcCombineKernel<<<Blocks(n4, kThreads), kThreads, 0, context.stream()>>>(
        static_cast<const float4*>(res->data), static_cast<const float4*>(node->src[1]->data),
        static_cast<const float*>(node->src[2]->data), static_cast<float4*>(node->data),
        static_cast<int>(res->ne[0] / 4), hc, n4, 1.0f / static_cast<float>(hc), 0.0f);
  });
}

std::expected<void, KernelFailure> RunHcNorm(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckHcNorm(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* x = node->src[0];
    const int width = static_cast<int>(x->ne[0]);
    const int hc = static_cast<int>(x->ne[1]);
    const dim3 grid(static_cast<unsigned>(x->ne[2]), static_cast<unsigned>(hc));
    const auto* xs = static_cast<const float*>(x->data);
    const auto* w = static_cast<const float*>(node->src[1]->data);
    const float eps = JitllmOpEps(node);
    if (node->type == GGML_TYPE_BF16) {
      HcNormKernel<nv_bfloat16><<<grid, kNormThreads, 0, context.stream()>>>(
          xs, w, static_cast<nv_bfloat16*>(node->data), width, hc, eps);
    } else {
      HcNormKernel<float><<<grid, kNormThreads, 0, context.stream()>>>(
          xs, w, static_cast<float*>(node->data), width, hc, eps);
    }
  });
}

std::expected<void, KernelFailure> RunHcMix(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckHcMix(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* x = node->src[0];
    const int hc = static_cast<int>(x->ne[1]);
    HcMixKernel<<<static_cast<unsigned>(x->ne[2]), kNormThreads, 0, context.stream()>>>(
        static_cast<const float*>(x->data), static_cast<const float*>(node->src[1]->data),
        static_cast<const float*>(node->src[2]->data), static_cast<float*>(node->data),
        static_cast<int>(x->ne[0]), hc, JitllmOpEps(node), 1.0f / static_cast<float>(hc), 0.0f);
  });
}

std::expected<void, KernelFailure> RunMoeGlu(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMoeGlu(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* ids = node->src[2];
    const std::int64_t total4 = ggml_nelements(node) / 4;
    MoeGluKernel<<<Blocks(total4, kThreads), kThreads, 0, context.stream()>>>(
        static_cast<const float4*>(node->src[0]->data),
        static_cast<const float4*>(node->src[1]->data), static_cast<const std::int32_t*>(ids->data),
        static_cast<const float*>(node->src[3]->data),
        static_cast<const float*>(node->src[4]->data), static_cast<float4*>(node->data),
        static_cast<int>(node->ne[0] / 4), static_cast<int>(node->ne[1]),
        static_cast<int>(ids->nb[1] / sizeof(std::int32_t)), static_cast<int>(node->src[3]->ne[0]),
        total4);
  });
}

std::expected<void, KernelFailure> RunMoeCombine(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMoeCombine(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* down = node->src[0];
    const ggml_tensor* ids = node->src[1];
    const std::int64_t total4 = ggml_nelements(node) / 4;
    MoeCombineKernel<<<Blocks(total4, kThreads), kThreads, 0, context.stream()>>>(
        static_cast<const float4*>(down->data), static_cast<const std::int32_t*>(ids->data),
        static_cast<const float*>(node->src[2]->data),
        static_cast<const float*>(node->src[3]->data),
        static_cast<const float4*>(node->src[4]->data),
        static_cast<const float*>(node->src[5]->data), static_cast<float4*>(node->data),
        static_cast<int>(down->ne[0] / 4), static_cast<int>(down->ne[1]),
        static_cast<int>(ids->nb[1] / sizeof(std::int32_t)), static_cast<int>(node->src[2]->ne[0]),
        total4);
  });
}

std::expected<void, KernelFailure> RunGatedDeltaNetColumns(LaunchContext& launch,
                                                           ggml_tensor* node) {
  if (auto checked = CheckGatedDeltaNetColumns(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* q = node->src[0];
    const ggml_tensor* v = node->src[2];
    const ggml_tensor* beta = node->src[4];
    constexpr int kS = 128;
    const int heads = static_cast<int>(v->ne[1]);
    const int tokens = static_cast<int>(v->ne[2]);
    const auto f = [](std::size_t bytes) {
      return static_cast<std::int64_t>(bytes / sizeof(float));
    };
    auto* dst = static_cast<float*>(node->data);
    const dim3 block(32, 4);
    const dim3 grid(static_cast<unsigned>(heads), kS / (4 * kGdnColumns));
    GdnColumnsKernel<kGdnColumns><<<grid, block, 0, context.stream()>>>(
        static_cast<const float*>(q->data), static_cast<const float*>(node->src[1]->data),
        static_cast<const float*>(v->data), static_cast<const float*>(node->src[3]->data),
        static_cast<const float*>(beta->data), static_cast<const float*>(node->src[5]->data), dst,
        dst + (static_cast<std::int64_t>(kS) * heads * tokens), heads, static_cast<int>(q->ne[1]),
        tokens, f(q->nb[1]), f(q->nb[2]), f(v->nb[1]), f(v->nb[2]), f(beta->nb[1]), f(beta->nb[2]),
        1.0f / sqrtf(static_cast<float>(kS)));
  });
}

std::expected<void, KernelFailure> RunGatedDeltaNetLanes(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGatedDeltaNetLanes(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* q = node->src[0];
    const ggml_tensor* v = node->src[2];
    const ggml_tensor* beta = node->src[4];
    constexpr int kS = 128;
    const int heads = static_cast<int>(v->ne[1]);
    const int tokens = static_cast<int>(v->ne[2]);
    const auto f = [](std::size_t bytes) {
      return static_cast<std::int64_t>(bytes / sizeof(float));
    };
    auto* dst = static_cast<float*>(node->data);
    const dim3 block(32, kGdnBlockColumns / 4);
    const dim3 grid(static_cast<unsigned>(heads), kS / kGdnBlockColumns);
    GdnLanesKernel<<<grid, block, 0, context.stream()>>>(
        static_cast<const float*>(q->data), static_cast<const float*>(node->src[1]->data),
        static_cast<const float*>(v->data), static_cast<const float*>(node->src[3]->data),
        static_cast<const float*>(beta->data), static_cast<const float*>(node->src[5]->data), dst,
        dst + (static_cast<std::int64_t>(kS) * heads * tokens), heads, static_cast<int>(q->ne[1]),
        tokens, f(q->nb[1]), f(q->nb[2]), f(v->nb[1]), f(v->nb[2]), f(beta->nb[1]), f(beta->nb[2]),
        1.0f / sqrtf(static_cast<float>(kS)));
  });
}

std::expected<void, KernelFailure> RunGdnConv(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGdnConv(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* x = node->src[0];
    const dim3 grid(static_cast<unsigned>(x->ne[0] / 128), static_cast<unsigned>(x->ne[1]));
    GdnConvKernel<<<grid, 128, 0, context.stream()>>>(
        static_cast<const float*>(x->data), static_cast<const float*>(node->src[1]->data),
        static_cast<const float*>(node->src[2]->data), static_cast<float*>(node->data),
        static_cast<int>(x->ne[0]), JitllmOpInt(node, 0), JitllmOpFloat(node, 2),
        JitllmOpFloat(node, 3), 0.0f);
  });
}

std::expected<void, KernelFailure> RunGdnNormGate(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGdnNormGate(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* o = node->src[0];
    const int rows = static_cast<int>(o->ne[1] * o->ne[2]);
    const auto* x = static_cast<const float*>(o->data);
    const auto* w = static_cast<const float*>(node->src[1]->data);
    const auto* z = static_cast<const float*>(node->src[2]->data);
    const float eps = JitllmOpEps(node);
    const unsigned blocks = static_cast<unsigned>((rows + 7) / 8);
    if (node->type == GGML_TYPE_BF16) {
      GdnNormGateKernel<nv_bfloat16><<<blocks, 256, 0, context.stream()>>>(
          x, w, z, static_cast<nv_bfloat16*>(node->data), rows, eps);
    } else {
      GdnNormGateKernel<float><<<blocks, 256, 0, context.stream()>>>(
          x, w, z, static_cast<float*>(node->data), rows, eps);
    }
  });
}

std::expected<void, KernelFailure> RunBf16(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckBf16(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const std::int64_t n = ggml_nelements(node);
    Bf16Kernel<<<Blocks(n, kThreads), kThreads, 0, context.stream()>>>(
        static_cast<const float*>(node->src[0]->data), static_cast<nv_bfloat16*>(node->data), n);
  });
}

std::expected<void, KernelFailure> RunGemmBf16(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGemmBf16(node); !checked) {
    return checked;
  }
  if (launch.cublas() == nullptr) {
    return Refused("the launch context lends no cuBLAS handle");
  }
  // cuBLAS writes its workspace while it reads the operands.
  if (auto clear = CheckClearOf(node, launch.cublas()->workspace().base,
                                launch.cublas()->workspace().size.value());
      !clear) {
    return clear;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* weights = node->src[0];
    const ggml_tensor* x = node->src[1];
    const int k = static_cast<int>(weights->ne[0]);
    const int n = static_cast<int>(weights->ne[1]);
    const int t = static_cast<int>(x->ne[1]);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // ggml_cuda_mul_mat_cublas_impl's call for BF16 weights with F32 output
    // (its lda and ldb the packed rows, ldc the output's).
    CUBLAS_CHECK(cublasGemmEx(context.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &alpha,
                              weights->data, CUDA_R_16BF, k, x->data, CUDA_R_16BF, k, &beta,
                              node->data, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  });
}

}  // namespace jitllm::kernels::ggml
