// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// FlashAttention-2's forward pass (Dao, 2023), written for jitLLM (ops.h
// FlashAttention): BF16, head dimension 128, no mask. Each block takes 128
// query rows of one head, eight warps of 16 rows each; the block streams
// the head's keys and values in tiles of 64 through shared memory, two
// stages deep (cp.async), and each warp keeps its queries, running maxima,
// sums and output in registers. Products are mma.sync m16n8k16 BF16 with
// F32 accumulation; shared memory rows are XOR-swizzled in 16-byte chunks,
// so that ldmatrix reads eight rows without bank conflicts. The last key
// tile is zero-filled and masked past kv_rows.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#include <string>

#include "kernels/image/ops.h"

namespace jitllm::kernels::image {
namespace {

constexpr int kD = 128;
constexpr int kBr = 128;  // query rows per block
constexpr int kBc = 64;   // keys per tile
constexpr int kWarps = kBr / 16;
constexpr int kThreads = kWarps * 32;
constexpr int kChunks = kD * 2 / 16;  // 16-byte chunks per row: 16
constexpr int kTileBytes = kBc * kD * 2;
constexpr int kSmem = 4 * kTileBytes;  // K and V, two stages each

__device__ __forceinline__ std::uint32_t Swizzle(int row, int chunk) {
  return static_cast<std::uint32_t>(row * kChunks + (chunk ^ (row & 7))) * 16U;
}

__device__ __forceinline__ void CpAsync16(std::uint32_t dst, const void* src, bool valid) {
  const int size = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(size));
}
__device__ __forceinline__ void CpAsyncCommit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int kPending>
__device__ __forceinline__ void CpAsyncWait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}

__device__ __forceinline__ void LdMatrixX4(std::uint32_t address, std::uint32_t& r0,
                                           std::uint32_t& r1, std::uint32_t& r2,
                                           std::uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(address));
}
__device__ __forceinline__ void LdMatrixX4Trans(std::uint32_t address, std::uint32_t& r0,
                                                std::uint32_t& r1, std::uint32_t& r2,
                                                std::uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(address));
}

// d += a (16x16 row) * b (16x8 col), BF16 in, F32 accumulate.
__device__ __forceinline__ void Mma(float (&d)[4], const std::uint32_t (&a)[4], std::uint32_t b0,
                                    std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two floats as a BF16 pair (round to nearest even), low half first.
__device__ __forceinline__ std::uint32_t PackBf16(float lo, float hi) {
  std::uint32_t out = 0;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "f"(hi), "f"(lo));
  return out;
}

__global__ void __launch_bounds__(kThreads, 1)
    FlashForwardKernel(const std::uint16_t* __restrict__ q, std::int64_t q_stride,
                       const std::uint16_t* __restrict__ k, std::int64_t k_stride,
                       const std::uint16_t* __restrict__ v, std::int64_t v_stride,
                       std::uint16_t* __restrict__ out, std::int64_t out_stride, int q_rows,
                       int kv_rows, int group, float scale_log2) {
  extern __shared__ __align__(128) unsigned char smem[];
  const std::uint32_t base = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  const std::uint32_t k_smem[2] = {base, base + kTileBytes};
  const std::uint32_t v_smem[2] = {base + 2 * kTileBytes, base + 3 * kTileBytes};
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int group_id = lane >> 2;
  const int tig = lane & 3;
  const int head = static_cast<int>(blockIdx.y);
  const int kv_head = head / group;
  const int m0 = static_cast<int>(blockIdx.x) * kBr + warp * 16;

  const std::uint16_t* kh = k + static_cast<std::int64_t>(kv_head) * kD;
  const std::uint16_t* vh = v + static_cast<std::int64_t>(kv_head) * kD;
  const int tiles = (kv_rows + kBc - 1) / kBc;

  // Stage one key and value tile: 64 rows x 16 chunks each, 4 per thread.
  auto load_tile = [&](int tile, int stage) {
    const int key0 = tile * kBc;
#pragma unroll
    for (int i = 0; i < kBc * kChunks / kThreads; ++i) {
      const int item = tid + i * kThreads;
      const int row = item / kChunks;
      const int chunk = item % kChunks;
      const int key = key0 + row;
      const bool valid = key < kv_rows;
      const std::int64_t at = static_cast<std::int64_t>(valid ? key : 0);
      CpAsync16(k_smem[stage] + Swizzle(row, chunk), kh + at * k_stride + chunk * 8, valid);
      CpAsync16(v_smem[stage] + Swizzle(row, chunk), vh + at * v_stride + chunk * 8, valid);
    }
    CpAsyncCommit();
  };

  load_tile(0, 0);

  // This warp's queries as A fragments, 8 k-steps of 16 dimensions.
  std::uint32_t qa[8][4];
  {
    const int r0 = m0 + group_id;
    const int r1 = r0 + 8;
    const std::uint16_t* q0 =
        q + static_cast<std::int64_t>(r0 < q_rows ? r0 : 0) * q_stride + head * kD;
    const std::uint16_t* q1 =
        q + static_cast<std::int64_t>(r1 < q_rows ? r1 : 0) * q_stride + head * kD;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int c = kk * 16 + tig * 2;
      qa[kk][0] = r0 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q0 + c) : 0U;
      qa[kk][1] = r1 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q1 + c) : 0U;
      qa[kk][2] = r0 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q0 + c + 8) : 0U;
      qa[kk][3] = r1 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q1 + c + 8) : 0U;
    }
  }

  float o[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
  }
  float row_max[2] = {-INFINITY, -INFINITY};
  float row_sum[2] = {0.0f, 0.0f};

  // ldmatrix lanes: lane l addresses row (l % 8) of matrix (l / 8).
  const int lm_row = lane % 8;
  const int lm_mat = lane / 8;

  for (int tile = 0; tile < tiles; ++tile) {
    const int stage = tile & 1;
    if (tile + 1 < tiles) {
      load_tile(tile + 1, stage ^ 1);
      CpAsyncWait<1>();
    } else {
      CpAsyncWait<0>();
    }
    __syncthreads();

    // S = Q K^T: 16 x 64 per warp, 8 n-tiles of 8 keys.
    float s[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
    }
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        // Matrices: (keys np*16 + 0..7, dims kk*16 + 0..7), (same keys,
        // dims +8), (keys np*16 + 8..15, dims kk*16), (those keys, +8).
        const int key = np * 16 + (lm_mat >> 1) * 8 + lm_row;
        const int chunk = kk * 2 + (lm_mat & 1);
        std::uint32_t b0, b1, b2, b3;
        LdMatrixX4(k_smem[stage] + Swizzle(key, chunk), b0, b1, b2, b3);
        Mma(s[np * 2], qa[kk], b0, b1);
        Mma(s[np * 2 + 1], qa[kk], b2, b3);
      }
    }
    // Keys past the end: -inf.
    if (tile * kBc + kBc > kv_rows) {
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int key = tile * kBc + n * 8 + tig * 2;
        if (key >= kv_rows) {
          s[n][0] = s[n][2] = -INFINITY;
        }
        if (key + 1 >= kv_rows) {
          s[n][1] = s[n][3] = -INFINITY;
        }
      }
    }
    // Online softmax: rows group_id (elements 0, 1) and group_id + 8 (2, 3).
    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      tile_max[0] = fmaxf(tile_max[0], fmaxf(s[n][0], s[n][1]));
      tile_max[1] = fmaxf(tile_max[1], fmaxf(s[n][2], s[n][3]));
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffU, tile_max[r], 1));
      tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffU, tile_max[r], 2));
    }
    float correction[2];
    float scaled_max[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float next = fmaxf(row_max[r], tile_max[r]);
      correction[r] = exp2f((row_max[r] - next) * scale_log2);
      row_max[r] = next;
      scaled_max[r] = next * scale_log2;
    }
    float tile_sum[2] = {0.0f, 0.0f};
    std::uint32_t pa[4][4];  // P as A fragments: 4 k-steps of 16 keys
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const float p0 = exp2f(s[n][0] * scale_log2 - scaled_max[0]);
      const float p1 = exp2f(s[n][1] * scale_log2 - scaled_max[0]);
      const float p2 = exp2f(s[n][2] * scale_log2 - scaled_max[1]);
      const float p3 = exp2f(s[n][3] * scale_log2 - scaled_max[1]);
      tile_sum[0] += p0 + p1;
      tile_sum[1] += p2 + p3;
      const int j = n / 2;
      if (n % 2 == 0) {
        pa[j][0] = PackBf16(p0, p1);
        pa[j][1] = PackBf16(p2, p3);
      } else {
        pa[j][2] = PackBf16(p0, p1);
        pa[j][3] = PackBf16(p2, p3);
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      row_sum[r] = row_sum[r] * correction[r] + tile_sum[r];
    }
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      o[n][0] *= correction[0];
      o[n][1] *= correction[0];
      o[n][2] *= correction[1];
      o[n][3] *= correction[1];
    }
    // O += P V: 16 n-tiles of 8 dimensions, 4 k-steps of 16 keys.
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int np = 0; np < 8; ++np) {
        // Transposed matrices: (keys j*16 + 0..7, dims np*16 + 0..7), (keys
        // +8, same dims), (keys 0..7, dims +8), (keys +8, dims +8).
        const int key = j * 16 + (lm_mat & 1) * 8 + lm_row;
        const int chunk = np * 2 + (lm_mat >> 1);
        std::uint32_t b0, b1, b2, b3;
        LdMatrixX4Trans(v_smem[stage] + Swizzle(key, chunk), b0, b1, b2, b3);
        Mma(o[np * 2], pa[j], b0, b1);
        Mma(o[np * 2 + 1], pa[j], b2, b3);
      }
    }
    __syncthreads();  // the stage is refilled next iteration
  }

  // The row sums over the four threads of each row, then O / l.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    row_sum[r] += __shfl_xor_sync(0xffffffffU, row_sum[r], 1);
    row_sum[r] += __shfl_xor_sync(0xffffffffU, row_sum[r], 2);
  }
  const float inv[2] = {1.0f / row_sum[0], 1.0f / row_sum[1]};
  const int r0 = m0 + group_id;
  const int r1 = r0 + 8;
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    const int c = n * 8 + tig * 2;
    if (r0 < q_rows) {
      *reinterpret_cast<std::uint32_t*>(out + static_cast<std::int64_t>(r0) * out_stride +
                                        head * kD + c) =
          PackBf16(o[n][0] * inv[0], o[n][1] * inv[0]);
    }
    if (r1 < q_rows) {
      *reinterpret_cast<std::uint32_t*>(out + static_cast<std::int64_t>(r1) * out_stride +
                                        head * kD + c) =
          PackBf16(o[n][2] * inv[1], o[n][3] * inv[1]);
    }
  }
}

}  // namespace

Status FlashAttention(const Bf16* q, std::int64_t q_stride, const Bf16* k, std::int64_t k_stride,
                      const Bf16* v, std::int64_t v_stride, Bf16* out, std::int64_t out_stride,
                      std::int64_t q_rows, std::int64_t kv_rows, std::int64_t heads,
                      std::int64_t kv_heads, float scale, Stream stream) {
  const auto aligned = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; };
  if (q_rows <= 0 || q_rows > (std::int64_t{1} << 30) || kv_rows <= 0 ||
      kv_rows > (std::int64_t{1} << 30) || heads <= 0 || heads > 65535 || kv_heads <= 0 ||
      heads % kv_heads != 0 || q_stride < heads * kD || k_stride < kv_heads * kD ||
      v_stride < kv_heads * kD || out_stride < heads * kD || q_stride % 8 != 0 ||
      k_stride % 8 != 0 || v_stride % 8 != 0 || out_stride % 8 != 0 || !aligned(q) || !aligned(k) ||
      !aligned(v) || !aligned(out)) {
    return std::unexpected(std::string("FlashAttention: sizes, strides or alignment"));
  }
  // A positive, finite scale: the running maximum is taken over unscaled
  // scores, and masked keys (-inf) times 0 would be NaN.
  if (!(scale > 0.0f) || !std::isfinite(scale)) {
    return std::unexpected(std::string("FlashAttention: the scale must be positive and finite"));
  }
  // Per call: the attribute is per device, and a cached flag would race
  // between threads.
  if (cudaFuncSetAttribute(FlashForwardKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kSmem) != cudaSuccess) {
    return std::unexpected(std::string("FlashAttention: shared memory limit"));
  }
  const dim3 grid(static_cast<unsigned>((q_rows + kBr - 1) / kBr), static_cast<unsigned>(heads));
  constexpr float kLog2e = 1.4426950408889634f;
  FlashForwardKernel<<<grid, kThreads, kSmem, static_cast<cudaStream_t>(stream)>>>(
      q, q_stride, k, k_stride, v, v_stride, out, out_stride, static_cast<int>(q_rows),
      static_cast<int>(kv_rows), static_cast<int>(heads / kv_heads), scale * kLog2e);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return std::unexpected(std::format("FlashAttention: {}", cudaGetErrorString(error)));
  }
  return {};
}

}  // namespace jitllm::kernels::image
