// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// jitLLM's own operations on GGML tensors, for the formats GGML has no type
// for (M3, Qwen3.8 Flash Next's ModelOpt checkpoint; the A/B that chose them
// is docs/experiments/qwen38-native/README.md):
//
//   jitllm.mxfp8.mul_mat_vec  y[n, t] = W[n, :] · x[:, t] for up to 8
//                             columns t, W in MXFP8 (E4M3 codes, one E8M0
//                             scale per 32 along k), each 32-element block's
//                             dot product in F32, then scaled, the blocks
//                             summed in F32; memory-bound decode products.
//   jitllm.mxfp8.dequant      W as BF16 [k, n], exactly (an E4M3 value times
//                             a power of two) wherever that is a BF16 value,
//                             for GGML's float products (cuBLAS) over wider
//                             batches. E8M0 0xFF and E4M3 0x7F/0xFF are NaN.
//   jitllm.nvfp4.get_rows     rows of a table in ModelOpt NVFP4, each row its
//                             v/2 code bytes (element 2i in the low nibble)
//                             then its v/16 E4M3 scales, times the table's
//                             global F32 scale, as F32 [v, ids]: Qwen3.8's
//                             n-gram embedding lookup.
//
// Each is a GGML_OP_CUSTOM node (ggml_custom_4d) whose function pointer
// names the operation; the function itself is never called (GGML's CPU
// backend never runs these graphs). The builders make the nodes; the plan
// (graph_plan.h) names the implementation from the kind; the checks here
// are what each implementation refuses on the host before anything is
// queued (D-086), and the launchers are in jitllm_ops.cu.
//
// Byte tensors are GGML_TYPE_I8: MXFP8 codes [k, n] and scales [k/32, n],
// NVFP4 table rows [v/2 + v/16, rows].

#ifndef JITLLM_KERNELS_GGML_JITLLM_OPS_H_
#define JITLLM_KERNELS_GGML_JITLLM_OPS_H_

#include <cstdint>
#include <expected>

#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

enum class JitllmOp : std::uint8_t {
  kNone,
  kMxfp8MulMatVec,
  kMxfp8Dequant,
  kNvfp4Rows,
};

// The operation a GGML_OP_CUSTOM node names, or kNone.
JitllmOp JitllmOpOf(const ggml_tensor* node);

// The most columns jitllm.mxfp8.mul_mat_vec takes.
inline constexpr std::int64_t kMxfp8VecColumns = 8;

// Builders. `codes` I8 [k, n], `scales` I8 [k / 32, n], `x` F32 [k, t].
ggml_tensor* Mxfp8MulMatVec(ggml_context* context, ggml_tensor* codes, ggml_tensor* scales,
                            ggml_tensor* x);
ggml_tensor* Mxfp8Dequant(ggml_context* context, ggml_tensor* codes, ggml_tensor* scales);
// `table` I8 [values / 2 + values / 16, rows], `ids` I32 [n], `scale` F32
// [1]: F32 [values, n].
ggml_tensor* Nvfp4Rows(ggml_context* context, ggml_tensor* table, ggml_tensor* ids,
                       ggml_tensor* scale, std::int64_t values);

// The host checks: operands bound, typed and shaped as above, packed where
// the kernels read them with vector loads (codes and x rows 16-byte
// aligned), extents within the kernels' 32-bit indexing, and outputs
// disjoint from their operands. The row lookup's ids are not read here:
// the caller builds them within the table (model/qwen38.h's n-gram hash
// over offsets and sizes the binding checked); the kernel writes NaN for
// an id outside the table rather than read out of bounds.
std::expected<void, KernelFailure> CheckMxfp8MulMatVec(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMxfp8Dequant(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckNvfp4Rows(const ggml_tensor* node);

// The launchers (CUDA builds, jitllm_ops.cu): the check, then one kernel
// on the context's stream. None draws scratch.
class LaunchContext;
std::expected<void, KernelFailure> RunMxfp8MulMatVec(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMxfp8Dequant(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunNvfp4Rows(LaunchContext& launch, ggml_tensor* node);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_JITLLM_OPS_H_
