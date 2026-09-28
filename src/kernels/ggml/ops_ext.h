// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML-derived implementations of the operations DeepSeek V4 Flash
// (GGUF) and Qwen3.8 Flash add beyond the backend proof's (ops.h), over the
// K-C launch context (D-053). As there, each checks every precondition its
// GGML launcher asserts (validate_ext.h), so an unsupported operand is a
// rejection and never an abort, and queues upstream's kernels on the
// context's stream. An operation that draws scratch from the context's pool
// computes its bound first, as upstream's launcher will draw it, so that
// the pool's bound is never exceeded (launch.h).
//
// The quantized matrix products come as GGML's two kernel families, vector
// (MMVQ) and tile (MMQ), which the plan selects between as upstream's
// routing would (SelectMulMatQ); flash attention comes as the tensor-core
// (MMA) kernels for head dimensions 256 and 512, grouped 8 query heads to a
// KV head, as upstream's MMA dispatch groups them on the GB10
// (fattn.cu:218-268). Upstream would take its vector kernel instead for a
// D = 256 decode row over fewer than 8,192 cells (fattn.cu:610-618), which
// is not built; the MMA kernel runs those rows here.
//
// Row indices and expert ids read from device memory are the plan's to
// bound (validate_ext.h).

#ifndef JITLLM_KERNELS_GGML_OPS_EXT_H_
#define JITLLM_KERNELS_GGML_OPS_EXT_H_

#include <cstdint>
#include <expected>

#include "ggml.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"

namespace jitllm::kernels::ggml {

// Quantized matrix products: a ggml_mul_mat or ggml_mul_mat_id node with
// weights of a compiled quantized type (validate_ext.h CheckMulMatQ,
// CheckMulMatIdQ, QuantMulMatPath).
// What upstream's routing chooses on the context's device
// (ggml-cuda.cu:1864-1871 for mul_mat, 1924-1942 for mul_mat_id), refused
// where it would take neither family.
std::expected<QuantMulMatPath, KernelFailure> SelectMulMatQ(const LaunchContext& launch,
                                                            const ggml_tensor* node);
// The pool scratch each family's launcher draws for a node it takes, each
// block from a 256-byte boundary in the launcher's allocation order: MMVQ's
// Q8_1 activations (mmvq.cu:1484-1486); MMQ's expert maps for mul_mat_id,
// its quantized activations and the stream-k fixup buffer
// (mmq.cu:199-278, mmq.cuh:1446-1455).
std::expected<std::uint64_t, KernelFailure> PlanMulMatVecQ(const LaunchContext& launch,
                                                           const ggml_tensor* node);
std::expected<std::uint64_t, KernelFailure> PlanMulMatQ(const LaunchContext& launch,
                                                        const ggml_tensor* node);
// Launches the family on a node upstream routes to it: ggml_cuda_mul_mat_vec_q
// or ggml_cuda_mul_mat_q, with the node's ids for mul_mat_id.
std::expected<void, KernelFailure> MulMatVecQ(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> MulMatQ(LaunchContext& launch, ggml_tensor* node);

// A ggml_mul_mat node with GGML_HINT_SRC0_IS_HADAMARD, as upstream runs it:
// the fast Walsh-Hadamard transform of the activations (ggml_cuda_op_fwht).
std::expected<void, KernelFailure> MulMatHadamard(LaunchContext& launch, ggml_tensor* node);

// Elementwise and row operations, F32 (validate_ext.h). None draws scratch.
std::expected<void, KernelFailure> Unary(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Scale(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Clamp(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Fill(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Repeat(LaunchContext& launch, ggml_tensor* node);
// ggml_sub and ggml_div nodes with broadcasting, as validate.h's CheckBinary
// takes add and mul.
std::expected<void, KernelFailure> Sub(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Div(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Concat(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> SumRows(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> SwiGluClamp(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RopeExt(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> GetRowsExt(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> SetRowsExt(LaunchContext& launch, ggml_tensor* node);

// MoE routing and the indexer's selection, in jitLLM's build without CUB
// (third_party/patches/ggml/0001). Argsort takes the bitonic kernel only,
// refused if a padded row does not fit the device's shared memory. Top-k
// takes upstream's radix select for rows over 1,024 (its HIP path: the k
// indices in no particular order, ties broken by atomics), else the
// bitonic argsort (the k largest in descending order), drawing the scratch
// PlanTopK computes.
std::expected<void, KernelFailure> Argsort(LaunchContext& launch, ggml_tensor* node);
std::expected<std::uint64_t, KernelFailure> PlanTopK(const LaunchContext& launch,
                                                     const ggml_tensor* node);
std::expected<void, KernelFailure> TopK(LaunchContext& launch, ggml_tensor* node);

// Qwen3.8's linear attention: the causal convolution (unfused) and the
// fused gated delta rule.
std::expected<void, KernelFailure> SsmConv(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> GatedDeltaNet(LaunchContext& launch, ggml_tensor* node);

// DeepSeek V4's sparse-attention indexer and hyper-connections.
std::expected<void, KernelFailure> LightningIndexer(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> HcComb(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> HcPre(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> HcPost(LaunchContext& launch, ggml_tensor* node);

// Flash attention through the tensor-core kernels (validate_ext.h
// CheckFlashAttnMma): what the launch does on the context's device. The
// kernel instance is ggml_cuda_flash_attn_ext_mma_f16_case<D, D, columns,
// 8>, columns 1, 2, 4 or 8 as upstream picks them for the query rows
// (fattn.cu:131-164); sparse is the gather of at most n_kv_max unmasked
// cells per row (D 512 and one column only); stream-k splits the cells over
// `blocks` blocks, and the fixup buffer and the mask pre-pass's or the
// sparse indices' buffer come from the pool.
struct FlashAttnMmaPlan {
  int head = 0;     // D
  int columns = 0;  // ncols1
  bool sparse = false;
  bool mask_prepass = false;
  int blocks = 0;
  std::uint64_t scratch = 0;
};
std::expected<FlashAttnMmaPlan, KernelFailure> PlanFlashAttnMma(const LaunchContext& launch,
                                                                const ggml_tensor* node);
std::expected<void, KernelFailure> FlashAttnMma(LaunchContext& launch, ggml_tensor* node);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_OPS_EXT_H_
