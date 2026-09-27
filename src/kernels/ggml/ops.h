// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// GGML-derived operation implementations over the K-C launch context
// (D-053; docs/backend-proof.md#dispatch-and-implementations-d-053). Each
// takes operation nodes built with GGML's graph functions (tensors.h) and
// bound to jitLLM memory, checks every precondition its GGML launcher
// asserts, so that an unsupported operand is a rejection and never an
// abort, and queues the launcher's kernels on the context's stream.
// GGML's graph functions assert their own shape rules when they build a
// node, so nodes are built only from shapes a plan has validated. The
// operand checks are in validate.h, which every profile builds.
//
// Matrix multiplication comes as separate implementations, one per GGML
// kernel family, since the plan, not GGML's routing, selects among them.
// Each accepts only operands that upstream's selection would route to it,
// which is where upstream validated it.

#ifndef JITLLM_KERNELS_GGML_OPS_H_
#define JITLLM_KERNELS_GGML_OPS_H_

#include <expected>

#include "ggml.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

// A ggml_rms_norm node over F32 rows.
std::expected<void, KernelFailure> RmsNorm(LaunchContext& launch, ggml_tensor* norm);

// A ggml_rms_norm node and the ggml_mul that scales it, as GGML's fused
// launcher: one kernel writes the product to `mul`, and `norm` is never
// written. Fusion is the plan's choice (D-053).
std::expected<void, KernelFailure> RmsNormMul(LaunchContext& launch, ggml_tensor* norm,
                                              ggml_tensor* mul);

// ggml_add and ggml_mul nodes with broadcasting, all F32.
std::expected<void, KernelFailure> Add(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Mul(LaunchContext& launch, ggml_tensor* node);

// A ggml_mul_mat node, F32 activations and output: GGML's vector kernel
// (MMVF) and its tensor-core kernel for up to 16 columns (MMF).
std::expected<void, KernelFailure> MulMatVecF(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> MulMatF(LaunchContext& launch, ggml_tensor* node);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_OPS_H_
