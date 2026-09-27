// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The operand checks of the GGML-derived implementations (ops.h), on the
// host and in every build profile, so the CPU build tests them without a
// GPU. Each refuses what its GGML launcher would assert on or index past,
// as a kRejected failure: operand types and layouts, empty tensors, extents
// computed with checked arithmetic, 32-bit indexing and grid limits,
// alignment of bases and strides, views that no longer follow their
// source, and outputs that share bytes with an input other than exactly in
// place. The elementwise and row operations take packed operands only
// (strides of a dense tensor of their shape). Upstream's kernel-family
// selection, which needs the device, stays in ops.cu.

#ifndef JITLLM_KERNELS_GGML_VALIDATE_H_
#define JITLLM_KERNELS_GGML_VALIDATE_H_

#include <expected>

#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

// A ggml_rms_norm node over packed F32 rows, writing its own output.
std::expected<void, KernelFailure> CheckRmsNorm(const ggml_tensor* norm);
// A ggml_rms_norm node and the ggml_mul that scales it, for GGML's fused
// launcher, which writes the product and never the norm.
std::expected<void, KernelFailure> CheckRmsNormMul(const ggml_tensor* norm, const ggml_tensor* mul);
// A ggml_add or ggml_mul node (op), all F32, with broadcasting.
std::expected<void, KernelFailure> CheckBinary(const ggml_tensor* node, ggml_op op);
// A ggml_mul_mat node, F16, BF16 or F32 weights and F32 activations and
// output: what both MMVF and MMF need.
std::expected<void, KernelFailure> CheckMulMat(const ggml_tensor* node);
// CheckMulMat plus MMF's own limits: at most 16 columns and paired strides.
std::expected<void, KernelFailure> CheckMulMatF(const ggml_tensor* node);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_VALIDATE_H_
