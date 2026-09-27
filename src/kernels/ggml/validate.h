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

#include <cstdint>
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

// GGML's cuBLAS matrix multiplication (ggml_cuda_mul_mat_cublas_impl in
// ggml-cuda.cu), as it would run for one node: what it converts into
// scratch, which cuBLAS entry point it calls with which leading dimensions,
// and the scratch it draws. The device chooses the compute type and output
// precision (ops.h); this is everything that follows from them.
enum class CublasGemm : std::uint8_t {
  kSgemm,                 // one F32 matrix
  kGemmEx,                // one matrix
  kGemmStridedBatchedEx,  // channels and samples at fixed strides, no broadcast
  kGemmBatchedEx,         // pointer arrays, built on the device into scratch
};
enum class CublasOperand : std::uint8_t {
  kDirect,     // read in place: already the compute type
  kConverted,  // converted element by element into scratch, strides kept
  kPacked,     // gathered into packed scratch
};
struct CublasMulMat {
  ggml_type compute = GGML_TYPE_F32;
  bool f32_output = false;  // cuBLAS writes F32 into the node; otherwise
                            // compute-type output goes through scratch
  CublasOperand weights = CublasOperand::kDirect;
  CublasOperand input = CublasOperand::kDirect;
  CublasGemm gemm = CublasGemm::kGemmEx;
  // Element strides as cuBLAS sees them, after any conversion.
  std::int64_t s01 = 0, s02 = 0, s03 = 0;
  std::int64_t s11 = 0, s12 = 0, s13 = 0;
  // The largest power of two, up to 256, that divides every address and
  // byte stride cuBLAS is given (scratch blocks are 256-aligned). cuBLAS
  // chooses kernels by alignment, so it is part of the executed plan. It
  // counts every stride, even those one matrix leaves unused, so two plans
  // cuBLAS runs alike may record different alignments; never the reverse.
  std::uint64_t alignment = 256;
  // The pool's high-water mark: each block from a 256-byte boundary, in
  // the launcher's allocation order.
  std::uint64_t scratch = 0;
};

// Whether no byte of the node or its two sources lies in the device range
// [base, base + size): a launcher's scratch or a library's workspace, which
// the launch writes while it reads the operands.
std::expected<void, KernelFailure> CheckClearOf(const ggml_tensor* node, std::uint64_t base,
                                                std::uint64_t size);

// CheckMulMat, then the plan for `compute` (F32, F16 or BF16) and
// `f32_output`, refusing what cuBLAS would refuse (leading dimensions below
// k, more than INT_MAX matrices) and an output that is not packed.
std::expected<CublasMulMat, KernelFailure> CheckMulMatCublas(const ggml_tensor* node,
                                                             ggml_type compute, bool f32_output);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_VALIDATE_H_
