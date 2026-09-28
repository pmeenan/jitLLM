// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/image/gemm.h"

#include <cublas_v2.h>

#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <string>

namespace jitllm::kernels::image {
namespace {

constexpr std::int64_t kIntMax = std::numeric_limits<int>::max();

bool Fits(std::int64_t v) { return v > 0 && v <= kIntMax; }

// Column-major C (m x n) = op(A) op(B), as cuBLAS sees row-major operands.
Status Gemm(cublasContext* handle, cublasOperation_t op_a, cublasOperation_t op_b, std::int64_t m,
            std::int64_t n, std::int64_t k, const void* a, std::int64_t lda, const void* b,
            std::int64_t ldb, void* c, cudaDataType_t c_type, std::int64_t ldc, float beta,
            const char* what) {
  if (handle == nullptr || !Fits(m) || !Fits(n) || !Fits(k) || !Fits(lda) || !Fits(ldb) ||
      !Fits(ldc)) {
    return std::unexpected(std::format("{}: sizes beyond cuBLAS's int", what));
  }
  const float alpha = 1.0f;
  const cublasStatus_t status = cublasGemmEx(
      handle, op_a, op_b, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), &alpha, a,
      CUDA_R_16BF, static_cast<int>(lda), b, CUDA_R_16BF, static_cast<int>(ldb), &beta, c, c_type,
      static_cast<int>(ldc), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return std::unexpected(
        std::format("{}: cublasGemmEx failed ({})", what, static_cast<int>(status)));
  }
  return {};
}

}  // namespace

Status Linear(cublasContext* handle, const Bf16* x, std::int64_t ldx, const Bf16* w,
              std::int64_t ldw, Bf16* out, std::int64_t ldo, std::int64_t m, std::int64_t n,
              std::int64_t k, bool accumulate) {
  if (ldx < k || ldw < k || ldo < n) {
    return std::unexpected(std::string("Linear: strides"));
  }
  return Gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, w, ldw, x, ldx, out, CUDA_R_16BF, ldo,
              accumulate ? 1.0f : 0.0f, "Linear");
}

Status LinearF32(cublasContext* handle, const Bf16* x, std::int64_t ldx, const Bf16* w,
                 std::int64_t ldw, float* out, std::int64_t ldo, std::int64_t m, std::int64_t n,
                 std::int64_t k) {
  if (ldx < k || ldw < k || ldo < n) {
    return std::unexpected(std::string("LinearF32: strides"));
  }
  return Gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, w, ldw, x, ldx, out, CUDA_R_32F, ldo, 0.0f,
              "LinearF32");
}

Status ConvProduct(cublasContext* handle, const Bf16* w, std::int64_t out_channels,
                   std::int64_t inner, const Bf16* col, std::int64_t ldc, Bf16* out,
                   std::int64_t ldo, std::int64_t pixels, bool accumulate) {
  if (ldc < pixels || ldo < pixels) {
    return std::unexpected(std::string("ConvProduct: strides"));
  }
  // Column-major, col is A (lda = ldc) and out is C (ldc = ldo).
  // NOLINTNEXTLINE(readability-suspicious-call-argument)
  return Gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, pixels, out_channels, inner, col, ldc, w, inner,
              out, CUDA_R_16BF, ldo, accumulate ? 1.0f : 0.0f, "ConvProduct");
}

Status ScoresQtK(cublasContext* handle, const Bf16* q, const Bf16* k, float* scores,
                 std::int64_t channels, std::int64_t tokens, std::int64_t ld) {
  if (ld < tokens) {
    return std::unexpected(std::string("ScoresQtK: strides"));
  }
  return Gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, tokens, tokens, channels, k, ld, q, ld, scores,
              CUDA_R_32F, tokens, 0.0f, "ScoresQtK");
}

Status ValuesTimesProbs(cublasContext* handle, const Bf16* v, const Bf16* probs, Bf16* out,
                        std::int64_t channels, std::int64_t tokens, std::int64_t ld) {
  if (ld < tokens) {
    return std::unexpected(std::string("ValuesTimesProbs: strides"));
  }
  return Gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, tokens, channels, tokens, probs, tokens, v, ld, out,
              CUDA_R_16BF, ld, 0.0f, "ValuesTimesProbs");
}

}  // namespace jitllm::kernels::image
