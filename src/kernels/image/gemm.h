// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// BF16 matrix products on cuBLAS for the Qwen-Image-2.1 pipeline (M3):
// BF16 operands, F32 accumulation (CUBLAS_COMPUTE_32F), BF16 or F32 out,
// through cublasGemmEx with CUBLAS_GEMM_DEFAULT_TENSOR_OP, the call PyTorch
// makes for a BF16 nn.Linear (at::cuda::blas::gemm<at::BFloat16>). The
// handle is jitLLM's (kernels/ggml/cublas.h): created on the stream the
// products queue on, with its declared workspace.

#ifndef JITLLM_KERNELS_IMAGE_GEMM_H_
#define JITLLM_KERNELS_IMAGE_GEMM_H_

#include <cstdint>

#include "kernels/image/ops.h"

struct cublasContext;

namespace jitllm::kernels::image {

// A linear layer over row-major matrices: out[m, n] = x[m, k] . w[n, k]^T
// (+ out when accumulate), x rows `ldx` apart, w rows `ldw` apart, out rows
// `ldo` apart.
Status Linear(cublasContext* handle, const Bf16* x, std::int64_t ldx, const Bf16* w,
              std::int64_t ldw, Bf16* out, std::int64_t ldo, std::int64_t m, std::int64_t n,
              std::int64_t k, bool accumulate = false);
Status LinearF32(cublasContext* handle, const Bf16* x, std::int64_t ldx, const Bf16* w,
                 std::int64_t ldw, float* out, std::int64_t ldo, std::int64_t m, std::int64_t n,
                 std::int64_t k);

// A channels-first convolution as a product: out[co, p] = w[co, kk] .
// col[kk, p] (+ out when accumulate), for `pixels` output pixels whose
// column block starts at col (rows of `ldc` elements) and output block at
// out (rows of `ldo` elements).
Status ConvProduct(cublasContext* handle, const Bf16* w, std::int64_t out_channels,
                   std::int64_t inner, const Bf16* col, std::int64_t ldc, Bf16* out,
                   std::int64_t ldo, std::int64_t pixels, bool accumulate);

// Single-head attention's two products, channels-first q, k, v [c, n]:
// scores[i, j] = sum_c q[c, i] k[c, j] (F32), and out[c, i] = sum_j
// probs[i, j] v[c, j].
Status ScoresQtK(cublasContext* handle, const Bf16* q, const Bf16* k, float* scores,
                 std::int64_t channels, std::int64_t tokens, std::int64_t ld);
Status ValuesTimesProbs(cublasContext* handle, const Bf16* v, const Bf16* probs, Bf16* out,
                        std::int64_t channels, std::int64_t tokens, std::int64_t ld);

}  // namespace jitllm::kernels::image

#endif  // JITLLM_KERNELS_IMAGE_GEMM_H_
