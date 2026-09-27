// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML-derived implementations' operand checks (kernels/ggml/
// validate.h), in every profile and without a GPU: the dense Qwen2 shapes
// pass, and malformed shapes, boundary sizes, misalignment, stale views
// and aliasing are refused; the cuBLAS path's plans follow upstream's
// launcher and draw the scratch GGML's pool recorded in P0. Nothing here
// touches the addresses bound.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <utility>

#include "ggml.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"

namespace {

using jitllm::kernels::ggml::CheckBinary;
using jitllm::kernels::ggml::CheckClearOf;
using jitllm::kernels::ggml::CheckMulMat;
using jitllm::kernels::ggml::CheckMulMatCublas;
using jitllm::kernels::ggml::CheckMulMatF;
using jitllm::kernels::ggml::CheckRmsNorm;
using jitllm::kernels::ggml::CheckRmsNormMul;
using jitllm::kernels::ggml::CheckRmsNormThenMul;
using jitllm::kernels::ggml::CublasGemm;
using jitllm::kernels::ggml::CublasOperand;
using jitllm::kernels::ggml::KernelError;
using jitllm::kernels::ggml::TensorArena;

constexpr std::int64_t kWidth = 896;
constexpr std::uint64_t kBase = 1ULL << 40;  // never dereferenced
constexpr std::uint64_t kSlot = 1ULL << 36;  // far enough apart for any operand here

class GgmlValidateTest : public ::testing::Test {
 protected:
  ggml_context* context() { return arena_.context(); }
  // A tensor bound to its own slot, clear of every other.
  ggml_tensor* Bound(ggml_tensor* tensor) {
    TensorArena::Bind(tensor, kBase + (next_++ * kSlot));
    return tensor;
  }
  ggml_tensor* F32(std::int64_t ne0, std::int64_t ne1 = 1) {
    return Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F32, ne0, ne1));
  }
  template <typename T>
  static void Rejected(const std::expected<T, jitllm::kernels::ggml::KernelFailure>& checked) {
    ASSERT_FALSE(checked.has_value());
    EXPECT_EQ(checked.error().error, KernelError::kRejected);
  }

  TensorArena arena_ = TensorArena::Create(64).value();
  std::uint64_t next_ = 0;
};

TEST_F(GgmlValidateTest, TheDenseQwen2ShapesPass) {
  ggml_tensor* x = F32(kWidth, 5);
  ggml_tensor* w = F32(kWidth);
  ggml_tensor* norm = Bound(ggml_rms_norm(context(), x, 1e-6f));
  EXPECT_TRUE(CheckRmsNorm(norm).has_value());
  ggml_tensor* scaled = Bound(ggml_mul(context(), norm, w));
  EXPECT_TRUE(CheckBinary(scaled, GGML_OP_MUL).has_value());
  ggml_tensor* fused_norm = ggml_rms_norm(context(), x, 1e-6f);  // never written
  EXPECT_TRUE(CheckRmsNormMul(fused_norm, Bound(ggml_mul(context(), fused_norm, w))).has_value());
  EXPECT_TRUE(CheckBinary(Bound(ggml_add(context(), x, scaled)), GGML_OP_ADD).has_value());
  EXPECT_TRUE(CheckBinary(Bound(ggml_add(context(), x, w)), GGML_OP_ADD).has_value());
  ggml_tensor* in_place = ggml_add_inplace(context(), x, scaled);  // exactly x
  EXPECT_TRUE(CheckBinary(in_place, GGML_OP_ADD).has_value());

  ggml_tensor* m = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, kWidth, 4864));
  ggml_tensor* product = Bound(ggml_mul_mat(context(), m, scaled));
  EXPECT_TRUE(CheckMulMat(product).has_value());
  EXPECT_TRUE(CheckMulMatF(product).has_value());
  ggml_tensor* column = ggml_view_2d(context(), scaled, kWidth, 1, scaled->nb[1], 0);
  EXPECT_TRUE(CheckMulMat(Bound(ggml_mul_mat(context(), m, column))).has_value());
  ggml_tensor* wide = F32(kWidth, 17);
  Rejected(CheckMulMatF(Bound(ggml_mul_mat(context(), m, wide))));  // MMF takes 16
  EXPECT_TRUE(CheckMulMat(Bound(ggml_mul_mat(context(), m, wide))).has_value());
}

// GGML's row kernel steps its column index by up to 1,024 past the row's
// last element in 32-bit arithmetic.
TEST_F(GgmlValidateTest, RmsNormBoundsIncludeTheKernelsLoopStep) {
  constexpr std::int64_t kLargest = std::numeric_limits<std::int32_t>::max() - 1024;
  ggml_tensor* fits = Bound(ggml_new_tensor_1d(context(), GGML_TYPE_F32, kLargest));
  EXPECT_TRUE(CheckRmsNorm(Bound(ggml_rms_norm(context(), fits, 0.0f))).has_value());
  ggml_tensor* over = Bound(ggml_new_tensor_1d(context(), GGML_TYPE_F32, kLargest + 1));
  Rejected(CheckRmsNorm(Bound(ggml_rms_norm(context(), over, 0.0f))));
  ggml_tensor* weight = Bound(ggml_new_tensor_1d(context(), GGML_TYPE_F32, kLargest + 1));
  ggml_tensor* fused_norm = ggml_rms_norm(context(), over, 0.0f);
  Rejected(CheckRmsNormMul(fused_norm, Bound(ggml_mul(context(), fused_norm, weight))));
}

TEST_F(GgmlValidateTest, EmptyMisalignedUnpackedAndWrappedOperandsAreRefused) {
  ggml_tensor* empty = F32(0);
  Rejected(CheckBinary(Bound(ggml_add(context(), empty, empty)), GGML_OP_ADD));

  ggml_tensor* odd = ggml_new_tensor_1d(context(), GGML_TYPE_F32, kWidth);
  TensorArena::Bind(odd, kBase + (next_++ * kSlot) + 2);
  Rejected(CheckRmsNorm(Bound(ggml_rms_norm(context(), odd, 0.0f))));

  // A dimension of one with an unpacked stride.
  ggml_tensor* big = F32(4096);
  ggml_tensor* loose = ggml_view_4d(context(), big, 4, 1, 2, 1, 8, 16, 32, 0);
  Rejected(CheckBinary(Bound(ggml_add(context(), loose, F32(4))), GGML_OP_ADD));

  // Strides that wrap, which ggml_nbytes sums to a few hundred bytes.
  ggml_tensor* small = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 64, 16));
  ggml_tensor* wrapped =
      ggml_view_4d(context(), big, 64, 1, 2, 2, 256, (~std::size_t{0} - (std::size_t{1} << 33)) + 1,
                   (std::size_t{1} << 33) + 256, 0);
  Rejected(CheckMulMat(Bound(ggml_mul_mat(context(), small, wrapped))));

  // An odd channel stride under paired loads.
  ggml_tensor* m = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, kWidth, 1024));
  ggml_tensor* channels = ggml_view_3d(context(), big, kWidth, 1, 2, kWidth * sizeof(float),
                                       (kWidth + 1) * sizeof(float), 0);
  Rejected(CheckMulMat(Bound(ggml_mul_mat(context(), m, channels))));
}

TEST_F(GgmlValidateTest, ShapesMustFollowFromTheOperands) {
  ggml_tensor* x = F32(kWidth, 2);
  ggml_tensor* m = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, kWidth, 1024));
  ggml_tensor* product = Bound(ggml_mul_mat(context(), m, x));
  product->ne[0] = 512;  // edited after GGML built it
  Rejected(CheckMulMat(product));
  ggml_tensor* sum = Bound(ggml_add(context(), x, x));
  sum->src[1] = F32(3);  // no longer broadcasts
  Rejected(CheckBinary(sum, GGML_OP_ADD));
  Rejected(CheckBinary(sum, GGML_OP_MUL));  // not the node's operation
}

TEST_F(GgmlValidateTest, OutputsMayAliasAnInputOnlyExactlyInPlace) {
  ggml_tensor* x = F32(kWidth, 2);
  // One row into its input.
  ggml_tensor* shifted = ggml_rms_norm(context(), x, 0.0f);
  TensorArena::Bind(shifted, reinterpret_cast<std::uintptr_t>(x->data) + x->nb[1]);
  Rejected(CheckRmsNorm(shifted));
  ggml_tensor* in_place = ggml_rms_norm_inplace(context(), x, 0.0f);
  EXPECT_TRUE(CheckRmsNorm(in_place).has_value());
  // Into a transposed view of an input.
  ggml_tensor* square = F32(4, 4);
  Rejected(CheckBinary(ggml_add_inplace(context(), ggml_transpose(context(), square), F32(4, 4)),
                       GGML_OP_ADD));
  // A matrix product never in place.
  ggml_tensor* m = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F32, kWidth, kWidth));
  ggml_tensor* product = ggml_mul_mat(context(), m, x);
  TensorArena::Bind(product, reinterpret_cast<std::uintptr_t>(x->data));
  Rejected(CheckMulMat(product));
}

TEST_F(GgmlValidateTest, TheFusedNormNeverReadsWhatItDoesNotWrite) {
  ggml_tensor* x = F32(kWidth, 2);
  ggml_tensor* norm = Bound(ggml_rms_norm(context(), x, 0.0f));
  Rejected(CheckRmsNormMul(norm, Bound(ggml_mul(context(), norm, norm))));
  // An in-place norm is a view of x: scaling it by x reads the norm's bytes.
  ggml_tensor* in_place = ggml_rms_norm_inplace(context(), x, 0.0f);
  Rejected(CheckRmsNormMul(in_place, Bound(ggml_mul(context(), in_place, x))));
}

// The unfused implementation writes the norm, so the norm must be memory of
// its own: it may not alter the input or the weight, which the fused
// implementation leaves alone.
TEST_F(GgmlValidateTest, TheUnfusedNormWritesOnlyItsOwnIntermediate) {
  ggml_tensor* x = F32(kWidth, 5);
  ggml_tensor* w = F32(kWidth);
  ggml_tensor* norm = Bound(ggml_rms_norm(context(), x, 1e-6f));
  ggml_tensor* scaled = Bound(ggml_mul(context(), norm, w));
  EXPECT_TRUE(CheckRmsNormThenMul(norm, scaled).has_value());
  // The same nodes pass the fused implementation's check: one plan's
  // operands serve either implementation.
  EXPECT_TRUE(CheckRmsNormMul(norm, scaled).has_value());
  // The mul in place over the norm is still the norm's own memory.
  EXPECT_TRUE(CheckRmsNormThenMul(norm, ggml_mul_inplace(context(), norm, w)).has_value());

  // Not this norm's mul, or no norm at all.
  ggml_tensor* other = Bound(ggml_rms_norm(context(), x, 1e-6f));
  Rejected(CheckRmsNormThenMul(other, scaled));
  Rejected(CheckRmsNormThenMul(norm, Bound(ggml_add(context(), norm, w))));
  Rejected(CheckRmsNormThenMul(nullptr, scaled));
  Rejected(CheckRmsNormThenMul(norm, nullptr));
  // Unbound: the unfused norm needs memory.
  ggml_tensor* unbound = ggml_rms_norm(context(), x, 1e-6f);
  Rejected(CheckRmsNormThenMul(unbound, Bound(ggml_mul(context(), unbound, w))));
  // In place over its input: the fused implementation leaves x alone.
  ggml_tensor* in_place = ggml_rms_norm_inplace(context(), x, 1e-6f);
  Rejected(CheckRmsNormThenMul(in_place, Bound(ggml_mul(context(), in_place, w))));
  // Over the weight, which the mul would then read.
  ggml_tensor* wide = F32(kWidth, 5);
  ggml_tensor* weight_view = ggml_view_1d(context(), wide, kWidth, 0);
  ggml_tensor* over_weight = ggml_rms_norm(context(), x, 1e-6f);
  TensorArena::Bind(over_weight, reinterpret_cast<std::uintptr_t>(wide->data));
  Rejected(CheckRmsNormThenMul(over_weight, Bound(ggml_mul(context(), over_weight, weight_view))));
}

TEST_F(GgmlValidateTest, AViewThatNoLongerFollowsItsSourceIsRefused) {
  ggml_tensor* source = F32(kWidth);
  ggml_tensor* view = ggml_view_1d(context(), source, kWidth, 0);
  EXPECT_TRUE(CheckRmsNorm(Bound(ggml_rms_norm(context(), view, 0.0f))).has_value());
  TensorArena::Bind(source, kBase + (next_++ * kSlot));
  Rejected(CheckRmsNorm(Bound(ggml_rms_norm(context(), view, 0.0f))));
}

// GGML's pool peaks for the output head in the FP16 bridge, recorded in
// P0 (docs/experiments/backend-proof-p0/README.md): the F16 copy of the
// input and the F16 output temporary, at 17, 32 and 512 rows.
TEST_F(GgmlValidateTest, TheOutputHeadPlanDrawsTheRecordedPoolPeaks) {
  ggml_tensor* head = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, kWidth, 151936));
  for (const auto& [rows, peak] : {std::pair<std::int64_t, std::uint64_t>{17, 5'196'288},
                                   std::pair<std::int64_t, std::uint64_t>{32, 9'781'248},
                                   std::pair<std::int64_t, std::uint64_t>{512, 156'499'968}}) {
    ggml_tensor* logits = Bound(ggml_mul_mat(context(), head, F32(kWidth, rows)));
    const auto plan = CheckMulMatCublas(logits, GGML_TYPE_F16, /*f32_output=*/false);
    ASSERT_TRUE(plan.has_value()) << plan.error().detail;
    EXPECT_EQ(plan->gemm, CublasGemm::kGemmEx);
    EXPECT_EQ(plan->weights, CublasOperand::kDirect);
    EXPECT_EQ(plan->input, CublasOperand::kConverted);
    EXPECT_EQ(plan->s01, kWidth);
    EXPECT_EQ(plan->s11, kWidth);
    EXPECT_EQ(plan->scratch, peak) << rows;
    EXPECT_EQ(plan->alignment, 256U);
    // An F32 output (upstream's choice on Volta) needs no temporary.
    EXPECT_EQ(CheckMulMatCublas(logits, GGML_TYPE_F16, true)->scratch,
              static_cast<std::uint64_t>(kWidth * rows * 2));
  }
}

// Attention without flash attention, as llama.cpp builds it: K and V views
// of an F16 cache, grouped over 14 query heads by 2 KV heads.
TEST_F(GgmlValidateTest, AttentionPlansFollowTheLauncher) {
  constexpr std::int64_t kHead = 64;
  constexpr std::int64_t kCells = 256;
  constexpr std::int64_t kTokens = 32;
  ggml_tensor* k_cache = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 2 * kHead, 512));
  ggml_tensor* k =
      ggml_permute(context(),
                   ggml_view_3d(context(), k_cache, kHead, 2, kCells,
                                ggml_row_size(GGML_TYPE_F16, kHead), k_cache->nb[1], 0),
                   0, 2, 1, 3);
  ggml_tensor* q = ggml_permute(
      context(), Bound(ggml_new_tensor_3d(context(), GGML_TYPE_F32, kHead, 14, kTokens)), 0, 2, 1,
      3);
  ggml_tensor* kq = ggml_mul_mat(context(), k, q);
  ASSERT_TRUE(ggml_prec_set_acc(kq, GGML_PREC_F32));
  Bound(kq);
  // F32 compute: the K view, its bytes exactly its elements, converted in
  // place order; grouping needs pointer arrays.
  auto plan = CheckMulMatCublas(kq, GGML_TYPE_F32, false);
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  EXPECT_EQ(plan->gemm, CublasGemm::kGemmBatchedEx);
  EXPECT_EQ(plan->weights, CublasOperand::kConverted);
  EXPECT_EQ(plan->input, CublasOperand::kDirect);
  EXPECT_EQ(plan->s01, 2 * kHead);  // a cell's row of both heads
  constexpr std::uint64_t kKeys = kHead * kCells * 2 * sizeof(float);
  EXPECT_EQ(plan->scratch, kKeys + 256 + (std::uint64_t{14} * 8));  // then 2 × 14 and 14 pointers

  // A view of one head of three has gaps: gathered into packed rows.
  ggml_tensor* wide_cache = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 3 * kHead, 512));
  ggml_tensor* gapped =
      ggml_permute(context(),
                   ggml_view_3d(context(), wide_cache, kHead, 2, kCells,
                                ggml_row_size(GGML_TYPE_F16, kHead), wide_cache->nb[1], 0),
                   0, 2, 1, 3);
  ggml_tensor* kq_gapped = ggml_mul_mat(context(), gapped, q);
  ASSERT_TRUE(ggml_prec_set_acc(kq_gapped, GGML_PREC_F32));
  plan = CheckMulMatCublas(Bound(kq_gapped), GGML_TYPE_F32, false);
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  EXPECT_EQ(plan->weights, CublasOperand::kPacked);
  EXPECT_EQ(plan->s01, kHead);
  EXPECT_EQ(plan->s02, kHead * kCells);

  // KQV: the transposed V view read in place, the scores converted to F16,
  // and an F16 output temporary.
  ggml_tensor* v_cache = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 512, 2 * kHead));
  ggml_tensor* v =
      ggml_view_3d(context(), v_cache, kCells, kHead, 2, v_cache->nb[1], v_cache->nb[1] * kHead, 0);
  plan = CheckMulMatCublas(Bound(ggml_mul_mat(context(), v, kq)), GGML_TYPE_F16, false);
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  EXPECT_EQ(plan->gemm, CublasGemm::kGemmBatchedEx);
  EXPECT_EQ(plan->weights, CublasOperand::kDirect);
  EXPECT_EQ(plan->input, CublasOperand::kConverted);
  EXPECT_EQ(plan->s01, 512);
  // Its F16 output rows are 64 elements, 128 bytes apart.
  EXPECT_EQ(plan->alignment, 128U);

  // Without grouping, packed operands take the strided entry point.
  ggml_tensor* keys = Bound(ggml_new_tensor_3d(context(), GGML_TYPE_F16, kHead, kCells, 2));
  ggml_tensor* rows = Bound(ggml_new_tensor_3d(context(), GGML_TYPE_F32, kHead, kTokens, 2));
  plan = CheckMulMatCublas(Bound(ggml_mul_mat(context(), keys, rows)), GGML_TYPE_F16, false);
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  EXPECT_EQ(plan->gemm, CublasGemm::kGemmStridedBatchedEx);
  // And one F32 matrix, Sgemm.
  ggml_tensor* square = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F32, kHead, kHead));
  plan = CheckMulMatCublas(Bound(ggml_mul_mat(context(), square, F32(kHead, kTokens))),
                           GGML_TYPE_F32, false);
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  EXPECT_EQ(plan->gemm, CublasGemm::kSgemm);
  EXPECT_EQ(plan->scratch, 0U);
}

TEST_F(GgmlValidateTest, OperandsMustClearAWorkspace) {
  ggml_tensor* m = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 128, 64));
  ggml_tensor* x = F32(128, 32);
  ggml_tensor* y = Bound(ggml_mul_mat(context(), m, x));
  const auto at = [](const ggml_tensor* tensor) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(tensor->data));
  };
  EXPECT_TRUE(CheckClearOf(y, at(y) + ggml_nbytes(y), 4096).has_value());  // just past
  EXPECT_TRUE(CheckClearOf(y, at(m) - 4096, 4096).has_value());            // just before
  EXPECT_TRUE(CheckClearOf(y, at(m), 0).has_value());                      // no workspace
  for (const ggml_tensor* tensor : {m, x, y}) {
    Rejected(CheckClearOf(y, at(tensor) + ggml_nbytes(tensor) - 1, 4096));  // the last byte
    Rejected(CheckClearOf(y, at(tensor) - 4095, 4096));                     // the first byte
  }
}

TEST_F(GgmlValidateTest, WhatCublasWouldRefuseIsRefused) {
  ggml_tensor* m = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 128, 64));
  ggml_tensor* x = F32(128, 32);
  // Weight rows closer than k elements: cuBLAS needs lda >= k.
  ggml_tensor* overlapping = ggml_view_2d(context(), m, 128, 32, 64 * sizeof(ggml_fp16_t), 0);
  Rejected(CheckMulMatCublas(Bound(ggml_mul_mat(context(), overlapping, x)), GGML_TYPE_F16, false));
  // An output that is a strided view.
  ggml_tensor* big = F32(128, 64);
  ggml_tensor* product = ggml_mul_mat(context(), m, x);
  ggml_tensor* out = ggml_view_2d(context(), big, 64, 32, big->nb[1], 0);
  TensorArena::Bind(product, reinterpret_cast<std::uintptr_t>(out->data));
  product->nb[1] = out->nb[1];
  product->nb[2] = out->nb[1] * 32;
  product->nb[3] = product->nb[2];
  Rejected(CheckMulMatCublas(product, GGML_TYPE_F16, false));
  // A compute type cuBLAS does not take here, and a routing hint.
  ggml_tensor* plain = Bound(ggml_mul_mat(context(), m, x));
  EXPECT_TRUE(CheckMulMatCublas(plain, GGML_TYPE_F16, false).has_value());
  Rejected(CheckMulMatCublas(plain, GGML_TYPE_Q8_0, false));
  plain->op_params[1] = GGML_HINT_SRC0_IS_HADAMARD;
  Rejected(CheckMulMat(plain));
}

}  // namespace
