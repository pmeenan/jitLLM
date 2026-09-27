// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML-derived implementations' operand checks (kernels/ggml/
// validate.h), in every profile and without a GPU: the dense Qwen2 shapes
// pass, and malformed shapes, boundary sizes, misalignment, stale views
// and aliasing are refused. Nothing here touches the addresses bound.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "ggml.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"

namespace {

using jitllm::kernels::ggml::CheckBinary;
using jitllm::kernels::ggml::CheckMulMat;
using jitllm::kernels::ggml::CheckMulMatF;
using jitllm::kernels::ggml::CheckRmsNorm;
using jitllm::kernels::ggml::CheckRmsNormMul;
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
  static void Rejected(const std::expected<void, jitllm::kernels::ggml::KernelFailure>& checked) {
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

TEST_F(GgmlValidateTest, AViewThatNoLongerFollowsItsSourceIsRefused) {
  ggml_tensor* source = F32(kWidth);
  ggml_tensor* view = ggml_view_1d(context(), source, kWidth, 0);
  EXPECT_TRUE(CheckRmsNorm(Bound(ggml_rms_norm(context(), view, 0.0f))).has_value());
  TensorArena::Bind(source, kBase + (next_++ * kSlot));
  Rejected(CheckRmsNorm(Bound(ggml_rms_norm(context(), view, 0.0f))));
}

}  // namespace
