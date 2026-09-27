// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The locked ExLlamaV3 GEMM kernels link into jitLLM's build (every CUDA
// profile, no GPU; docs/backend-proof.md, P1): upstream's compilation units
// for the mcg codebook at K = 4, 5, 6 and 8 each define their GEMM and
// multi-GEMM tables, for FP16 and FP32 outputs, with a kernel for each of
// tile shapes 1 to 4. exl3_kernels_test.cc loads them on a GB10.

#include "exl3_tables.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using jitllm::tests::exl3::AllKernels;
using jitllm::tests::exl3::kRates;
using jitllm::tests::exl3::kShapes;
using jitllm::tests::exl3::RateTables;

TEST(Exl3KernelTablesTest, EveryRateHasAKernelPerShape) {
  static_assert(kShapes == 4);
  for (const RateTables& rate : kRates) {
    SCOPED_TRACE("K=" + std::to_string(rate.bits));
    EXPECT_EQ(rate.gemm_fp16[0], nullptr);
    EXPECT_EQ(rate.gemm_fp32[0], nullptr);
    EXPECT_EQ(rate.mgemm_fp16[0], nullptr);
    EXPECT_EQ(rate.mgemm_fp32[0], nullptr);
    for (int shape = 1; shape <= kShapes; ++shape) {
      SCOPED_TRACE("shape " + std::to_string(shape));
      EXPECT_NE(rate.gemm_fp16[shape], nullptr);
      EXPECT_NE(rate.gemm_fp32[shape], nullptr);
      EXPECT_NE(rate.mgemm_fp16[shape], nullptr);
      EXPECT_NE(rate.mgemm_fp32[shape], nullptr);
    }
  }
  // 4 rates x 4 shapes x {GEMM, multi-GEMM} x {FP16, FP32}, all distinct.
  EXPECT_EQ(AllKernels().size(), 64U);
}

}  // namespace
