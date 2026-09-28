// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Greedy and seeded sampling (execution/sampling.h).

#include "execution/sampling.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

#include "expected_error.h"

namespace {

namespace ex = jitllm::execution;
using jitllm::test_support::Failed;

// Random123's published known-answer vectors for Philox4x32-10 (its
// tests/kat_vectors).
TEST(Philox, KnownAnswers) {
  EXPECT_EQ(ex::Philox4x32({0, 0, 0, 0}, {0, 0}),
            (std::array<std::uint32_t, 4>{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}));
  EXPECT_EQ(
      ex::Philox4x32({0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}, {0xffffffff, 0xffffffff}),
      (std::array<std::uint32_t, 4>{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}));
  EXPECT_EQ(
      ex::Philox4x32({0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}, {0xa4093822, 0x299f31d0}),
      (std::array<std::uint32_t, 4>{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}));
}

TEST(Uniform, InRangeAndKeyedByEveryField) {
  const double u = ex::UniformAt({1, 2, 3});
  EXPECT_GE(u, 0.0);
  EXPECT_LT(u, 1.0);
  EXPECT_EQ(u, ex::UniformAt({1, 2, 3}));
  EXPECT_NE(u, ex::UniformAt({2, 2, 3}));
  EXPECT_NE(u, ex::UniformAt({1, 3, 3}));
  EXPECT_NE(u, ex::UniformAt({1, 2, 4}));
  double sum = 0;
  for (std::uint64_t p = 0; p < 100000; ++p) {
    sum += ex::UniformAt({7, 0, p});
  }
  EXPECT_NEAR(sum / 100000, 0.5, 0.01);
}

TEST(Greedy, TakesTheLowestIdAmongEqualMaxima) {
  const std::vector<float> logits = {1.0F, 3.0F, -2.0F, 3.0F};
  EXPECT_EQ(ex::Greedy(logits), 1);
  const std::vector<float> masked = {-INFINITY, -INFINITY, 0.5F};
  EXPECT_EQ(ex::Greedy(masked), 2);
}

TEST(Greedy, RefusesBadLogits) {
  EXPECT_EQ(Failed(ex::Greedy(std::vector<float>{})), ex::SamplingError::kNoLogits);
  EXPECT_EQ(Failed(ex::Greedy(std::vector<float>{-INFINITY, -INFINITY})),
            ex::SamplingError::kNoLogits);
  EXPECT_EQ(Failed(ex::Greedy(std::vector<float>{1.0F, NAN})), ex::SamplingError::kInvalidLogits);
  EXPECT_EQ(Failed(ex::Greedy(std::vector<float>{1.0F, INFINITY})),
            ex::SamplingError::kInvalidLogits);
}

TEST(Sample, TemperatureZeroIsGreedy) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {0.1F, 0.9F, 0.3F};
  for (std::uint64_t p = 0; p < 20; ++p) {
    EXPECT_EQ(ex::Sample(logits, {.temperature = 0}, {5, 0, p}, scratch), 1);
  }
}

TEST(Sample, IsReproducibleFromItsKey) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {1.0F, 1.2F, 0.8F, 1.1F, 0.0F};
  for (std::uint64_t p = 0; p < 50; ++p) {
    const auto a = ex::Sample(logits, {}, {9, 1, p}, scratch);
    const auto b = ex::Sample(logits, {}, {9, 1, p}, scratch);
    ASSERT_TRUE(a.has_value());
    EXPECT_EQ(a, b);
  }
}

TEST(Sample, FollowsTheSoftmax) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {std::log(0.5F), std::log(0.3F), std::log(0.2F)};
  std::map<int, int> counts;
  constexpr int kDraws = 60000;
  for (int p = 0; p < kDraws; ++p) {
    ++counts[*ex::Sample(logits, {}, {3, 0, static_cast<std::uint64_t>(p)}, scratch)];
  }
  EXPECT_NEAR(counts[0] / double{kDraws}, 0.5, 0.01);
  EXPECT_NEAR(counts[1] / double{kDraws}, 0.3, 0.01);
  EXPECT_NEAR(counts[2] / double{kDraws}, 0.2, 0.01);
}

TEST(Sample, FiltersAsDocumented) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {std::log(0.5F), std::log(0.3F), std::log(0.15F),
                                     std::log(0.05F)};
  auto seen = [&](ex::SamplingParams params) {
    std::map<int, int> counts;
    for (std::uint64_t p = 0; p < 4000; ++p) {
      ++counts[*ex::Sample(logits, params, {11, 0, p}, scratch)];
    }
    return counts;
  };
  EXPECT_EQ(seen({.top_k = 1}).size(), 1U);
  EXPECT_EQ(seen({.top_k = 2}).size(), 2U);
  EXPECT_EQ(seen({.top_p = 0.45F}).size(), 1U);  // 0.5 reaches 0.45
  EXPECT_EQ(seen({.top_p = 0.79F}).size(), 2U);  // 0.5 + 0.3
  EXPECT_EQ(seen({.min_p = 0.5F}).size(), 2U);   // 0.3 >= 0.25, 0.15 is not
  EXPECT_EQ(seen({.min_p = 0.05F}).size(), 4U);
}

TEST(Sample, TiesOrderByIdNotBySortStability) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits(1000, 0.0F);
  const auto a = ex::Sample(logits, {.top_k = 10}, {1, 0, 0}, scratch);
  ASSERT_TRUE(a.has_value());
  EXPECT_LT(*a, 10);  // top_k keeps the ten lowest IDs among equals
}

TEST(Sample, RefusesBadParameters) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {0.0F, 1.0F};
  EXPECT_EQ(Failed(ex::Sample(logits, {.temperature = -1}, {}, scratch)),
            ex::SamplingError::kInvalidParams);
  EXPECT_EQ(Failed(ex::Sample(logits, {.temperature = NAN}, {}, scratch)),
            ex::SamplingError::kInvalidParams);
  EXPECT_EQ(Failed(ex::Sample(logits, {.top_p = 0}, {}, scratch)),
            ex::SamplingError::kInvalidParams);
  EXPECT_EQ(Failed(ex::Sample(logits, {.top_p = 1.5F}, {}, scratch)),
            ex::SamplingError::kInvalidParams);
  EXPECT_EQ(Failed(ex::Sample(logits, {.min_p = -0.1F}, {}, scratch)),
            ex::SamplingError::kInvalidParams);
  EXPECT_EQ(Failed(ex::Sample(std::vector<float>{NAN}, {}, {}, scratch)),
            ex::SamplingError::kInvalidLogits);
}

// Speculative sampling with a greedy drafter: the verdict's token is
// distributed as Sample's, the draft accepted with its probability.
TEST(VerifyDraft, PreservesTheDistribution) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {2.0F, 1.0F, 0.0F, -1.0F, 0.5F};
  std::vector<double> p(logits.size());
  double total = 0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    p[i] = std::exp(static_cast<double>(logits[i]));
    total += p[i];
  }
  for (double& x : p) {
    x /= total;
  }
  constexpr std::uint64_t kDraws = 200000;
  for (const std::int32_t draft : {0, 1, 3}) {
    std::vector<double> seen(logits.size(), 0.0);
    std::uint64_t accepted = 0;
    for (std::uint64_t s = 0; s < kDraws; ++s) {
      auto v = ex::VerifyDraft(logits, draft, {}, {.seed = s, .stream = 0, .position = 9}, scratch);
      ASSERT_TRUE(v.has_value());
      seen[static_cast<std::size_t>(v->token)] += 1.0;
      accepted += v->accepted ? 1 : 0;
      // A rejected draft is never the token.
      EXPECT_TRUE(v->accepted || v->token != draft);
    }
    EXPECT_NEAR(static_cast<double>(accepted) / kDraws, p[static_cast<std::size_t>(draft)], 0.01)
        << draft;
    for (std::size_t i = 0; i < p.size(); ++i) {
      EXPECT_NEAR(seen[i] / kDraws, p[i], 0.01) << draft << " " << i;
    }
  }
  // Reproducible from the key.
  const auto a = ex::VerifyDraft(logits, 1, {}, {.seed = 3, .stream = 1, .position = 2}, scratch);
  const auto b = ex::VerifyDraft(logits, 1, {}, {.seed = 3, .stream = 1, .position = 2}, scratch);
  ASSERT_TRUE(a && b);
  EXPECT_EQ(a->accepted, b->accepted);
  EXPECT_EQ(a->token, b->token);
}

TEST(VerifyDraft, GreedyAndFilteredDrafts) {
  std::vector<ex::SamplingCandidate> scratch;
  const std::vector<float> logits = {0.0F, 3.0F, 1.0F};
  // Temperature 0: accepted exactly when the draft is the greedy token.
  auto yes = ex::VerifyDraft(logits, 1, {.temperature = 0}, {}, scratch);
  auto no = ex::VerifyDraft(logits, 2, {.temperature = 0}, {}, scratch);
  ASSERT_TRUE(yes && no);
  EXPECT_TRUE(yes->accepted);
  EXPECT_EQ(yes->token, 1);
  EXPECT_FALSE(no->accepted);
  EXPECT_EQ(no->token, 1);
  // A draft top-k filters out has probability 0: always rejected, and the
  // replacement is from what top-k keeps.
  for (std::uint64_t s = 0; s < 1000; ++s) {
    auto v =
        ex::VerifyDraft(logits, 0, {.top_k = 2}, {.seed = s, .stream = 0, .position = 0}, scratch);
    ASSERT_TRUE(v.has_value());
    EXPECT_FALSE(v->accepted);
    EXPECT_NE(v->token, 0);
  }
  // A draft holding all the weight is always accepted.
  auto only =
      ex::VerifyDraft(logits, 1, {.top_k = 1}, {.seed = 5, .stream = 0, .position = 0}, scratch);
  ASSERT_TRUE(only.has_value());
  EXPECT_TRUE(only->accepted);
  EXPECT_EQ(Failed(ex::VerifyDraft(logits, 1, {.top_p = 0}, {}, scratch)),
            ex::SamplingError::kInvalidParams);
}

}  // namespace
