// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "execution/sampling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

namespace jitllm::execution {

std::string_view SamplingErrorName(SamplingError e) {
  switch (e) {
    case SamplingError::kNoLogits:
      return "no-logits";
    case SamplingError::kInvalidLogits:
      return "invalid-logits";
    case SamplingError::kInvalidParams:
      return "invalid-params";
  }
  return "unknown";
}

std::array<std::uint32_t, 4> Philox4x32(std::array<std::uint32_t, 4> c,
                                        std::array<std::uint32_t, 2> k) {
  constexpr std::uint64_t kM0 = 0xD2511F53U;
  constexpr std::uint64_t kM1 = 0xCD9E8D57U;
  constexpr std::uint32_t kW0 = 0x9E3779B9U;
  constexpr std::uint32_t kW1 = 0xBB67AE85U;
  for (int round = 0; round < 10; ++round) {
    const std::uint64_t p0 = kM0 * c[0];
    const std::uint64_t p1 = kM1 * c[2];
    const auto hi0 = static_cast<std::uint32_t>(p0 >> 32U);
    const auto lo0 = static_cast<std::uint32_t>(p0);
    const auto hi1 = static_cast<std::uint32_t>(p1 >> 32U);
    const auto lo1 = static_cast<std::uint32_t>(p1);
    c = {hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0};
    k[0] += kW0;
    k[1] += kW1;
  }
  return c;
}

double UniformAt(const SamplingKey& key) {
  const std::array<std::uint32_t, 4> counter = {
      static_cast<std::uint32_t>(key.position), static_cast<std::uint32_t>(key.position >> 32U),
      static_cast<std::uint32_t>(key.stream), static_cast<std::uint32_t>(key.stream >> 32U)};
  const std::array<std::uint32_t, 2> k = {static_cast<std::uint32_t>(key.seed),
                                          static_cast<std::uint32_t>(key.seed >> 32U)};
  const auto r = Philox4x32(counter, k);
  const std::uint64_t bits = (static_cast<std::uint64_t>(r[0] >> 5U) << 26U) | (r[1] >> 6U);
  return static_cast<double>(bits) * 0x1.0p-53;
}

namespace {

std::expected<void, SamplingError> CheckLogits(std::span<const float> logits) {
  if (logits.empty()) {
    return std::unexpected(SamplingError::kNoLogits);
  }
  bool finite = false;
  for (const float x : logits) {
    if (std::isnan(x) || (std::isinf(x) && x > 0)) {
      return std::unexpected(SamplingError::kInvalidLogits);
    }
    finite = finite || std::isfinite(x);
  }
  if (!finite) {
    return std::unexpected(SamplingError::kNoLogits);
  }
  return {};
}

}  // namespace

std::expected<std::int32_t, SamplingError> Greedy(std::span<const float> logits) {
  if (auto c = CheckLogits(logits); !c) {
    return std::unexpected(c.error());
  }
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) {
      best = i;
    }
  }
  return static_cast<std::int32_t>(best);
}

std::expected<std::int32_t, SamplingError> Sample(std::span<const float> logits,
                                                  const SamplingParams& p, const SamplingKey& key,
                                                  std::vector<SamplingCandidate>& scratch) {
  // NaN fails every comparison, so each range is checked for it too.
  if (!std::isfinite(p.temperature) || p.temperature < 0 || std::isnan(p.top_p) || p.top_p <= 0 ||
      p.top_p > 1 || std::isnan(p.min_p) || p.min_p < 0 || p.min_p > 1) {
    return std::unexpected(SamplingError::kInvalidParams);
  }
  if (p.temperature == 0) {
    return Greedy(logits);
  }
  if (auto c = CheckLogits(logits); !c) {
    return std::unexpected(c.error());
  }
  scratch.clear();
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (std::isfinite(logits[i])) {
      scratch.push_back(
          {static_cast<double>(logits[i]) / p.temperature, static_cast<std::int32_t>(i)});
    }
  }
  auto before = [](const SamplingCandidate& a, const SamplingCandidate& b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
  };
  std::size_t keep = scratch.size();
  if (p.top_k != 0 && p.top_k < keep) {
    keep = p.top_k;
    std::nth_element(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(keep - 1),
                     scratch.end(), before);
    scratch.resize(keep);
  }
  std::ranges::sort(scratch, before);
  // Softmax, as unnormalized weights relative to the largest.
  const double top = scratch[0].value;
  double total = 0;
  for (auto& c : scratch) {
    c.value = std::exp(c.value - top);
    total += c.value;
  }
  // min-p: drop what is less likely than min_p times the most likely.
  if (p.min_p > 0) {
    const double floor = static_cast<double>(p.min_p) * scratch[0].value;
    while (keep > 1 && scratch[keep - 1].value < floor) {
      total -= scratch[keep - 1].value;
      --keep;
    }
  }
  // top-p: the shortest prefix whose probability reaches top_p.
  if (p.top_p < 1) {
    double cumulative = 0;
    const double target = static_cast<double>(p.top_p) * total;
    std::size_t n = 0;
    while (n < keep) {
      cumulative += scratch[n].value;
      ++n;
      if (cumulative >= target) {
        break;
      }
    }
    keep = n;
    total = cumulative;
  }
  const double target = UniformAt(key) * total;
  double cumulative = 0;
  for (std::size_t i = 0; i < keep; ++i) {
    cumulative += scratch[i].value;
    if (target < cumulative) {
      return scratch[i].id;
    }
  }
  return scratch[keep - 1].id;
}

}  // namespace jitllm::execution
