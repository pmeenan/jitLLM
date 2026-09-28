// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// What the DeepSeek V4 harnesses share beyond the engine's planning
// (engine/dsv4_plan.h, under the harnesses' names in engine_names.h): the
// token lines and logits helpers.

#ifndef JITLLM_BENCHMARKS_DSV4_COMMON_H_
#define JITLLM_BENCHMARKS_DSV4_COMMON_H_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "engine_names.h"

namespace jitllm::benchmarks {

// A file of token lines: `name<TAB>ids...`, or ids alone.
struct TokenLine {
  std::string name;
  std::vector<std::int32_t> ids;
};
std::expected<std::vector<TokenLine>, std::string> ReadTokenLines(const std::filesystem::path& p);

// -log softmax(row)[target], in double.
double Nll(std::span<const float> row, std::int32_t target);
std::expected<void, std::string> WriteFloats(const std::filesystem::path& p,
                                             std::span<const float> v);

}  // namespace jitllm::benchmarks

#endif  // JITLLM_BENCHMARKS_DSV4_COMMON_H_
