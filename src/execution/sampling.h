// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Choosing the next token from a step's logits, on the host: greedy, and
// seeded sampling that is reproducible from its key alone.
//
// Greedy takes the highest logit, the lowest ID among equals (as
// torch.argmax and llama.cpp's greedy sampler do). Sampling applies, in
// order: temperature (0 means greedy), top-k, softmax, min-p, top-p; then
// draws from what remains with one uniform number. Candidates are ordered by
// probability, then by ID, so ties never depend on sort stability.
//
// The uniform number is Philox4x32-10 (Salmon et al., SC'11) keyed by the
// seed, over a counter made of the stream and the position: the same (seed,
// stream, position) always draws the same token, whatever came before, so a
// speculative verifier can redraw any position and a restored request
// continues exactly (D-068).

#ifndef JITLLM_EXECUTION_SAMPLING_H_
#define JITLLM_EXECUTION_SAMPLING_H_

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

namespace jitllm::execution {

struct SamplingParams {
  float temperature = 1.0F;  // 0: greedy
  std::uint32_t top_k = 0;   // 0: off
  float top_p = 1.0F;        // (0, 1]; 1: off
  float min_p = 0.0F;        // [0, 1]; 0: off
};

struct SamplingKey {
  std::uint64_t seed = 0;
  std::uint64_t stream = 0;    // e.g. the request's sequence
  std::uint64_t position = 0;  // the generated token's position
};

enum class SamplingError : std::uint8_t {
  kNoLogits,       // empty, or no finite logit
  kInvalidLogits,  // a NaN or +infinity logit
  kInvalidParams,  // a parameter outside its range
};

std::string_view SamplingErrorName(SamplingError e);

// The greedy choice.
std::expected<std::int32_t, SamplingError> Greedy(std::span<const float> logits);

// A seeded draw. `scratch` is reused between calls to avoid allocating.
struct SamplingCandidate {
  double value;
  std::int32_t id;
};
std::expected<std::int32_t, SamplingError> Sample(std::span<const float> logits,
                                                  const SamplingParams& params,
                                                  const SamplingKey& key,
                                                  std::vector<SamplingCandidate>& scratch);

// Philox4x32-10's block for a 128-bit counter and 64-bit key.
std::array<std::uint32_t, 4> Philox4x32(std::array<std::uint32_t, 4> counter,
                                        std::array<std::uint32_t, 2> key);

// The uniform number in [0, 1) a key draws, with 53 random bits.
double UniformAt(const SamplingKey& key);

}  // namespace jitllm::execution

#endif  // JITLLM_EXECUTION_SAMPLING_H_
