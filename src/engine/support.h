// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Small helpers every engine file uses (docs/engine.md): an error result,
// an address as a pointer and back, rounding up, seconds of a duration, and
// a list of problems joined into one error. Header-only.

#ifndef JITLLM_ENGINE_SUPPORT_H_
#define JITLLM_ENGINE_SUPPORT_H_

#include <chrono>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>

namespace jitllm::engine::support {

inline std::unexpected<std::string> Error(std::string what) {
  return std::unexpected(std::move(what));
}

inline void* Pointer(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

inline std::uint64_t Address(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer);
}

// `bytes` rounded up to a multiple of `to`.
inline std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) {
  return (bytes + to - 1) / to * to;
}

inline double Seconds(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double>(d).count();
}

// Every problem in one error, "; "-separated; success if there is none.
inline std::expected<void, std::string> Joined(std::span<const std::string> problems) {
  if (problems.empty()) {
    return {};
  }
  std::string all;
  for (const std::string& problem : problems) {
    all += (all.empty() ? "" : "; ") + problem;
  }
  return Error(all);
}

}  // namespace jitllm::engine::support

#endif  // JITLLM_ENGINE_SUPPORT_H_
