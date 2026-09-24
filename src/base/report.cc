// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "base/report.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>

namespace jitllm::base {

std::string FormatBytes(std::uint64_t bytes) {
  static constexpr std::array<std::string_view, 6> kUnits = {"KiB", "MiB", "GiB",
                                                             "TiB", "PiB", "EiB"};
  if (bytes < 1024) {
    return std::format("{} B", bytes);
  }
  std::size_t unit = 0;
  std::uint64_t scale = 1024;
  while (unit + 1 < kUnits.size() && bytes / scale >= 1024) {
    scale *= 1024;
    ++unit;
  }
  if (bytes % scale == 0) {
    return std::format("{} {}", bytes / scale, kUnits.at(unit));
  }
  double value = static_cast<double>(bytes) / static_cast<double>(scale);
  // Just under the next unit rounds up to it: "1.0 GiB", never "1024.0 MiB".
  if (std::round(value * 10) >= 10240 && unit + 1 < kUnits.size()) {
    value /= 1024;
    ++unit;
  }
  return std::format("{:.1f} {}", value, kUnits.at(unit));
}

}  // namespace jitllm::base
