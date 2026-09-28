// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The serving commands in a build without CUDA: nothing to serve with.

#include <cstdio>
#include <string_view>

#include "runtime/commands.h"

namespace jitllm::runtime {

int RunServing(const config::NodeConfig& /*config*/, const config::RuntimeRoles& /*roles*/,
               const CommandOptions& /*command*/, std::FILE* /*out*/, std::FILE* log) {
  constexpr std::string_view kLine =
      "jitllm-runtime: this build has no GPU support, so no serving commands\n";
  (void)std::fwrite(kLine.data(), 1, kLine.size(), log);
  return 1;  // kExitFailure
}

}  // namespace jitllm::runtime
