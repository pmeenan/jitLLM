// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <span>
#include <string_view>
#include <vector>

#include "cli/cli.h"

int main(int argc, char** argv) {
  const std::span<char*> all(argv, static_cast<std::size_t>(argc));
  const std::vector<std::string_view> args(all.begin() + (argc > 0 ? 1 : 0), all.end());
  return jitllm::cli::Run(args, stdout, stderr);
}
