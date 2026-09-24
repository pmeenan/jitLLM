// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"

#include <cstdio>
#include <format>
#include <span>
#include <string>
#include <string_view>

#include "base/build_info.h"

namespace jitllm::cli {
namespace {

constexpr std::string_view kUsage =
    "Usage: jitllm --version\n"
    "       jitllm --help\n";

constexpr std::string_view kHelp =
    "Usage: jitllm --version\n"
    "       jitllm --help\n"
    "\n"
    "jitLLM, a just-in-time LLM inference engine.\n"
    "\n"
    "  --version  print the version, commit, license profile, SDK and target\n"
    "  --help     print this help\n";

// Writes all of text and flushes it; false if either fails.
bool WriteAll(std::FILE* stream, std::string_view text) {
  return std::fwrite(text.data(), 1, text.size(), stream) == text.size() &&
         std::fflush(stream) == 0;
}

int Print(std::FILE* out, std::FILE* err, std::string_view text) {
  if (WriteAll(out, text)) {
    return kExitOk;
  }
  WriteAll(err, "jitllm: cannot write to standard output\n");
  return kExitFailure;
}

int UsageError(std::FILE* err, std::string_view message) {
  WriteAll(err, std::format("jitllm: {}\n{}", message, kUsage));
  return kExitUsage;
}

}  // namespace

std::string VersionText(const base::BuildInfo& info) {
  std::string commit = info.commit.empty() ? std::string("unknown") : std::string(info.commit);
  if (info.modified) {
    commit += " (with uncommitted changes)";
  }
  return std::format("jitllm {}\ncommit: {}\nlicense profile: {}\nSDK: {}\ntarget: {}\n",
                     info.version, commit, info.license_profile, info.sdk, info.target);
}

int Run(std::span<const std::string_view> args, std::FILE* out, std::FILE* err) {
  if (args.empty()) {
    return UsageError(err, "no option given");
  }
  const std::string_view option = args.front();
  if (option != "--version" && option != "--help" && option != "-h") {
    return UsageError(err, std::format("unknown option '{}'", option));
  }
  if (args.size() > 1) {
    return UsageError(err, std::format("unexpected argument '{}' after {}", args[1], option));
  }
  if (option == "--version") {
    return Print(out, err, VersionText(base::GetBuildInfo()));
  }
  return Print(out, err, kHelp);
}

}  // namespace jitllm::cli
