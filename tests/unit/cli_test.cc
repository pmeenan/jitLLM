// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The `jitllm` command's options and output (D-062), and the surface
// versions it is built with.

#include "cli/cli.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/build_info.h"
#include "base/surface_versions.h"

namespace {

using ::testing::HasSubstr;
using ::testing::MatchesRegex;
using ::testing::StartsWith;

// A FILE* that collects what is written to it.
class Capture {
 public:
  Capture() : stream_(open_memstream(&buffer_, &size_)) {}
  Capture(const Capture&) = delete;
  Capture& operator=(const Capture&) = delete;
  Capture(Capture&&) = delete;
  Capture& operator=(Capture&&) = delete;
  ~Capture() {
    if (stream_ != nullptr) {
      (void)std::fclose(stream_);
    }
    std::free(buffer_);  // open_memstream allocates it
  }

  std::FILE* stream() const { return stream_; }

  std::string text() {
    EXPECT_EQ(std::fflush(stream_), 0);
    return {buffer_, size_};
  }

 private:
  char* buffer_ = nullptr;
  std::size_t size_ = 0;
  std::FILE* stream_ = nullptr;
};

struct Result {
  int status;
  std::string out;
  std::string err;
};

Result RunWith(std::vector<std::string_view> args) {
  Capture out;
  Capture err;
  const int status = jitllm::cli::Run(args, out.stream(), err.stream());
  return {.status = status, .out = out.text(), .err = err.text()};
}

TEST(Cli, VersionPrintsTheBuild) {
  const Result result = RunWith({"--version"});
  EXPECT_EQ(result.status, jitllm::cli::kExitOk);
  EXPECT_EQ(result.out, jitllm::cli::VersionText(jitllm::base::GetBuildInfo()));
  EXPECT_EQ(result.err, "");
}

TEST(Cli, VersionTextFormat) {
  const jitllm::base::BuildInfo info{.version = "0.2.0-dev.7+g0123456789ab.dirty",
                                     .commit = "0123456789abcdef0123456789abcdef01234567",
                                     .modified = true,
                                     .license_profile = "core",
                                     .sdk = "aarch64-0123456789abcdef",
                                     .target = "aarch64-linux-gnu"};
  EXPECT_EQ(jitllm::cli::VersionText(info),
            "jitllm 0.2.0-dev.7+g0123456789ab.dirty\n"
            "commit: 0123456789abcdef0123456789abcdef01234567 (with uncommitted changes)\n"
            "license profile: core\n"
            "SDK: aarch64-0123456789abcdef\n"
            "target: aarch64-linux-gnu\n");
}

TEST(Cli, VersionTextWithoutGit) {
  const jitllm::base::BuildInfo info{.version = "0.2.0-dev+unknown",
                                     .commit = "",
                                     .modified = false,
                                     .license_profile = "core+example",
                                     .sdk = "x86_64-0123456789abcdef",
                                     .target = "x86_64-linux-gnu"};
  EXPECT_THAT(
      jitllm::cli::VersionText(info),
      StartsWith("jitllm 0.2.0-dev+unknown\ncommit: unknown\nlicense profile: core+example\n"));
}

TEST(Cli, Help) {
  for (const std::string_view option : {"--help", "-h"}) {
    const Result result = RunWith({option});
    EXPECT_EQ(result.status, jitllm::cli::kExitOk) << option;
    EXPECT_THAT(result.out, StartsWith("Usage: jitllm --version\n")) << option;
    EXPECT_EQ(result.err, "") << option;
  }
}

TEST(Cli, UsageErrors) {
  const std::vector<std::pair<std::vector<std::string_view>, std::string>> cases = {
      {{}, "jitllm: no option given\n"},
      {{"--verison"}, "jitllm: unknown option '--verison'\n"},
      {{"version"}, "jitllm: unknown option 'version'\n"},
      {{"--version", "--help"}, "jitllm: unexpected argument '--help' after --version\n"},
      {{"--help", "x"}, "jitllm: unexpected argument 'x' after --help\n"},
  };
  for (const auto& [args, message] : cases) {
    const Result result = RunWith(args);
    EXPECT_EQ(result.status, jitllm::cli::kExitUsage) << message;
    EXPECT_EQ(result.out, "") << message;
    EXPECT_THAT(result.err, StartsWith(message));
    EXPECT_THAT(result.err, HasSubstr("Usage: jitllm --version\n")) << message;
  }
}

struct CloseFile {
  void operator()(std::FILE* stream) const { (void)std::fclose(stream); }
};

// A version nobody saw is a failure: the output is flushed and checked.
TEST(Cli, WriteFailureFails) {
  const std::unique_ptr<std::FILE, CloseFile> full(std::fopen("/dev/full", "w"));
  ASSERT_NE(full, nullptr);
  Capture err;
  const std::vector<std::string_view> args = {"--version"};
  EXPECT_EQ(jitllm::cli::Run(args, full.get(), err.stream()), jitllm::cli::kExitFailure);
  EXPECT_EQ(err.text(), "jitllm: cannot write to standard output\n");
}

// What the build generated: D-062's forms, for this checkout's project(VERSION).
TEST(BuildInfo, IsThisBuild) {
  const jitllm::base::BuildInfo& info = jitllm::base::GetBuildInfo();
  const std::string version(info.version);
  EXPECT_THAT(version, StartsWith(JITLLM_TEST_PROJECT_VERSION));
  EXPECT_THAT(version, MatchesRegex(R"([0-9]+\.[0-9]+\.[0-9]+)"
                                    R"((-dev\.[0-9]+\+g[0-9a-f]{12,}(\.dirty)?|-dev\+unknown)?)"));
  if (info.commit.empty()) {
    EXPECT_THAT(version, testing::EndsWith("-dev+unknown"));
    EXPECT_FALSE(info.modified);
  } else {
    EXPECT_THAT(std::string(info.commit), MatchesRegex("[0-9a-f]{40}|[0-9a-f]{64}"));
    EXPECT_EQ(info.modified, version.ends_with(".dirty"));
  }
  EXPECT_THAT(std::string(info.license_profile), MatchesRegex(R"(core(\+[a-z0-9_-]+)*)"));
  EXPECT_FALSE(info.sdk.empty());
  EXPECT_FALSE(info.target.empty());
}

// Surface versions change only deliberately: a bump comes with a CHANGELOG
// entry and a decisions.md entry (D-062), and with this test.
TEST(SurfaceVersions, Pinned) { EXPECT_EQ(jitllm::surface::kReasoningSignatureVersion, 1U); }

}  // namespace
