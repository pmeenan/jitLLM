// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The `jitllm` command's commands, options and output (D-062), and the
// surface versions it is built with.

#include "cli/cli.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "base/build_info.h"
#include "base/report.h"
#include "base/surface_versions.h"
#include "cli/doctor.h"

namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::MatchesRegex;
using ::testing::Not;
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
    EXPECT_THAT(result.out,
                StartsWith("Usage: jitllm doctor [--config FILE]\n       jitllm --version\n"))
        << option;
    EXPECT_THAT(result.out, HasSubstr("\n  doctor ")) << option;
    EXPECT_EQ(result.err, "") << option;
  }
}

TEST(Cli, UsageErrors) {
  const std::vector<std::pair<std::vector<std::string_view>, std::string>> cases = {
      {{}, "jitllm: no command or option given\n"},
      {{"--verison"}, "jitllm: unknown command or option '--verison'\n"},
      {{"version"}, "jitllm: unknown command or option 'version'\n"},
      {{"Doctor"}, "jitllm: unknown command or option 'Doctor'\n"},
      {{"--version", "--help"}, "jitllm: unexpected argument '--help' after --version\n"},
      {{"--help", "x"}, "jitllm: unexpected argument 'x' after --help\n"},
      {{"doctor", "--help"}, "jitllm: unexpected argument '--help' after doctor\n"},
      {{"doctor", "--config"}, "jitllm: --config needs a file\n"},
      {{"doctor", "--config", ""}, "jitllm: --config needs a file\n"},
      {{"doctor", "--config", "a.toml", "x"}, "jitllm: unexpected argument 'x' after doctor\n"},
      {{"--version", "--config", "a.toml"},
       "jitllm: unexpected argument '--config' after --version\n"},
  };
  for (const auto& [args, message] : cases) {
    const Result result = RunWith(args);
    EXPECT_EQ(result.status, jitllm::cli::kExitUsage) << message;
    EXPECT_EQ(result.out, "") << message;
    EXPECT_THAT(result.err, StartsWith(message));
    EXPECT_THAT(result.err, HasSubstr("Usage: jitllm doctor [--config FILE]\n")) << message;
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

// Whatever this host is, doctor prints a report whose summary agrees with
// its exit status. smoke.doctor checks the binary on a GB10.
TEST(Doctor, ExitStatusFollowsTheReport) {
  const Result result = RunWith({"doctor"});
  EXPECT_EQ(result.err, "");
  EXPECT_THAT(result.out, StartsWith("build\n  version: "));
  const bool clean = result.out.contains("\ndoctor: no problems, ");
  EXPECT_EQ(result.status, clean ? jitllm::cli::kExitOk : jitllm::cli::kExitFailure) << result.out;
  EXPECT_EQ(clean, !result.out.contains("\nproblem: ")) << result.out;
}

TEST(Doctor, WriteFailureFails) {
  const std::unique_ptr<std::FILE, CloseFile> full(std::fopen("/dev/full", "w"));
  ASSERT_NE(full, nullptr);
  Capture err;
  const std::vector<std::string_view> args = {"doctor"};
  EXPECT_EQ(jitllm::cli::Run(args, full.get(), err.stream()), jitllm::cli::kExitFailure);
  EXPECT_EQ(err.text(), "jitllm: cannot write to standard output\n");
}

TEST(Doctor, Build) {
  const jitllm::base::BuildInfo info{.version = "0.2.0-dev.7+g0123456789ab.dirty",
                                     .commit = "0123456789abcdef0123456789abcdef01234567",
                                     .modified = true,
                                     .license_profile = "core",
                                     .sdk = "aarch64-0123456789abcdef",
                                     .target = "aarch64-linux-gnu"};
  jitllm::base::Report report;
  jitllm::cli::DescribeBuild(info, report);
  ASSERT_EQ(report.sections.size(), 1U);
  const jitllm::base::ReportSection& build = report.sections[0];
  EXPECT_EQ(build.title, "build");
  std::string text;
  for (const auto& line : build.lines) {
    text += line.key + ": " + line.value + "\n";
  }
  EXPECT_THAT(text, StartsWith("version: 0.2.0-dev.7+g0123456789ab.dirty\n"
                               "commit: 0123456789abcdef0123456789abcdef01234567 (with uncommitted "
                               "changes)\n"
                               "license profile: core\n"
                               "SDK: aarch64-0123456789abcdef\n"
                               "target: aarch64-linux-gnu\n"));
  // The test is built by the compiler that built the command.
  EXPECT_THAT(text, HasSubstr(std::format("\ncompiler: Clang {}.{}.{}\n", __clang_major__,
                                          __clang_minor__, __clang_patchlevel__)));
  EXPECT_THAT(text, HasSubstr("C++ runtime: libstdc++ from GCC 16, linked statically (D-060)\n"));
  EXPECT_TRUE(report.problems.empty());
}

TEST(Doctor, ControlCharactersAreEscaped) {
  EXPECT_EQ(jitllm::cli::Printable("a\nproblem: b\t\x7f\x1b[31m"),
            "a\\x0aproblem: b\\x09\\x7f\\x1b[31m");
  EXPECT_EQ(jitllm::cli::Printable("GB10 caf\xc3\xa9"), "GB10 caf\xc3\xa9");
  jitllm::base::Report report;
  report.AddSection("t\n").Add("k\n", "v\nproblem: forged");
  report.warnings = {"w\nproblem: forged"};
  EXPECT_EQ(jitllm::cli::DoctorText(report),
            "t\\x0a\n  k\\x0a: v\\x0aproblem: forged\n\n"
            "warning: w\\x0aproblem: forged\ndoctor: no problems, 1 warning\n");
}

// The build and host sections go out before the device probe runs, and a
// failed write stops the run.
TEST(Doctor, WritesInStages) {
  std::vector<std::string> parts;
  jitllm::base::Report report;
  const jitllm::cli::DoctorOptions options{.config = "/nonexistent/jitllm.toml"};
  ASSERT_TRUE(jitllm::cli::Doctor("/nonexistent", options, report, [&](std::string_view text) {
    parts.emplace_back(text);
    return true;
  }));
  ASSERT_EQ(parts.size(), 2U);
  EXPECT_THAT(parts[0], StartsWith("build\n"));
  EXPECT_THAT(parts[0], HasSubstr("\nhost\n"));
  EXPECT_THAT(parts[0], HasSubstr("\nRDMA\n"));
  EXPECT_THAT(parts[0], HasSubstr("\nconfiguration\n"));
  EXPECT_THAT(parts[0], Not(HasSubstr("doctor: ")));
  EXPECT_THAT(parts[1], HasSubstr("\ndoctor: "));
  EXPECT_EQ(parts[0] + parts[1], jitllm::cli::DoctorText(report));

  jitllm::base::Report stopped;
  int calls = 0;
  EXPECT_FALSE(jitllm::cli::Doctor("/nonexistent", options, stopped, [&](std::string_view) {
    ++calls;
    return false;
  }));
  EXPECT_EQ(calls, 1);
  // build, host, RDMA, configuration (which failed): no device probe
  EXPECT_EQ(stopped.sections.size(), 4U);
}

// Scratch trees in the build tree (JITLLM_TEST_SCRATCH), whose parents no
// other user shares.
class Scratch {
 public:
  Scratch() {
    const char* base = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const std::filesystem::path parent = base != nullptr ? base : ::testing::TempDir();
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    std::string pattern = (parent / "cli-XXXXXX").string();
    if (::mkdtemp(pattern.data()) != nullptr) {
      path_ = pattern;
    } else {
      ADD_FAILURE() << "cannot create a directory from " << pattern;
    }
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  Scratch(Scratch&&) = delete;
  Scratch& operator=(Scratch&&) = delete;
  ~Scratch() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

TEST(Doctor, ReportsTheConfigurationAndStorage) {
  const Scratch scratch;
  const std::filesystem::path config = scratch.path() / "jitllm.toml";
  std::ofstream(config) << std::format("schema_version = 2\n[storage]\ndata_dir = \"{}\"\n",
                                       (scratch.path() / "data").string());
  std::filesystem::create_directories(scratch.path() / "data/models");
  jitllm::base::Report report;
  jitllm::cli::DescribeConfiguration({.config = config}, report);
  const std::string text = jitllm::cli::DoctorText(report);
  EXPECT_THAT(text, HasSubstr("configuration\n  runtime's user: uid "));
  EXPECT_THAT(text, HasSubstr("\n  file: " + config.string() + "\n  node: standalone\n"));
  EXPECT_THAT(text,
              HasSubstr("\nstorage\n  data_dir: " + (scratch.path() / "data").string() + "\n"));
  EXPECT_THAT(text, HasSubstr("  installed: " + (scratch.path() / "data/models").string() +
                              ", owner uid "));
  EXPECT_THAT(text, HasSubstr("  spill: " + (scratch.path() / "data/spill").string() +
                              ", not created yet"));
  EXPECT_THAT(text, HasSubstr("  checkpoints: " + (scratch.path() / "data/checkpoints").string() +
                              " (job processes only; not examined)"));
}

TEST(Doctor, AnInvalidConfigurationIsAProblem) {
  const Scratch scratch;
  const std::filesystem::path config = scratch.path() / "jitllm.toml";
  std::ofstream(config) << "schema_version = 2\nstorage.spil = 1\n";
  jitllm::base::Report report;
  jitllm::cli::DescribeConfiguration({.config = config}, report);
  EXPECT_THAT(report.problems,
              ElementsAre("configuration: " + config.string() + ":2:16: unknown key storage.spil"));
  jitllm::base::Report missing;
  jitllm::cli::DescribeConfiguration({.config = scratch.path() / "other.toml"}, missing);
  EXPECT_THAT(missing.problems, ElementsAre(HasSubstr("other.toml: does not exist")));
}

TEST(Doctor, Text) {
  jitllm::base::Report report;
  jitllm::base::ReportSection& first = report.AddSection("first");
  first.Add("a", "1");
  first.Add("b c", "two words");
  report.AddSection("empty");
  EXPECT_EQ(jitllm::cli::DoctorText(report),
            "first\n  a: 1\n  b c: two words\n\nempty\n\ndoctor: no problems, 0 warnings\n");
  report.problems = {"p1", "p2"};
  report.warnings = {"w1"};
  EXPECT_EQ(jitllm::cli::DoctorText(report),
            "first\n  a: 1\n  b c: two words\n\nempty\n\n"
            "problem: p1\nproblem: p2\nwarning: w1\ndoctor: 2 problems, 1 warning\n");
  report.problems = {"p1"};
  report.warnings = {"w1", "w2"};
  EXPECT_THAT(jitllm::cli::DoctorText(report),
              testing::EndsWith("doctor: 1 problem, 2 warnings\n"));
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
