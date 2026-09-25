// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The platform module: small file reads and the host probe, on fake /proc,
// /sys and /dev trees.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>

#include "base/report.h"
#include "platform/direct_io.h"
#include "platform/files.h"
#include "platform/host_probe.h"
#include "platform/path_trust.h"

namespace {

namespace fs = std::filesystem;
using ::testing::AnyOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

// A directory for one test's fake tree, removed afterwards.
class FakeRoot {
 public:
  FakeRoot() {
    std::string pattern = (fs::path(::testing::TempDir()) / "jitllm-platform-XXXXXX").string();
    if (::mkdtemp(pattern.data()) != nullptr) {
      path_ = pattern;
    } else {
      ADD_FAILURE() << "cannot create a directory from " << pattern;
    }
  }
  FakeRoot(const FakeRoot&) = delete;
  FakeRoot& operator=(const FakeRoot&) = delete;
  FakeRoot(FakeRoot&&) = delete;
  FakeRoot& operator=(FakeRoot&&) = delete;
  ~FakeRoot() {
    if (!path_.empty()) {
      std::error_code error;
      fs::remove_all(path_, error);
    }
  }

  const fs::path& path() const { return path_; }

  // Writes text to a file below the root, creating its directories.
  // Nothing is created outside a root that exists.
  void Write(std::string_view name, std::string_view text) const {
    ASSERT_FALSE(path_.empty());
    const fs::path file = path_ / name;
    std::error_code error;
    fs::create_directories(file.parent_path(), error);
    ASSERT_FALSE(error) << error.message();
    std::ofstream out(file, std::ios::binary);
    out << text;
    ASSERT_TRUE(out.good()) << file;
  }

  void Directory(std::string_view name) const {
    ASSERT_FALSE(path_.empty());
    std::error_code error;
    fs::create_directories(path_ / name, error);
    ASSERT_FALSE(error) << error.message();
  }

 private:
  fs::path path_;
};

// The value of a line in a report section, or nullopt.
std::optional<std::string> Value(const jitllm::base::Report& report, std::string_view title,
                                 std::string_view key) {
  for (const auto& section : report.sections) {
    if (section.title != title) {
      continue;
    }
    for (const auto& line : section.lines) {
      if (line.key == key) {
        return line.value;
      }
    }
  }
  return std::nullopt;
}

TEST(Files, ReadSmallFile) {
  const FakeRoot root;
  ASSERT_FALSE(root.path().empty());
  root.Write("a", "hello\nworld\n");
  EXPECT_EQ(jitllm::platform::ReadSmallFile(root.path() / "a"), "hello\nworld\n");
  root.Write("empty", "");
  EXPECT_EQ(jitllm::platform::ReadSmallFile(root.path() / "empty"), "");
}

TEST(Files, ReadSmallFileLimit) {
  const FakeRoot root;
  root.Write("four", "1234");
  EXPECT_EQ(jitllm::platform::ReadSmallFile(root.path() / "four", 4), "1234");
  const auto longer = jitllm::platform::ReadSmallFile(root.path() / "four", 3);
  ASSERT_FALSE(longer);
  EXPECT_EQ(longer.error(), std::errc::file_too_large);
}

TEST(Files, ReadSmallFileErrors) {
  const FakeRoot root;
  const auto missing = jitllm::platform::ReadSmallFile(root.path() / "missing");
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error(), std::errc::no_such_file_or_directory);
  const auto directory = jitllm::platform::ReadSmallFile(root.path());
  ASSERT_FALSE(directory);
  EXPECT_EQ(directory.error(), std::errc::is_a_directory);
}

TEST(Files, ReadFirstLine) {
  const FakeRoot root;
  root.Write("a", "580.178.04  \r\nnext\n");
  EXPECT_EQ(jitllm::platform::ReadFirstLine(root.path() / "a"), "580.178.04");
  root.Write("b", "no newline");
  EXPECT_EQ(jitllm::platform::ReadFirstLine(root.path() / "b"), "no newline");
  root.Write("c", "\n");
  EXPECT_EQ(jitllm::platform::ReadFirstLine(root.path() / "c"), "");
  EXPECT_FALSE(jitllm::platform::ReadFirstLine(root.path() / "missing"));
}

TEST(Files, ListDirectory) {
  const FakeRoot root;
  root.Write("d/b", "");
  root.Write("d/a", "");
  root.Directory("d/C");
  EXPECT_THAT(jitllm::platform::ListDirectory(root.path() / "d").value(),
              ElementsAre("C", "a", "b"));
  root.Directory("empty");
  EXPECT_THAT(jitllm::platform::ListDirectory(root.path() / "empty").value(), IsEmpty());
  const auto missing = jitllm::platform::ListDirectory(root.path() / "missing");
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error(), std::errc::no_such_file_or_directory);
}

TEST(Meminfo, Values) {
  constexpr std::string_view kMeminfo =
      "MemTotal:       127598832 kB\n"
      "MemFree:        1 kB\n"
      "MemAvailable:   123791920 kB\n"
      "HugePages_Total:       0\n";
  using jitllm::platform::MeminfoBytes;
  EXPECT_EQ(MeminfoBytes(kMeminfo, "MemTotal"), std::uint64_t{127598832} * 1024);
  EXPECT_EQ(MeminfoBytes(kMeminfo, "MemAvailable"), std::uint64_t{123791920} * 1024);
  EXPECT_EQ(MeminfoBytes(kMeminfo, "HugePages_Total"), 0U);
  EXPECT_EQ(MeminfoBytes(kMeminfo, "Mem"), std::nullopt);
  EXPECT_EQ(MeminfoBytes(kMeminfo, "SwapTotal"), std::nullopt);
  EXPECT_EQ(MeminfoBytes("MemTotal: 12 MB\n", "MemTotal"), std::nullopt);
  EXPECT_EQ(MeminfoBytes("MemTotal: x kB\n", "MemTotal"), std::nullopt);
  EXPECT_EQ(MeminfoBytes("MemTotal: 18014398509481984 kB\n", "MemTotal"), std::nullopt);
  EXPECT_EQ(MeminfoBytes("MemTotal: 99999999999999999999 kB\n", "MemTotal"), std::nullopt);
  EXPECT_EQ(MeminfoBytes("", "MemTotal"), std::nullopt);
}

TEST(KernelModule, Found) {
  const FakeRoot root;
  root.Write("sys/module/nvidia/version", "580.178.04\n");
  root.Directory("sys/module/nvidia_uvm");
  const auto nvidia = jitllm::platform::FindKernelModule(root.path(), "nvidia");
  EXPECT_TRUE(nvidia.loaded);
  EXPECT_EQ(nvidia.version, "580.178.04");
  const auto uvm = jitllm::platform::FindKernelModule(root.path(), "nvidia_uvm");
  EXPECT_TRUE(uvm.loaded);
  EXPECT_EQ(uvm.version, "");
  const auto fs_module = jitllm::platform::FindKernelModule(root.path(), "nvidia_fs");
  EXPECT_FALSE(fs_module.loaded);
}

// A Spark-like tree: two RDMA devices, one port each.
void WriteSparkLike(const FakeRoot& root) {
  root.Write("proc/meminfo", "MemTotal: 127598832 kB\nMemAvailable: 123791920 kB\n");
  root.Write("proc/sys/fs/protected_hardlinks", "1\n");
  for (const auto& [device, netdev, verbs, state] :
       {std::tuple{"rocep1s0f0", "enp1s0f0np0", "uverbs0", "1: DOWN"},
        std::tuple{"rocep1s0f1", "enp1s0f1np1", "uverbs1", "4: ACTIVE"}}) {
    const std::string dir = std::string("sys/class/infiniband/") + device;
    root.Write(dir + "/ports/1/state", std::string(state) + "\n");
    root.Write(dir + "/ports/1/link_layer", "Ethernet\n");
    root.Write(dir + "/ports/1/rate", "200 Gb/sec (2X NDR)\n");
    root.Directory(dir + "/device/net/" + netdev);
    root.Directory(dir + "/device/infiniband_verbs/" + verbs);
    root.Write(std::string("dev/infiniband/") + verbs, "");
  }
  root.Write("dev/infiniband/rdma_cm", "");
}

TEST(DescribeHost, SparkLike) {
  const FakeRoot root;
  WriteSparkLike(root);
  jitllm::base::Report report;
  jitllm::platform::DescribeHost(root.path(), report);
  EXPECT_EQ(Value(report, "host", "memory"), "121.7 GiB total, 118.1 GiB available");
  EXPECT_EQ(Value(report, "host", "fs.protected_hardlinks"), "1");
  EXPECT_TRUE(Value(report, "host", "glibc"));
  EXPECT_TRUE(Value(report, "host", "kernel"));
  EXPECT_EQ(Value(report, "host", "page size"),
            jitllm::base::FormatBytes(static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE))));
  EXPECT_EQ(Value(report, "RDMA", "rocep1s0f0 port 1"),
            "DOWN, Ethernet, 200 Gb/sec (2X NDR), enp1s0f0np0, /dev/infiniband/uverbs0 read-write");
  EXPECT_EQ(
      Value(report, "RDMA", "rocep1s0f1 port 1"),
      "ACTIVE, Ethernet, 200 Gb/sec (2X NDR), enp1s0f1np1, /dev/infiniband/uverbs1 read-write");
  EXPECT_EQ(Value(report, "RDMA", "/dev/infiniband/rdma_cm"), "read-write");
  EXPECT_EQ(Value(report, "RDMA", "device access checked for"),
            "uid " + std::to_string(::geteuid()));
  EXPECT_THAT(report.problems, IsEmpty());
  EXPECT_THAT(report.warnings, IsEmpty());
}

TEST(DescribeHost, NothingThere) {
  const FakeRoot root;
  jitllm::base::Report report;
  jitllm::platform::DescribeHost(root.path(), report);
  EXPECT_EQ(Value(report, "host", "memory"), "unknown");
  EXPECT_EQ(Value(report, "host", "fs.protected_hardlinks"), "unknown");
  EXPECT_EQ(Value(report, "RDMA", "devices"), "none");
  EXPECT_THAT(report.problems, IsEmpty());
  ASSERT_EQ(report.warnings.size(), 2U);
  EXPECT_THAT(report.warnings[0], HasSubstr("cannot read /proc/meminfo"));
  EXPECT_THAT(report.warnings[1], HasSubstr("cannot read fs.protected_hardlinks"));
}

TEST(DescribeHost, Warnings) {
  const FakeRoot root;
  root.Write("proc/meminfo", "MemFree: 1 kB\n");
  root.Write("proc/sys/fs/protected_hardlinks", "0\n");
  root.Write("sys/class/infiniband/mlx5_0/ports/1/state", "4: ACTIVE\n");
  root.Directory("sys/class/infiniband/mlx5_0/device/infiniband_verbs/uverbs0");
  root.Directory("sys/class/infiniband/mlx5_1");  // no ports
  jitllm::base::Report report;
  jitllm::platform::DescribeHost(root.path(), report);
  EXPECT_EQ(Value(report, "RDMA", "mlx5_0 port 1"),
            "ACTIVE, unknown link layer, unknown rate, /dev/infiniband/uverbs0 No such file or "
            "directory");
  EXPECT_EQ(Value(report, "RDMA", "mlx5_1"), "no ports");
  EXPECT_EQ(Value(report, "RDMA", "/dev/infiniband/rdma_cm"), "No such file or directory");
  EXPECT_THAT(report.problems, IsEmpty());
  ASSERT_EQ(report.warnings.size(), 5U);
  EXPECT_EQ(report.warnings[0], "/proc/meminfo has no MemTotal or MemAvailable");
  EXPECT_THAT(report.warnings[1], HasSubstr("fs.protected_hardlinks is 0, not 1"));
  EXPECT_THAT(report.warnings[2], HasSubstr("RDMA device mlx5_0: this user cannot open "
                                            "/dev/infiniband/uverbs0"));
  EXPECT_THAT(report.warnings[3], HasSubstr("RDMA device mlx5_1 has no verbs device"));
  EXPECT_THAT(report.warnings[4], HasSubstr("/dev/infiniband/rdma_cm"));
}

// A device node this user cannot open (not testable as root, who can).
TEST(DescribeHost, InaccessibleVerbs) {
  if (::geteuid() == 0) {
    GTEST_SKIP() << "root can open any file";
  }
  const FakeRoot root;
  WriteSparkLike(root);
  ASSERT_EQ(::chmod((root.path() / "dev/infiniband/uverbs1").c_str(), 0), 0);
  jitllm::base::Report report;
  jitllm::platform::DescribeHost(root.path(), report);
  EXPECT_EQ(Value(report, "RDMA", "rocep1s0f1 port 1"),
            "ACTIVE, Ethernet, 200 Gb/sec (2X NDR), enp1s0f1np1, /dev/infiniband/uverbs1 "
            "Permission denied");
  ASSERT_EQ(report.warnings.size(), 1U);
  EXPECT_THAT(report.warnings[0], HasSubstr("RDMA device rocep1s0f1"));
}

// Scratch directories in the build tree (JITLLM_TEST_SCRATCH), whose
// parents no other user shares, as they may /tmp's.
class Scratch {
 public:
  Scratch() {
    const char* base = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const fs::path parent = base != nullptr ? fs::path(base) : fs::path(::testing::TempDir());
    std::error_code error;
    fs::create_directories(parent, error);
    std::string pattern = (parent / "trust-XXXXXX").string();
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
    fs::permissions(path_, fs::perms::owner_all, error);
    fs::remove_all(path_, error);
  }
  const fs::path& path() const { return path_; }

 private:
  fs::path path_;
};

TEST(WalkTrusted, ResolvesLinksAndMissingTails) {
  const Scratch scratch;
  const fs::path& root = scratch.path();
  fs::create_directories(root / "real/sub");
  fs::create_directory_symlink("real", root / "rel");
  fs::create_directory_symlink(root / "real", root / "abs");
  std::ofstream(root / "real/sub/file") << "x";
  const uid_t me = ::geteuid();

  auto walked = jitllm::platform::WalkTrusted(root / "rel/sub/file", me, false);
  ASSERT_TRUE(walked.has_value()) << walked.error();
  EXPECT_TRUE(walked->exists);
  EXPECT_EQ(walked->resolved, fs::canonical(root / "real/sub/file"));
  EXPECT_TRUE(S_ISREG(walked->status.st_mode));

  walked = jitllm::platform::WalkTrusted(root / "abs/sub/../missing/deeper", me, false);
  ASSERT_TRUE(walked.has_value()) << walked.error();
  EXPECT_FALSE(walked->exists);
  EXPECT_EQ(walked->resolved, fs::canonical(root / "real") / "missing/deeper");
  EXPECT_EQ(walked->existing, fs::canonical(root / "real"));

  // A link as the last component is refused unless allowed.
  fs::create_symlink("real/sub/file", root / "link");
  walked = jitllm::platform::WalkTrusted(root / "link", me, false);
  ASSERT_FALSE(walked.has_value());
  EXPECT_THAT(walked.error(), HasSubstr("is a symbolic link"));
  walked = jitllm::platform::WalkTrusted(root / "link", me, true);
  ASSERT_TRUE(walked.has_value()) << walked.error();
  EXPECT_EQ(walked->resolved, fs::canonical(root / "real/sub/file"));
}

TEST(WalkTrusted, RefusesLoopsAndNonDirectories) {
  const Scratch scratch;
  const fs::path& root = scratch.path();
  fs::create_symlink("b", root / "a");
  fs::create_symlink("a", root / "b");
  auto walked = jitllm::platform::WalkTrusted(root / "a/x", ::geteuid(), false);
  ASSERT_FALSE(walked.has_value());
  EXPECT_THAT(walked.error(), HasSubstr("too many symbolic links"));
  std::ofstream(root / "file") << "x";
  walked = jitllm::platform::WalkTrusted(root / "file/x", ::geteuid(), false);
  ASSERT_FALSE(walked.has_value());
  EXPECT_THAT(walked.error(), HasSubstr("file is not a directory"));
  // ".." after a missing component: the kernel stops at the missing one.
  fs::create_symlink("missing/../../x", root / "dotdot");
  walked = jitllm::platform::WalkTrusted(root / "dotdot/y", ::geteuid(), false);
  ASSERT_FALSE(walked.has_value());
  EXPECT_THAT(walked.error(), HasSubstr("does not exist, and the path goes on to '..'"));
  walked = jitllm::platform::WalkTrusted("relative/path", ::geteuid(), false);
  ASSERT_FALSE(walked.has_value());
  EXPECT_THAT(walked.error(), HasSubstr("not an absolute path"));
}

TEST(WalkTrusted, RefusesWhatOthersCanChange) {
  const Scratch scratch;
  const fs::path& root = scratch.path();
  fs::create_directories(root / "open/inner");
  const uid_t me = ::geteuid();
  ASSERT_EQ(::chmod((root / "open").c_str(), 0757), 0);
  auto walked = jitllm::platform::WalkTrusted(root / "open/inner", me, false);
  ASSERT_FALSE(walked.has_value());
  EXPECT_THAT(walked.error(), HasSubstr("open can be changed by users other than root and uid"));
  // A sticky directory is passable: others cannot replace what it holds.
  ASSERT_EQ(::chmod((root / "open").c_str(), 01777), 0);
  walked = jitllm::platform::WalkTrusted(root / "open/inner", me, false);
  EXPECT_TRUE(walked.has_value()) << walked.error();
  // But not as a directory to add files to.
  struct stat status{};
  ASSERT_EQ(::stat((root / "open").c_str(), &status), 0);
  const auto private_dir = jitllm::platform::CheckPrivateDirectory(root / "open", status, me);
  ASSERT_FALSE(private_dir.has_value());
  EXPECT_THAT(private_dir.error(), HasSubstr("can add files to"));
  // Entries another user owns are untrusted: here, trusting root alone.
  if (me != 0) {
    walked = jitllm::platform::WalkTrusted(root / "open/inner", 0, false);
    ASSERT_FALSE(walked.has_value());
    // Under qemu-user, "/" is the sysroot, which this user owns: the walk
    // stops there instead.
    EXPECT_THAT(walked.error(),
                AnyOf(HasSubstr(std::format("is owned by uid {}, not root", me)),
                      HasSubstr(std::format("/ is not owned by root and private (uid {}", me))));
  }
}

TEST(DirectIo, DescribesFilesystems) {
  auto proc = jitllm::platform::DescribeFilesystem("/proc");
  ASSERT_TRUE(proc.has_value()) << proc.error();
  EXPECT_FALSE(proc->accepted);
  EXPECT_FALSE(jitllm::platform::DescribeFilesystem("/nonexistent/jitllm").has_value());
  auto refused = jitllm::platform::ProbeDirectIo("/proc");
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("not a local block-device filesystem"));
}

TEST(DirectIo, ProbesTheBuildTreesFilesystem) {
  const Scratch scratch;
  auto filesystem = jitllm::platform::DescribeFilesystem(scratch.path());
  ASSERT_TRUE(filesystem.has_value()) << filesystem.error();
  if (!filesystem->accepted) {
    GTEST_SKIP() << "the build tree is on " << filesystem->type
                 << ", which the storage roles refuse";
  }
  auto probe = jitllm::platform::ProbeDirectIo(scratch.path());
  ASSERT_TRUE(probe.has_value()) << probe.error();
  EXPECT_LE(probe->offset_alignment, jitllm::platform::kDirectIoAlignment);
  // The probe's file had no name: nothing is left behind.
  EXPECT_TRUE(fs::is_empty(scratch.path()));
}

}  // namespace
