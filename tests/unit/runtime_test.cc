// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The runtime module's startup steps on scratch trees, and the platform
// module's process services it relies on: the crash policy, lock files and
// readiness notification.

#include "runtime/runtime.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "platform/crash_policy.h"
#include "platform/files.h"
#include "platform/lock_file.h"
#include "platform/sd_notify.h"

namespace {

namespace fs = std::filesystem;
using ::testing::HasSubstr;

class Scratch {
 public:
  Scratch() {
    const char* base = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const fs::path parent = base != nullptr ? fs::path(base) : fs::path(::testing::TempDir());
    std::error_code error;
    fs::create_directories(parent, error);
    std::string pattern = (parent / "runtime-XXXXXX").string();
    if (::mkdtemp(pattern.data()) != nullptr) {
      path_ = pattern;
      (void)::chmod(path_.c_str(), 0755);
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
    fs::remove_all(path_, error);
  }
  const fs::path& path() const { return path_; }

  // A standalone configuration whose roles live here.
  jitllm::runtime::Options Options(std::string_view extra = "") const {
    const fs::path config = path_ / "jitllm.toml";
    std::ofstream(config) << std::format("schema_version = 2\n{}[storage]\ndata_dir = \"{}\"\n",
                                         extra, (path_ / "data").string());
    return {.config = config, .config_given = true, .anchor = path_ / "enrollment", .help = false};
  }

 private:
  fs::path path_;
};

// Start()'s log, and its exit status if it refused.
struct Outcome {
  std::string log;
  int status = -1;  // -1: started
};

Outcome StartWith(const jitllm::runtime::Options& options) {
  char* buffer = nullptr;
  std::size_t size = 0;
  std::FILE* log = ::open_memstream(&buffer, &size);
  Outcome outcome;
  {
    auto started = jitllm::runtime::Start(options, log);
    if (!started) {
      outcome.status = started.error();
    }
  }
  (void)std::fclose(log);
  outcome.log.assign(buffer, size);
  std::free(buffer);  // NOLINT(cppcoreguidelines-no-malloc)
  return outcome;
}

TEST(RuntimeArguments, Parse) {
  const std::vector<std::string_view> args = {"--config", "a.toml", "--anchor", "/x/anchor"};
  auto options = jitllm::runtime::ParseArguments(args);
  ASSERT_TRUE(options.has_value()) << options.error();
  EXPECT_EQ(options->config, "a.toml");
  EXPECT_TRUE(options->config_given);
  EXPECT_EQ(options->anchor, "/x/anchor");
  auto defaults = jitllm::runtime::ParseArguments({});
  ASSERT_TRUE(defaults.has_value());
  EXPECT_EQ(defaults->config, "/etc/jitllm/jitllm.toml");
  EXPECT_FALSE(defaults->config_given);
  EXPECT_EQ(defaults->anchor, "/var/lib/jitllm/enrollment");
  for (const std::vector<std::string_view>& bad :
       {std::vector<std::string_view>{"--config"}, {"--anchor", ""}, {"serve"}}) {
    EXPECT_FALSE(jitllm::runtime::ParseArguments(bad).has_value());
  }
}

// A development host has no GB10, so a CUDA build refuses at the platform
// step there; the CPU-only build and a Spark get to readiness.
TEST(RuntimeStart, CreatesItsRolesAndTakesTheLock) {
  const Scratch scratch;
  const Outcome outcome = StartWith(scratch.Options());
  EXPECT_THAT(outcome.log,
              HasSubstr("storage: installed " + (scratch.path() / "data/models").string()));
  EXPECT_TRUE(fs::is_regular_file(scratch.path() / "data/spill/.jitllm-spill"));
  EXPECT_TRUE(fs::is_regular_file(scratch.path() / "enrollment.lock"));
  if (outcome.status != -1) {
    EXPECT_EQ(outcome.status, jitllm::runtime::kExitHostNotReady);
    EXPECT_THAT(outcome.log, HasSubstr("refusing to start: this host cannot run this build now"));
  }
}

TEST(RuntimeStart, RefusesWhileTheAnchorExists) {
  const Scratch scratch;
  const auto options = scratch.Options();
  std::ofstream(options.anchor) << "anchor";
  const Outcome outcome = StartWith(options);
  EXPECT_EQ(outcome.status, jitllm::runtime::kExitRefused);
  EXPECT_THAT(outcome.log,
              HasSubstr("the enrollment anchor " + options.anchor.string() + " exists"));
  EXPECT_FALSE(fs::exists(scratch.path() / "data"));
}

TEST(RuntimeStart, RefusesAnInvalidOrMemberConfiguration) {
  const Scratch scratch;
  Outcome outcome = StartWith(scratch.Options("bogus = 1\n"));
  EXPECT_EQ(outcome.status, jitllm::runtime::kExitRefused);
  EXPECT_THAT(outcome.log, HasSubstr(":2:9: unknown key bogus"));
  outcome = StartWith(scratch.Options(
      "cluster_file = \"/etc/jitllm/cluster.toml\"\nnode_id = "
      "\"af564a6b-8b4e-4528-8140-e50b92b40002\"\n"
      "credentials.ca_file = \"/c/ca.pem\"\ncredentials.certificate_file = \"/c/n.pem\"\n"
      "credentials.private_key_file = \"/c/k.pem\"\ncontrol.port = 7443\n"));
  EXPECT_EQ(outcome.status, jitllm::runtime::kExitRefused);
  EXPECT_THAT(outcome.log, HasSubstr("a cluster member's, and this build has no cluster support"));
  jitllm::runtime::Options missing = scratch.Options();
  missing.config = scratch.path() / "missing.toml";
  outcome = StartWith(missing);
  EXPECT_EQ(outcome.status, jitllm::runtime::kExitRefused);
  EXPECT_THAT(outcome.log, HasSubstr("missing.toml: does not exist"));
}

TEST(RuntimeStart, OneRuntimePerNode) {
  const Scratch scratch;
  const auto options = scratch.Options();
  fs::path lock_path = options.anchor;
  lock_path += ".lock";
  auto held = jitllm::platform::LockFile::Acquire(lock_path, ::geteuid());
  ASSERT_TRUE(held.has_value()) << held.error();
  const Outcome outcome = StartWith(options);
  EXPECT_EQ(outcome.status, jitllm::runtime::kExitRefused);
  EXPECT_THAT(outcome.log, HasSubstr("enrollment.lock is locked: another process holds it"));
  EXPECT_FALSE(fs::exists(scratch.path() / "data"));
}

TEST(LockFile, RefusesLinksAndOpenModes) {
  const Scratch scratch;
  std::ofstream(scratch.path() / "open.lock") << "";
  ASSERT_EQ(::chmod((scratch.path() / "open.lock").c_str(), 0644), 0);
  auto lock = jitllm::platform::LockFile::Acquire(scratch.path() / "open.lock", ::geteuid());
  ASSERT_FALSE(lock.has_value());
  EXPECT_THAT(lock.error(), HasSubstr("with mode 0600, not uid"));
  fs::create_symlink(scratch.path() / "elsewhere", scratch.path() / "link.lock");
  lock = jitllm::platform::LockFile::Acquire(scratch.path() / "link.lock", ::geteuid());
  ASSERT_FALSE(lock.has_value());
  EXPECT_THAT(lock.error(), HasSubstr("symbolic link"));
  EXPECT_FALSE(fs::exists(scratch.path() / "elsewhere"));
  // Released with its last descriptor.
  {
    auto first = jitllm::platform::LockFile::Acquire(scratch.path() / "a.lock", ::geteuid());
    ASSERT_TRUE(first.has_value()) << first.error();
  }
  EXPECT_TRUE(
      jitllm::platform::LockFile::Acquire(scratch.path() / "a.lock", ::geteuid()).has_value());
}

// Runs body in a child with the crash policy installed; returns its wait
// status.
template <typename Body>
int CrashChild(Body body) {
  (void)std::fflush(nullptr);
  const pid_t child = ::fork();
  if (child == 0) {
    if (!jitllm::platform::InstallCrashPolicy("crash-test")) {
      ::_exit(99);
    }
    body();
    ::_exit(98);
  }
  int status = 0;
  EXPECT_EQ(::waitpid(child, &status, 0), child);
  return status;
}

// D-014: a crash never becomes a core dump. The process ends with an exit
// status, so the kernel never starts one, whatever core_pattern says.
TEST(CrashPolicy, FatalSignalsExitWithoutACoreDump) {
  for (const int signal : {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGQUIT, SIGSYS, SIGTRAP}) {
    const int status = CrashChild([signal] { (void)std::raise(signal); });
    EXPECT_TRUE(WIFEXITED(status)) << signal << ": wait status " << status;
    EXPECT_EQ(WEXITSTATUS(status), jitllm::platform::kFatalSignalExitBase + signal) << signal;
  }
  const int aborted = CrashChild([] { std::abort(); });
  EXPECT_TRUE(WIFEXITED(aborted));
  EXPECT_EQ(WEXITSTATUS(aborted), jitllm::platform::kFatalSignalExitBase + SIGABRT);
}

TEST(CrashPolicy, StackOverflowStillExits) {
  const int status = CrashChild([] {
    // Recurses until the stack runs out; the flag keeps the compiler from
    // proving it never ends.
    struct Recurse {
      static int Deeper(int depth, const volatile bool& go_on) {
        std::array<volatile char, 4096> frame{};
        frame[0] = static_cast<char>(depth);
        return go_on ? Deeper(depth + 1, go_on) + frame[0] : 0;
      }
    };
    static volatile bool go_on = true;
    (void)Recurse::Deeper(0, go_on);
  });
  EXPECT_TRUE(WIFEXITED(status)) << "wait status " << status;
  EXPECT_EQ(WEXITSTATUS(status), jitllm::platform::kFatalSignalExitBase + SIGSEGV);
}

// A thread that installs its own signal stack is covered too.
TEST(CrashPolicy, AThreadsStackOverflowStillExits) {
  const int status = CrashChild([] {
    std::thread worker([] {
      if (!jitllm::platform::InstallThreadSignalStack()) {
        ::_exit(97);
      }
      struct Recurse {
        static int Deeper(int depth, const volatile bool& go_on) {
          std::array<volatile char, 4096> frame{};
          frame[0] = static_cast<char>(depth);
          return go_on ? Deeper(depth + 1, go_on) + frame[0] : 0;
        }
      };
      static volatile bool go_on = true;
      (void)Recurse::Deeper(0, go_on);
    });
    worker.join();
  });
  EXPECT_TRUE(WIFEXITED(status)) << "wait status " << status;
  EXPECT_EQ(WEXITSTATUS(status), jitllm::platform::kFatalSignalExitBase + SIGSEGV);
}

TEST(CrashPolicy, MarkingTwiceIsFine) {
  const int status = CrashChild([] {
    ::_exit(jitllm::platform::MarkNonDumpable() && jitllm::platform::MarkNonDumpable() ? 0 : 1);
  });
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(CrashPolicy, MarksTheProcessNonDumpable) {
  const int status = CrashChild([] {
    auto filter = jitllm::platform::ReadFirstLine("/proc/self/coredump_filter");
    const bool filtered = filter && std::strtoul(filter->c_str(), nullptr, 16) == 0;
    ::_exit(::prctl(PR_GET_DUMPABLE) == 0 && filtered ? 0 : 1);
  });
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(NotifyServiceManager, SendsToTheSocket) {
  const Scratch scratch;
  const fs::path socket_path = scratch.path() / "notify";
  const int fd = ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  ASSERT_GE(fd, 0);
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if (socket_path.string().size() >= sizeof(address.sun_path)) {
    (void)::close(fd);
    GTEST_SKIP() << "the scratch path is too long for a socket";
  }
  (void)socket_path.string().copy(address.sun_path, sizeof(address.sun_path) - 1);
  ASSERT_EQ(::bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)), 0);  // NOLINT
  ASSERT_EQ(::setenv("NOTIFY_SOCKET", socket_path.c_str(), 1), 0);  // NOLINT(concurrency-mt-unsafe)
  auto sent = jitllm::platform::NotifyServiceManager("READY=1");
  (void)::unsetenv("NOTIFY_SOCKET");  // NOLINT(concurrency-mt-unsafe)
  ASSERT_TRUE(sent.has_value()) << sent.error();
  EXPECT_TRUE(*sent);
  std::array<char, 64> received{};
  const ssize_t got = ::recv(fd, received.data(), received.size(), MSG_DONTWAIT);
  (void)::close(fd);
  EXPECT_EQ(std::string_view(received.data(), got > 0 ? static_cast<std::size_t>(got) : 0),
            "READY=1");
  auto unset = jitllm::platform::NotifyServiceManager("READY=1");
  ASSERT_TRUE(unset.has_value());
  EXPECT_FALSE(*unset);
}

}  // namespace
