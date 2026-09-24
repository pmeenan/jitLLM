// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The toolchain contract, checked in a binary the build produced: C++23 with
// the GCC 16.2 library (D-032, D-060), Clang 22, no exceptions (D-066), and
// the profile's explicit CPU baseline (D-011). Runs in every profile, cross
// builds included (under qemu-user or on a Spark).

#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <expected>
#include <flat_map>
#include <mdspan>
#include <print>
#include <ranges>
#include <vector>

static_assert(__cplusplus >= 202302L, "C++23 throughout (D-032)");
#if !defined(__clang__) || __clang_major__ != 22
#error "jitLLM code compiles with the SDK's Clang 22 (D-032)"
#endif
#if __GLIBCXX__ != 20260807
#error "jitLLM uses the GCC 16.2.0 C++ runtime (D-060)"
#endif
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
#error "jitLLM builds with -fno-exceptions (D-066)"
#endif

// The target is the profile's, at its explicit baseline: nothing that
// -march=native on a build host would add (D-011).
#ifdef JITLLM_TEST_TARGET_x86_64
#if !defined(__x86_64__) || defined(__SSE3__) || defined(__AVX__)
#error "x86-64 builds target the x86-64 baseline"
#endif
#elifdef JITLLM_TEST_TARGET_aarch64
#if !defined(__aarch64__) || defined(__ARM_FEATURE_ATOMICS) || defined(__ARM_FEATURE_SVE)
#error "AArch64 builds target armv8-a"
#endif
#else
#error "unknown target: define JITLLM_TEST_TARGET_<arch>"
#endif

namespace {

class Checks {
 public:
  void Expect(bool ok, const char* what) {
    if (!ok) {
      std::println(stderr, "FAIL: {}", what);
      ++failures_;
    }
  }
  [[nodiscard]] int failures() const { return failures_; }

 private:
  int failures_ = 0;
};

void CheckLibrary(Checks& checks) {
  const std::flat_map<int, int> map{{2, 20}, {1, 10}};
  checks.Expect(map.begin()->second == 10, "std::flat_map");

  std::vector<int> values{1, 2, 3, 4, 5, 6};
  const std::mdspan grid(values.data(), 2, 3);
  checks.Expect(grid[1, 2] == 6, "std::mdspan");

  auto half = [](int x) -> std::expected<int, int> {
    if (x % 2 != 0) {
      return std::unexpected(x);
    }
    return x / 2;
  };
  checks.Expect(half(4).value_or(0) == 2 && !half(3).has_value(), "std::expected");

  const auto evens = values | std::views::filter([](int x) { return x % 2 == 0; }) |
                     std::ranges::to<std::vector>();
  checks.Expect(evens.size() == 3, "std::ranges::to");
}

// Under -fno-exceptions a library path that would throw ends the process
// (D-066). The child makes itself non-dumpable first, so the abort leaves no
// core file or crash report.
void CheckThrowPathTerminates(Checks& checks) {
  const pid_t child = fork();
  if (child == 0) {
    const rlimit no_core{.rlim_cur = 0, .rlim_max = 0};
    setrlimit(RLIMIT_CORE, &no_core);
    prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);
    if (std::freopen("/dev/null", "w", stderr) == nullptr) {
      _exit(2);
    }
    const std::vector<int> empty;
    std::println("{}", empty.at(1));  // Out of range: throws, which terminates.
    _exit(0);
  }
  int status = 0;
  const bool waited = child > 0 && waitpid(child, &status, 0) == child;
  checks.Expect(waited && WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
                "a throwing library path aborts the process");
}

}  // namespace

int main() {
  Checks checks;
  CheckLibrary(checks);
  CheckThrowPathTerminates(checks);
  if (checks.failures() != 0) {
    return 1;
  }
  std::println("toolchain contract: C++ {}, libstdc++ {}, Clang {}, no exceptions", __cplusplus,
               __GLIBCXX__, __clang_version__);
  return 0;
}
