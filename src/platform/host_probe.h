// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The Linux half of the capability probe that `jitllm doctor` reports
// (D-026, D-063): the kernel, glibc, memory totals, the hard-link
// protection that D-063's checkpoint store relies on, and RDMA devices with
// this user's access to them. It only reads; it changes nothing on the host.
//
// Files under /proc, /sys and /dev are read beneath `root`, which is "/"
// for this host; tests pass a fake tree. The kernel release, glibc version
// and page size always come from the running process.

#ifndef JITLLM_PLATFORM_HOST_PROBE_H_
#define JITLLM_PLATFORM_HOST_PROBE_H_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "base/report.h"

namespace jitllm::platform {

struct KernelModule {
  bool loaded = false;
  // What /sys/module/<name>/version says, or empty.
  std::string version;
};

// Whether a kernel module (or a built-in one with parameters) is present.
KernelModule FindKernelModule(const std::filesystem::path& root, std::string_view name);

// A /proc/meminfo value in bytes, such as MemTotal's.
std::optional<std::uint64_t> MeminfoBytes(std::string_view meminfo, std::string_view key);

// Adds the `host` and `RDMA` sections, with their warnings, to report.
void DescribeHost(const std::filesystem::path& root, base::Report& report);

}  // namespace jitllm::platform

#endif  // JITLLM_PLATFORM_HOST_PROBE_H_
