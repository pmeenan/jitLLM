// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// `jitllm doctor`: the capability probe (D-026, D-063). It reports what this
// binary is, the host, the device driver and devices, and RDMA, and fails
// if this host cannot run this build as designed. It only reads and
// queries: it writes no file and allocates no device memory (the driver's
// own initialization may load its kernel modules; cuda_probe.h).

#ifndef JITLLM_CLI_DOCTOR_H_
#define JITLLM_CLI_DOCTOR_H_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

#include "base/build_info.h"
#include "base/report.h"

namespace jitllm::cli {

// Adds the `build` section: the version, commit, license profile, SDK and
// target, and the compiler and C++ runtime that built this binary.
void DescribeBuild(const base::BuildInfo& info, base::Report& report);

// Runs every probe into report, reading /proc, /sys and /dev beneath root
// ("/"), and hands write() the report text in two parts: the build and
// host sections before the device probe starts, so a driver that hangs
// still leaves them, and then the rest. False if a write fails.
bool Doctor(const std::filesystem::path& root, base::Report& report,
            const std::function<bool(std::string_view)>& write);

// The report as `jitllm doctor` prints it: each section's facts, then the
// problems and warnings, then a summary line. Control characters in the
// report's text are escaped (\xNN), so a value cannot forge a line.
std::string DoctorText(const base::Report& report);

// DoctorText()'s parts: the sections from `first` on, and the rest.
std::string SectionsText(const base::Report& report, std::size_t first);
std::string SummaryText(const base::Report& report);

// text with each control character written as \xNN.
std::string Printable(std::string_view text);

}  // namespace jitllm::cli

#endif  // JITLLM_CLI_DOCTOR_H_
