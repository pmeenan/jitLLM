// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "cli/doctor.h"

#include <cstddef>
#include <filesystem>
#include <format>
#include <functional>
#include <string>
#include <string_view>

#include "base/build_info.h"
#include "base/report.h"
#include "platform/host_probe.h"
#include "providers/device_probe.h"

namespace jitllm::cli {
namespace {

std::string Count(std::size_t count, std::string_view noun) {
  return std::format("{} {}{}", count, noun, count == 1 ? "" : "s");
}

}  // namespace

void DescribeBuild(const base::BuildInfo& info, base::Report& report) {
  base::ReportSection& build = report.AddSection("build");
  build.Add("version", std::string(info.version));
  std::string commit = info.commit.empty() ? std::string("unknown") : std::string(info.commit);
  if (info.modified) {
    commit += " (with uncommitted changes)";
  }
  build.Add("commit", commit);
  build.Add("license profile", std::string(info.license_profile));
  build.Add("SDK", std::string(info.sdk));
  build.Add("target", std::string(info.target));
  build.Add("compiler",
            std::format("Clang {}.{}.{}", __clang_major__, __clang_minor__, __clang_patchlevel__));
  build.Add("C++ runtime",
            std::format("libstdc++ from GCC {}, linked statically (D-060)", _GLIBCXX_RELEASE));
}

bool Doctor(const std::filesystem::path& root, base::Report& report,
            const std::function<bool(std::string_view)>& write) {
  DescribeBuild(base::GetBuildInfo(), report);
  platform::DescribeHost(root, report);
  // Out before the driver is touched: a wedged GPU can hang cuInit.
  if (!write(SectionsText(report, 0))) {
    return false;
  }
  const std::size_t printed = report.sections.size();
  providers::DescribeDevices(root, report);
  return write(SectionsText(report, printed) + SummaryText(report));
}

std::string Printable(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (const char c : text) {
    const auto byte = static_cast<unsigned char>(c);
    if (byte < 0x20 || byte == 0x7f) {
      out += std::format("\\x{:02x}", byte);
    } else {
      out += c;
    }
  }
  return out;
}

std::string SectionsText(const base::Report& report, std::size_t first) {
  std::string text;
  for (std::size_t i = first; i < report.sections.size(); ++i) {
    const base::ReportSection& section = report.sections[i];
    text += Printable(section.title) + "\n";
    for (const base::ReportLine& line : section.lines) {
      text += std::format("  {}: {}\n", Printable(line.key), Printable(line.value));
    }
    text += "\n";
  }
  return text;
}

std::string SummaryText(const base::Report& report) {
  std::string text;
  for (const std::string& problem : report.problems) {
    text += std::format("problem: {}\n", Printable(problem));
  }
  for (const std::string& warning : report.warnings) {
    text += std::format("warning: {}\n", Printable(warning));
  }
  const std::string problems =
      report.problems.empty() ? "no problems" : Count(report.problems.size(), "problem");
  text += std::format("doctor: {}, {}\n", problems, Count(report.warnings.size(), "warning"));
  return text;
}

std::string DoctorText(const base::Report& report) {
  return SectionsText(report, 0) + SummaryText(report);
}

}  // namespace jitllm::cli
