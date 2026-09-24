// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Small reads of the kernel's text interfaces (/proc, /sys) and of
// directories, with errors as values (D-066).

#ifndef JITLLM_PLATFORM_FILES_H_
#define JITLLM_PLATFORM_FILES_H_

#include <cstddef>
#include <expected>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace jitllm::platform {

inline constexpr std::size_t kSmallFileLimit = std::size_t{64} * 1024;

// Reads a whole file of at most limit bytes, reading to its end rather
// than trusting its reported size, which /proc and /sys files misstate. A
// longer file is an error (std::errc::file_too_large).
std::expected<std::string, std::error_code> ReadSmallFile(const std::filesystem::path& path,
                                                          std::size_t limit = kSmallFileLimit);

// The first line of such a file, without its line ending or trailing
// whitespace.
std::expected<std::string, std::error_code> ReadFirstLine(const std::filesystem::path& path);

// The names in a directory, sorted bytewise.
std::expected<std::vector<std::string>, std::error_code> ListDirectory(
    const std::filesystem::path& path);

}  // namespace jitllm::platform

#endif  // JITLLM_PLATFORM_FILES_H_
