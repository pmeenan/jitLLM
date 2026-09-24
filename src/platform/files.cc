// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/files.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace jitllm::platform {
namespace {

std::error_code LastError() { return {errno, std::generic_category()}; }

}  // namespace

std::expected<std::string, std::error_code> ReadSmallFile(const std::filesystem::path& path,
                                                          std::size_t limit) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOCTTY | O_NONBLOCK);
  if (fd < 0) {
    return std::unexpected(LastError());
  }
  std::string text;
  std::error_code error;
  // One byte past the limit tells a file of exactly `limit` bytes from a longer one.
  text.resize(limit + 1);
  std::size_t size = 0;
  while (size < text.size()) {
    const ssize_t got = ::read(fd, text.data() + size, text.size() - size);
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      error = LastError();
      break;
    }
    if (got == 0) {
      break;
    }
    size += static_cast<std::size_t>(got);
  }
  (void)::close(fd);  // read-only: nothing to lose
  if (error) {
    return std::unexpected(error);
  }
  if (size > limit) {
    return std::unexpected(std::make_error_code(std::errc::file_too_large));
  }
  text.resize(size);
  return text;
}

std::expected<std::string, std::error_code> ReadFirstLine(const std::filesystem::path& path) {
  auto text = ReadSmallFile(path);
  if (!text) {
    return text;
  }
  std::string line = text->substr(0, text->find('\n'));
  const std::size_t end = line.find_last_not_of(" \t\r\n");
  line.resize(end == std::string::npos ? 0 : end + 1);
  return line;
}

std::expected<std::vector<std::string>, std::error_code> ListDirectory(
    const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::directory_iterator it(path, error);
  std::vector<std::string> names;
  for (const std::filesystem::directory_iterator end; !error && it != end; it.increment(error)) {
    names.push_back(it->path().filename().string());
  }
  if (error) {
    return std::unexpected(error);
  }
  std::ranges::sort(names);
  return names;
}

}  // namespace jitllm::platform
