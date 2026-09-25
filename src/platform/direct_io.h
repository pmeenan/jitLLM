// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// What the storage roles that the runtime pages from and spills to must
// offer (D-034, D-054, D-055): a local block-device filesystem, writable,
// on which direct I/O works at the 4 KiB alignment of D-056's artifacts.

#ifndef JITLLM_PLATFORM_DIRECT_IO_H_
#define JITLLM_PLATFORM_DIRECT_IO_H_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>

namespace jitllm::platform {

// The largest direct-I/O alignment the storage roles may require: D-056
// aligns artifact data to 4 KiB.
inline constexpr std::uint32_t kDirectIoAlignment = 4096;

struct FilesystemFacts {
  // The kernel's name for it where known ("ext4"), else its magic number.
  std::string type;
  // One of the local block-device filesystems the roles accept: ext4, XFS
  // or Btrfs. Network, FUSE, overlay and memory-backed ones are not.
  bool accepted = false;
  bool read_only = false;
};

// The filesystem holding path, which must exist.
std::expected<FilesystemFacts, std::string> DescribeFilesystem(const std::filesystem::path& path);

struct DirectIoFacts {
  // What statx reports for a file there (0 if the kernel does not say):
  // the alignment direct I/O needs for memory and for file offsets.
  std::uint32_t memory_alignment = 0;
  std::uint32_t offset_alignment = 0;
};

// D-034's probe: creates an unnamed file in directory (O_TMPFILE, so
// nothing is left behind), opened for direct I/O, then writes and reads
// back one 4 KiB-aligned block. Fails if the filesystem is not accepted,
// if direct I/O is refused or needs more than 4 KiB alignment, or if the
// data does not come back.
std::expected<DirectIoFacts, std::string> ProbeDirectIo(const std::filesystem::path& directory);

}  // namespace jitllm::platform

#endif  // JITLLM_PLATFORM_DIRECT_IO_H_
