// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Whole reads over the storage provider (D-034, D-048;
// docs/async-model.md): a range of a file into memory, all of it or an
// explicit failure. The storage lane owns one reader over its provider.
//
//   - Alignment is checked before anything starts: memory, file offset and
//     length must be multiples of the direct-I/O alignment (4 KiB for
//     D-056's artifacts).
//   - A read is split into requests of at most `request_bytes`. A short
//     transfer continues from where it stopped; one that ends at the file's
//     end, or at an unaligned point, ends the read as end of file.
//   - Interrupted or retry-later errors are retried, a bounded number of
//     times per read; any other error fails the read.
//   - Duplicate reads coalesce: a read for a key already in flight, with
//     the same range and memory, adds a waiter (a bounded list) instead of
//     reading again.
//   - Withdrawing a waiter removes only its interest. When the last one
//     leaves, the read stops starting requests and asks the provider to
//     cancel those in flight, but it still drains: it ends only when every
//     request it started has completed, so its memory is then untouched.
//   - A request the provider could not start waits for the next Poll; one
//     whose start is unknown is waited for like any other.
// A read that fails or is cancelled has no usable contents: the caller
// publishes nothing from it.

#ifndef JITLLM_PROVIDERS_DIRECT_READER_H_
#define JITLLM_PROVIDERS_DIRECT_READER_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <string>
#include <vector>

#include "providers/storage.h"

namespace jitllm::providers {

struct ReaderSettings {
  std::uint32_t alignment = 4096;
  std::uint32_t request_bytes = 2U << 20U;  // the most one request moves
  std::uint32_t retries = 3;                // transient errors per read
  std::size_t reads = 64;                   // reads in flight at once
  std::size_t waiters = 8;                  // per read
};

struct ReadSpec {
  int fd = -1;
  std::uint64_t offset = 0;
  std::byte* memory = nullptr;
  std::uint64_t length = 0;
  bool operator==(const ReadSpec&) const = default;
};

enum class ReadError : std::uint8_t {
  kFull,            // as many reads in flight as allowed
  kUnaligned,       // memory, offset or length off the alignment
  kInvalidRange,    // the file or memory range cannot be represented
  kMismatch,        // the key is in flight for a different range or memory
  kTooManyWaiters,  // the read's waiter list is full
  kDraining,        // cancelled and still draining: retry once it ends
  kUnknownRead,     // no such read, or not a waiter of it
};

std::string ToString(ReadError error);

enum class ReadOutcome : std::uint8_t {
  kComplete,
  kEndOfFile,  // the file ended before the range did
  kFailed,     // an I/O error; `error` holds its errno
  kCancelled,  // every waiter withdrew
};

struct FinishedRead {
  std::uint64_t key = 0;
  ReadOutcome outcome = ReadOutcome::kComplete;
  std::uint64_t bytes = 0;  // transferred into memory
  int error = 0;
  std::vector<std::uint64_t> waiters;
};

class DirectReader {
 public:
  // `settings` must be sane: a non-zero alignment, and requests a non-zero
  // multiple of it, with non-zero read and waiter limits (a configuration
  // error is fatal). Accepted requests must be drained before destruction.
  DirectReader(Storage& storage, ReaderSettings settings);
  ~DirectReader();

  DirectReader(const DirectReader&) = delete;
  DirectReader& operator=(const DirectReader&) = delete;
  DirectReader(DirectReader&&) = delete;
  DirectReader& operator=(DirectReader&&) = delete;

  // Starts reading `spec` for `key`, or joins the read in flight for it.
  // True if it joined. Repeating the same waiter is idempotent.
  std::expected<bool, ReadError> Read(std::uint64_t key, const ReadSpec& spec,
                                      std::uint64_t waiter);
  std::expected<void, ReadError> Withdraw(std::uint64_t key, std::uint64_t waiter);

  // Starts what can start, harvests completions (waiting for one, with
  // `wait`, if anything is in flight), and returns the reads that ended.
  std::vector<FinishedRead> Poll(bool wait);

  std::size_t reads() const { return reads_.size(); }

 private:
  struct Piece {
    std::uint64_t start = 0;  // within the read
    std::uint64_t length = 0;
    std::uint64_t done = 0;
    bool in_flight = false;
    std::uint64_t token = 0;
  };
  struct Reading {
    ReadSpec spec;
    std::vector<std::uint64_t> waiters;
    std::vector<Piece> pieces;
    std::uint32_t retries = 0;
    bool stopping = false;  // failed, cancelled or at end of file: start nothing more
    ReadOutcome outcome = ReadOutcome::kComplete;
    int error = 0;
  };

  void Start(std::uint64_t key, Reading& reading);
  static void Stop(Reading& reading, ReadOutcome outcome, int error);
  static bool Finished(const Reading& reading);

  Storage& storage_;
  ReaderSettings settings_;
  std::map<std::uint64_t, Reading> reads_;  // by key
  // token -> (key, piece)
  std::map<std::uint64_t, std::pair<std::uint64_t, std::size_t>> tokens_;
  std::uint64_t next_token_ = 1;
};

}  // namespace jitllm::providers

#endif  // JITLLM_PROVIDERS_DIRECT_READER_H_
