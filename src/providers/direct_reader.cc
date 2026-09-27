// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/direct_reader.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "base/check.h"
#include "providers/storage.h"

namespace jitllm::providers {

std::string ToString(ReadError error) {
  switch (error) {
    case ReadError::kFull:
      return "as many reads are in flight as allowed";
    case ReadError::kUnaligned:
      return "the memory, offset or length is not aligned for direct I/O";
    case ReadError::kInvalidRange:
      return "the file or memory range cannot be represented";
    case ReadError::kMismatch:
      return "the key is being read into a different range or memory";
    case ReadError::kTooManyWaiters:
      return "the read has as many waiters as allowed";
    case ReadError::kDraining:
      return "the read was cancelled and is still draining";
    case ReadError::kUnknownRead:
      return "no such read or waiter";
  }
  return "unknown read error";
}

DirectReader::DirectReader(Storage& storage, ReaderSettings settings)
    : storage_(storage), settings_(settings) {
  base::Check(settings.alignment > 0 && settings.request_bytes > 0 &&
                  settings.request_bytes % settings.alignment == 0 && settings.reads > 0 &&
                  settings.waiters > 0,
              "direct reads need aligned non-zero requests and non-zero read and waiter limits");
}

DirectReader::~DirectReader() {
  base::Check(tokens_.empty(), "a direct reader destroyed with requests in flight");
}

std::expected<bool, ReadError> DirectReader::Read(std::uint64_t key, const ReadSpec& spec,
                                                  std::uint64_t waiter) {
  if (const auto found = reads_.find(key); found != reads_.end()) {
    Reading& reading = found->second;
    if (reading.spec != spec) {
      return std::unexpected(ReadError::kMismatch);
    }
    if (reading.stopping && reading.waiters.empty()) {
      return std::unexpected(ReadError::kDraining);
    }
    if (std::ranges::find(reading.waiters, waiter) != reading.waiters.end()) {
      return true;
    }
    if (reading.waiters.size() >= settings_.waiters) {
      return std::unexpected(ReadError::kTooManyWaiters);
    }
    reading.waiters.push_back(waiter);
    return true;
  }
  const auto address = reinterpret_cast<std::uintptr_t>(spec.memory);
  if (spec.memory == nullptr || spec.length == 0 || address % settings_.alignment != 0 ||
      spec.offset % settings_.alignment != 0 || spec.length % settings_.alignment != 0) {
    return std::unexpected(ReadError::kUnaligned);
  }
  if (spec.length > std::numeric_limits<std::uint64_t>::max() - spec.offset ||
      spec.length > std::numeric_limits<std::uintptr_t>::max() - address ||
      spec.length > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    return std::unexpected(ReadError::kInvalidRange);
  }
  if (reads_.size() >= settings_.reads) {
    return std::unexpected(ReadError::kFull);
  }
  Reading reading;
  reading.spec = spec;
  reading.arrival = next_arrival_++;
  reading.waiters.push_back(waiter);
  for (std::uint64_t start = 0; start < spec.length; start += settings_.request_bytes) {
    reading.pieces.push_back(
        Piece{.start = start,
              .length = std::min<std::uint64_t>(settings_.request_bytes, spec.length - start),
              .done = 0,
              .in_flight = false,
              .token = 0});
  }
  Queue(key, reads_.emplace(key, std::move(reading)).first->second);
  StartQueued();  // behind any older read still waiting for room
  return false;
}

std::expected<void, ReadError> DirectReader::Withdraw(std::uint64_t key, std::uint64_t waiter) {
  const auto found = reads_.find(key);
  if (found == reads_.end()) {
    return std::unexpected(ReadError::kUnknownRead);
  }
  Reading& reading = found->second;
  const auto position = std::ranges::find(reading.waiters, waiter);
  if (position == reading.waiters.end()) {
    return std::unexpected(ReadError::kUnknownRead);
  }
  reading.waiters.erase(position);
  if (reading.waiters.empty() && !reading.stopping) {
    Stop(reading, ReadOutcome::kCancelled, 0);
    for (const Piece& piece : reading.pieces) {
      if (piece.in_flight) {
        (void)storage_.Cancel(piece.token);  // best effort: it still completes
      }
    }
    settled_.push_back(key);  // with nothing in flight, it has ended
  }
  return {};
}

void DirectReader::Stop(Reading& reading, ReadOutcome outcome, int error) {
  if (!reading.stopping) {
    reading.stopping = true;
    reading.outcome = outcome;
    reading.error = error;
  }
}

void DirectReader::Queue(std::uint64_t key, Reading& reading) {
  if (!reading.queued && !reading.stopping) {
    reading.queued = true;
    queue_.emplace(reading.arrival, key);
  }
}

void DirectReader::StartQueued() {
  while (!queue_.empty()) {
    const auto oldest = queue_.begin();
    Reading& reading = reads_.at(oldest->second);
    if (!Start(oldest->second, reading)) {
      return;  // the provider is full: the rest wait, in order
    }
    reading.queued = false;
    queue_.erase(oldest);
  }
}

bool DirectReader::Start(std::uint64_t key, Reading& reading) {
  if (reading.stopping) {
    return true;
  }
  for (std::size_t i = 0; i < reading.pieces.size(); ++i) {
    Piece& piece = reading.pieces[i];
    if (piece.in_flight || piece.done == piece.length) {
      continue;
    }
    const std::uint64_t token = next_token_++;
    const IoRequest request{
        .token = token,
        .kind = IoKind::kRead,
        .fd = reading.spec.fd,
        .offset = reading.spec.offset + piece.start + piece.done,
        .memory = reading.spec.memory + piece.start +
                  piece.done,  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        .length = static_cast<std::uint32_t>(piece.length - piece.done),
    };
    if (storage_.Submit(request) == Submission::kNotStarted) {
      return false;  // the provider is full: the next Poll tries again
    }
    // Accepted or unknown: either way a completion is owed.
    piece.in_flight = true;
    piece.token = token;
    tokens_.emplace(token, std::pair(key, i));
  }
  return true;
}

bool DirectReader::Finished(const Reading& reading) {
  return std::ranges::all_of(reading.pieces, [&reading](const Piece& piece) {
    return !piece.in_flight && (reading.stopping || piece.done == piece.length);
  });
}

std::vector<FinishedRead> DirectReader::Poll(bool wait) {
  StartQueued();
  std::array<IoCompletion, 64> completions{};
  std::size_t harvested = storage_.Harvest(completions, wait);
  while (harvested > 0) {
    for (std::size_t c = 0; c < harvested; ++c) {
      const IoCompletion& completion = completions[c];
      const auto token = tokens_.find(completion.token);
      if (token == tokens_.end()) {
        continue;  // not ours; the provider reports only what it accepted
      }
      const auto [key, index] = token->second;
      tokens_.erase(token);
      Reading& reading = reads_.at(key);
      Piece& piece = reading.pieces[index];
      piece.in_flight = false;
      settled_.push_back(key);
      const std::int64_t result = completion.result;
      if (result < 0) {
        const int error = static_cast<int>(-result);
        if (error == ECANCELED) {
          Stop(reading, ReadOutcome::kCancelled, 0);
        } else if ((error == EINTR || error == EAGAIN) && reading.retries < settings_.retries) {
          ++reading.retries;  // transient: started again below, ahead of later reads
          Queue(key, reading);
        } else {
          Stop(reading, ReadOutcome::kFailed, error);
        }
        continue;
      }
      const auto transferred = static_cast<std::uint64_t>(result);
      piece.done += std::min(transferred, piece.length - piece.done);
      const bool aligned = (piece.done % settings_.alignment) == 0;
      if (piece.done < piece.length && (transferred == 0 || !aligned)) {
        Stop(reading, ReadOutcome::kEndOfFile, 0);  // the file ended here
      } else if (piece.done < piece.length) {
        Queue(key, reading);  // a short transfer continues, ahead of later reads
      }
    }
    harvested = storage_.Harvest(completions, false);
  }
  StartQueued();
  std::vector<FinishedRead> finished;
  for (const std::uint64_t key : settled_) {
    const auto it = reads_.find(key);
    if (it == reads_.end() || !Finished(it->second)) {
      continue;  // ended already (listed twice), or still reading
    }
    Reading& reading = it->second;
    if (reading.queued) {
      queue_.erase(reading.arrival);  // stopping: nothing more will start
    }
    std::uint64_t bytes = 0;
    for (const Piece& piece : reading.pieces) {
      bytes += piece.done;
    }
    finished.push_back(
        FinishedRead{.key = it->first,
                     .outcome = reading.stopping ? reading.outcome : ReadOutcome::kComplete,
                     .bytes = bytes,
                     .error = reading.error,
                     .waiters = std::move(reading.waiters)});
    reads_.erase(it);
  }
  settled_.clear();
  return finished;
}

}  // namespace jitllm::providers
