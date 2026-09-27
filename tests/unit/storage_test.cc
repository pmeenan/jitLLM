// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The storage provider and whole reads over it (D-034, D-048): short
// transfers, alignment, retries, cancellation that drains, and coalesced
// duplicate reads, on the scripted fake; and the io_uring provider
// against a real direct-I/O file and a pipe whose read never completes
// (skipped where there is no io_uring, as under qemu-user or a container
// policy that denies the syscall).

#include "providers/storage.h"

#include <fcntl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "expected_error.h"
#include "platform/direct_io.h"
#include "providers/direct_reader.h"
#include "providers/fake/fake_storage.h"
#include "providers/uring_storage.h"

#ifdef __SANITIZE_THREAD__
#define JITLLM_TEST_TSAN 1
#elifdef __has_feature
#if __has_feature(thread_sanitizer)
#define JITLLM_TEST_TSAN 1
#endif
#endif

namespace {

using jitllm::providers::DirectReader;
using jitllm::providers::FinishedRead;
using jitllm::providers::IoCompletion;
using jitllm::providers::IoKind;
using jitllm::providers::IoRequest;
using jitllm::providers::ReadError;
using jitllm::providers::ReaderSettings;
using jitllm::providers::ReadOutcome;
using jitllm::providers::ReadSpec;
using jitllm::providers::Submission;
using jitllm::providers::UringStorage;
using jitllm::providers::fake::FakeStorage;
using jitllm::test_support::Failed;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

constexpr std::uint64_t kAlignment = 4096;

std::vector<std::byte> Pattern(std::size_t size) {
  std::vector<std::byte> bytes(size);
  for (std::size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<std::byte>((i * 7) + (i >> 12));
  }
  return bytes;
}

// Aligned memory for direct I/O.
struct Buffer {
  explicit Buffer(std::size_t size) : size(size) {
    data = static_cast<std::byte*>(std::aligned_alloc(kAlignment, size));
    std::memset(data, 0xee, size);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&&) = delete;
  Buffer& operator=(Buffer&&) = delete;
  ~Buffer() { std::free(data); }  // NOLINT(cppcoreguidelines-no-malloc)
  std::byte* data;
  std::size_t size;
};

ReaderSettings Settings() {
  return ReaderSettings{.alignment = static_cast<std::uint32_t>(kAlignment),
                        .request_bytes = static_cast<std::uint32_t>(4 * kAlignment),
                        .retries = 2,
                        .reads = 4,
                        .waiters = 2};
}

std::vector<FinishedRead> PollUntilDone(DirectReader& reader) {
  std::vector<FinishedRead> all;
  for (int i = 0; i < 16 && reader.reads() > 0; ++i) {
    for (FinishedRead& finished : reader.Poll(false)) {
      all.push_back(std::move(finished));
    }
  }
  return all;
}

class ReaderTest : public ::testing::Test {
 protected:
  FakeStorage storage_{8, static_cast<std::uint32_t>(kAlignment)};
  DirectReader reader_{storage_, Settings()};
  std::vector<std::byte> contents_ = Pattern(10 * kAlignment);
  int fd_ = storage_.AddFile(contents_);
  Buffer buffer_{10 * kAlignment};
};

TEST_F(ReaderTest, AWholeReadSplitsAndCompletes) {
  const ReadSpec spec{.fd = fd_, .offset = 0, .memory = buffer_.data, .length = 8 * kAlignment};
  EXPECT_FALSE(reader_.Read(1, spec, 100).value());
  const auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kComplete);
  EXPECT_EQ(finished[0].bytes, 8 * kAlignment);
  EXPECT_THAT(finished[0].waiters, ElementsAre(100));
  EXPECT_EQ(std::memcmp(buffer_.data, contents_.data(), 8 * kAlignment), 0);
  EXPECT_EQ(storage_.submitted().size(), 2U);  // two requests of four blocks
}

TEST_F(ReaderTest, ShortTransfersContinueWhereTheyStopped) {
  storage_.ScriptNext({.submission = Submission::kAccepted,
                       .result = static_cast<std::int64_t>(kAlignment),
                       .hold = false});
  const ReadSpec spec{
      .fd = fd_, .offset = kAlignment, .memory = buffer_.data, .length = 4 * kAlignment};
  ASSERT_TRUE(reader_.Read(1, spec, 100).has_value());
  const auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kComplete);
  ASSERT_EQ(storage_.submitted().size(), 2U);
  EXPECT_EQ(storage_.submitted()[1].offset, 2 * kAlignment);  // the remainder
  EXPECT_EQ(storage_.submitted()[1].length, 3 * kAlignment);
  EXPECT_EQ(std::memcmp(buffer_.data, contents_.data() + kAlignment, 4 * kAlignment), 0);
}

TEST_F(ReaderTest, TheEndOfTheFileEndsTheRead) {
  const ReadSpec spec{
      .fd = fd_, .offset = 8 * kAlignment, .memory = buffer_.data, .length = 4 * kAlignment};
  ASSERT_TRUE(reader_.Read(1, spec, 100).has_value());
  const auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kEndOfFile);
  EXPECT_EQ(finished[0].bytes, 2 * kAlignment);
}

TEST_F(ReaderTest, TransientErrorsRetryAndOthersFailAfterDraining) {
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = -EINTR, .hold = false});
  const ReadSpec spec{.fd = fd_, .offset = 0, .memory = buffer_.data, .length = 4 * kAlignment};
  ASSERT_TRUE(reader_.Read(1, spec, 100).has_value());
  auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kComplete);

  // A hard error on one request fails the read, but only once the other
  // request in flight has completed.
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = -EIO, .hold = false});
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = std::nullopt, .hold = true});
  const ReadSpec two{.fd = fd_, .offset = 0, .memory = buffer_.data, .length = 8 * kAlignment};
  ASSERT_TRUE(reader_.Read(2, two, 100).has_value());
  EXPECT_THAT(reader_.Poll(false), IsEmpty());  // the held request still owns its memory
  ASSERT_TRUE(storage_.Release(storage_.submitted().back().token));
  finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kFailed);
  EXPECT_EQ(finished[0].error, EIO);
}

TEST_F(ReaderTest, DuplicatesCoalesceAndTheLastWaiterCancels) {
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = std::nullopt, .hold = true});
  const ReadSpec spec{.fd = fd_, .offset = 0, .memory = buffer_.data, .length = 4 * kAlignment};
  EXPECT_FALSE(reader_.Read(1, spec, 100).value());
  EXPECT_TRUE(reader_.Read(1, spec, 100).value());  // repeating an interest is idempotent
  EXPECT_TRUE(reader_.Read(1, spec, 101).value());  // joined: no second request
  EXPECT_EQ(Failed(reader_.Read(1, spec, 102)), ReadError::kTooManyWaiters);
  ReadSpec other = spec;
  other.offset = kAlignment;
  EXPECT_EQ(Failed(reader_.Read(1, other, 103)), ReadError::kMismatch);
  EXPECT_EQ(storage_.submitted().size(), 1U);
  ASSERT_TRUE(reader_.Withdraw(1, 100).has_value());  // one waiter leaves: the read goes on
  EXPECT_EQ(storage_.in_flight(), 1U);
  ASSERT_TRUE(reader_.Withdraw(1, 101).has_value());  // the last: cancelled, draining
  EXPECT_EQ(Failed(reader_.Read(1, spec, 104)), ReadError::kDraining);
  const auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kCancelled);
  EXPECT_THAT(finished[0].waiters, IsEmpty());
  EXPECT_EQ(storage_.in_flight(), 0U);
  EXPECT_EQ(Failed(reader_.Withdraw(1, 101)), ReadError::kUnknownRead);
}

TEST_F(ReaderTest, AlignmentIsCheckedAndAFullProviderWaits) {
  const ReadSpec unaligned{
      .fd = fd_, .offset = 512, .memory = buffer_.data, .length = 4 * kAlignment};
  EXPECT_EQ(Failed(reader_.Read(1, unaligned, 100)), ReadError::kUnaligned);
  const ReadSpec odd{
      .fd = fd_, .offset = 0, .memory = buffer_.data + 512, .length = 4 * kAlignment};
  EXPECT_EQ(Failed(reader_.Read(1, odd, 100)), ReadError::kUnaligned);
  // The provider refuses the first attempt; the next poll starts it.
  storage_.ScriptNext(
      {.submission = Submission::kNotStarted, .result = std::nullopt, .hold = false});
  const ReadSpec spec{.fd = fd_, .offset = 0, .memory = buffer_.data, .length = 4 * kAlignment};
  ASSERT_TRUE(reader_.Read(1, spec, 100).has_value());
  EXPECT_TRUE(storage_.submitted().empty());
  const auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kComplete);
}

TEST(ReaderOrderTest, ReadsStartInArrivalOrderAndContinuationsGoFirst) {
  // One request at a time, so every read after the first waits for room.
  FakeStorage storage{1, static_cast<std::uint32_t>(kAlignment)};
  DirectReader reader{storage, Settings()};
  const int fd = storage.AddFile(Pattern(16 * kAlignment));
  Buffer buffer{16 * kAlignment};
  const auto spec = [&](std::uint64_t block) {
    return ReadSpec{.fd = fd,
                    .offset = block * kAlignment,
                    .memory = buffer.data + (block * kAlignment),
                    .length = 2 * kAlignment};
  };
  // The first read's request moves only one block: its remainder must
  // start before the later reads.
  storage.ScriptNext({.submission = Submission::kAccepted,
                      .result = static_cast<std::int64_t>(kAlignment),
                      .hold = false});
  // Keys in an order unlike their arrival (the storage lane's keys are
  // mailbox indices, which are reused out of order).
  ASSERT_TRUE(reader.Read(9, spec(0), 100).has_value());
  ASSERT_TRUE(reader.Read(2, spec(2), 100).has_value());
  ASSERT_TRUE(reader.Read(7, spec(4), 100).has_value());
  ASSERT_TRUE(reader.Read(4, spec(6), 100).has_value());
  const auto finished = PollUntilDone(reader);
  ASSERT_EQ(finished.size(), 4U);
  std::vector<std::uint64_t> offsets;
  for (const auto& request : storage.submitted()) {
    offsets.push_back(request.offset / kAlignment);
  }
  EXPECT_THAT(offsets, ElementsAre(0, 1, 2, 4, 6));
  std::vector<std::uint64_t> keys;
  for (const auto& read : finished) {
    EXPECT_EQ(read.outcome, ReadOutcome::kComplete);
    keys.push_back(read.key);
  }
  EXPECT_THAT(keys, ElementsAre(9, 2, 7, 4));
}

TEST(ReaderOrderTest, AWithdrawnReadThatNeverStartedEndsWithoutHoldingUpTheOthers) {
  // One request at a time, and the first read's is held: the two later
  // reads wait, in order, and nothing of them has started.
  FakeStorage storage{1, static_cast<std::uint32_t>(kAlignment)};
  DirectReader reader{storage, Settings()};
  const int fd = storage.AddFile(Pattern(16 * kAlignment));
  Buffer buffer{16 * kAlignment};
  const auto spec = [&](std::uint64_t block) {
    return ReadSpec{.fd = fd,
                    .offset = block * kAlignment,
                    .memory = buffer.data + (block * kAlignment),
                    .length = 2 * kAlignment};
  };
  storage.ScriptNext({.submission = Submission::kAccepted, .result = std::nullopt, .hold = true});
  ASSERT_TRUE(reader.Read(1, spec(0), 100).has_value());
  ASSERT_TRUE(reader.Read(2, spec(2), 100).has_value());
  ASSERT_TRUE(reader.Read(3, spec(4), 100).has_value());
  ASSERT_EQ(storage.submitted().size(), 1U);
  // The last read, queued behind one that cannot start, is withdrawn: with
  // nothing in flight it ends at the next poll, and leaves the queue.
  ASSERT_TRUE(reader.Withdraw(3, 100).has_value());
  auto finished = reader.Poll(false);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].key, 3U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kCancelled);
  EXPECT_EQ(finished[0].bytes, 0U);
  EXPECT_EQ(storage.submitted().size(), 1U);
  ASSERT_TRUE(storage.Release(storage.submitted().back().token));
  finished = PollUntilDone(reader);
  std::vector<std::uint64_t> keys;
  for (const auto& read : finished) {
    EXPECT_EQ(read.outcome, ReadOutcome::kComplete);
    keys.push_back(read.key);
  }
  EXPECT_THAT(keys, ElementsAre(1, 2));
  std::vector<std::uint64_t> offsets;
  for (const auto& request : storage.submitted()) {
    offsets.push_back(request.offset / kAlignment);
  }
  EXPECT_THAT(offsets, ElementsAre(0, 2));
  EXPECT_EQ(reader.reads(), 0U);
}

TEST_F(ReaderTest, OverflowingRangesAreRefusedBeforeAnyIo) {
  const ReadSpec wrapped_file{
      .fd = fd_,
      .offset = std::numeric_limits<std::uint64_t>::max() - (kAlignment - 1),
      .memory = buffer_.data,
      .length = 2 * kAlignment};
  EXPECT_EQ(Failed(reader_.Read(1, wrapped_file, 100)), ReadError::kInvalidRange);
  const ReadSpec wrapped_memory{
      .fd = fd_,
      .offset = 0,
      .memory = reinterpret_cast<std::byte*>(  // NOLINT(performance-no-int-to-ptr)
          std::numeric_limits<std::uintptr_t>::max() - (kAlignment - 1)),
      .length = 2 * kAlignment};
  EXPECT_EQ(Failed(reader_.Read(2, wrapped_memory, 100)), ReadError::kInvalidRange);
  EXPECT_TRUE(storage_.submitted().empty());
}

TEST_F(ReaderTest, AReadStartingPastTheFileEndsWithoutTouchingMemory) {
  const ReadSpec spec{
      .fd = fd_, .offset = 100 * kAlignment, .memory = buffer_.data, .length = kAlignment};
  ASSERT_TRUE(reader_.Read(1, spec, 100).has_value());
  const auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kEndOfFile);
  EXPECT_EQ(finished[0].bytes, 0U);
  EXPECT_EQ(buffer_.data[0], std::byte{0xee});
}

TEST(ReaderDeathTest, AReaderCannotForgetAcceptedRequests) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeStorage storage(1, static_cast<std::uint32_t>(kAlignment));
  const int fd = storage.AddFile(Pattern(kAlignment));
  Buffer buffer(kAlignment);
  storage.ScriptNext({.submission = Submission::kAccepted, .result = std::nullopt, .hold = true});
  EXPECT_DEATH(
      {
        DirectReader reader(storage, Settings());
        ASSERT_TRUE(
            reader.Read(1, {.fd = fd, .offset = 0, .memory = buffer.data, .length = kAlignment}, 1)
                .has_value());
      },
      "direct reader destroyed with requests in flight");
}

// The io_uring provider against a real file opened for direct I/O.
class UringTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto storage = UringStorage::Create(8);
    if (!storage && (storage.error() == std::errc::function_not_supported ||
                     storage.error() == std::errc::operation_not_permitted)) {
      GTEST_SKIP() << "io_uring is unavailable or denied by host policy: "
                   << storage.error().message();
    }
    ASSERT_TRUE(storage.has_value()) << storage.error().message();
    storage_ = std::move(*storage);
    const char* base = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const std::filesystem::path directory =
        base != nullptr ? std::filesystem::path(base) : std::filesystem::path(::testing::TempDir());
    std::filesystem::create_directories(directory);
    filesystem_ = jitllm::platform::DescribeFilesystem(directory)
                      .value_or(jitllm::platform::FilesystemFacts{})
                      .type;
    fd_ = ::open(directory.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    ASSERT_GE(fd_, 0) << std::strerror(errno);  // NOLINT(concurrency-mt-unsafe)
    Buffer staging(contents_.size());
    std::memcpy(staging.data, contents_.data(), contents_.size());
    ASSERT_EQ(::pwrite(fd_, staging.data, contents_.size(), 0),
              static_cast<ssize_t>(contents_.size()));
  }
  void TearDown() override {
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
  }

  std::unique_ptr<UringStorage> storage_;
  std::vector<std::byte> contents_ = Pattern(16 * kAlignment);
  int fd_ = -1;
  std::string filesystem_;
};

TEST_F(UringTest, DirectReadsLandInPlace) {
  Buffer buffer(16 * kAlignment);
  DirectReader reader(*storage_, Settings());
  const ReadSpec spec{.fd = fd_, .offset = 0, .memory = buffer.data, .length = 16 * kAlignment};
  ASSERT_TRUE(reader.Read(7, spec, 1).has_value());
  std::vector<FinishedRead> finished;
  for (int i = 0; i < 100 && finished.empty(); ++i) {
    finished = reader.Poll(true);
  }
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kComplete);
  EXPECT_EQ(finished[0].bytes, 16 * kAlignment);
  EXPECT_EQ(std::memcmp(buffer.data, contents_.data(), contents_.size()), 0);
  // Reading past the end is a short transfer: end of file.
  const ReadSpec past{
      .fd = fd_, .offset = 12 * kAlignment, .memory = buffer.data, .length = 8 * kAlignment};
  ASSERT_TRUE(reader.Read(8, past, 1).has_value());
  finished.clear();
  for (int i = 0; i < 100 && finished.empty(); ++i) {
    finished = reader.Poll(true);
  }
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kEndOfFile);
  EXPECT_EQ(finished[0].bytes, 4 * kAlignment);
}

TEST_F(UringTest, UnalignedDirectIoFailsAndCancellationStillCompletes) {
  Buffer buffer(8 * kAlignment);
  // The kernel refuses a misaligned direct-I/O offset; the request still
  // completes.
  ASSERT_EQ(storage_->Submit(IoRequest{.token = 1,
                                       .kind = IoKind::kRead,
                                       .fd = fd_,
                                       .offset = 1,
                                       .memory = buffer.data,
                                       .length = static_cast<std::uint32_t>(kAlignment)}),
            Submission::kAccepted);
  // A cancellation of an unknown token is refused; of a live one, accepted.
  EXPECT_EQ(storage_->Cancel(99), Submission::kNotStarted);
  ASSERT_NE(storage_->Submit(IoRequest{.token = 2,
                                       .kind = IoKind::kRead,
                                       .fd = fd_,
                                       .offset = 0,
                                       .memory = buffer.data,
                                       .length = 4 * kAlignment}),
            Submission::kNotStarted);
  (void)storage_->Cancel(2);
  std::array<IoCompletion, 8> completions{};
  std::size_t seen = 0;
  std::int64_t unaligned = 0;
  for (int i = 0; i < 100 && storage_->in_flight() > 0; ++i) {
    const std::size_t count = storage_->Harvest(std::span(completions).subspan(seen), true);
    for (std::size_t c = seen; c < seen + count; ++c) {
      if (completions[c].token == 1) {
        unaligned = completions[c].result;
      }
    }
    seen += count;
  }
  EXPECT_EQ(storage_->in_flight(), 0U);
  EXPECT_EQ(seen, 2U);  // both originals, never the cancellation's own result
  // Btrfs serves misaligned direct I/O through the page cache instead of
  // refusing it (RE-018); ext4 and XFS refuse it.
  if (filesystem_ != "btrfs") {
    EXPECT_EQ(unaligned, -EINVAL);
  }
}

// Cancellations count as in flight until their own completions arrive, so
// an owner that drains to zero, even one slot at a time, can then destroy
// the provider.
TEST_F(UringTest, DrainingIncludesCancellations) {
  Buffer buffer(4 * kAlignment);
  ASSERT_NE(storage_->Submit(IoRequest{.token = 5,
                                       .kind = IoKind::kRead,
                                       .fd = fd_,
                                       .offset = 0,
                                       .memory = buffer.data,
                                       .length = static_cast<std::uint32_t>(4 * kAlignment)}),
            Submission::kNotStarted);
  if (storage_->Cancel(5) != Submission::kNotStarted) {
    EXPECT_EQ(storage_->in_flight(), 2U);
  }
  std::array<IoCompletion, 1> one{};
  std::size_t originals = 0;
  for (int i = 0; i < 100 && storage_->in_flight() > 0; ++i) {
    originals += storage_->Harvest(one, true);
  }
  EXPECT_EQ(originals, 1U);
  EXPECT_EQ(storage_->in_flight(), 0U);
  storage_.reset();  // drained: no abort
}

// A read of an empty pipe never completes on its own. Wake, from another
// thread, ends the Harvest waiting for it, so the lane can take a new
// command, such as the cancellation that then drains it.
TEST_F(UringTest, WakeEndsAHarvestWaitingForAReadThatNeverCompletes) {
  std::array<int, 2> pipe_fds{-1, -1};
  ASSERT_EQ(::pipe2(pipe_fds.data(), O_CLOEXEC), 0);
  Buffer buffer(kAlignment);
  ASSERT_NE(storage_->Submit(IoRequest{.token = 9,
                                       .kind = IoKind::kRead,
                                       .fd = pipe_fds[0],
                                       .offset = 0,
                                       .memory = buffer.data,
                                       .length = static_cast<std::uint32_t>(kAlignment)}),
            Submission::kNotStarted);
  std::atomic<bool> returned{false};
  std::size_t harvested = 0;
  std::array<IoCompletion, 4> completions{};
  {
    std::jthread waiter([&] {
      harvested = storage_->Harvest(completions, true);
      returned.store(true);
    });
    // Most likely waiting by now; the wake must end the wait either way.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    storage_->Wake();
    storage_->Wake();  // coalesced
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!returned.load() && std::chrono::steady_clock::now() < give_up) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!returned.load()) {
      ADD_FAILURE() << "Wake did not end the Harvest";
      const std::vector<std::byte> fill(kAlignment);
      ASSERT_EQ(::write(pipe_fds[1], fill.data(), fill.size()), static_cast<ssize_t>(kAlignment));
    }
  }
  EXPECT_EQ(harvested, 0U);
  ASSERT_EQ(storage_->in_flight(), 1U);
  // The cancellation is what drains it.
  ASSERT_NE(storage_->Cancel(9), Submission::kNotStarted);
  std::int64_t result = 0;
  for (int i = 0; i < 100 && storage_->in_flight() > 0; ++i) {
    const std::size_t count = storage_->Harvest(completions, true);
    for (std::size_t c = 0; c < count; ++c) {
      if (completions.at(c).token == 9) {
        result = completions.at(c).result;
      }
    }
  }
  EXPECT_EQ(storage_->in_flight(), 0U);
  EXPECT_EQ(result, -ECANCELED);
  (void)::close(pipe_fds[0]);
  (void)::close(pipe_fds[1]);
}

// The lane's side of the wake protocol under many producers: each publishes
// a command (here a counter), then wakes; the lane waits in Harvest for a
// read that never completes, and after every return looks for commands
// before it waits again. A wake lost among the coalesced ones leaves the
// lane asleep with a command published, which shows as a producer that is
// never answered (bounded: the test then ends the read itself).
TEST_F(UringTest, NoWakeIsLostAmongManyProducers) {
#ifdef JITLLM_TEST_TSAN
  constexpr std::uint64_t kCommands = 2000;  // per producer
#else
  constexpr std::uint64_t kCommands = 20000;
#endif
  constexpr std::uint64_t kProducers = 4;
  std::array<int, 2> pipe_fds{-1, -1};
  ASSERT_EQ(::pipe2(pipe_fds.data(), O_CLOEXEC), 0);
  Buffer buffer(kAlignment);
  ASSERT_NE(storage_->Submit(IoRequest{.token = 5,
                                       .kind = IoKind::kRead,
                                       .fd = pipe_fds[0],
                                       .offset = 0,
                                       .memory = buffer.data,
                                       .length = static_cast<std::uint32_t>(kAlignment)}),
            Submission::kNotStarted);
  std::atomic<std::uint64_t> published{0};
  std::atomic<std::uint64_t> answered{0};
  std::atomic<bool> stop{false};
  std::atomic<bool> lost{false};
  std::size_t completions_seen = 0;
  {
    std::jthread lane([&] {
      std::array<IoCompletion, 4> completions{};
      while (!stop.load()) {
        const std::uint64_t now = published.load();
        if (now > answered.load()) {
          answered.store(now);  // "took the commands"
          continue;
        }
        completions_seen += storage_->Harvest(completions, true);
      }
    });
    {
      std::vector<std::jthread> producers;
      producers.reserve(kProducers);
      for (std::uint64_t p = 0; p < kProducers; ++p) {
        producers.emplace_back([&] {
          for (std::uint64_t i = 0; i < kCommands && !lost.load(); ++i) {
            if (i % 16 == 0) {
              // Now and then, long enough for the lane to wait in the kernel.
              std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
            const std::uint64_t mine = published.fetch_add(1) + 1;
            storage_->Wake();
            const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (answered.load() < mine) {
              if (std::chrono::steady_clock::now() > give_up) {
                lost.store(true);
                return;
              }
              std::this_thread::yield();
            }
          }
        });
      }
    }
    stop.store(true);
    storage_->Wake();
    if (lost.load()) {
      // Release the lane so the test can end.
      const std::vector<std::byte> fill(kAlignment);
      EXPECT_EQ(::write(pipe_fds[1], fill.data(), fill.size()), static_cast<ssize_t>(kAlignment));
    }
  }
  EXPECT_FALSE(lost.load()) << "a wake was lost: the lane slept with a command published";
  if (!lost.load()) {
    EXPECT_EQ(answered.load(), kCommands * kProducers);
  }
  // Drain the read: cancelled unless the test had to end it.
  std::array<IoCompletion, 4> completions{};
  if (storage_->in_flight() > 0) {
    ASSERT_NE(storage_->Cancel(5), Submission::kNotStarted);
  }
  for (int i = 0; i < 100 && storage_->in_flight() > 0; ++i) {
    completions_seen += storage_->Harvest(completions, true);
  }
  EXPECT_EQ(storage_->in_flight(), 0U);
  EXPECT_EQ(completions_seen, 1U);
  (void)::close(pipe_fds[0]);
  (void)::close(pipe_fds[1]);
}

}  // namespace
