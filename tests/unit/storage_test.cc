// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The storage provider and whole reads over it (D-034, D-048): short
// transfers, alignment, retries, cancellation that drains, and coalesced
// duplicate reads, on the scripted fake; and the io_uring provider
// against a real direct-I/O file (skipped where there is no io_uring, as
// under qemu-user or a container policy that denies the syscall).

#include "providers/storage.h"

#include <fcntl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "platform/direct_io.h"
#include "providers/direct_reader.h"
#include "providers/fake/fake_storage.h"
#include "providers/uring_storage.h"

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
  EXPECT_EQ(reader_.Read(1, spec, 102).error(), ReadError::kTooManyWaiters);
  ReadSpec other = spec;
  other.offset = kAlignment;
  EXPECT_EQ(reader_.Read(1, other, 103).error(), ReadError::kMismatch);
  EXPECT_EQ(storage_.submitted().size(), 1U);
  ASSERT_TRUE(reader_.Withdraw(1, 100).has_value());  // one waiter leaves: the read goes on
  EXPECT_EQ(storage_.in_flight(), 1U);
  ASSERT_TRUE(reader_.Withdraw(1, 101).has_value());  // the last: cancelled, draining
  EXPECT_EQ(reader_.Read(1, spec, 104).error(), ReadError::kDraining);
  const auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kCancelled);
  EXPECT_THAT(finished[0].waiters, IsEmpty());
  EXPECT_EQ(storage_.in_flight(), 0U);
  EXPECT_EQ(reader_.Withdraw(1, 101).error(), ReadError::kUnknownRead);
}

TEST_F(ReaderTest, AlignmentIsCheckedAndAFullProviderWaits) {
  const ReadSpec unaligned{
      .fd = fd_, .offset = 512, .memory = buffer_.data, .length = 4 * kAlignment};
  EXPECT_EQ(reader_.Read(1, unaligned, 100).error(), ReadError::kUnaligned);
  const ReadSpec odd{
      .fd = fd_, .offset = 0, .memory = buffer_.data + 512, .length = 4 * kAlignment};
  EXPECT_EQ(reader_.Read(1, odd, 100).error(), ReadError::kUnaligned);
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

TEST_F(ReaderTest, OverflowingRangesAreRefusedBeforeAnyIo) {
  const ReadSpec wrapped_file{
      .fd = fd_,
      .offset = std::numeric_limits<std::uint64_t>::max() - (kAlignment - 1),
      .memory = buffer_.data,
      .length = 2 * kAlignment};
  EXPECT_EQ(reader_.Read(1, wrapped_file, 100).error(), ReadError::kInvalidRange);
  const ReadSpec wrapped_memory{
      .fd = fd_,
      .offset = 0,
      .memory = reinterpret_cast<std::byte*>(  // NOLINT(performance-no-int-to-ptr)
          std::numeric_limits<std::uintptr_t>::max() - (kAlignment - 1)),
      .length = 2 * kAlignment};
  EXPECT_EQ(reader_.Read(2, wrapped_memory, 100).error(), ReadError::kInvalidRange);
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

}  // namespace
