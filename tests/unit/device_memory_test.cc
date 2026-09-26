// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The device-memory provider's shared rules (VmmProvider), on the fake
// (docs/architecture.md#providers): explicit reservations, backing mapped
// whole into holes, access granted explicitly, and absent backing that
// faults when touched.

#include "providers/device_memory.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "base/bytes.h"
#include "providers/fake/fake_device_memory.h"

namespace {

using jitllm::base::Bytes;
using jitllm::base::operator""_MiB;
using jitllm::providers::Access;
using jitllm::providers::BackingId;
using jitllm::providers::BackingKind;
using jitllm::providers::ProviderError;
using jitllm::providers::ReservationId;
using jitllm::providers::fake::FakeDeviceMemory;
using jitllm::providers::fake::kPoison;
using jitllm::providers::fake::Operation;

constexpr std::size_t kDevice = 0;
constexpr std::size_t kHost = 1;

class DeviceMemoryTest : public ::testing::Test {
 protected:
  std::byte* At(ReservationId reservation, Bytes offset) {
    const std::uint64_t address = memory_.RangeOf(reservation).value().base + offset.value();
    return reinterpret_cast<std::byte*>(address);  // NOLINT(performance-no-int-to-ptr)
  }

  FakeDeviceMemory memory_{2_MiB, 16_MiB};
};

TEST_F(DeviceMemoryTest, BackingIsMappedWholeAndAccessIsExplicit) {
  ASSERT_EQ(memory_.Classes().size(), 2U);
  EXPECT_EQ(memory_.Classes()[kHost].kind, BackingKind::kHost);
  const ReservationId reservation = memory_.Reserve(8_MiB).value();
  EXPECT_EQ(memory_.RangeOf(reservation).value().base % (2_MiB).value(), 0U);
  const BackingId backing = memory_.Create(kHost, 4_MiB).value();
  ASSERT_TRUE(memory_.Map(reservation, 2_MiB, backing).has_value());
  ASSERT_TRUE(memory_.SetAccess(reservation, 2_MiB, 4_MiB, Access::kReadWrite).has_value());
  std::byte* data = At(reservation, 2_MiB);
  EXPECT_EQ(data[0], kPoison);  // fresh backing holds nothing to rely on
  std::memset(data, 7, (4_MiB).value());
  // The same backing, unmapped and mapped elsewhere, keeps its contents.
  ASSERT_TRUE(memory_.Unmap(reservation, 2_MiB, 4_MiB).has_value());
  ASSERT_TRUE(memory_.Map(reservation, 4_MiB, backing).has_value());
  ASSERT_TRUE(memory_.SetAccess(reservation, 4_MiB, 4_MiB, Access::kRead).has_value());
  EXPECT_EQ(At(reservation, 4_MiB)[(4_MiB).value() - 1], std::byte{7});
  ASSERT_TRUE(memory_.Unmap(reservation, 4_MiB, 4_MiB).has_value());
  ASSERT_TRUE(memory_.Release(backing).has_value());
  ASSERT_TRUE(memory_.Free(reservation).has_value());
  EXPECT_EQ(memory_.reservations(), 0U);
  EXPECT_EQ(memory_.backings(), 0U);
  EXPECT_EQ(memory_.in_use(), Bytes());
}

TEST_F(DeviceMemoryTest, MisuseIsRefused) {
  EXPECT_EQ(memory_.Reserve(Bytes(4096)).error().error, ProviderError::kInvalid);  // not a granule
  const ReservationId reservation = memory_.Reserve(6_MiB).value();
  const BackingId a = memory_.Create(kDevice, 4_MiB).value();
  const BackingId b = memory_.Create(kDevice, 2_MiB).value();
  EXPECT_EQ(memory_.Create(kDevice, Bytes(4096)).error().error, ProviderError::kInvalid);
  EXPECT_EQ(memory_.Create(7, 2_MiB).error().error, ProviderError::kInvalid);
  EXPECT_EQ(memory_.Map(reservation, 4_MiB, a).error().error,
            ProviderError::kInvalid);  // runs outside
  EXPECT_EQ(memory_.Map(reservation, Bytes(4096), a).error().error,
            ProviderError::kInvalid);  // unaligned
  ASSERT_TRUE(memory_.Map(reservation, 0_MiB, a).has_value());
  EXPECT_EQ(memory_.Map(reservation, 2_MiB, b).error().error, ProviderError::kInvalid);  // overlaps
  EXPECT_EQ(memory_.Map(reservation, 4_MiB, a).error().error,
            ProviderError::kInvalid);  // mapped twice
  ASSERT_TRUE(memory_.Map(reservation, 4_MiB, b).has_value());
  // Unmap and access take whole mappings only.
  EXPECT_EQ(memory_.Unmap(reservation, 0_MiB, 2_MiB).error().error, ProviderError::kInvalid);
  EXPECT_EQ(memory_.SetAccess(reservation, 2_MiB, 4_MiB, Access::kRead).error().error,
            ProviderError::kInvalid);
  EXPECT_TRUE(memory_.SetAccess(reservation, 0_MiB, 6_MiB, Access::kRead).has_value());
  // Nothing mapped is released or freed.
  EXPECT_EQ(memory_.Release(a).error().error, ProviderError::kInvalid);
  EXPECT_EQ(memory_.Free(reservation).error().error, ProviderError::kInvalid);
  ASSERT_TRUE(memory_.Unmap(reservation, 0_MiB, 6_MiB).has_value());  // both at once
  ASSERT_TRUE(memory_.Release(a).has_value());
  EXPECT_EQ(memory_.Release(a).error().error, ProviderError::kInvalid);  // stale
  ASSERT_TRUE(memory_.Release(b).has_value());
  ASSERT_TRUE(memory_.Free(reservation).has_value());
  EXPECT_EQ(memory_.RangeOf(reservation).error().error, ProviderError::kInvalid);
}

TEST_F(DeviceMemoryTest, CapacityAndScriptedFailuresChangeNothing) {
  const BackingId all = memory_.Create(kHost, 16_MiB).value();
  EXPECT_EQ(memory_.Create(kHost, 2_MiB).error().error, ProviderError::kOutOfMemory);
  ASSERT_TRUE(memory_.Release(all).has_value());
  const ReservationId reservation = memory_.Reserve(2_MiB).value();
  const BackingId backing = memory_.Create(kHost, 2_MiB).value();
  memory_.FailNext(Operation::kMap, ProviderError::kFailed);
  EXPECT_EQ(memory_.Map(reservation, 0_MiB, backing).error().error, ProviderError::kFailed);
  ASSERT_TRUE(
      memory_.Map(reservation, 0_MiB, backing).has_value());  // nothing changed: it maps now
  ASSERT_TRUE(memory_.Unmap(reservation, 0_MiB, 2_MiB).has_value());
  ASSERT_TRUE(memory_.Release(backing).has_value());
  ASSERT_TRUE(memory_.Free(reservation).has_value());
}

TEST(DeviceMemoryRangeTest, AlignmentPaddingCannotWrapAReservation) {
  constexpr std::uint64_t kGranularity = std::uint64_t{3} * 4096;
  FakeDeviceMemory memory(Bytes(kGranularity), Bytes(0));
  const Bytes largest((std::numeric_limits<std::uint64_t>::max() / kGranularity) * kGranularity);
  EXPECT_EQ(memory.Reserve(largest).error().error, ProviderError::kOutOfMemory);
  EXPECT_EQ(memory.reservations(), 0U);
}

// An unknown outcome is a fault, not a retry: what it touched is
// undetermined, and every later call on it is refused, even when the
// driver did what it was asked.
TEST_F(DeviceMemoryTest, UnknownOutcomesAreNeverRetried) {
  const ReservationId reservation = memory_.Reserve(4_MiB).value();
  const BackingId first = memory_.Create(kHost, 2_MiB).value();
  const BackingId second = memory_.Create(kHost, 2_MiB).value();
  memory_.FailNext(Operation::kMap, ProviderError::kUnknown, /*applied=*/true);
  EXPECT_EQ(memory_.Map(reservation, 0_MiB, first).error().error, ProviderError::kUnknown);
  EXPECT_TRUE(memory_.Undetermined(reservation));
  EXPECT_TRUE(memory_.Undetermined(first));
  EXPECT_EQ(memory_.RangeOf(reservation).error().error, ProviderError::kInvalid);
  EXPECT_EQ(memory_.Map(reservation, 2_MiB, second).error().error, ProviderError::kInvalid);
  EXPECT_EQ(memory_.Release(first).error().error, ProviderError::kInvalid);
  EXPECT_EQ(memory_.Free(reservation).error().error, ProviderError::kInvalid);
  EXPECT_FALSE(memory_.Undetermined(second));
  // A release whose outcome is unknown is never repeated.
  memory_.FailNext(Operation::kRelease, ProviderError::kUnknown, /*applied=*/true);
  EXPECT_EQ(memory_.Release(second).error().error, ProviderError::kUnknown);
  EXPECT_EQ(memory_.Release(second).error().error, ProviderError::kInvalid);
  // A create whose outcome is unknown may have made backing: it stays
  // charged, with no handle anyone can use.
  memory_.FailNext(Operation::kCreate, ProviderError::kUnknown, /*applied=*/true);
  EXPECT_EQ(memory_.Create(kHost, 2_MiB).error().error, ProviderError::kUnknown);
  // What is left is the owner's to quarantine; the fake cleans up at the end.
  EXPECT_EQ(memory_.backings(), 3U);
  EXPECT_EQ(memory_.UndeterminedBytes(), 6_MiB);
}

TEST_F(DeviceMemoryTest, UnknownAccessKeepsItsBackingChargedAsUndetermined) {
  const ReservationId reservation = memory_.Reserve(4_MiB).value();
  const BackingId first = memory_.Create(kHost, 2_MiB).value();
  const BackingId second = memory_.Create(kHost, 2_MiB).value();
  ASSERT_TRUE(memory_.Map(reservation, 0_MiB, first).has_value());
  ASSERT_TRUE(memory_.Map(reservation, 2_MiB, second).has_value());
  memory_.FailNext(Operation::kSetAccess, ProviderError::kUnknown, /*applied=*/true);
  EXPECT_EQ(memory_.SetAccess(reservation, 0_MiB, 4_MiB, Access::kReadWrite).error().error,
            ProviderError::kUnknown);
  EXPECT_TRUE(memory_.Undetermined(first));
  EXPECT_TRUE(memory_.Undetermined(second));
  EXPECT_EQ(memory_.UndeterminedBytes(), 4_MiB);
  EXPECT_EQ(memory_.RangeOf(reservation).error().error, ProviderError::kInvalid);
  EXPECT_EQ(memory_.Unmap(reservation, 0_MiB, 4_MiB).error().error, ProviderError::kInvalid);
}

// Touching absent backing is a bug, and the fake makes it fault: reserved
// address space, a mapping with no access yet, and an unmapped range.
using DeviceMemoryDeathTest = DeviceMemoryTest;

TEST_F(DeviceMemoryDeathTest, AbsentBackingFaults) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const ReservationId reservation = memory_.Reserve(4_MiB).value();
  volatile std::byte* hole = At(reservation, 0_MiB);
  EXPECT_DEATH((void)hole[0], "");
  const BackingId backing = memory_.Create(kDevice, 2_MiB).value();
  ASSERT_TRUE(memory_.Map(reservation, 2_MiB, backing).has_value());
  volatile std::byte* no_access = At(reservation, 2_MiB);
  EXPECT_DEATH((void)no_access[0], "");
  ASSERT_TRUE(memory_.SetAccess(reservation, 2_MiB, 2_MiB, Access::kRead).has_value());
  const std::byte seen = no_access[0];
  EXPECT_EQ(seen, kPoison);
  EXPECT_DEATH(no_access[0] = std::byte{1}, "");  // read-only
  ASSERT_TRUE(memory_.Unmap(reservation, 2_MiB, 2_MiB).has_value());
  EXPECT_DEATH((void)no_access[0], "");
  ASSERT_TRUE(memory_.Release(backing).has_value());
  ASSERT_TRUE(memory_.Free(reservation).has_value());
}

}  // namespace
