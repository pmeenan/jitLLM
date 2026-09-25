// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The base module: diagnostic reports.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "base/report.h"

namespace {

using jitllm::base::FormatBytes;

TEST(FormatBytes, BelowOneKibibyteIsBytes) {
  EXPECT_EQ(FormatBytes(0), "0 B");
  EXPECT_EQ(FormatBytes(1023), "1023 B");
}

TEST(FormatBytes, ExactUnitsAreWhole) {
  EXPECT_EQ(FormatBytes(1024), "1 KiB");
  EXPECT_EQ(FormatBytes(4096), "4 KiB");
  EXPECT_EQ(FormatBytes(std::uint64_t{2} << 20), "2 MiB");
  EXPECT_EQ(FormatBytes(std::uint64_t{3} << 40), "3 TiB");
  EXPECT_EQ(FormatBytes(std::uint64_t{1} << 63), "8 EiB");
}

TEST(FormatBytes, OthersHaveOneDecimal) {
  EXPECT_EQ(FormatBytes(1536), "1.5 KiB");
  EXPECT_EQ(FormatBytes(std::uint64_t{127598832} * 1024), "121.7 GiB");
  EXPECT_EQ(FormatBytes(std::numeric_limits<std::uint64_t>::max()), "16.0 EiB");
}

TEST(FormatBytes, RoundsUpToTheNextUnit) {
  EXPECT_EQ(FormatBytes((std::uint64_t{1} << 30) - 1), "1.0 GiB");
  EXPECT_EQ(FormatBytes((std::uint64_t{1} << 20) - 1), "1.0 MiB");
  EXPECT_EQ(FormatBytes((std::uint64_t{1023} << 20) + (std::uint64_t{900} << 10)), "1023.9 MiB");
}

TEST(Report, SectionsKeepTheirOrder) {
  jitllm::base::Report report;
  report.AddSection("first").Add("a", "1");
  jitllm::base::ReportSection& second = report.AddSection("second");
  second.Add("b", "2");
  second.Add("c", "3");
  ASSERT_EQ(report.sections.size(), 2U);
  EXPECT_EQ(report.sections[0].title, "first");
  ASSERT_EQ(report.sections[1].lines.size(), 2U);
  EXPECT_EQ(report.sections[1].lines[1].key, "c");
  EXPECT_EQ(report.sections[1].lines[1].value, "3");
}

TEST(Printable, EscapesWhatCouldForgeOrHideText) {
  using jitllm::base::Printable;
  EXPECT_EQ(Printable("plain caf\xc3\xa9"), "plain caf\xc3\xa9");
  EXPECT_EQ(Printable("a\nb\tc\x7f"), "a\\x0ab\\x09c\\x7f");
  EXPECT_EQ(Printable("\xc2\x9b"
                      "31m"),
            "\\u009b31m");  // C1 CSI
  // NOLINTNEXTLINE(misc-misleading-bidirectional): the override is the input
  EXPECT_EQ(Printable("x\xe2\x80\xae"
                      "y"),
            "x\\u202ey");                                   // right-to-left override
  EXPECT_EQ(Printable("\xff\xc3"), "\\xff\\xc3");           // not UTF-8
  EXPECT_EQ(Printable("\xe0\x80\x80"), "\\xe0\\x80\\x80");  // overlong
}

TEST(Printable, EscapesSeparatorsAndTags) {
  using jitllm::base::Printable;
  EXPECT_EQ(Printable("a\xe2\x80\xa8"
                      "b"),
            "a\\u2028b");                                // line separator
  EXPECT_EQ(Printable("\xf3\xa0\x80\x81"), "\\ue0001");  // a tag character
}

}  // namespace
