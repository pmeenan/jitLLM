// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The commitment ledger against D-050's worked cases
// (docs/reservation-policy.md#worked-cases-and-implementation-gates), and
// the victim-selection baseline.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"
#include "memory/commitment.h"
#include "memory/materialize.h"
#include "memory/victims.h"

namespace {

using jitllm::base::Bytes;
using jitllm::base::operator""_MiB;
using jitllm::catalog::Catalog;
using jitllm::catalog::ExtentId;
using jitllm::catalog::MemoryClass;
using jitllm::catalog::Recovery;
using jitllm::memory::CommitmentError;
using jitllm::memory::CommitmentLedger;
using jitllm::memory::Envelope;
using jitllm::memory::GrantId;
using ::testing::ElementsAre;

Envelope E(std::uint64_t retained, std::uint64_t phase) {
  return {.retained = Bytes(retained), .phase = Bytes(phase)};
}

constexpr jitllm::catalog::DomainId kSpark(0, 1);
constexpr jitllm::catalog::DomainId kSparkB(1, 1);
constexpr Bytes kBudget = 1024_MiB;

jitllm::memory::CommitmentTotals Totals(const CommitmentLedger& ledger) {
  const auto totals = ledger.Totals(kSpark);
  EXPECT_TRUE(totals.has_value());
  return totals.value_or(jitllm::memory::CommitmentTotals{});
}

CommitmentLedger Ledger(std::uint64_t budget) {
  CommitmentLedger ledger;
  EXPECT_TRUE(ledger.AddDomain(kSpark, Bytes(budget)).has_value());
  return ledger;
}

// B=100, F=10, J=0: A and B (R=20, E=50) fit serially; a third (1, 1) is
// deferred; (41, 50) is impossible even alone.
TEST(CommitmentLedger, TheWorkedCases) {
  CommitmentLedger ledger = Ledger(100);
  ASSERT_TRUE(ledger.AddFixed(kSpark, Bytes(10)).has_value());
  const GrantId a = ledger.Grant(kSpark, E(20, 50)).value();
  const GrantId b = ledger.Grant(kSpark, E(20, 50)).value();
  EXPECT_EQ(Totals(ledger).required, Bytes(100));
  EXPECT_EQ(ledger.Grant(kSpark, E(1, 1)).error(), CommitmentError::kDoesNotFit);
  // Both phases at once would need 10 + 40 + 100.
  const std::vector<GrantId> cohort = {a, b};
  EXPECT_EQ(ledger.SetCohort(kSpark, cohort).error(), CommitmentError::kDoesNotFit);
  const std::vector<GrantId> alone = {a};
  EXPECT_TRUE(ledger.SetCohort(kSpark, alone).has_value());

  CommitmentLedger empty = Ledger(100);
  ASSERT_TRUE(empty.AddFixed(kSpark, Bytes(10)).has_value());
  EXPECT_EQ(empty.Grant(kSpark, E(41, 50)).error(), CommitmentError::kDoesNotFit);
}

TEST(CommitmentLedger, ReplacementIsAtomicAndRetirementNeverRefused) {
  CommitmentLedger ledger = Ledger(100);
  const GrantId a = ledger.Grant(kSpark, E(20, 50)).value();
  EXPECT_EQ(ledger.Replace(a, E(60, 50)).error(), CommitmentError::kDoesNotFit);
  EXPECT_EQ(ledger.EnvelopeOf(a).value_or(Envelope{}).retained, Bytes(20));  // the old grant stands
  ASSERT_TRUE(ledger.Replace(a, E(30, 50)).has_value());
  EXPECT_EQ(Totals(ledger).required, Bytes(80));
  ASSERT_TRUE(ledger.Retire(a).has_value());
  EXPECT_EQ(ledger.Retire(a).error(), CommitmentError::kUnknownGrant);
  EXPECT_EQ(ledger.Replace(a, E(1, 1)).error(), CommitmentError::kUnknownGrant);
  EXPECT_EQ(Totals(ledger).required, Bytes(0));
}

// Every change is checked against the active cohort, not only the serial
// rule (docs/reservation-policy.md: "recheck the serial and any
// active-cohort inequalities").
TEST(CommitmentLedger, ChangesAreCheckedAgainstTheActiveCohort) {
  CommitmentLedger ledger = Ledger(100);
  const GrantId a = ledger.Grant(kSpark, E(0, 40)).value();
  const GrantId b = ledger.Grant(kSpark, E(0, 40)).value();
  const std::vector<GrantId> both = {a, b};
  ASSERT_TRUE(ledger.SetCohort(kSpark, both).has_value());
  EXPECT_EQ(ledger.Replace(a, E(0, 70)).error(), CommitmentError::kDoesNotFit);  // 70 + 40
  EXPECT_EQ(ledger.AddFixed(kSpark, Bytes(21)).error(), CommitmentError::kDoesNotFit);
  EXPECT_EQ(ledger.SetBudget(kSpark, Bytes(79)).error(), CommitmentError::kDoesNotFit);
  // Retiring a member leaves the cohort, and never fails.
  ASSERT_TRUE(ledger.Retire(b).has_value());
  EXPECT_EQ(Totals(ledger).cohort_phase, Bytes(40));
  EXPECT_TRUE(ledger.Replace(a, E(0, 70)).has_value());
}

// A budget reduction waits or is refused; it never drops below claims.
TEST(CommitmentLedger, TheBudgetNeverDropsBelowClaims) {
  CommitmentLedger ledger = Ledger(100);
  ASSERT_TRUE(ledger.AddBackground(kSpark, Bytes(10)).has_value());
  ASSERT_TRUE(ledger.Grant(kSpark, E(20, 50)).has_value());
  EXPECT_EQ(ledger.SetBudget(kSpark, Bytes(79)).error(), CommitmentError::kDoesNotFit);
  EXPECT_EQ(Totals(ledger).budget, Bytes(100));
  ASSERT_TRUE(ledger.SetBudget(kSpark, Bytes(80)).has_value());
  // F and J increases are checked too; releases are not.
  EXPECT_EQ(ledger.AddFixed(kSpark, Bytes(1)).error(), CommitmentError::kDoesNotFit);
  EXPECT_EQ(ledger.AddBackground(kSpark, Bytes(1)).error(), CommitmentError::kDoesNotFit);
  ASSERT_TRUE(ledger.ReleaseBackground(kSpark, Bytes(10)).has_value());
  EXPECT_EQ(ledger.ReleaseBackground(kSpark, Bytes(1)).error(), CommitmentError::kUnderflow);
  EXPECT_EQ(ledger.ReleaseFixed(kSpark, Bytes(1)).error(), CommitmentError::kUnderflow);
}

TEST(CommitmentLedger, ArithmeticOverflowIsRefused) {
  CommitmentLedger ledger = Ledger(UINT64_MAX);
  ASSERT_TRUE(ledger.Grant(kSpark, E(UINT64_MAX - 1, 1)).has_value());
  EXPECT_EQ(ledger.Grant(kSpark, E(1, 0)).error(), CommitmentError::kOverflow);
  EXPECT_EQ(ledger.AddFixed(kSpark, Bytes(UINT64_MAX)).error(), CommitmentError::kOverflow);
}

TEST(CommitmentLedger, ResumptionOverflowDefersAChangeThatFitsNow) {
  CommitmentLedger ledger = Ledger(UINT64_MAX);
  const GrantId paused = ledger.Grant(kSpark, E(0, UINT64_MAX - 10)).value();
  const GrantId peer = ledger.Grant(kSpark, E(0, 10)).value();
  const GrantId substitute = ledger.Grant(kSpark, E(0, 1)).value();
  const std::vector<GrantId> running = {peer, substitute};
  const std::vector<GrantId> resuming = {paused, peer};
  ASSERT_TRUE(ledger.SetCohort(kSpark, running).has_value());
  ASSERT_TRUE(ledger.SetResumption(kSpark, resuming, substitute).has_value());
  EXPECT_EQ(ledger.Replace(peer, E(0, 20)).error(), CommitmentError::kBreaksResumption);
  const auto envelope = ledger.EnvelopeOf(peer);
  ASSERT_TRUE(envelope.has_value());
  EXPECT_EQ(envelope.value_or(Envelope{}).phase, Bytes(10));
}

// Domains are separate budgets, and a grant belongs to one of them.
TEST(CommitmentLedger, DomainsAreSeparate) {
  CommitmentLedger ledger = Ledger(100);
  ASSERT_TRUE(ledger.AddDomain(kSparkB, Bytes(50)).has_value());
  EXPECT_EQ(ledger.AddDomain(kSparkB, Bytes(50)).error(), CommitmentError::kUnknownDomain);
  const GrantId here = ledger.Grant(kSpark, E(0, 80)).value();
  EXPECT_EQ(ledger.Grant(kSparkB, E(0, 80)).error(), CommitmentError::kDoesNotFit);
  const GrantId there = ledger.Grant(kSparkB, E(0, 40)).value();
  EXPECT_NE(here, there);
  EXPECT_EQ(ledger.DomainOf(there), kSparkB);
  const std::vector<GrantId> mixed = {here, there};
  EXPECT_EQ(ledger.SetCohort(kSpark, mixed).error(), CommitmentError::kUnknownGrant);
  const std::vector<GrantId> twice = {here, here};
  EXPECT_EQ(ledger.SetCohort(kSpark, twice).error(), CommitmentError::kUnknownGrant);
  EXPECT_EQ(ledger.Grant(jitllm::catalog::DomainId(9, 1), E(1, 1)).error(),
            CommitmentError::kUnknownDomain);
}

class VictimTest : public ::testing::Test {
 protected:
  void SetUp() override { domain_ = catalog_.AddDomain("spark"); }

  ExtentId Resident(MemoryClass memory_class, Recovery recovery, std::uint32_t chunk,
                    std::uint64_t used) {
    jitllm::catalog::ExtentDescriptor descriptor{.domain = domain_,
                                                 .memory_class = memory_class,
                                                 .recovery = recovery,
                                                 .size = 2_MiB,
                                                 .content = {}};
    descriptor.content.chunk = chunk;
    const ExtentId extent = catalog_.AddExtent(descriptor).value();
    const auto ticket = catalog_.BeginLoad(extent, kBudget).value();
    EXPECT_TRUE(catalog_.CompleteLoad(ticket).has_value());
    if (used != 0) {
      const std::vector<ExtentId> only = {extent};
      const auto lease = catalog_.AcquireLease(catalog_.ClosureOfExtents(only).value()).value();
      EXPECT_TRUE(catalog_.RecordUse(lease, used).has_value());
      EXPECT_TRUE(catalog_.ReleaseLease(lease).has_value());
    }
    return extent;
  }

  Catalog catalog_;
  jitllm::catalog::DomainId domain_;
};

TEST_F(VictimTest, DiscardedFirstThenLeastRecentlyUsedThenContent) {
  const ExtentId recent = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 0, 9);
  const ExtentId old_b = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 5, 2);
  const ExtentId old_a = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 3, 2);
  const ExtentId scratch = Resident(MemoryClass::kScratch, Recovery::kDiscardable, 0, 20);
  const ExtentId never = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 7, 0);
  auto plan = jitllm::memory::SelectVictims(catalog_, domain_, 10_MiB);
  std::vector<ExtentId> order;
  order.reserve(plan.victims.size());
  for (const auto& victim : plan.victims) {
    order.push_back(victim.extent);
  }
  EXPECT_THAT(order, ElementsAre(scratch, never, old_a, old_b, recent));
  EXPECT_TRUE(plan.sufficient);
  EXPECT_EQ(plan.credited, 10_MiB);
  EXPECT_EQ(plan.passed_over, 0U);
  // Only what is needed; the rest are passed over.
  plan = jitllm::memory::SelectVictims(catalog_, domain_, 3_MiB);
  ASSERT_EQ(plan.victims.size(), 2U);
  EXPECT_EQ(plan.passed_over, 3U);
  EXPECT_EQ(plan.victims[0].memory_class, MemoryClass::kScratch);
}

TEST_F(VictimTest, NeverHeldPinnedOrProtected) {
  const ExtentId leased = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 0, 1);
  const ExtentId state = Resident(MemoryClass::kLiveState, Recovery::kPreserve, 1, 1);
  const ExtentId pending = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 2, 1);
  const ExtentId free = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 3, 1);
  const std::vector<ExtentId> only = {leased};
  (void)catalog_.AcquireLease(catalog_.ClosureOfExtents(only).value()).value();
  (void)catalog_
      .AddExtent({.domain = domain_,
                  .memory_class = MemoryClass::kUnknown,
                  .recovery = Recovery::kPinned,
                  .size = 2_MiB,
                  .content = {}},
                 /*resident=*/true)
      .value();
  (void)state;
  const std::vector<ExtentId> protect = {pending};
  const auto plan = jitllm::memory::SelectVictims(catalog_, domain_, 8_MiB, protect);
  ASSERT_EQ(plan.victims.size(), 1U);
  EXPECT_EQ(plan.victims[0].extent, free);
  EXPECT_FALSE(plan.sufficient);  // not enough: nothing should be evicted for it
}

TEST_F(VictimTest, TheChoiceIsDeterministic) {
  for (std::uint32_t chunk = 0; chunk < 16; ++chunk) {
    (void)Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 15 - chunk, 1);
  }
  const auto first = jitllm::memory::SelectVictims(catalog_, domain_, 8_MiB);
  const auto second = jitllm::memory::SelectVictims(catalog_, domain_, 8_MiB);
  ASSERT_EQ(first.victims.size(), 4U);
  for (std::size_t i = 0; i < first.victims.size(); ++i) {
    EXPECT_EQ(first.victims[i].extent, second.victims[i].extent);
    EXPECT_EQ(catalog_.Describe(first.victims[i].extent).value().descriptor.content.chunk, i);
  }
}

class MaterializeTest : public VictimTest {
 protected:
  ExtentId Absent(std::uint32_t chunk) {
    jitllm::catalog::ExtentDescriptor descriptor{.domain = domain_,
                                                 .memory_class = MemoryClass::kWeights,
                                                 .recovery = Recovery::kFromArtifact,
                                                 .size = 2_MiB,
                                                 .content = {}};
    descriptor.content.chunk = chunk;
    return catalog_.AddExtent(descriptor).value();
  }

  jitllm::catalog::Closure Of(const std::vector<ExtentId>& extents) {
    return catalog_.ClosureOfExtents(extents).value();
  }
};

TEST_F(MaterializeTest, AResidentClosureIsReady) {
  const ExtentId a = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 0, 1);
  const auto plan = jitllm::memory::PlanMaterialization(catalog_, domain_, 2_MiB, Of({a}));
  EXPECT_TRUE(plan.feasible);
  EXPECT_TRUE(plan.ready);
  EXPECT_EQ(plan.missing, Bytes());
}

// Only the missing dependencies are materialized, and victims are chosen
// only for the shortfall against B, never from the closure or protected.
TEST_F(MaterializeTest, OnlyTheShortfallIsReclaimed) {
  const ExtentId needed = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 0, 1);
  const ExtentId kept = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 1, 2);
  const ExtentId old = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 2, 3);
  const ExtentId newer = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 3, 4);
  const ExtentId x = Absent(10);
  const ExtentId y = Absent(11);
  const std::vector<ExtentId> protect = {kept};
  // 8 MiB occupied, 4 MiB missing, B = 10 MiB: 2 MiB short.
  auto plan =
      jitllm::memory::PlanMaterialization(catalog_, domain_, 10_MiB, Of({needed, x, y}), protect);
  EXPECT_THAT(plan.load, ElementsAre(x, y));
  EXPECT_EQ(plan.missing, 4_MiB);
  EXPECT_EQ(plan.shortfall, 2_MiB);
  ASSERT_EQ(plan.victims.victims.size(), 1U);
  EXPECT_EQ(plan.victims.victims[0].extent, old);  // not `needed` (older) or `kept`
  EXPECT_TRUE(plan.feasible);
  EXPECT_FALSE(plan.ready);
  // Loads wait for the victim's eviction to complete.
  EXPECT_EQ(catalog_.BeginLoad(x, 10_MiB).value_or(jitllm::catalog::Ticket{}).extent, x);
  EXPECT_EQ(catalog_.BeginLoad(y, 10_MiB).error(), jitllm::catalog::CatalogError::kOverBudget);
  auto evict = catalog_.BeginEvict(old).value();
  EXPECT_EQ(catalog_.BeginLoad(y, 10_MiB).error(), jitllm::catalog::CatalogError::kOverBudget);
  ASSERT_TRUE(catalog_.CompleteEvict(evict).has_value());
  EXPECT_TRUE(catalog_.BeginLoad(y, 10_MiB).has_value());
  plan =
      jitllm::memory::PlanMaterialization(catalog_, domain_, 10_MiB, Of({needed, x, y}), protect);
  EXPECT_THAT(plan.loading, ElementsAre(x, y));
  EXPECT_EQ(plan.shortfall, Bytes());
  EXPECT_TRUE(plan.feasible);
  (void)newer;
}

TEST_F(MaterializeTest, InsufficientVictimsMakeItInfeasible) {
  const ExtentId leased = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 0, 1);
  const std::vector<ExtentId> only = {leased};
  (void)catalog_.AcquireLease(catalog_.ClosureOfExtents(only).value()).value();
  const ExtentId x = Absent(10);
  const auto plan = jitllm::memory::PlanMaterialization(catalog_, domain_, 3_MiB, Of({x}));
  EXPECT_EQ(plan.shortfall, 1_MiB);
  EXPECT_FALSE(plan.victims.sufficient);
  EXPECT_FALSE(plan.feasible);
}

TEST_F(MaterializeTest, InFlightQuarantinedAndStaleExtents) {
  const ExtentId evicting = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 0, 1);
  const ExtentId loading = Absent(1);
  const ExtentId broken = Absent(2);
  (void)catalog_.BeginEvict(evicting).value();
  (void)catalog_.BeginLoad(loading, kBudget).value();
  const auto failed = catalog_.BeginLoad(broken, kBudget).value();
  auto plan =
      jitllm::memory::PlanMaterialization(catalog_, domain_, kBudget, Of({evicting, loading}));
  EXPECT_THAT(plan.cancel, ElementsAre(evicting));
  EXPECT_THAT(plan.loading, ElementsAre(loading));
  EXPECT_TRUE(plan.feasible);
  EXPECT_FALSE(plan.ready);
  ASSERT_TRUE(catalog_.FailLoad(failed, /*completion_known=*/false).has_value());
  plan = jitllm::memory::PlanMaterialization(catalog_, domain_, kBudget, Of({broken}));
  EXPECT_THAT(plan.quarantined, ElementsAre(broken));
  EXPECT_FALSE(plan.feasible);
  // Invalidated state cannot be restored by loading.
  const ExtentId state = Resident(MemoryClass::kLiveState, Recovery::kPreserve, 3, 0);
  const auto taken = Of({state});
  ASSERT_TRUE(catalog_.InvalidateContents(state).has_value());
  plan = jitllm::memory::PlanMaterialization(catalog_, domain_, kBudget, taken);
  EXPECT_THAT(plan.stale, ElementsAre(state));
  EXPECT_FALSE(plan.feasible);
}

}  // namespace
