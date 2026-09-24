#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "zebra_bt/recovery_manager.hpp"

using zebra_bt::FailureType;
using zebra_bt::RecoveryAction;
using zebra_bt::RecoveryManager;

// ---------------------------------------------------------------------------
// PICK_FAILED path: RETRY up to max_pick_retries, then ESCALATE.
// Default max_pick_retries = 3, so calls 1-3 RETRY, call 4 ESCALATE.
// ---------------------------------------------------------------------------

TEST(RecoveryManagerPickFailed, FirstFailureRetries)
{
  RecoveryManager rm;
  EXPECT_EQ(rm.decide("p0e", FailureType::PICK_FAILED), RecoveryAction::RETRY);
}

TEST(RecoveryManagerPickFailed, ThirdFailureStillRetries)
{
  RecoveryManager rm;
  rm.decide("p0e", FailureType::PICK_FAILED);   // 1
  rm.decide("p0e", FailureType::PICK_FAILED);   // 2
  EXPECT_EQ(rm.decide("p0e", FailureType::PICK_FAILED), RecoveryAction::RETRY);  // 3
}

TEST(RecoveryManagerPickFailed, FourthFailureEscalates)
{
  RecoveryManager rm;
  rm.decide("p0e", FailureType::PICK_FAILED);   // 1
  rm.decide("p0e", FailureType::PICK_FAILED);   // 2
  rm.decide("p0e", FailureType::PICK_FAILED);   // 3
  EXPECT_EQ(rm.decide("p0e", FailureType::PICK_FAILED), RecoveryAction::ESCALATE);  // 4
}

// ---------------------------------------------------------------------------
// PART_LOST path: RELOCATE up to max_relocate_attempts, then ESCALATE.
// Default max_relocate_attempts = 2, so calls 1-2 RELOCATE, call 3 ESCALATE.
// ---------------------------------------------------------------------------

TEST(RecoveryManagerPartLost, FirstLossRelocates)
{
  RecoveryManager rm;
  EXPECT_EQ(rm.decide("p0e", FailureType::PART_LOST), RecoveryAction::RELOCATE);
}

TEST(RecoveryManagerPartLost, SecondLossStillRelocates)
{
  RecoveryManager rm;
  rm.decide("p0e", FailureType::PART_LOST);
  EXPECT_EQ(rm.decide("p0e", FailureType::PART_LOST), RecoveryAction::RELOCATE);
}

TEST(RecoveryManagerPartLost, ThirdLossEscalates)
{
  RecoveryManager rm;
  rm.decide("p0e", FailureType::PART_LOST);
  rm.decide("p0e", FailureType::PART_LOST);
  EXPECT_EQ(rm.decide("p0e", FailureType::PART_LOST), RecoveryAction::ESCALATE);
}

// ---------------------------------------------------------------------------
// Independent failure counters: pick failures and lost events on the same
// part must not interfere with each other.
// ---------------------------------------------------------------------------

TEST(RecoveryManagerIndependentCounters, PickFailuresDoNotConsumeRelocateBudget)
{
  RecoveryManager rm;
  // Burn two pick failures (still RETRY territory)
  rm.decide("p0e", FailureType::PICK_FAILED);
  rm.decide("p0e", FailureType::PICK_FAILED);
  // Now a loss event: should still have its full relocate budget
  EXPECT_EQ(rm.decide("p0e", FailureType::PART_LOST), RecoveryAction::RELOCATE);
}

TEST(RecoveryManagerIndependentCounters, LostEventsDoNotConsumePickBudget)
{
  RecoveryManager rm;
  rm.decide("p0e", FailureType::PART_LOST);
  // Still gets the full pick retry allowance
  EXPECT_EQ(rm.decide("p0e", FailureType::PICK_FAILED), RecoveryAction::RETRY);
}

// ---------------------------------------------------------------------------
// Per-part isolation: escalating one part doesn't affect another.
// ---------------------------------------------------------------------------

TEST(RecoveryManagerPerPart, EscalatingOnePartDoesNotAffectAnother)
{
  RecoveryManager rm;
  // Escalate part A completely
  for (int i = 0; i < 4; ++i) {
    rm.decide("p0e", FailureType::PICK_FAILED);
  }
  // Part B is fresh, first failure should still RETRY
  EXPECT_EQ(rm.decide("p0f", FailureType::PICK_FAILED), RecoveryAction::RETRY);
}

// ---------------------------------------------------------------------------
// escalatedParts() list behaviour.
// ---------------------------------------------------------------------------

TEST(RecoveryManagerEscalatedList, EmptyInitially)
{
  RecoveryManager rm;
  EXPECT_TRUE(rm.escalatedParts().empty());
}

TEST(RecoveryManagerEscalatedList, ContainsPartAfterEscalation)
{
  RecoveryManager rm;
  for (int i = 0; i < 4; ++i) {
    rm.decide("p0e", FailureType::PICK_FAILED);
  }
  const auto escalated = rm.escalatedParts();
  ASSERT_EQ(escalated.size(), 1u);
  EXPECT_EQ(escalated[0], "p0e");
}

TEST(RecoveryManagerEscalatedList, DoesNotDuplicateOnRepeatedEscalation)
{
  RecoveryManager rm;
  // Trip escalation four times in a row on the same part
  for (int i = 0; i < 7; ++i) {
    rm.decide("p0e", FailureType::PICK_FAILED);
  }
  // The part should appear once, not four+ times
  EXPECT_EQ(rm.escalatedParts().size(), 1u);
}

TEST(RecoveryManagerEscalatedList, TracksMultiplePartsIndependently)
{
  RecoveryManager rm;
  for (int i = 0; i < 4; ++i) rm.decide("p0e", FailureType::PICK_FAILED);
  for (int i = 0; i < 4; ++i) rm.decide("p0f", FailureType::PICK_FAILED);
  EXPECT_EQ(rm.escalatedParts().size(), 2u);
}

// ---------------------------------------------------------------------------
// Custom thresholds: construction args must actually take effect.
// ---------------------------------------------------------------------------

TEST(RecoveryManagerCustomThresholds, ZeroPickRetriesEscalatesImmediately)
{
  RecoveryManager rm(/*max_pick_retries=*/0, /*max_relocate_attempts=*/2);
  EXPECT_EQ(rm.decide("p0e", FailureType::PICK_FAILED), RecoveryAction::ESCALATE);
}

TEST(RecoveryManagerCustomThresholds, ZeroRelocateAttemptsEscalatesImmediately)
{
  RecoveryManager rm(/*max_pick_retries=*/3, /*max_relocate_attempts=*/0);
  EXPECT_EQ(rm.decide("p0e", FailureType::PART_LOST), RecoveryAction::ESCALATE);
}

TEST(RecoveryManagerCustomThresholds, HigherThresholdRetriesLonger)
{
  RecoveryManager rm(/*max_pick_retries=*/5, /*max_relocate_attempts=*/2);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(rm.decide("p0e", FailureType::PICK_FAILED), RecoveryAction::RETRY);
  }
  EXPECT_EQ(rm.decide("p0e", FailureType::PICK_FAILED), RecoveryAction::ESCALATE);
}