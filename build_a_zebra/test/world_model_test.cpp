#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "zebra_bt/world_model.hpp"

using zebra_bt::PartStatus;
using zebra_bt::WorldModel;

// ---------------------------------------------------------------------------
// Fixture: rclcpp must be initialised once per process, but each test wants
// a fresh node and a fresh WorldModel.
// ---------------------------------------------------------------------------
class WorldModelTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()    { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  void SetUp() override
  {
    // Unique node name per test to avoid "node already exists" warnings
    // (each test creates a node in the same process).
    static int counter = 0;
    node_ = std::make_shared<rclcpp::Node>(
      "wm_test_" + std::to_string(++counter));
  }

  rclcpp::Node::SharedPtr node_;
  std::vector<std::string> parts_{"p0e", "p0f", "p0g"};

  std::shared_ptr<WorldModel> make_wm(double timeout_sec = 2.0)
  {
    return std::make_shared<WorldModel>(node_, parts_, timeout_sec);
  }
};

// ---------------------------------------------------------------------------
// Initial state: every tracked part starts UNKNOWN.
// ---------------------------------------------------------------------------

TEST_F(WorldModelTest, AllPartsStartUnknown)
{
  auto wm = make_wm();
  for (const auto & id : parts_) {
    EXPECT_EQ(wm->getPartState(id).status, PartStatus::UNKNOWN);
    EXPECT_EQ(wm->getPartState(id).pick_attempts, 0);
    EXPECT_FALSE(wm->getPartState(id).seen_by_perception);
  }
}

// ---------------------------------------------------------------------------
// Status setter/getter.
// ---------------------------------------------------------------------------

TEST_F(WorldModelTest, SetStatusRoundTrip)
{
  auto wm = make_wm();
  wm->setPartStatus("p0e", PartStatus::LOCATED);
  EXPECT_EQ(wm->getPartState("p0e").status, PartStatus::LOCATED);

  wm->setPartStatus("p0e", PartStatus::PICKED);
  EXPECT_EQ(wm->getPartState("p0e").status, PartStatus::PICKED);
}

TEST_F(WorldModelTest, AllStatusValuesSupported)
{
  auto wm = make_wm();
  const PartStatus statuses[] = {
    PartStatus::UNKNOWN, PartStatus::LOCATED, PartStatus::LOST,
    PartStatus::PICKED,  PartStatus::PICK_FAILED, PartStatus::PLACED,
    PartStatus::ESCALATED,
  };
  for (auto s : statuses) {
    wm->setPartStatus("p0e", s);
    EXPECT_EQ(wm->getPartState("p0e").status, s);
  }
}

// ---------------------------------------------------------------------------
// Position setter/getter.
// ---------------------------------------------------------------------------

TEST_F(WorldModelTest, SetPositionRoundTrip)
{
  auto wm = make_wm();
  wm->setPartPosition("p0e", 0.51, -0.32, -0.10);
  const auto state = wm->getPartState("p0e");
  EXPECT_NEAR(state.position.x, 0.51, 1e-9);
  EXPECT_NEAR(state.position.y, -0.32, 1e-9);
  EXPECT_NEAR(state.position.z, -0.10, 1e-9);
}

TEST_F(WorldModelTest, IndependentPositionsPerPart)
{
  auto wm = make_wm();
  wm->setPartPosition("p0e", 1.0, 0.0, 0.0);
  wm->setPartPosition("p0f", 2.0, 0.0, 0.0);
  EXPECT_NEAR(wm->getPartState("p0e").position.x, 1.0, 1e-9);
  EXPECT_NEAR(wm->getPartState("p0f").position.x, 2.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Pick attempts.
// ---------------------------------------------------------------------------

TEST_F(WorldModelTest, IncrementPickAttempts)
{
  auto wm = make_wm();
  EXPECT_EQ(wm->incrementPickAttempts("p0e"), 1);
  EXPECT_EQ(wm->incrementPickAttempts("p0e"), 2);
  EXPECT_EQ(wm->incrementPickAttempts("p0e"), 3);
}

TEST_F(WorldModelTest, ResetPickAttempts)
{
  auto wm = make_wm();
  wm->incrementPickAttempts("p0e");
  wm->incrementPickAttempts("p0e");
  wm->resetPickAttempts("p0e");
  EXPECT_EQ(wm->getPartState("p0e").pick_attempts, 0);
}

TEST_F(WorldModelTest, PickAttemptsArePerPart)
{
  auto wm = make_wm();
  wm->incrementPickAttempts("p0e");
  wm->incrementPickAttempts("p0e");
  EXPECT_EQ(wm->getPartState("p0e").pick_attempts, 2);
  EXPECT_EQ(wm->getPartState("p0f").pick_attempts, 0);
}

// ---------------------------------------------------------------------------
// allPartsResolved() logic.
// ---------------------------------------------------------------------------

TEST_F(WorldModelTest, NotResolvedInitially)
{
  auto wm = make_wm();
  EXPECT_FALSE(wm->allPartsResolved());
}

TEST_F(WorldModelTest, NotResolvedWhenOnePartPlacedAndOthersUnknown)
{
  auto wm = make_wm();
  wm->setPartStatus("p0e", PartStatus::PLACED);
  EXPECT_FALSE(wm->allPartsResolved());
}

TEST_F(WorldModelTest, ResolvedWhenAllPlaced)
{
  auto wm = make_wm();
  for (const auto & id : parts_) {
    wm->setPartStatus(id, PartStatus::PLACED);
  }
  EXPECT_TRUE(wm->allPartsResolved());
}

TEST_F(WorldModelTest, ResolvedWhenAllEscalated)
{
  auto wm = make_wm();
  for (const auto & id : parts_) {
    wm->setPartStatus(id, PartStatus::ESCALATED);
  }
  EXPECT_TRUE(wm->allPartsResolved());
}

TEST_F(WorldModelTest, ResolvedWhenMixOfPlacedAndEscalated)
{
  auto wm = make_wm();
  wm->setPartStatus("p0e", PartStatus::PLACED);
  wm->setPartStatus("p0f", PartStatus::ESCALATED);
  wm->setPartStatus("p0g", PartStatus::PLACED);
  EXPECT_TRUE(wm->allPartsResolved());
}

TEST_F(WorldModelTest, NotResolvedWhenAnyPartPicked)
{
  auto wm = make_wm();
  wm->setPartStatus("p0e", PartStatus::PLACED);
  wm->setPartStatus("p0f", PartStatus::PLACED);
  wm->setPartStatus("p0g", PartStatus::PICKED);
  EXPECT_FALSE(wm->allPartsResolved());
}

// ---------------------------------------------------------------------------
// Staleness — the feature you built.
//
// Two cases:
//   A) Part was NEVER seen by external perception -> setPartStatus(LOCATED)
//      alone must NOT go stale. This prevents LocatePart from thrashing.
//   B) Part WAS seen by external perception -> after the timeout, it must
//      demote to UNKNOWN.
//
// Case A is testable directly. Case B needs an actual ROS 2 message to hit
// the subscription, so we publish one and spin.
// ---------------------------------------------------------------------------

TEST_F(WorldModelTest, SelfLocatedPartDoesNotGoStale)
{
  auto wm = make_wm(/*timeout_sec=*/0.05);
  wm->setPartStatus("p0e", PartStatus::LOCATED);

  // Wait clearly longer than the timeout.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Should still be LOCATED because it was never seen by external perception.
  EXPECT_EQ(wm->getPartState("p0e").status, PartStatus::LOCATED);
}

TEST_F(WorldModelTest, PickedPartDoesNotGoStale)
{
  auto wm = make_wm(/*timeout_sec=*/0.05);
  wm->setPartStatus("p0e", PartStatus::PICKED);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Only LOCATED parts demote. PICKED is a decision, not a time-sensitive fact.
  EXPECT_EQ(wm->getPartState("p0e").status, PartStatus::PICKED);
}

TEST_F(WorldModelTest, PlacedPartDoesNotGoStale)
{
  auto wm = make_wm(/*timeout_sec=*/0.05);
  wm->setPartStatus("p0e", PartStatus::PLACED);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  EXPECT_EQ(wm->getPartState("p0e").status, PartStatus::PLACED);
}

TEST_F(WorldModelTest, ExternalPerceptionGoesStaleAfterTimeout)
{
  auto wm = make_wm(/*timeout_sec=*/0.1);

  // Publish one perception message so the WorldModel marks p0e as seen.
  auto pub = node_->create_publisher<std_msgs::msg::String>(
    "/zebra/perception_updates", 10);
  std_msgs::msg::String msg;
  msg.data = "p0e,LOCATED,0.51,-0.32,-0.10";
  pub->publish(msg);

  // Spin until the callback has run.
  for (int i = 0; i < 50; ++i) {
    rclcpp::spin_some(node_);
    if (wm->getPartState("p0e").seen_by_perception) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(wm->getPartState("p0e").seen_by_perception);
  EXPECT_EQ(wm->getPartState("p0e").status, PartStatus::LOCATED);

  // Wait past the timeout, then read again — this triggers the demote.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  EXPECT_EQ(wm->getPartState("p0e").status, PartStatus::UNKNOWN);
}

TEST_F(WorldModelTest, StalePositionIsPreservedEvenAfterDemote)
{
  // Demoting status to UNKNOWN should NOT erase the last known position.
  // The BT may still find the (stale) coordinates useful for a re-locate.
  auto wm = make_wm(/*timeout_sec=*/0.1);
  auto pub = node_->create_publisher<std_msgs::msg::String>(
    "/zebra/perception_updates", 10);
  std_msgs::msg::String msg;
  msg.data = "p0e,LOCATED,0.51,-0.32,-0.10";
  pub->publish(msg);

  for (int i = 0; i < 50; ++i) {
    rclcpp::spin_some(node_);
    if (wm->getPartState("p0e").seen_by_perception) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  const auto state = wm->getPartState("p0e");
  EXPECT_EQ(state.status, PartStatus::UNKNOWN);
  EXPECT_NEAR(state.position.x, 0.51, 1e-6);
  EXPECT_NEAR(state.position.y, -0.32, 1e-6);
  EXPECT_NEAR(state.position.z, -0.10, 1e-6);
}

// ---------------------------------------------------------------------------
// Perception callback: unknown part IDs are ignored, known ones update.
// ---------------------------------------------------------------------------

TEST_F(WorldModelTest, UnknownPartInPerceptionIsIgnored)
{
  auto wm = make_wm();
  auto pub = node_->create_publisher<std_msgs::msg::String>(
    "/zebra/perception_updates", 10);
  std_msgs::msg::String msg;
  msg.data = "not_a_real_part,LOCATED,0.0,0.0,0.0";
  pub->publish(msg);
  for (int i = 0; i < 20; ++i) rclcpp::spin_some(node_);

  // Nothing should change for the tracked parts.
  for (const auto & id : parts_) {
    EXPECT_EQ(wm->getPartState(id).status, PartStatus::UNKNOWN);
  }
}