#include <chrono>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_cpp/loggers/abstract_logger.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include <nlohmann/json.hpp>

#include "zebra_bt/disturbance_generator.hpp"
#include "zebra_bt/recovery_manager.hpp"
#include "zebra_bt/task_model.hpp"
#include "zebra_bt/world_model.hpp"

using namespace std::chrono_literals;
using zebra_bt::PartStatus;
using zebra_bt::RecoveryManager;
using zebra_bt::WorldModel;

using WorldModelPtr = std::shared_ptr<WorldModel>;
using RecoveryManagerPtr = std::shared_ptr<RecoveryManager>;

namespace zebra_bt
{

// ---------------------------------------------------------------------
// SkillBridge — sends JSON commands, tracks status replies.
//
// Topics:
//   PUB  /zebra/skill_commands    {"command_id","skill","part_id","target"}
//   SUB  /zebra/skill_status      {"command_id","status"[,"message"]}
// ---------------------------------------------------------------------
class SkillBridge
{
public:
  explicit SkillBridge(const rclcpp::Node::SharedPtr & node)
  : node_(node)
  {
    command_pub_ = node_->create_publisher<std_msgs::msg::String>(
      "/zebra/skill_commands", 10);

    status_sub_ = node_->create_subscription<std_msgs::msg::String>(
      "/zebra/skill_status", 10,
      std::bind(&SkillBridge::statusCallback, this, std::placeholders::_1));
  }

  std::string send(
    const std::string & skill,
    const std::string & part,
    const geometry_msgs::msg::Point & target)
  {
    const std::string command_id =
      "zebra-" + std::to_string(++next_command_id_);

    nlohmann::json payload = {
      {"command_id", command_id},
      {"skill", skill},
      {"part_id", part},
      {"target", {
        {"x", target.x},
        {"y", target.y},
        {"z", target.z}
      }}
    };

    std_msgs::msg::String message;
    message.data = payload.dump();
    command_pub_->publish(message);

    RCLCPP_INFO(
      node_->get_logger(),
      "[SKILL]   -> %s %s at (%.2f, %.2f, %.2f)  [%s]",
      skill.c_str(), part.c_str(),
      target.x, target.y, target.z,
      command_id.c_str());

    return command_id;
  }

  // Returns an empty string while no status has arrived.
  std::string status(const std::string & command_id)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = statuses_.find(command_id);
    if (it == statuses_.end()) {
      return "";
    }
    return it->second;
  }

private:
  void statusCallback(const std_msgs::msg::String::SharedPtr message)
  {
    try {
      const auto payload = nlohmann::json::parse(message->data);

      const std::string command_id =
        payload.at("command_id").get<std::string>();

      const std::string execution_status =
        payload.at("status").get<std::string>();

      std::lock_guard<std::mutex> lock(mutex_);
      statuses_[command_id] = execution_status;

      RCLCPP_INFO(
        node_->get_logger(),
        "[SKILL]   <- %s  %s",
        command_id.c_str(),
        execution_status.c_str());
    } catch (const std::exception & error) {
      RCLCPP_WARN(
        node_->get_logger(),
        "[SKILL]   ignoring malformed status message: %s",
        error.what());
    }
  }

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr command_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr status_sub_;

  std::mutex mutex_;
  std::unordered_map<std::string, std::string> statuses_;
  unsigned long next_command_id_{0};
};

using SkillBridgePtr = std::shared_ptr<SkillBridge>;

// ---------------------------------------------------------------------
// QuietLogger — same role as BT::StdCoutLogger, but only prints
// outcome-level transitions. Filters out the IDLE <-> RUNNING noise
// that makes the raw tick trace unreadable.
// ---------------------------------------------------------------------
class QuietLogger : public BT::StatusChangeLogger
{
public:
  explicit QuietLogger(BT::TreeNode * root)
  : BT::StatusChangeLogger(root)
  {}

  void callback(
    BT::Duration,
    const BT::TreeNode & node,
    BT::NodeStatus prev,
    BT::NodeStatus curr) override
  {
    if (prev != BT::NodeStatus::RUNNING) {
      return;
    }

    std::cout << "    [BT] " << std::left << std::setw(24)
              << node.name() << "  "
              << BT::toStr(prev) << " -> " << BT::toStr(curr)
              << "\n";
  }

  void flush() override {}   // <-- ADD THIS LINE
};
// ---------------------------------------------------------------------
// Condition nodes — read WorldModel only. No logging.
// ---------------------------------------------------------------------

class IsPartPlaced : public BT::ConditionNode
{
public:
  IsPartPlaced(
    const std::string & name,
    const BT::NodeConfig & config,
    WorldModelPtr wm)
  : BT::ConditionNode(name, config), wm_(wm)
  {}

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<std::string>("part")};
  }

  BT::NodeStatus tick() override
  {
    std::string part;
    getInput("part", part);

    return wm_->getPartState(part).status == PartStatus::PLACED
      ? BT::NodeStatus::SUCCESS
      : BT::NodeStatus::FAILURE;
  }

private:
  WorldModelPtr wm_;
};

class IsPartLocated : public BT::ConditionNode
{
public:
  IsPartLocated(
    const std::string & name,
    const BT::NodeConfig & config,
    WorldModelPtr wm)
  : BT::ConditionNode(name, config), wm_(wm)
  {}

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<std::string>("part")};
  }

  BT::NodeStatus tick() override
  {
    std::string part;
    getInput("part", part);

    const auto status = wm_->getPartState(part).status;
    const bool located =
      status == PartStatus::LOCATED || status == PartStatus::PICKED;

    return located ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }

private:
  WorldModelPtr wm_;
};

class IsPartEscalated : public BT::ConditionNode
{
public:
  IsPartEscalated(
    const std::string & name,
    const BT::NodeConfig & config,
    WorldModelPtr wm)
  : BT::ConditionNode(name, config), wm_(wm)
  {}

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<std::string>("part")};
  }

  BT::NodeStatus tick() override
  {
    std::string part;
    getInput("part", part);

    return wm_->getPartState(part).status == PartStatus::ESCALATED
      ? BT::NodeStatus::SUCCESS
      : BT::NodeStatus::FAILURE;
  }

private:
  WorldModelPtr wm_;
};

// ---------------------------------------------------------------------
// Action nodes.
// ---------------------------------------------------------------------

class LocatePart : public BT::StatefulActionNode
{
public:
  LocatePart(
    const std::string & name,
    const BT::NodeConfig & config,
    WorldModelPtr wm,
    RecoveryManagerPtr rm,
    rclcpp::Logger logger)
  : BT::StatefulActionNode(name, config),
    wm_(wm), rm_(rm), logger_(logger)
  {}

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<std::string>("part")};
  }

  BT::NodeStatus onStart() override
  {
    getInput("part", part_);
    search_ticks_ = 0;

    const auto state = wm_->getPartState(part_);

    // Only a genuine LOST counts as a real failure worth remembering.
    // UNKNOWN (first look at a fresh part) is normal startup, not a failure.
    if (state.status == PartStatus::LOST) {
      const auto action = rm_->decide(part_, zebra_bt::FailureType::PART_LOST);

      RCLCPP_WARN(
        logger_,
        "[RECOVER] %s -> %s",
        part_.c_str(),
        zebra_bt::toString(action).c_str());

      if (action == zebra_bt::RecoveryAction::ESCALATE) {
        wm_->setPartStatus(part_, PartStatus::ESCALATED);
        return BT::NodeStatus::FAILURE;
      }
    }

    RCLCPP_INFO(
      logger_,
      "[BT]      LocatePart: searching for %s",
      part_.c_str());

    return BT::NodeStatus::RUNNING;
  }

  BT::NodeStatus onRunning() override
  {
    if (++search_ticks_ < 2) {
      return BT::NodeStatus::RUNNING;
    }

    auto state = wm_->getPartState(part_);

    if (state.status != PartStatus::LOCATED) {
      // NOTE: hardcoded nominal table location. Replace with real
      // perception once the simulator publishes /zebra/perception_updates.
      wm_->setPartPosition(part_, 0.40, 0.0, 0.32);
    }

    wm_->setPartStatus(part_, PartStatus::LOCATED);

    RCLCPP_INFO(
      logger_,
      "[BT]      LocatePart: located %s",
      part_.c_str());

    return BT::NodeStatus::SUCCESS;
  }

  void onHalted() override {}

private:
  WorldModelPtr wm_;
  RecoveryManagerPtr rm_;
  rclcpp::Logger logger_;
  std::string part_;
  int search_ticks_{0};
};

class PickPart : public BT::StatefulActionNode
{
public:
  PickPart(
    const std::string & name,
    const BT::NodeConfig & config,
    WorldModelPtr wm,
    SkillBridgePtr bridge,
    rclcpp::Logger logger)
  : BT::StatefulActionNode(name, config),
    wm_(wm), bridge_(bridge), logger_(logger)
  {}

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<std::string>("part")};
  }

  BT::NodeStatus onStart() override
  {
    getInput("part", part_);

    const auto state = wm_->getPartState(part_);

    if (state.status == PartStatus::LOST) {
      RCLCPP_WARN(
        logger_,
        "[BT]      PickPart: %s is LOST, cannot pick",
        part_.c_str());

      return BT::NodeStatus::FAILURE;
    }

    attempt_ = wm_->incrementPickAttempts(part_);

    command_id_ = bridge_->send("pick", part_, state.position);

    wait_ticks_ = 0;

    return BT::NodeStatus::RUNNING;
  }

  BT::NodeStatus onRunning() override
  {
    const std::string result = bridge_->status(command_id_);

    if (result == "SUCCEEDED") {
      wm_->setPartStatus(part_, PartStatus::PICKED);
      wm_->resetPickAttempts(part_);

      RCLCPP_INFO(
        logger_,
        "[BT]      PickPart: picked %s (attempt %d)",
        part_.c_str(), attempt_);

      return BT::NodeStatus::SUCCESS;
    }

    if (result == "FAILED") {
      wm_->setPartStatus(part_, PartStatus::PICK_FAILED);

      RCLCPP_WARN(
        logger_,
        "[BT]      PickPart: %s failed (attempt %d)",
        part_.c_str(), attempt_);

      return BT::NodeStatus::FAILURE;
    }

    // 30 seconds at the existing 2 Hz tick rate.
    if (++wait_ticks_ > 60) {
      wm_->setPartStatus(part_, PartStatus::PICK_FAILED);

      RCLCPP_WARN(
        logger_,
        "[BT]      PickPart: timed out waiting for %s",
        part_.c_str());

      return BT::NodeStatus::FAILURE;
    }

    return BT::NodeStatus::RUNNING;
  }

  void onHalted() override {}

private:
  WorldModelPtr wm_;
  SkillBridgePtr bridge_;
  rclcpp::Logger logger_;

  std::string part_;
  std::string command_id_;

  int attempt_{0};
  int wait_ticks_{0};
};

class PlacePart : public BT::StatefulActionNode
{
public:
  PlacePart(
    const std::string & name,
    const BT::NodeConfig & config,
    WorldModelPtr wm,
    SkillBridgePtr bridge,
    rclcpp::Logger logger)
  : BT::StatefulActionNode(name, config),
    wm_(wm), bridge_(bridge), logger_(logger)
  {}

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<std::string>("part")};
  }

  BT::NodeStatus onStart() override
  {
    getInput("part", part_);

    const auto state = wm_->getPartState(part_);

    if (state.status != PartStatus::PICKED) {
      RCLCPP_WARN(
        logger_,
        "[BT]      PlacePart: cannot place %s, not held",
        part_.c_str());

      return BT::NodeStatus::FAILURE;
    }

    command_id_ = bridge_->send("place", part_, state.position);
    wait_ticks_ = 0;

    return BT::NodeStatus::RUNNING;
  }

  BT::NodeStatus onRunning() override
  {
    const std::string result = bridge_->status(command_id_);

    if (result == "SUCCEEDED") {
      wm_->setPartStatus(part_, PartStatus::PLACED);

      RCLCPP_INFO(
        logger_,
        "[BT]      PlacePart: %s placed",
        part_.c_str());

      return BT::NodeStatus::SUCCESS;
    }

    if (result == "FAILED" || ++wait_ticks_ > 60) {
      // A failed place returns the part to the locate/pick workflow.
      wm_->setPartStatus(part_, PartStatus::LOST);

      RCLCPP_WARN(
        logger_,
        "[BT]      PlacePart: %s failed or timed out",
        part_.c_str());

      return BT::NodeStatus::FAILURE;
    }

    return BT::NodeStatus::RUNNING;
  }

  void onHalted() override {}

private:
  WorldModelPtr wm_;
  SkillBridgePtr bridge_;
  rclcpp::Logger logger_;

  std::string part_;
  std::string command_id_;

  int wait_ticks_{0};
};

class RecoveryPolicy : public BT::DecoratorNode
{
public:
  RecoveryPolicy(
    const std::string & name,
    const BT::NodeConfig & config,
    WorldModelPtr wm,
    RecoveryManagerPtr rm,
    rclcpp::Logger logger)
  : BT::DecoratorNode(name, config),
    wm_(wm), rm_(rm), logger_(logger)
  {}

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<std::string>("part")};
  }

  BT::NodeStatus tick() override
  {
    std::string part;
    getInput("part", part);

    setStatus(BT::NodeStatus::RUNNING);

    const BT::NodeStatus child_status = child_node_->executeTick();

    if (child_status == BT::NodeStatus::SUCCESS) {
      resetChild();
      return BT::NodeStatus::SUCCESS;
    }

    if (child_status == BT::NodeStatus::RUNNING) {
      return BT::NodeStatus::RUNNING;
    }

    resetChild();

    const auto state = wm_->getPartState(part);

    const auto failure_type =
      state.status == PartStatus::LOST
      ? zebra_bt::FailureType::PART_LOST
      : zebra_bt::FailureType::PICK_FAILED;

    const auto action = rm_->decide(part, failure_type);

    RCLCPP_WARN(
      logger_,
      "[RECOVER] %s -> %s",
      part.c_str(),
      zebra_bt::toString(action).c_str());

    if (action == zebra_bt::RecoveryAction::ESCALATE) {
      wm_->setPartStatus(part, PartStatus::ESCALATED);
      return BT::NodeStatus::FAILURE;
    }

    if (action == zebra_bt::RecoveryAction::RELOCATE) {
      // Return FAILURE so the parent Sequence restarts next tick at
      // EnsureLocated, causing LocatePart to run again.
      return BT::NodeStatus::FAILURE;
    }

    // RETRY: retry PickPart on the next BT tick.
    return BT::NodeStatus::RUNNING;
  }

  void halt() override
  {
    resetChild();
    resetStatus();
  }

private:
  WorldModelPtr wm_;
  RecoveryManagerPtr rm_;
  rclcpp::Logger logger_;
};

}  // namespace zebra_bt

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("zebra_bt_node");

  std::string bom_file;
  std::vector<std::string> part_ids;

  try {
    bom_file =
      ament_index_cpp::get_package_share_directory("build_a_zebra") +
      "/resources/bom.json";

    zebra_bt::TaskModel task_model(bom_file);

    RCLCPP_INFO(
      node->get_logger(),
      "Task Model loaded. Build target: %s (%zu parts)",
      task_model.buildTarget().c_str(),
      task_model.buildOrder().size());

    for (const auto & part : task_model.buildOrder()) {
      part_ids.push_back(part.part_id);

      RCLCPP_INFO(
        node->get_logger(),
        "  -> %s (%s)",
        part.part_id.c_str(),
        part.role.c_str());
    }
  } catch (const std::exception & error) {
    RCLCPP_ERROR(
      node->get_logger(),
      "Failed to load Task Model: %s",
      error.what());
    rclcpp::shutdown();
    return 1;
  }

  auto world_model = std::make_shared<WorldModel>(node, part_ids, /*perception_timeout_sec=*/2.0);

  auto recovery_manager = std::make_shared<RecoveryManager>(
    /*max_pick_retries=*/3,
    /*max_relocate_attempts=*/2);

  auto skill_bridge = std::make_shared<zebra_bt::SkillBridge>(node);

  const std::string disturbances_file =
    ament_index_cpp::get_package_share_directory("build_a_zebra") +
    "/resources/disturbances.json";

  zebra_bt::DisturbanceGenerator disturbance_generator(
    node, disturbances_file);

  BT::BehaviorTreeFactory factory;

  factory.registerBuilder<zebra_bt::IsPartPlaced>(
    "IsPartPlaced",
    [world_model](
      const std::string & name,
      const BT::NodeConfig & config)
    {
      return std::make_unique<zebra_bt::IsPartPlaced>(name, config, world_model);
    });

  factory.registerBuilder<zebra_bt::IsPartLocated>(
    "IsPartLocated",
    [world_model](
      const std::string & name,
      const BT::NodeConfig & config)
    {
      return std::make_unique<zebra_bt::IsPartLocated>(name, config, world_model);
    });

  factory.registerBuilder<zebra_bt::IsPartEscalated>(
    "IsPartEscalated",
    [world_model](
      const std::string & name,
      const BT::NodeConfig & config)
    {
      return std::make_unique<zebra_bt::IsPartEscalated>(name, config, world_model);
    });

  factory.registerBuilder<zebra_bt::LocatePart>(
    "LocatePart",
    [world_model, recovery_manager, node](
      const std::string & name,
      const BT::NodeConfig & config)
    {
      return std::make_unique<zebra_bt::LocatePart>(
        name, config, world_model, recovery_manager, node->get_logger());
    });

  factory.registerBuilder<zebra_bt::PickPart>(
    "PickPart",
    [world_model, skill_bridge, node](
      const std::string & name,
      const BT::NodeConfig & config)
    {
      return std::make_unique<zebra_bt::PickPart>(
        name, config, world_model, skill_bridge, node->get_logger());
    });

  factory.registerBuilder<zebra_bt::PlacePart>(
    "PlacePart",
    [world_model, skill_bridge, node](
      const std::string & name,
      const BT::NodeConfig & config)
    {
      return std::make_unique<zebra_bt::PlacePart>(
        name, config, world_model, skill_bridge, node->get_logger());
    });

  factory.registerBuilder<zebra_bt::RecoveryPolicy>(
    "RecoveryPolicy",
    [world_model, recovery_manager, node](
      const std::string & name,
      const BT::NodeConfig & config)
    {
      return std::make_unique<zebra_bt::RecoveryPolicy>(
        name, config, world_model, recovery_manager, node->get_logger());
    });

  std::string tree_file;
  try {
    tree_file =
      ament_index_cpp::get_package_share_directory("build_a_zebra") +
      "/trees/zebra_tree.xml";
  } catch (const std::exception & error) {
    RCLCPP_ERROR(
      node->get_logger(),
      "Could not locate package share directory: %s",
      error.what());
    rclcpp::shutdown();
    return 1;
  }

  auto tree = factory.createTreeFromFile(tree_file);

  // Filtered logger — only shows RUNNING -> SUCCESS/FAILURE transitions.
  // Replaces BT::StdCoutLogger, which printed the whole IDLE/RUNNING wall.
  zebra_bt::QuietLogger quiet_logger(tree.rootNode());
  quiet_logger.setEnabled(true);

  RCLCPP_INFO(
    node->get_logger(),
    "Zebra Behavior Tree started. Ticking at 2 Hz...");

  const int STATUS_EVERY = 10;   // ticks between world-state boards (5 s @ 2 Hz)

  auto print_status_board = [&](int tick) {
    std::cout << "\n──── World state @ tick " << tick << " ────\n";
    for (const auto & id : part_ids) {
      const auto st = world_model->getPartState(id);
      std::printf(
        "  %-10s  %-12s  attempts=%d\n",
        id.c_str(),
        zebra_bt::toString(st.status).c_str(),
        st.pick_attempts);
    }
    std::cout << "──────────────────────────────────────\n\n";
  };

  rclcpp::Rate rate(2.0);
  int tick_count = 0;

  print_status_board(tick_count);

  while (rclcpp::ok() && !world_model->allPartsResolved()) {
    rclcpp::spin_some(node);

    if (tick_count % disturbance_generator.checkIntervalTicks() == 0) {
      disturbance_generator.maybeTrigger(world_model, part_ids);
    }

    tree.tickOnce();
    tick_count++;

    if (tick_count % STATUS_EVERY == 0) {
      print_status_board(tick_count);
    }

    rate.sleep();
  }

  // ---- Final summary table ------------------------------------------
  std::printf("\n");
  std::printf("══════════════════════════════════════════\n");
  std::printf("  BUILD SUMMARY\n");
  std::printf("══════════════════════════════════════════\n");
  std::printf("  %-10s  %-12s  %-8s\n", "PART", "STATUS", "ATTEMPTS");
  std::printf("  ──────────  ────────────  ────────\n");

  int placed = 0;
  int escalated = 0;

  for (const auto & id : part_ids) {
    const auto st = world_model->getPartState(id);

    std::printf(
      "  %-10s  %-12s  %-8d\n",
      id.c_str(),
      zebra_bt::toString(st.status).c_str(),
      st.pick_attempts);

    if (st.status == PartStatus::PLACED)    ++placed;
    if (st.status == PartStatus::ESCALATED) ++escalated;
  }

  std::printf("  ──────────  ────────────  ────────\n");
  std::printf(
    "  %d placed, %d escalated, %zu total\n",
    placed, escalated, part_ids.size());
  std::printf("══════════════════════════════════════════\n\n");

  if (escalated == 0) {
    RCLCPP_INFO(node->get_logger(), "Zebra fully assembled.");
  } else {
    RCLCPP_ERROR(
      node->get_logger(),
      "Zebra build incomplete: %d part(s) need human help.",
      escalated);
  }

  rclcpp::shutdown();
  return escalated > 0 ? 1 : 0;
}