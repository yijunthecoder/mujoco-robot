#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
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
using RolesMapPtr = std::shared_ptr<const std::map<std::string, std::string>>;

namespace zebra_bt
{
  // ---------------------------------------------------------------------
  // Assembly pose: where each part goes when it's placed.
  // Legs go on the table, body stacks on top of legs, head on top of body.
  // Tune these to match the physical assembly point in the sim.
  // ---------------------------------------------------------------------
  static geometry_msgs::msg::Point placeTargetFor(const std::string & part_id)
  {
    constexpr double TABLE_X      = 0.4148;
    constexpr double TABLE_Y      = 0.0;
    constexpr double TABLE_Z      = -0.1026;
    constexpr double BRICK_HEIGHT = 0.04;

    geometry_msgs::msg::Point p;
    p.x = TABLE_X;
    p.y = TABLE_Y;

    if (part_id == "31111p0e") {          // legs
      p.z = TABLE_Z;
    } else if (part_id == "31111p0f") {   // body
      p.z = TABLE_Z + BRICK_HEIGHT;
    } else if (part_id == "31111p0g") {   // head
      p.z = TABLE_Z + 2 * BRICK_HEIGHT;
    } else {
      p.z = TABLE_Z;
    }
    return p;
  }

// ---------------------------------------------------------------------
// Human-readable part label: "legs" for "31111p0e".
// ---------------------------------------------------------------------
static std::string prettyPart(const std::string & id, const RolesMapPtr & roles)
{
  if (roles) {
    auto it = roles->find(id);
    if (it != roles->end()) return it->second;
  }
  return id;
}

// ---------------------------------------------------------------------
// SkillBridge — sends JSON commands, tracks status replies.
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
    return command_id;
  }

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
    } catch (const std::exception &) {
      // Ignore malformed status messages.
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
// QuietLogger — prints only RUNNING -> SUCCESS/FAILURE transitions.
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
    if (prev != BT::NodeStatus::RUNNING) return;

    std::cout << "    [BT] " << std::left << std::setw(24)
              << node.name() << "  "
              << BT::toStr(prev) << " -> " << BT::toStr(curr)
              << "\n";
  }

  void flush() override {}
};

// ---------------------------------------------------------------------
// Condition nodes.
// ---------------------------------------------------------------------

class IsPartPlaced : public BT::ConditionNode
{
public:
  IsPartPlaced(const std::string & name, const BT::NodeConfig & config,
               WorldModelPtr wm)
  : BT::ConditionNode(name, config), wm_(wm) {}

  static BT::PortsList providedPorts()
  { return {BT::InputPort<std::string>("part")}; }

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
  IsPartLocated(const std::string & name, const BT::NodeConfig & config,
                WorldModelPtr wm)
  : BT::ConditionNode(name, config), wm_(wm) {}

  static BT::PortsList providedPorts()
  { return {BT::InputPort<std::string>("part")}; }

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
  IsPartEscalated(const std::string & name, const BT::NodeConfig & config,
                  WorldModelPtr wm)
  : BT::ConditionNode(name, config), wm_(wm) {}

  static BT::PortsList providedPorts()
  { return {BT::InputPort<std::string>("part")}; }

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
  LocatePart(const std::string & name, const BT::NodeConfig & config,
             WorldModelPtr wm, RecoveryManagerPtr rm,
             RolesMapPtr roles, rclcpp::Logger logger)
  : BT::StatefulActionNode(name, config),
    wm_(wm), rm_(rm), roles_(roles), logger_(logger) {}

  static BT::PortsList providedPorts()
  { return {BT::InputPort<std::string>("part")}; }

  BT::NodeStatus onStart() override
  {
    getInput("part", part_);
    search_ticks_ = 0;

    const auto state = wm_->getPartState(part_);

    if (state.status == PartStatus::LOST) {
      const auto action = rm_->decide(part_, zebra_bt::FailureType::PART_LOST);

      RCLCPP_WARN(
        logger_,
        "[RECOVER] %s (lost): decision = %s",
        prettyPart(part_, roles_).c_str(),
        zebra_bt::toString(action).c_str());

      if (action == zebra_bt::RecoveryAction::ESCALATE) {
        wm_->setPartStatus(part_, PartStatus::ESCALATED);
        RCLCPP_ERROR(
          logger_,
          "[ABORT]   %s: too many losses, escalating to human",
          prettyPart(part_, roles_).c_str());
        return BT::NodeStatus::FAILURE;
      }
    }

    RCLCPP_INFO(
      logger_,
      "[LOCATE]  %s: searching...",
      prettyPart(part_, roles_).c_str());

    return BT::NodeStatus::RUNNING;
  }

  BT::NodeStatus onRunning() override
  {
    ++search_ticks_;

    const auto state = wm_->getPartState(part_);

    // Success: perception has located the part.
    if (state.seen_by_perception && state.status == PartStatus::LOCATED) {
      RCLCPP_INFO(
        logger_,
        "[LOCATE]  %s: found at (%.2f, %.2f, %.2f)",
        prettyPart(part_, roles_).c_str(),
        state.position.x, state.position.y, state.position.z);
      return BT::NodeStatus::SUCCESS;
    }

    // Timeout: no perception within budget. Mark LOST so RecoveryPolicy
    // (called from onStart next tick) decides RETRY / RELOCATE / ESCALATE.
    if (search_ticks_ > locate_timeout_ticks_) {
      RCLCPP_WARN(
        logger_,
        "[LOCATE]  %s: no perception after %.1fs -- marking LOST",
        prettyPart(part_, roles_).c_str(),
        locate_timeout_ticks_ / 2.0);
      wm_->setPartStatus(part_, PartStatus::LOST);
      return BT::NodeStatus::FAILURE;
    }

    // Still waiting.
    return BT::NodeStatus::RUNNING;
  }

  void onHalted() override {}

private:
  WorldModelPtr wm_;
  RecoveryManagerPtr rm_;
  RolesMapPtr roles_;
  rclcpp::Logger logger_;
  std::string part_;
  int search_ticks_{0};
  static constexpr int locate_timeout_ticks_{20};   // 10 s at 2 Hz
};

class PickPart : public BT::StatefulActionNode
{
public:
  PickPart(const std::string & name, const BT::NodeConfig & config,
           WorldModelPtr wm, SkillBridgePtr bridge,
           RolesMapPtr roles, rclcpp::Logger logger)
  : BT::StatefulActionNode(name, config),
    wm_(wm), bridge_(bridge), roles_(roles), logger_(logger) {}

  static BT::PortsList providedPorts()
  { return {BT::InputPort<std::string>("part")}; }

  BT::NodeStatus onStart() override
  {
    getInput("part", part_);
    const auto state = wm_->getPartState(part_);

    if (state.status == PartStatus::LOST) {
      RCLCPP_ERROR(
        logger_,
        "[PICK]    %s: CANNOT pick - part is LOST",
        prettyPart(part_, roles_).c_str());
      return BT::NodeStatus::FAILURE;
    }

    attempt_ = wm_->incrementPickAttempts(part_);
    command_id_ = bridge_->send("pick", part_, state.position);

    RCLCPP_INFO(
      logger_,
      "[PICK]    %s: attempt %d, target (%.2f, %.2f, %.2f)  [%s]",
      prettyPart(part_, roles_).c_str(),
      attempt_,
      state.position.x, state.position.y, state.position.z,
      command_id_.c_str());

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
        "[PICK]    %s: SUCCESS (attempt %d)",
        prettyPart(part_, roles_).c_str(), attempt_);
      return BT::NodeStatus::SUCCESS;
    }

    if (result == "FAILED") {
      wm_->setPartStatus(part_, PartStatus::PICK_FAILED);
      RCLCPP_WARN(
        logger_,
        "[PICK]    %s: FAILED (attempt %d) -- executor reported failure",
        prettyPart(part_, roles_).c_str(), attempt_);
      return BT::NodeStatus::FAILURE;
    }

    if (++wait_ticks_ > 60) {
      wm_->setPartStatus(part_, PartStatus::PICK_FAILED);
      RCLCPP_WARN(
        logger_,
        "[PICK]    %s: TIMEOUT after 30s -- no reply from executor",
        prettyPart(part_, roles_).c_str());
      return BT::NodeStatus::FAILURE;
    }

    return BT::NodeStatus::RUNNING;
  }

  void onHalted() override {}

private:
  WorldModelPtr wm_;
  SkillBridgePtr bridge_;
  RolesMapPtr roles_;
  rclcpp::Logger logger_;
  std::string part_;
  std::string command_id_;
  int attempt_{0};
  int wait_ticks_{0};
};

class PlacePart : public BT::StatefulActionNode
{
public:
  PlacePart(const std::string & name, const BT::NodeConfig & config,
            WorldModelPtr wm, SkillBridgePtr bridge,
            RolesMapPtr roles, rclcpp::Logger logger)
  : BT::StatefulActionNode(name, config),
    wm_(wm), bridge_(bridge), roles_(roles), logger_(logger) {}

  static BT::PortsList providedPorts()
  { return {BT::InputPort<std::string>("part")}; }

  BT::NodeStatus onStart() override
  {
    getInput("part", part_);
    const auto state = wm_->getPartState(part_);

    if (state.status != PartStatus::PICKED) {
      RCLCPP_ERROR(
        logger_,
        "[PLACE]   %s: CANNOT place - part is not currently held",
        prettyPart(part_, roles_).c_str());
      return BT::NodeStatus::FAILURE;
    }
    
    const auto place_target = placeTargetFor(part_);
    command_id_ = bridge_->send("place", part_, place_target);
    RCLCPP_INFO(
      logger_,
      "[PLACE]   %s: placing at (%.2f, %.2f, %.2f)  [%s]",
      prettyPart(part_, roles_).c_str(),
      place_target.x, place_target.y, place_target.z,
      command_id_.c_str());

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
        "[PLACE]   %s: SUCCESS -- part is on the zebra",
        prettyPart(part_, roles_).c_str());
      return BT::NodeStatus::SUCCESS;
    }

    if (result == "FAILED" || ++wait_ticks_ > 60) {
      wm_->setPartStatus(part_, PartStatus::LOST);
      RCLCPP_WARN(
        logger_,
        "[PLACE]   %s: FAILED -- part returned to locate/pick workflow",
        prettyPart(part_, roles_).c_str());
      return BT::NodeStatus::FAILURE;
    }

    return BT::NodeStatus::RUNNING;
  }

  void onHalted() override {}

private:
  WorldModelPtr wm_;
  SkillBridgePtr bridge_;
  RolesMapPtr roles_;
  rclcpp::Logger logger_;
  std::string part_;
  std::string command_id_;
  int wait_ticks_{0};
};

class RecoveryPolicy : public BT::DecoratorNode
{
public:
  RecoveryPolicy(const std::string & name, const BT::NodeConfig & config,
                 WorldModelPtr wm, RecoveryManagerPtr rm,
                 RolesMapPtr roles, rclcpp::Logger logger)
  : BT::DecoratorNode(name, config),
    wm_(wm), rm_(rm), roles_(roles), logger_(logger) {}

  static BT::PortsList providedPorts()
  { return {BT::InputPort<std::string>("part")}; }

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
      "[RECOVER] %s: decision = %s",
      prettyPart(part, roles_).c_str(),
      zebra_bt::toString(action).c_str());

    if (action == zebra_bt::RecoveryAction::ESCALATE) {
      wm_->setPartStatus(part, PartStatus::ESCALATED);
      RCLCPP_ERROR(
        logger_,
        "[ABORT]   %s: retries exhausted, escalating to human",
        prettyPart(part, roles_).c_str());
      return BT::NodeStatus::FAILURE;
    }

    if (action == zebra_bt::RecoveryAction::RELOCATE) {
      RCLCPP_WARN(
        logger_,
        "[RECOVER] %s: will re-locate and retry",
        prettyPart(part, roles_).c_str());
      return BT::NodeStatus::FAILURE;
    }

    RCLCPP_WARN(
      logger_,
      "[RECOVER] %s: retrying the pick",
      prettyPart(part, roles_).c_str());
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
  RolesMapPtr roles_;
  rclcpp::Logger logger_;
};

}  // namespace zebra_bt

int main(int argc, char ** argv)
{
  // Human-readable log format. Must be set BEFORE rclcpp::init —
  // rcutils reads this env var once at startup.
  setenv("RCUTILS_CONSOLE_OUTPUT_FORMAT",
         "[{severity}] [{date_time_with_ms}] {message}",
         1);

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("zebra_bt_node");

  std::string bom_file;
  std::vector<std::string> part_ids;
  auto roles = std::make_shared<std::map<std::string, std::string>>();
  std::string build_target = "unknown";

  try {
    bom_file =
      ament_index_cpp::get_package_share_directory("build_a_zebra") +
      "/resources/bom.json";

    zebra_bt::TaskModel task_model(bom_file);
    build_target = task_model.buildTarget();

    std::printf("\n");
    std::printf("══════════════════════════════════════════\n");
    std::printf("  ZEBRA BUILD STARTED\n");
    std::printf("══════════════════════════════════════════\n");
    std::printf("  Animal : %s\n", build_target.c_str());
    std::printf("  Parts  :\n");
    for (const auto & part : task_model.buildOrder()) {
      part_ids.push_back(part.part_id);
      (*roles)[part.part_id] = part.role;
      std::printf("    %-6s  (%s)\n", part.role.c_str(), part.part_id.c_str());
    }
    std::printf("══════════════════════════════════════════\n\n");
  } catch (const std::exception & error) {
    RCLCPP_ERROR(node->get_logger(),
      "Failed to load Task Model: %s", error.what());
    rclcpp::shutdown();
    return 1;
  }

  auto world_model = std::make_shared<WorldModel>(
    node, part_ids, /*perception_timeout_sec=*/2.0);

  auto recovery_manager = std::make_shared<RecoveryManager>(
    /*max_pick_retries=*/3, /*max_relocate_attempts=*/2);

  auto skill_bridge = std::make_shared<zebra_bt::SkillBridge>(node);

  const std::string disturbances_file =
    ament_index_cpp::get_package_share_directory("build_a_zebra") +
    "/resources/disturbances.json";

  zebra_bt::DisturbanceGenerator disturbance_generator(
    node, disturbances_file);

  BT::BehaviorTreeFactory factory;
  const double TICK_HZ = 2.0;

  factory.registerBuilder<zebra_bt::IsPartPlaced>("IsPartPlaced",
    [world_model](const std::string & n, const BT::NodeConfig & c) {
      return std::make_unique<zebra_bt::IsPartPlaced>(n, c, world_model); });

  factory.registerBuilder<zebra_bt::IsPartLocated>("IsPartLocated",
    [world_model](const std::string & n, const BT::NodeConfig & c) {
      return std::make_unique<zebra_bt::IsPartLocated>(n, c, world_model); });

  factory.registerBuilder<zebra_bt::IsPartEscalated>("IsPartEscalated",
    [world_model](const std::string & n, const BT::NodeConfig & c) {
      return std::make_unique<zebra_bt::IsPartEscalated>(n, c, world_model); });

  factory.registerBuilder<zebra_bt::LocatePart>("LocatePart",
    [world_model, recovery_manager, roles, node]
    (const std::string & n, const BT::NodeConfig & c) {
      return std::make_unique<zebra_bt::LocatePart>(
        n, c, world_model, recovery_manager, roles, node->get_logger()); });

  factory.registerBuilder<zebra_bt::PickPart>("PickPart",
    [world_model, skill_bridge, roles, node]
    (const std::string & n, const BT::NodeConfig & c) {
      return std::make_unique<zebra_bt::PickPart>(
        n, c, world_model, skill_bridge, roles, node->get_logger()); });

  factory.registerBuilder<zebra_bt::PlacePart>("PlacePart",
    [world_model, skill_bridge, roles, node]
    (const std::string & n, const BT::NodeConfig & c) {
      return std::make_unique<zebra_bt::PlacePart>(
        n, c, world_model, skill_bridge, roles, node->get_logger()); });

  factory.registerBuilder<zebra_bt::RecoveryPolicy>("RecoveryPolicy",
    [world_model, recovery_manager, roles, node]
    (const std::string & n, const BT::NodeConfig & c) {
      return std::make_unique<zebra_bt::RecoveryPolicy>(
        n, c, world_model, recovery_manager, roles, node->get_logger()); });

  std::string tree_file =
    ament_index_cpp::get_package_share_directory("build_a_zebra") +
    "/trees/zebra_tree.xml";

  auto tree = factory.createTreeFromFile(tree_file);
  zebra_bt::QuietLogger quiet_logger(tree.rootNode());
  quiet_logger.setEnabled(true);

  RCLCPP_INFO(node->get_logger(), "Ticking at %.1f Hz...", TICK_HZ);

  const int STATUS_EVERY = int(TICK_HZ * 5);   // 5 seconds
  rclcpp::Rate rate(TICK_HZ);
  int tick_count = 0;

  auto print_status_board = [&](int tick) {
    const double secs = tick / TICK_HZ;
    std::printf("\n──── ZEBRA  ·  %.1fs  ·  step %d ────\n", secs, tick);
    std::printf("  %-8s  %-12s  %-8s  %s\n",
                "PART", "STATUS", "ATTEMPTS", "MEANING");
    std::printf("  ────────  ────────────  ────────  ───────\n");
    for (const auto & id : part_ids) {
      const auto st = world_model->getPartState(id);
      const std::string label = (*roles).count(id) ? (*roles)[id] : id;

      const char * meaning = "";
      switch (st.status) {
        case PartStatus::UNKNOWN:     meaning = "not yet located"; break;
        case PartStatus::LOCATED:     meaning = "position known";  break;
        case PartStatus::LOST:        meaning = "cannot find it";  break;
        case PartStatus::PICKED:      meaning = "in the gripper";  break;
        case PartStatus::PICK_FAILED: meaning = "pick failed, retrying"; break;
        case PartStatus::PLACED:      meaning = "on the zebra -- DONE"; break;
        case PartStatus::ESCALATED:   meaning = "gave up, needs human"; break;
      }
      std::printf("  %-8s  %-12s  %-8d  %s\n",
                  label.c_str(),
                  zebra_bt::toString(st.status).c_str(),
                  st.pick_attempts,
                  meaning);
    }
    std::cout << "──────────────────────────────────────\n\n";
  };

  print_status_board(0);

  // Read ALL waiting messages each tick, not just one (spin_some only takes
  // one message per subscription per call, so perception backed up).
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  while (rclcpp::ok() && !world_model->allPartsResolved()) {
    executor.spin_all(std::chrono::milliseconds(100));

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

  const bool completed = world_model->allPartsResolved();
  const double elapsed_sec = tick_count / TICK_HZ;

  std::printf("\n");
  std::printf("══════════════════════════════════════════\n");
  std::printf("  BUILD SUMMARY  ·  %s\n", build_target.c_str());
  std::printf("══════════════════════════════════════════\n");
  std::printf("  %-8s  %-12s  %-8s\n", "PART", "STATUS", "ATTEMPTS");
  std::printf("  ────────  ────────────  ────────\n");

  int placed = 0;
  int escalated = 0;

  for (const auto & id : part_ids) {
    const auto st = world_model->getPartState(id);
    const std::string label = (*roles).count(id) ? (*roles)[id] : id;
    std::printf("  %-8s  %-12s  %-8d\n",
                label.c_str(),
                zebra_bt::toString(st.status).c_str(),
                st.pick_attempts);
    if (st.status == PartStatus::PLACED)    ++placed;
    if (st.status == PartStatus::ESCALATED) ++escalated;
  }

  std::printf("  ────────  ────────────  ────────\n");
  std::printf("  %d placed, %d escalated, %zu total\n",
              placed, escalated, part_ids.size());
  std::printf("  elapsed: %.1f s  (%d ticks at %.1f Hz)\n",
              elapsed_sec, tick_count, TICK_HZ);
  std::printf("══════════════════════════════════════════\n\n");

  if (!completed) {
    RCLCPP_WARN(node->get_logger(),
      "Build aborted early (Ctrl+C). %d of %zu parts resolved.",
      placed + escalated, part_ids.size());
  } else if (escalated == 0) {
    RCLCPP_INFO(node->get_logger(), "Zebra fully assembled.");
  } else {
    RCLCPP_ERROR(node->get_logger(),
      "Zebra build incomplete: %d part(s) need human help.", escalated);
  }

  rclcpp::shutdown();
  return escalated > 0 ? 1 : 0;
}