#pragma once

#include <map>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>
#include <random>

#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace zebra_bt
{

// Every state the Behavior Tree cares about for a single Zebra part.
enum class PartStatus
{
  UNKNOWN,    // Never seen / no info yet                          | Enum 0
  LOCATED,    // Perception has a valid position for it            | Enum 1
  LOST,       // It was known before, Bit can no longer find i     | Enum 2
  PICKED,     // Gripper is currently holding it                   | Enum 3
  PICK_FAILED,// Last pick attempt failed; it's still on the table | Enum 4
  PLACED,     // Successfully assembled onto the zebra             | Enum 5
  ESCALATED   // Recovery gave up automatically; needs a human now | Enum 6
};

std::string toString(PartStatus status);

struct PartState
{
  std::string name;
  PartStatus status{PartStatus::UNKNOWN};
  geometry_msgs::msg::Point position;
  int pick_attempts{0};

  std::chrono::steady_clock::time_point last_update
  {
    std::chrono::steady_clock::now()
  };

  bool seen_by_perception{false};
};

// WorldModel is the single source of truth for where each Zebra part is and
// what state it's in. It subscribes to a ROS 2 topic where a
// perception/simulation node reports updates, and exposes thread-safe
// getters/setters that the Behavior Tree action/condition nodes read and
// write every tick.
class WorldModel
{
public:
  // part_ids: the real parts this build needs to track (e.g. from
  // TaskModel::buildOrder()), NOT hardcoded here -- so WorldModel works
  // for whichever animal/task TaskModel was configured with.
  WorldModel(const rclcpp::Node::SharedPtr & node,
            const std::vector<std::string> & part_ids,
            double perception_timeout_sec = 2.0);

  PartState getPartState(const std::string & part_name);
  void setPartStatus(const std::string & part_name, PartStatus status);
  void setPartPosition(const std::string & part_name, double x, double y, double z);
  int incrementPickAttempts(const std::string & part_name);
  void resetPickAttempts(const std::string & part_name);

  // Built-in demo/testing helper for randomly relocating or "losing" parts
  // has moved to its own class -- see zebra_bt/disturbance_generator.hpp.

  // True once every tracked part is either PLACED or ESCALATED -- i.e.
  // there's nothing more the tree can productively do automatically.
  // Used to decide when the main tick loop should stop.
  bool allPartsResolved();

private:
  void perceptionCallback(const std_msgs::msg::String::SharedPtr msg);
  void _demote_stale_locked();
  void _refresh_timestamp_locked(const std::string & part_name);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr perception_sub_;

  std::mutex mutex_;
  std::vector<std::string> part_ids_;
  std::map<std::string, PartState> parts_;
  double perception_timeout_sec_;
};

}  // namespace zebra_bt