#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "zebra_bt/world_model.hpp"

namespace zebra_bt
{

enum class DisturbanceType
{
  LOSS,         // part LOST, needs LocatePart to re-run       | Enum 0
  RELOCATION,   // part bumped on new position, still LOCATED  | Enum 1
  SENSOR_NOISE  // Tiny jitter, doesn't change status          | Enum 2
};

enum class Severity
{
  MINOR,  // Doesn't stop progress, only adds noise   | Enum 0
  MAJOR   // Interrupts the current plan              | Enum 1
};

std::string toString(DisturbanceType type);
std::string toString(Severity severity);

struct DisturbanceEvent
{
  DisturbanceType type;
  double probability;  // 0.0-1.0, checked independently per part per interval
  Severity severity;
  std::string description;
};

// DisturbanceGenerator is the environment's side of things: it doesn't know
// or care about the Behavior Tree, only about randomly perturbing parts in
// the WorldModel according to a configurable set of event types and rates,
// loaded from resources/disturbances.json (same pattern as TaskModel's bom.json).
class DisturbanceGenerator
{
public:
  // node: used only so disturbance events can be logged via RCLCPP_WARN
  // (shows up colored/highlighted in the console), so they're easy to spot
  // against the constant stream of BT tick-trace lines.
  DisturbanceGenerator(const rclcpp::Node::SharedPtr & node, const std::string & config_path);

  int checkIntervalTicks() const {return check_interval_ticks_;}

  // Rolls the configured events against one randomly chosen, not-yet-
  // resolved part. Applies at most one event (the first that fires) so a
  // single check doesn't pile multiple disturbances onto the same part at
  // once. Returns true if something happened (for logging by the caller).
  bool maybeTrigger(const std::shared_ptr<WorldModel> & wm, const std::vector<std::string> & part_ids);

private:
  rclcpp::Node::SharedPtr node_;
  std::vector<DisturbanceEvent> events_;
  int check_interval_ticks_;
  std::mt19937 rng_;
};

}  // namespace zebra_bt