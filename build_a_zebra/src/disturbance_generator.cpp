#include "zebra_bt/disturbance_generator.hpp"

#include <fstream>
<<<<<<< HEAD
=======
#include <iostream>
>>>>>>> 95f1652 (Add Zebra behavior tree and ROS skill bridge)
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace zebra_bt
{

<<<<<<< HEAD
using std::string;
using std::runtime_error;
using std::random_device;
using std::ifstream;
using std::shared_ptr;
using std::vector;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

string toString(DisturbanceType type)
=======
std::string toString(DisturbanceType type)
>>>>>>> 95f1652 (Add Zebra behavior tree and ROS skill bridge)
{
  switch (type) {
    case DisturbanceType::LOSS: return "LOSS";
    case DisturbanceType::RELOCATION: return "RELOCATION";
    case DisturbanceType::SENSOR_NOISE: return "SENSOR_NOISE";
  }
  return "UNKNOWN";
}

<<<<<<< HEAD
string toString(Severity severity)
=======
std::string toString(Severity severity)
>>>>>>> 95f1652 (Add Zebra behavior tree and ROS skill bridge)
{
  switch (severity) {
    case Severity::MINOR: return "minor";
    case Severity::MAJOR: return "major";
  }
  return "unknown";
}

namespace
{
<<<<<<< HEAD
DisturbanceType parseType(const string & s)
=======
DisturbanceType parseType(const std::string & s)
>>>>>>> 95f1652 (Add Zebra behavior tree and ROS skill bridge)
{
  if (s == "LOSS") return DisturbanceType::LOSS;
  if (s == "RELOCATION") return DisturbanceType::RELOCATION;
  if (s == "SENSOR_NOISE") return DisturbanceType::SENSOR_NOISE;
<<<<<<< HEAD
  throw runtime_error("DisturbanceGenerator: unknown event type '" + s + "'");
}

Severity parseSeverity(const string & s)
=======
  throw std::runtime_error("DisturbanceGenerator: unknown event type '" + s + "'");
}

Severity parseSeverity(const std::string & s)
>>>>>>> 95f1652 (Add Zebra behavior tree and ROS skill bridge)
{
  if (s == "major") return Severity::MAJOR;
  return Severity::MINOR;
}
}  // namespace

<<<<<<< HEAD
// [  Class Name  ]  :: [ Constructor Name ] ( [      Parameters     ] )
DisturbanceGenerator::DisturbanceGenerator(
  const rclcpp::Node::SharedPtr & node, const string & config_path)
: node_(node), rng_(random_device{}())
{
  ifstream file(config_path);   // Open file
  if (!file.is_open()) {
    throw runtime_error("DisturbanceGenerator: could not open config file: " + config_path);
  }

  nlohmann::json j;             // Creates empty, smart JSON object container named j
  file >> j;                    // Input raw text inside your open file stream and pours it directly into j
=======
DisturbanceGenerator::DisturbanceGenerator(const std::string & config_path)
: rng_(std::random_device{}())
{
  std::ifstream file(config_path);
  if (!file.is_open()) {
    throw std::runtime_error("DisturbanceGenerator: could not open config file: " + config_path);
  }

  nlohmann::json j;
  file >> j;
>>>>>>> 95f1652 (Add Zebra behavior tree and ROS skill bridge)

  check_interval_ticks_ = j.value("check_interval_ticks", 4);

  for (const auto & entry : j.at("events")) {
    DisturbanceEvent event;
<<<<<<< HEAD
    event.type = parseType(entry.at("type").get<string>());
=======
    event.type = parseType(entry.at("type").get<std::string>());
>>>>>>> 95f1652 (Add Zebra behavior tree and ROS skill bridge)
    event.probability = entry.at("probability").get<double>();
    event.severity = parseSeverity(entry.value("severity", "minor"));
    event.description = entry.value("description", "");
    events_.push_back(event);
  }
}

bool DisturbanceGenerator::maybeTrigger(
<<<<<<< HEAD
  const shared_ptr<WorldModel> & wm, const vector<string> & part_ids)
{
  // Function checks what is there and randomly disturb a part. 
  // so if the leg is already in place you cant disturb so it will be remove.
  // To do that you need wm so u can see it live
  vector<string> eligible;
=======
  const std::shared_ptr<WorldModel> & wm, const std::vector<std::string> & part_ids)
{
  // Only disturb parts that are still "in play" -- no point relocating
  // something already placed on the zebra or already given up on.
  std::vector<std::string> eligible;
>>>>>>> 95f1652 (Add Zebra behavior tree and ROS skill bridge)
  for (const auto & id : part_ids) {
    auto status = wm->getPartState(id).status;
    if (status != PartStatus::PLACED && status != PartStatus::ESCALATED) {
      eligible.push_back(id);
    }
  }
  if (eligible.empty()) return false;

<<<<<<< HEAD
  // Build the die (named part_pick)
  // ROLL the die by putting your random engine inside the parentheses: part_pick(rng_)
  uniform_int_distribution<int> part_pick(0, static_cast<int>(eligible.size()) - 1);
  const string & part = eligible[part_pick(rng_)];
  uniform_real_distribution<double> roll(0.0, 1.0);

  for (const auto & event : events_) {
    if (roll(rng_) >= event.probability) continue;  // this event type didn't fire
    
    auto state = wm->getPartState(part);
    
    // Forcefully changes that chosen part's status to LOST
    if (event.type == DisturbanceType::LOSS) {
      wm->setPartStatus(part, PartStatus::LOST);
      RCLCPP_WARN(
        node_->get_logger(),
        ">>> DISTURBANCE: %s -> LOSS (%s) -- %s <<<",
        part.c_str(), toString(event.severity).c_str(), event.description.c_str());
=======
  std::uniform_int_distribution<int> part_pick(0, static_cast<int>(eligible.size()) - 1);
  const std::string & part = eligible[part_pick(rng_)];
  std::uniform_real_distribution<double> roll(0.0, 1.0);

  for (const auto & event : events_) {
    if (roll(rng_) >= event.probability) continue;  // this event type didn't fire

    auto state = wm->getPartState(part);

    if (event.type == DisturbanceType::LOSS) {
      wm->setPartStatus(part, PartStatus::LOST);
      std::cout << "[Disturbance] " << part << ": LOSS (" << toString(event.severity)
                << ") -- " << event.description << "\n";
>>>>>>> 95f1652 (Add Zebra behavior tree and ROS skill bridge)
      return true;
    }

    // RELOCATION and SENSOR_NOISE only make sense on a part we currently
    // know the position of. If it's not LOCATED, skip this event type and
    // let the loop try the next configured event instead.
    if (state.status != PartStatus::LOCATED) continue;

    if (event.type == DisturbanceType::RELOCATION) {
<<<<<<< HEAD
      uniform_real_distribution<double> jitter(-0.10, 0.10);
      wm->setPartPosition(
        part, state.position.x + jitter(rng_), state.position.y + jitter(rng_), state.position.z);
      RCLCPP_WARN(
        node_->get_logger(),
        ">>> DISTURBANCE: %s -> RELOCATION (%s) -- %s <<<",
        part.c_str(), toString(event.severity).c_str(), event.description.c_str());
=======
      std::uniform_real_distribution<double> jitter(-0.10, 0.10);
      wm->setPartPosition(
        part, state.position.x + jitter(rng_), state.position.y + jitter(rng_), state.position.z);
      std::cout << "[Disturbance] " << part << ": RELOCATION (" << toString(event.severity)
                << ") -- " << event.description << "\n";
>>>>>>> 95f1652 (Add Zebra behavior tree and ROS skill bridge)
      return true;
    }

    if (event.type == DisturbanceType::SENSOR_NOISE) {
<<<<<<< HEAD
      uniform_real_distribution<double> jitter(-0.02, 0.02);
      wm->setPartPosition(
        part, state.position.x + jitter(rng_), state.position.y + jitter(rng_), state.position.z);
      RCLCPP_INFO(
        node_->get_logger(),
        "[Disturbance] %s: SENSOR_NOISE (%s) -- %s",
        part.c_str(), toString(event.severity).c_str(), event.description.c_str());
=======
      std::uniform_real_distribution<double> jitter(-0.02, 0.02);
      wm->setPartPosition(
        part, state.position.x + jitter(rng_), state.position.y + jitter(rng_), state.position.z);
      std::cout << "[Disturbance] " << part << ": SENSOR_NOISE (" << toString(event.severity)
                << ") -- " << event.description << "\n";
>>>>>>> 95f1652 (Add Zebra behavior tree and ROS skill bridge)
      return true;
    }
  }

  return false;
}

}  // namespace zebra_bt