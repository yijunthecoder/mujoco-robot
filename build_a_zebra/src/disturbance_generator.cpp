#include "zebra_bt/disturbance_generator.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace zebra_bt
{

using std::string;
using std::runtime_error;
using std::random_device;
using std::ifstream;
using std::shared_ptr;
using std::vector;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

string toString(DisturbanceType type)
{
  switch (type) {
    case DisturbanceType::LOSS: return "LOSS";
    case DisturbanceType::RELOCATION: return "RELOCATION";
    case DisturbanceType::SENSOR_NOISE: return "SENSOR_NOISE";
  }
  return "UNKNOWN";
}

string toString(Severity severity)
{
  switch (severity) {
    case Severity::MINOR: return "minor";
    case Severity::MAJOR: return "major";
  }
  return "unknown";
}

namespace
{
DisturbanceType parseType(const string & s)
{
  if (s == "LOSS") return DisturbanceType::LOSS;
  if (s == "RELOCATION") return DisturbanceType::RELOCATION;
  if (s == "SENSOR_NOISE") return DisturbanceType::SENSOR_NOISE;
  throw runtime_error("DisturbanceGenerator: unknown event type '" + s + "'");
}

Severity parseSeverity(const string & s)
{
  if (s == "major") return Severity::MAJOR;
  return Severity::MINOR;
}
}  // namespace

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

  check_interval_ticks_ = j.value("check_interval_ticks", 4);

  for (const auto & entry : j.at("events")) {
    DisturbanceEvent event;
    event.type = parseType(entry.at("type").get<string>());
    event.probability = entry.at("probability").get<double>();
    event.severity = parseSeverity(entry.value("severity", "minor"));
    event.description = entry.value("description", "");
    events_.push_back(event);
  }
}

bool DisturbanceGenerator::maybeTrigger(
  const shared_ptr<WorldModel> & wm, const vector<string> & part_ids)
{
  // Function checks what is there and randomly disturb a part. 
  // so if the leg is already in place you cant disturb so it will be remove.
  // To do that you need wm so u can see it live
  vector<string> eligible;
  for (const auto & id : part_ids) {
    auto status = wm->getPartState(id).status;
    if (status != PartStatus::PLACED && status != PartStatus::ESCALATED) {
      eligible.push_back(id);
    }
  }
  if (eligible.empty()) return false;

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
      return true;
    }

    // RELOCATION and SENSOR_NOISE only make sense on a part we currently
    // know the position of. If it's not LOCATED, skip this event type and
    // let the loop try the next configured event instead.
    if (state.status != PartStatus::LOCATED) continue;

    if (event.type == DisturbanceType::RELOCATION) {
      uniform_real_distribution<double> jitter(-0.10, 0.10);
      wm->setPartPosition(
        part, state.position.x + jitter(rng_), state.position.y + jitter(rng_), state.position.z);
      RCLCPP_WARN(
        node_->get_logger(),
        ">>> DISTURBANCE: %s -> RELOCATION (%s) -- %s <<<",
        part.c_str(), toString(event.severity).c_str(), event.description.c_str());
      return true;
    }

    if (event.type == DisturbanceType::SENSOR_NOISE) {
      uniform_real_distribution<double> jitter(-0.02, 0.02);
      wm->setPartPosition(
        part, state.position.x + jitter(rng_), state.position.y + jitter(rng_), state.position.z);
      RCLCPP_INFO(
        node_->get_logger(),
        "[Disturbance] %s: SENSOR_NOISE (%s) -- %s",
        part.c_str(), toString(event.severity).c_str(), event.description.c_str());
      return true;
    }
  }

  return false;
}

}  // namespace zebra_bt