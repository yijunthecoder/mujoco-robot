#include "zebra_bt/world_model.hpp"

#include <sstream>
#include <vector>

namespace zebra_bt
{

using std::string;
using std::stringstream;
using std::getline;
using std::vector;
using std::stod;
using std::bind;
using std::lock_guard;
using std::mutex;

string toString(PartStatus status)
{
  switch (status) {
    case PartStatus::UNKNOWN: return "UNKNOWN";
    case PartStatus::LOCATED: return "LOCATED";
    case PartStatus::LOST: return "LOST";
    case PartStatus::PICKED: return "PICKED";
    case PartStatus::PICK_FAILED: return "PICK_FAILED";
    case PartStatus::PLACED: return "PLACED";
    case PartStatus::ESCALATED: return "ESCALATED";
  }
  return "UNKNOWN";
}

WorldModel::WorldModel(const rclcpp::Node::SharedPtr & node,
                       const vector<string> & part_ids,
                       double perception_timeout_sec)
: node_(node), part_ids_(part_ids), perception_timeout_sec_(perception_timeout_sec)
{
  for (const auto & id : part_ids_) {
    PartState p;
    p.name = id;
    p.status = PartStatus::UNKNOWN;
    p.pick_attempts = 0;
    p.last_update = std::chrono::steady_clock::now();
    p.seen_by_perception = false;
    parts_[id] = p;
  }

  perception_sub_ = node_->create_subscription<std_msgs::msg::String>(
    "/zebra/perception_updates", 10,
    bind(&WorldModel::perceptionCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    node_->get_logger(),
    "[WORLD]   perception staleness timeout: %.1f s",
    perception_timeout_sec_);
}

void WorldModel::perceptionCallback(const std_msgs::msg::String::SharedPtr msg)
{
  stringstream ss(msg->data);
  string part, status_str, xs, ys, zs;
  if (!getline(ss, part, ',')) return;
  if (!getline(ss, status_str, ',')) return;
  getline(ss, xs, ',');
  getline(ss, ys, ',');
  getline(ss, zs, ',');

  PartStatus status = PartStatus::UNKNOWN;
  if (status_str == "LOCATED") status = PartStatus::LOCATED;
  else if (status_str == "LOST") status = PartStatus::LOST;
  else if (status_str == "PICKED") status = PartStatus::PICKED;
  else if (status_str == "PICK_FAILED") status = PartStatus::PICK_FAILED;
  else if (status_str == "PLACED") status = PartStatus::PLACED;
  else if (status_str == "ESCALATED") status = PartStatus::ESCALATED;

  lock_guard<mutex> lock(mutex_);
  auto it = parts_.find(part);
  if (it == parts_.end()) {
    RCLCPP_WARN(node_->get_logger(),
                "[WorldModel] Unknown part '%s' in perception update",
                part.c_str());
    return;
  }

  // Guard: don't let incoming perception downgrade a part the BT has
  // already moved past locate-stage. Once a part is PLACED or ESCALATED,
  // a stale or late LOCATED/LOST message should be ignored.
  const auto current = it->second.status;
  if ((current == PartStatus::PLACED || current == PartStatus::ESCALATED) &&
      (status == PartStatus::LOCATED || status == PartStatus::LOST ||
       status == PartStatus::PICKED)) {
    return;
  }

  it->second.status = status;
  if (!xs.empty() && !ys.empty() && !zs.empty()) {
    it->second.position.x = stod(xs);
    it->second.position.y = stod(ys);
    it->second.position.z = stod(zs);
  }

  it->second.last_update = std::chrono::steady_clock::now();
  it->second.seen_by_perception = true;

  RCLCPP_INFO(node_->get_logger(), "[WorldModel] %s -> %s",
              part.c_str(), toString(status).c_str());
}

PartState WorldModel::getPartState(const string & part_name)
{
  lock_guard<mutex> lock(mutex_);
  _demote_stale_locked();
  return parts_.at(part_name);
}

void WorldModel::setPartStatus(const string & part_name, PartStatus status)
{
  lock_guard<mutex> lock(mutex_);
  auto & p = parts_.at(part_name);
  p.status = status;

  if (status == PartStatus::LOCATED) {
    p.last_update = std::chrono::steady_clock::now();
  }
}

void WorldModel::setPartPosition(const string & part_name,
                                 double x, double y, double z)
{
  lock_guard<mutex> lock(mutex_);
  auto & p = parts_.at(part_name);
  p.position.x = x;
  p.position.y = y;
  p.position.z = z;
  p.last_update = std::chrono::steady_clock::now();
}

int WorldModel::incrementPickAttempts(const string & part_name)
{
  lock_guard<mutex> lock(mutex_);
  return ++parts_.at(part_name).pick_attempts;
}

void WorldModel::resetPickAttempts(const string & part_name)
{
  lock_guard<mutex> lock(mutex_);
  parts_.at(part_name).pick_attempts = 0;
}

bool WorldModel::allPartsResolved()
{
  lock_guard<mutex> lock(mutex_);
  _demote_stale_locked();
  for (const auto & id : part_ids_) {
    auto status = parts_.at(id).status;
    if (status != PartStatus::PLACED && status != PartStatus::ESCALATED) {
      return false;
    }
  }
  return true;
}

void WorldModel::_demote_stale_locked()
{
  const auto now = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::duration<double>(perception_timeout_sec_);

  for (auto & [id, p] : parts_) {
    if (p.status != PartStatus::LOCATED) continue;
    if (!p.seen_by_perception) continue;
    if ((now - p.last_update) <= timeout) continue;

    RCLCPP_WARN(
      node_->get_logger(),
      "[WORLD]   %s: LOCATED -> UNKNOWN (stale, no update in %.1fs)",
      id.c_str(), perception_timeout_sec_);

    p.status = PartStatus::UNKNOWN;
  }
}

}  // namespace zebra_bt