#include "zebra_bt/task_model.hpp"

#include <fstream>
#include <algorithm>
#include <nlohmann/json.hpp>

namespace zebra_bt
{

using std::string;
using std::runtime_error;
using std::ifstream;
using std::vector;
using std::sort;
using std::find;

TaskModel::TaskModel(const string & bom_file_path)
{
  ifstream file(bom_file_path);
  if (!file.is_open()) {
    throw runtime_error("TaskModel: could not open BOM file: " + bom_file_path);
  }

  nlohmann::json j;
  file >> j;

  build_target_ = j.at("build_target").get<string>();
  role_order_ = j.at("role_order").get<vector<string>>();

  for (const auto & entry : j.at("catalog")) {
    PartSpec spec;
    spec.part_id = entry.at("part_id").get<string>();
    spec.animal = entry.at("animal").get<string>();
    spec.role = entry.at("role").get<string>();
    spec.description = entry.value("description", "");
    catalog_.push_back(spec);
  }

  // Filter to the current build target, then sort by role_order.
  for (const auto & spec : catalog_) {
    if (spec.animal == build_target_) {
      build_order_.push_back(spec);
    }
  }

  sort(
    build_order_.begin(), build_order_.end(),
    [this](const PartSpec & a, const PartSpec & b) {
      auto ai = find(role_order_.begin(), role_order_.end(), a.role);
      auto bi = find(role_order_.begin(), role_order_.end(), b.role);
      return ai < bi;
    });

  if (build_order_.empty()) {
    throw runtime_error(
      "TaskModel: no catalog parts found for build_target '" + build_target_ + "'");
  }
}

string TaskModel::dependencyOf(const string & part_id) const
{
  for (size_t i = 0; i < build_order_.size(); ++i) {
    if (build_order_[i].part_id == part_id) {
      return i == 0 ? string{} : build_order_[i - 1].part_id;
    }
  }
  return {};  // not part of the current build -- no dependency to report
}

}  // namespace zebra_bt