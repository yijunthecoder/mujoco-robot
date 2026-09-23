#include "zebra_bt/recovery_manager.hpp"

#include <algorithm>

namespace zebra_bt
{

using std::string;
using std::lock_guard;
using std::mutex;
using std::find;
using std::vector;

string toString(RecoveryAction action)
{
  switch (action) {
    case RecoveryAction::RETRY: return "RETRY";
    case RecoveryAction::RELOCATE: return "RELOCATE";
    case RecoveryAction::ESCALATE: return "ESCALATE";
  }
  return "UNKNOWN";
}

RecoveryManager::RecoveryManager(int max_pick_retries, int max_relocate_attempts)
: max_pick_retries_(max_pick_retries), max_relocate_attempts_(max_relocate_attempts)
// [Class Variable] (  Input Value  ) ,  [Class Variable]      (    Input Value   )
{}

RecoveryAction RecoveryManager::decide(const string & part_id, FailureType failure)
{
  lock_guard<mutex> lock(mutex_);
  auto & hist = history_[part_id];

  if (failure == FailureType::PART_LOST) {
    hist.lost_events++;
    if (hist.lost_events <= max_relocate_attempts_) {
      return RecoveryAction::RELOCATE;
    }
  } else {  // PICK_FAILED
    hist.pick_failures++;
    if (hist.pick_failures <= max_pick_retries_) {
      return RecoveryAction::RETRY;
    }
  }

  // Exceeded the relevant limit -- give up on this part automatically.
  if (find(escalated_.begin(), escalated_.end(), part_id) == escalated_.end()) {
    escalated_.push_back(part_id);
  }
  return RecoveryAction::ESCALATE;
}

vector<string> RecoveryManager::escalatedParts()
{
  lock_guard<mutex> lock(mutex_);
  return escalated_;
}

}  // namespace zebra_bt