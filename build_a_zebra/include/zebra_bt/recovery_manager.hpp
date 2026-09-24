#pragma once

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace zebra_bt
{

// What kind of failure just happened. Different failure types deserve
// different responses -- a bad grip is not the same problem as a part
// that's gone missing.
enum class FailureType
{
  PICK_FAILED,  // gripper attempted the pick and it didn't take
  PART_LOST     // perception can no longer find the part at all
};

// What the tree should do in response, as decided by policy (not a
// hardcoded number baked into the tree XML).
enum class RecoveryAction
{
  RETRY,     // try the same action again
  RELOCATE,  // give up on the current position, re-search first
  ESCALATE   // give up automatically; this part now needs a human
};

std::string toString(RecoveryAction action);

// RecoveryManager is the policy layer: given a part and what just went
// wrong, it decides how to respond -- and remembers failure history per
// part so it can escalate once retrying stops being reasonable, instead
// of retrying forever or giving up too early.
class RecoveryManager
{
public:
  explicit RecoveryManager(int max_pick_retries = 3, int max_relocate_attempts = 2);

  RecoveryAction decide(const std::string & part_id, FailureType failure);

  // Parts that have been escalated and are no longer being automatically
  // retried -- useful for a final report of what needs human attention.
  std::vector<std::string> escalatedParts();

private:
  struct FailureHistory
  {
    int pick_failures{0};
    int lost_events{0};
  };

  std::mutex mutex_;
  std::map<std::string, FailureHistory> history_;
  std::vector<std::string> escalated_;
  int max_pick_retries_;
  int max_relocate_attempts_;
};

}  // namespace zebra_bt