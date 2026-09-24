#!/usr/bin/env bash
# Run the full zebra pick/place in one terminal: Victor's zebra_bt tree +
# our skill bridge (which opens the MuJoCo viewer, executes pick/place, and
# publishes perception itself - don't also run zebra_publisher.py).
#
# Every output line is labelled [tree] or [bridge]. Closing the MuJoCo
# window or pressing Ctrl+C stops both, so nothing is left running.
#
# Usage (from WSL Ubuntu):
#   bash scripts/run_zebra.sh
#   bash scripts/run_zebra.sh --arm left      # extra args go to the bridge

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
cd "$(dirname "$0")/.."

ros2 run build_a_zebra zebra_bt_node 2>&1 | sed -u 's/^/[tree]   /' &
trap 'pkill -P $$ 2>/dev/null; pkill -f zebra_bt_node 2>/dev/null' EXIT

python3 -u scripts/zebra_skill_bridge.py "$@" 2>&1 | sed -u 's/^/[bridge] /'
