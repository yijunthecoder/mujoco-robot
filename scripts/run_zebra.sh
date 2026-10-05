#!/usr/bin/env bash
# Run the full zebra pick/place in one terminal: Victor's zebra_bt tree +
# our skill bridge (which opens the MuJoCo viewer, executes pick/place, and
# publishes perception itself - don't also run zebra_publisher.py).
#
# Every output line is labelled [tree] or [bridge]. Closing the MuJoCo
# window or pressing Ctrl+C stops both, so nothing is left running.
#
# Usage (from WSL Ubuntu):
#   bash scripts/run_zebra.sh                 # one of demos 1-5, at random
#   bash scripts/run_zebra.sh --demo 3        # demo 3 (each demo builds a full zebra)
#   bash scripts/run_zebra.sh --arm left      # one arm for everything (default: nearest per brick); extra args go to the bridge
#   bash scripts/run_zebra.sh --scatter 4     # replay start seed 4 (demos 1-5 are seeds 32, 4, 6, 15, 23)
#   bash scripts/run_zebra.sh --random        # a new random start instead of a demo seed
#   bash scripts/run_zebra.sh --drop          # bricks dropped anywhere in reach (default: placed in the box, any way up)
#   bash scripts/run_zebra.sh --upright       # bricks upright anywhere in reach
#   bash scripts/run_zebra.sh --fixed-start   # bricks upright in the old fixed spots

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
cd "$(dirname "$0")/.."

ros2 run build_a_zebra zebra_bt_node 2>&1 | sed -u 's/^/[tree]   /' &
trap 'pkill -P $$ 2>/dev/null; pkill -f zebra_bt_node 2>/dev/null' EXIT

python3 -u scripts/zebra_skill_bridge.py "$@" 2>&1 | sed -u 's/^/[bridge] /'
