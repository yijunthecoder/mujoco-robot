#!/usr/bin/env bash
# Run the full zebra pick/place in one terminal: Victor's zebra_bt tree +
# our skill bridge (which opens the MuJoCo viewer, executes pick/place, and
# publishes perception itself - don't also run zebra_publisher.py).
#
# Every output line is labelled [tree] or [bridge] (and [vla] with --vla). Closing the MuJoCo
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
#   bash scripts/run_zebra.sh --vla --demo 1  # Victor's VLA tree: Gemini picks each skill (see below)
#
# --vla: Victor's tree where a vision-language model (Gemini) picks each next skill
# from the head-camera picture (VLADecide <-> his vla_planner_node.py, /vla/request and
# /vla/decision). It runs his VLA version from its own workspace, ~/ros2_ws_vla (the
# version is in ~/ros2_ws_vla/src/build_a_zebra/VICTOR_COMMIT), plus his planner, which
# needs a Google API key in ~/ros2_ws_vla/src/build_a_zebra/scripts/apikeys.py
# (GOOGLE_API_KEY = "..."; not in git - ask Victor). Without --vla: his tree without the
# VLA from ~/ros2_ws, as before (and as scripts/check_demos.sh uses). Labelled [vla].

# --vla is ours; every other argument goes to the bridge
vla=0
args=()
for a in "$@"; do
  if [ "$a" = "--vla" ]; then vla=1; else args+=("$a"); fi
done
ws=~/ros2_ws
if [ $vla = 1 ]; then
  ws=~/ros2_ws_vla
  planner=$ws/src/build_a_zebra/scripts/vla_planner_node.py
  if [ ! -f "$ws/install/setup.bash" ]; then
    echo "--vla: no $ws - build Victor's VLA version there first (see HANDOVER.md)"; exit 1
  fi
  if [ ! -f "$(dirname "$planner")/apikeys.py" ]; then
    echo "--vla: no $(dirname "$planner")/apikeys.py - put GOOGLE_API_KEY = \"...\" in it (ask Victor)"; exit 1
  fi
fi

source /opt/ros/humble/setup.bash
source $ws/install/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
cd "$(dirname "$0")/.."

trap 'pkill -P $$ 2>/dev/null; pkill -f zebra_bt_node 2>/dev/null; pkill -f vla_planner_node 2>/dev/null; rm -f "$ready"' EXIT

if [ $vla = 0 ]; then
  ros2 run build_a_zebra zebra_bt_node 2>&1 | sed -u 's/^/[tree]   /' &
  python3 -u scripts/zebra_skill_bridge.py "${args[@]}" 2>&1 | sed -u 's/^/[bridge] /'
  exit
fi

# --vla: the tree asks the planner as soon as it starts, and the planner drops a request
# that comes before the first camera picture (VLADecide then waits out its 60 s timeout).
# The bridge only publishes pictures once its cameras are calibrated (~20 s) - so start
# the planner and the bridge first, and the tree once the bridge says it's ready.
python3 -u "$planner" 2>&1 | sed -u 's/^/[vla]    /' &
ready=$(mktemp)
python3 -u scripts/zebra_skill_bridge.py "${args[@]}" 2>&1 | sed -u 's/^/[bridge] /' | tee "$ready" &
bridge=$!
until grep -q "Ready - watching" "$ready"; do
  kill -0 $bridge 2>/dev/null || exit 1  # the bridge stopped before it was ready (its error is above)
  sleep 0.5
done
ros2 run build_a_zebra zebra_bt_node 2>&1 | sed -u 's/^/[tree]   /' &
wait $bridge  # closing the MuJoCo window (or Ctrl+C) ends the bridge, then everything else
