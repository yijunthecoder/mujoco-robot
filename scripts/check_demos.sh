#!/usr/bin/env bash
# Check that the demos still build a full zebra - run this after every change.
#
# Runs demos 1-5 headless (no window) and 4x faster than real time, each with
# Victor's real tree (zebra_bt_node) on its own ROS domain, so they don't
# clash with each other or with a live run (domain 42). Then prints one line
# per demo: full zebra or not, time, each flip, placement accuracy, problems.
#
# Usage (from WSL Ubuntu, in the repo folder):
#   bash scripts/check_demos.sh            # all 5 demos (~3 min)
#   bash scripts/check_demos.sh 2 5        # just demos 2 and 5
#   PARALLEL=2 bash scripts/check_demos.sh # fewer at once (each needs ~1.3 GB RAM)
# Logs: logs/check_demos/demo_N.log (kept until the next check).

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
cd "$(dirname "$0")/.."

DEMOS=("$@"); [ ${#DEMOS[@]} -eq 0 ] && DEMOS=(1 2 3 4 5)
PARALLEL=${PARALLEL:-3}
SPEED=${SPEED:-4}
LIMIT=${LIMIT:-300}  # s per demo before giving up
OUT=logs/check_demos
mkdir -p $OUT
rm -f $OUT/demo_*.log

run_demo() {  # demo number, ROS domain
  local n=$1 log=$OUT/demo_$1.log
  export ROS_DOMAIN_ID=$2
  python3 -u scripts/zebra_skill_bridge.py --headless --speed $SPEED --demo $n 2>&1 \
    | sed -u 's/^/[bridge] /' >> $log &
  local bridge=$!
  sleep 8  # the bridge calibrates its cameras first
  ros2 run build_a_zebra zebra_bt_node 2>&1 | sed -u 's/^/[tree]   /' >> $log &
  local tree=$!
  for ((i = 0; i < LIMIT; i++)); do
    sleep 1
    grep -qsE "Zebra (fully )?assembled|placed, [0-9]+ escalated" $log 2>/dev/null && break
  done
  sleep 1
  # stop only this demo's processes, never a live run
  for p in $(pgrep -f "zebra_skill_bridge.py --headless --speed $SPEED --demo $n"); do
    pkill -P $p 2>/dev/null; kill $p 2>/dev/null  # its flip-planning process first
  done
  for p in $(pgrep -x zebra_bt_node); do
    [ "$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep ^ROS_DOMAIN_ID=)" = "ROS_DOMAIN_ID=$2" ] && kill $p
  done
  wait $bridge $tree 2>/dev/null
}

echo "checking demos ${DEMOS[*]} (${PARALLEL} at a time, ${SPEED}x speed) ..."
for ((k = 0; k < ${#DEMOS[@]}; k += PARALLEL)); do
  for ((j = 0; j < PARALLEL && k + j < ${#DEMOS[@]}; j++)); do
    run_demo ${DEMOS[k + j]} $((91 + j)) &
  done
  wait
done
python3 scripts/demo_report.py $OUT/demo_*.log
