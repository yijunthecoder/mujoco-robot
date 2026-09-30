#!/usr/bin/env python3
"""CLI: execute Victor's zebra_bt pick/place commands for zebra_legs in this
MuJoCo sim, over ROS2.

Requires ROS2 sourced first:
    source /opt/ros/humble/setup.bash

Usage:
    python scripts/zebra_skill_bridge.py
    python scripts/zebra_skill_bridge.py --arm left --gl osmesa
    python scripts/zebra_skill_bridge.py --scatter 7      # replay scatter seed 7
    python scripts/zebra_skill_bridge.py --upright        # set the bricks down upright instead of dropping them
    python scripts/zebra_skill_bridge.py --fixed-start    # old fixed brick spots (upright)

Also publishes perception (/zebra/perception_updates) from this same
simulation, so don't run scripts/zebra_publisher.py alongside it - that one
watches its own static copy of the scene. Run in a second terminal:
    ros2 run build_a_zebra zebra_bt_node     # Victor's tree
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjrobots import move_check
from mjrobots.zebra_skill_bridge import run_bridge

PART_NAMES = {"legs": "31111p0e", "body": "31111p0f", "head": "31111p0g"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gl", default="egl", help="preferred rendering backend (default: egl)")
    parser.add_argument("--scene", default=None, help="path to stationlite_pick_place.xml")
    parser.add_argument(
        "--arm", choices=["nearest", "left", "right"], default="nearest",
        help="which arm picks each brick: the nearest one (default), or always left/right",
    )
    parser.add_argument(
        "--fail-part", choices=sorted(PART_NAMES), default=None,
        help="test hook: send this part's picks to the wrong spot so they fail",
    )
    parser.add_argument(
        "--fail-offset", type=float, default=0.08,
        help="how far off (metres, +y) --fail-part's picks go (default: 0.08)",
    )
    parser.add_argument(
        "--fail-times", type=int, default=0,
        help="only fail --fail-part's first N picks, then pick normally (default: 0 = every pick)",
    )
    parser.add_argument(
        "--fail-bump", action="store_true",
        help="a missed pick also knocks the brick 5 cm away and perception loses it "
             "for 2s, so the tree re-locates instead of retrying",
    )
    parser.add_argument(
        "--knock-placed", choices=sorted(PART_NAMES), default=None,
        help="test hook: knock this part 8 cm off the stack right after its first place, "
             "so the place check fails it and the tree re-picks and re-places it",
    )
    parser.add_argument(
        "--knock-later", choices=sorted(PART_NAMES), default=None,
        help="test hook: knock this part 8 cm off the stack after it was placed and checked, "
             "so the stack check before the next place finds it gone",
    )
    parser.add_argument(
        "--scatter", type=int, default=None, metavar="SEED",
        help="replay this scatter: the bricks are dropped at random spots where the arm can reach "
             "them - a new random SEED every run unless given (it's printed); same SEED, same drop",
    )
    parser.add_argument(
        "--upright", action="store_true",
        help="set the bricks down upright at their scatter spots instead of dropping them from 15 cm "
             "with random tumbles (by default they land any way up; perception reports how)",
    )
    parser.add_argument(
        "--fixed-start", action="store_true",
        help="start the bricks upright in their old fixed square spots instead of scattering them",
    )
    parser.add_argument("--drop", action="store_true", help=argparse.SUPPRESS)  # the default now; old commands still work
    parser.add_argument(
        "--confirm-moves", choices=move_check.MODES, default="off",
        help="print each arm move's joint changes and wait for ENTER before it runs: "
             "'first' move only, 'all' moves, or 'off' (default)",
    )
    args = parser.parse_args()
    move_check.set_mode(args.confirm_moves)
    if args.drop and (args.fixed_start or args.upright):
        parser.error("--drop doesn't go with --fixed-start or --upright")
    if args.fixed_start:
        scatter_seed = None
    elif args.scatter is not None:
        scatter_seed = args.scatter
    else:
        scatter_seed = random.randrange(10_000)

    run_bridge(
        prefer_gl=args.gl, scene_path=args.scene, arm=args.arm,
        fault_part=PART_NAMES[args.fail_part] if args.fail_part else None,
        fault_offset=args.fail_offset,
        fault_times=args.fail_times,
        fault_bump=args.fail_bump,
        knock_placed=PART_NAMES[args.knock_placed] if args.knock_placed else None,
        knock_later=PART_NAMES[args.knock_later] if args.knock_later else None,
        scatter_seed=scatter_seed,
        drop=not (args.upright or args.fixed_start),
    )


if __name__ == "__main__":
    main()
