#!/usr/bin/env python3
"""CLI: execute Victor's zebra_bt pick/place commands for zebra_legs in this
MuJoCo sim, over ROS2.

Requires ROS2 sourced first:
    source /opt/ros/humble/setup.bash

Usage:
    python scripts/zebra_skill_bridge.py
    python scripts/zebra_skill_bridge.py --arm left --gl osmesa

Also publishes perception (/zebra/perception_updates) from this same
simulation, so don't run scripts/zebra_publisher.py alongside it - that one
watches its own static copy of the scene. Run in a second terminal:
    ros2 run build_a_zebra zebra_bt_node     # Victor's tree
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjrobots.zebra_skill_bridge import run_bridge

PART_NAMES = {"legs": "31111p0e", "body": "31111p0f", "head": "31111p0g"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gl", default="egl", help="preferred rendering backend (default: egl)")
    parser.add_argument("--scene", default=None, help="path to stationlite_pick_place.xml")
    parser.add_argument("--arm", choices=["left", "right"], default="right", help="which arm to move")
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
    args = parser.parse_args()

    run_bridge(
        prefer_gl=args.gl, scene_path=args.scene, arm=args.arm,
        fault_part=PART_NAMES[args.fail_part] if args.fail_part else None,
        fault_offset=args.fail_offset,
        fault_times=args.fail_times,
        fault_bump=args.fail_bump,
    )


if __name__ == "__main__":
    main()
