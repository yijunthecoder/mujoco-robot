#!/usr/bin/env python3
"""CLI: execute Victor's zebra_bt pick/place commands for zebra_legs in this
MuJoCo sim, over ROS2.

Requires ROS2 sourced first:
    source /opt/ros/humble/setup.bash

Usage:
    python scripts/zebra_skill_bridge.py
    python scripts/zebra_skill_bridge.py --arm left --gl osmesa

Run alongside (in separate terminals):
    python scripts/zebra_publisher.py       # reports zebra_legs' position
    ros2 run build_a_zebra zebra_bt_node     # Victor's tree, drives both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjrobots.zebra_skill_bridge import run_bridge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gl", default="egl", help="preferred rendering backend (default: egl)")
    parser.add_argument("--scene", default=None, help="path to stationlite_pick_place.xml")
    parser.add_argument("--arm", choices=["left", "right"], default="right", help="which arm to move")
    args = parser.parse_args()

    run_bridge(prefer_gl=args.gl, scene_path=args.scene, arm=args.arm)


if __name__ == "__main__":
    main()
