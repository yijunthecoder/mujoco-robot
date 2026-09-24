#!/usr/bin/env python3
"""CLI: publish calibrated block positions to Victor's zebra_bt over ROS2.

Requires ROS2 sourced first:
    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=42
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

Usage:
    python scripts/zebra_publisher.py
    python scripts/zebra_publisher.py --part-id 31111p0e --interval 1.0

Watch it yourself, without Victor's machine, in a second terminal:
    ros2 topic echo /zebra/perception_updates

Standalone perception only: it simulates its own static copy of the scene,
so it always reports the brick's spawn position. For a full pick/place run
use scripts/zebra_skill_bridge.py instead, which publishes perception from
the simulation the arm actually moves in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjrobots.zebra_publisher import DEFAULT_PART_ID, run_publisher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=None, help="path to stationlite_pick_place.xml")
    parser.add_argument("--part-id", default=DEFAULT_PART_ID, help=f"zebra part id to publish as (default: {DEFAULT_PART_ID})")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between updates (default: 1.0)")
    parser.add_argument("--n-calib", type=int, default=15, help="block positions used to fit (default: 15)")
    parser.add_argument("--pixel-noise", type=float, default=0.5, help="detector noise, pixels (default: 0.5)")
    parser.add_argument("--depth-noise", type=float, default=0.002, help="depth noise, metres (default: 0.002)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_publisher(
        part_id=args.part_id,
        interval=args.interval,
        n_calib=args.n_calib,
        pixel_sigma=args.pixel_noise,
        depth_sigma=args.depth_noise,
        seed=args.seed,
        scene_path=args.scene,
    )


if __name__ == "__main__":
    main()
