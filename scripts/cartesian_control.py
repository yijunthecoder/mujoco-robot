#!/usr/bin/env python3
"""CLI: move one stationlite arm to a target 3D point via runtime IK.

Usage:
    python scripts/cartesian_control.py --target 0.45 0.1 -0.05
    python scripts/cartesian_control.py --arm right --target 0.4 -0.2 -0.1
    python scripts/cartesian_control.py --gl osmesa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjrobots.cartesian_control import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gl", default="egl", help="preferred rendering backend (default: egl)")
    parser.add_argument("--scene", default=None, help="path to stationlite_pick_place.xml")
    parser.add_argument("--arm", choices=["left", "right"], default="left", help="which arm to move")
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=(0.4148, 0.0, -0.1216),
        metavar=("X", "Y", "Z"),
        help="target point in world coordinates (default: the table's middle stacking point)",
    )
    args = parser.parse_args()

    run_demo(prefer_gl=args.gl, scene_path=args.scene, arm=args.arm, target=tuple(args.target))


if __name__ == "__main__":
    main()
