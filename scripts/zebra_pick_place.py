#!/usr/bin/env python3
"""CLI: move the zebra_legs brick to the table's middle point via runtime IK.

Usage:
    python scripts/zebra_pick_place.py
    python scripts/zebra_pick_place.py --arm right
    python scripts/zebra_pick_place.py --gl osmesa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjrobots.zebra_pick_place import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gl", default="egl", help="preferred rendering backend (default: egl)")
    parser.add_argument("--scene", default=None, help="path to stationlite_pick_place.xml")
    parser.add_argument("--arm", choices=["left", "right"], default="right", help="which arm to move")
    args = parser.parse_args()

    run_demo(prefer_gl=args.gl, scene_path=args.scene, arm=args.arm)


if __name__ == "__main__":
    main()
