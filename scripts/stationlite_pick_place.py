#!/usr/bin/env python3
"""CLI: run the stationlite dual-arm pick-and-place demo (ball -> target).

Usage:
    python scripts/stationlite_pick_place.py
    python scripts/stationlite_pick_place.py --gl osmesa
    python scripts/stationlite_pick_place.py --scene /path/to/stationlite_pick_place.xml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjrobots.stationlite_pick_place import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gl", default="egl", help="preferred rendering backend (default: egl)")
    parser.add_argument("--scene", default=None, help="path to stationlite_pick_place.xml")
    args = parser.parse_args()

    run_demo(prefer_gl=args.gl, scene_path=args.scene)


if __name__ == "__main__":
    main()
