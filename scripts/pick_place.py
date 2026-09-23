#!/usr/bin/env python3
"""CLI: run the Panda pick-and-place demo (cube -> table) in the interactive viewer.

Usage:
    python scripts/pick_place.py
    python scripts/pick_place.py --gl osmesa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjrobots.pick_place import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gl", default="egl", help="preferred rendering backend (default: egl)")
    parser.add_argument("--robot", default="franka_emika_panda", help="menagerie robot folder name")
    args = parser.parse_args()

    run_demo(prefer_gl=args.gl, robot=args.robot)


if __name__ == "__main__":
    main()
