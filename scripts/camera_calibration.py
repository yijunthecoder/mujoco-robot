#!/usr/bin/env python3
"""CLI: calibrate the stationlite scene's 4 cameras against one reference block.

Usage:
    python scripts/camera_calibration.py                # prints each position's coordinates
    python scripts/camera_calibration.py --view          # also open the MuJoCo viewer
    python scripts/camera_calibration.py --pixel-noise 1.0 --depth-noise 0.005
    python scripts/camera_calibration.py --pixel-noise 0 --depth-noise 0   # noise-free sanity check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjrobots.camera_calibration import run_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=None, help="path to stationlite_pick_place.xml")
    parser.add_argument("--n-calib", type=int, default=15, help="block positions used to fit (default: 15)")
    parser.add_argument("--n-test", type=int, default=6, help="held-out block positions to check (default: 6)")
    parser.add_argument("--pixel-noise", type=float, default=0.5, help="detector noise, pixels (default: 0.5)")
    parser.add_argument("--depth-noise", type=float, default=0.002, help="depth noise, metres (default: 0.002)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view", action="store_true", help="open the MuJoCo viewer after calibrating")
    parser.add_argument("--gl", default="egl", help="preferred rendering backend for --view (default: egl)")
    args = parser.parse_args()

    run_calibration(
        scene_path=args.scene,
        n_calib=args.n_calib,
        n_test=args.n_test,
        pixel_sigma=args.pixel_noise,
        depth_sigma=args.depth_noise,
        seed=args.seed,
        view=args.view,
        prefer_gl=args.gl,
    )


if __name__ == "__main__":
    main()
