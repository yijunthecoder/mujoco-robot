#!/usr/bin/env python3
"""CLI: view any MuJoCo-loadable model file (MJCF .xml or URDF .urdf).

Unlike scripts/view.py (which only knows about mujoco_menagerie robots by
name), this takes a direct file path - useful for one-off/external models
like a downloaded URDF.

Usage:
    python scripts/view_file.py /path/to/robot.urdf
    python scripts/view_file.py /path/to/scene.xml --gl osmesa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mujoco
import mujoco.viewer

from mjrobots.gl import configure_gl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="path to a .urdf or .xml model file")
    parser.add_argument("--gl", default="egl", help="preferred rendering backend (default: egl)")
    args = parser.parse_args()

    configure_gl(args.gl)
    model = mujoco.MjModel.from_xml_path(args.path)
    data = mujoco.MjData(model)
    print(f"[mjrobots] loaded '{args.path}': {model.nbody} bodies, {model.njnt} joints")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
