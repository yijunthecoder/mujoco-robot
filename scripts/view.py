#!/usr/bin/env python3
"""CLI: view a MuJoCo Menagerie robot in the interactive viewer.

Usage:
    python scripts/view.py franka_emika_panda
    python scripts/view.py --list
    python scripts/view.py franka_emika_panda --gl osmesa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjrobots.models import list_models
from mjrobots.viewer import view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="menagerie folder name, e.g. franka_emika_panda")
    parser.add_argument("--list", action="store_true", help="list available models and exit")
    parser.add_argument("--gl", default="egl", help="preferred rendering backend (default: egl)")
    args = parser.parse_args()

    if args.list or not args.model:
        for name in list_models():
            print(name)
        sys.exit(0 if args.list else 1)

    view(args.model, prefer_gl=args.gl)


if __name__ == "__main__":
    main()
