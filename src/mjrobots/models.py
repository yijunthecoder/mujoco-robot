"""Locate the MuJoCo Menagerie clone and resolve robot model paths."""

from __future__ import annotations

import os
from pathlib import Path

def _candidate_roots() -> list[Path]:
    """Checked in order. Set MENAGERIE_ROOT to override without editing this file.

    No hardcoded username: a clone at .../Users/<anyone>/mujoco_test/
    mujoco_menagerie is found via a wildcard glob instead, so this works for
    whoever clones the repo, not just the machine it was written on. Both
    glob patterns are harmless no-ops on a platform where that root doesn't
    exist (e.g. the C:\\Users pattern under native Linux) - Path.glob()
    simply yields nothing rather than raising.
    """
    env = os.environ.get("MENAGERIE_ROOT")
    candidates: list[Path] = [Path(env)] if env else []
    candidates += sorted(Path("/mnt/c/Users").glob("*/mujoco_test/mujoco_menagerie"))
    candidates += sorted(Path(r"C:\Users").glob("*/mujoco_test/mujoco_menagerie"))
    candidates.append(Path(os.path.expanduser("~/mujoco_test/mujoco_menagerie")))
    return candidates


def menagerie_root() -> Path:
    for candidate in _candidate_roots():
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find mujoco_menagerie in any known location. "
        "Set the MENAGERIE_ROOT environment variable to its path."
    )


def list_models() -> list[str]:
    """Names of menagerie subfolders that contain a scene.xml."""
    root = menagerie_root()
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "scene.xml").exists()
    )


def scene_path(model_name: str) -> Path:
    """Resolve a model name to its scene.xml, e.g. 'franka_emika_panda'."""
    path = menagerie_root() / model_name / "scene.xml"
    if not path.exists():
        raise FileNotFoundError(
            f"No scene.xml found for '{model_name}'. "
            f"Available models: {', '.join(list_models())}"
        )
    return path
