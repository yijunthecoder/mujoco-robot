"""Locate the MuJoCo Menagerie clone and resolve robot model paths."""

from __future__ import annotations

import os
from pathlib import Path

# Checked in order. Set MENAGERIE_ROOT to override without editing this file.
_CANDIDATE_ROOTS = [
    os.environ.get("MENAGERIE_ROOT"),
    "/mnt/c/Users/pokem/mujoco_test/mujoco_menagerie",  # WSL2 view of the Windows clone
    r"C:\Users\pokem\mujoco_test\mujoco_menagerie",  # native Windows view
    os.path.expanduser("~/mujoco_test/mujoco_menagerie"),  # native WSL clone, if any
]


def menagerie_root() -> Path:
    for candidate in _CANDIDATE_ROOTS:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
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
