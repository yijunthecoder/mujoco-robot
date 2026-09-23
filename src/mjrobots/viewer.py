"""Load and view MuJoCo Menagerie robot models."""

from __future__ import annotations

import mujoco
import mujoco.viewer

from .gl import configure_gl
from .models import scene_path


def load(model_name: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a menagerie model by folder name, e.g. 'franka_emika_panda'."""
    model = mujoco.MjModel.from_xml_path(str(scene_path(model_name)))
    data = mujoco.MjData(model)
    return model, data


def view(model_name: str, prefer_gl: str = "egl") -> None:
    """Configure rendering, load `model_name`, and open the interactive viewer."""
    configure_gl(prefer_gl)
    model, data = load(model_name)
    print(f"[mjrobots] loaded '{model_name}': {model.nbody} bodies, {model.njnt} joints")
    mujoco.viewer.launch(model, data)
