"""Rendering-backend selection for MuJoCo.

Must run before any MuJoCo rendering context (an offscreen ``Renderer`` or
the interactive ``mujoco.viewer``) is created — MuJoCo reads ``MUJOCO_GL``
once, on first use, and ignores later changes.

Note on scope: ``MUJOCO_GL`` controls the backend for *offscreen* contexts
(``mujoco.Renderer``, used for headless frame capture). The interactive
viewer window always goes through GLFW, which reads ``LIBGL_ALWAYS_SOFTWARE``
directly (a Mesa driver setting, not MuJoCo-specific). The fallback branch
below sets that too, so it helps both cases.
"""

from __future__ import annotations

import os

_PROBE_XML = """
<mujoco>
  <worldbody>
    <geom type="sphere" size="0.1"/>
  </worldbody>
</mujoco>
"""


def _probe_renders(backend: str) -> bool:
    os.environ["MUJOCO_GL"] = backend
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_string(_PROBE_XML)
        renderer = mujoco.Renderer(model, height=8, width=8)
        renderer.close()
        return True
    except Exception:
        return False


def configure_gl(prefer: str = "egl", verbose: bool = True) -> str:
    if os.environ.get("MUJOCO_GL") or os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1":
        backend = os.environ.get("MUJOCO_GL", "libgl_software")
        if verbose:
            print(f"[mjrobots] using pre-set rendering backend: {backend}")
        return backend

    if _probe_renders(prefer): """" reaches this line if nothing was preset """"
        if verbose:
            print(f"[mjrobots] rendering backend: {prefer} (hardware)")
        return prefer

    if verbose:
        print(f"[mjrobots] '{prefer}' unavailable, falling back to software rendering")
    os.environ.pop("MUJOCO_GL", None)
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    return "libgl_software"
