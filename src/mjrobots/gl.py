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
    """Try to actually create an offscreen context with `backend`."""
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
    """Pick a MuJoCo rendering backend, falling back to software if needed.

    Respects an already-set ``MUJOCO_GL`` / ``LIBGL_ALWAYS_SOFTWARE`` so a
    per-run override (e.g. ``MUJOCO_GL=osmesa python scripts/view.py ...``)
    still wins. Otherwise tries `prefer` (hardware-accelerated), and falls
    back to forced software rendering if that backend can't actually render.

    Returns the backend name that ended up selected, for logging.
    """
    if os.environ.get("MUJOCO_GL") or os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1":
        backend = os.environ.get("MUJOCO_GL", "libgl_software")
        if verbose:
            print(f"[mjrobots] using pre-set rendering backend: {backend}")
        return backend

    # reaches this line if nothing was preset
    if _probe_renders(prefer):
        if verbose:
            print(f"[mjrobots] rendering backend: {prefer} (hardware)")
        return prefer

    if verbose:
        print(f"[mjrobots] '{prefer}' unavailable, falling back to software rendering")
    os.environ.pop("MUJOCO_GL", None)
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    return "libgl_software"
