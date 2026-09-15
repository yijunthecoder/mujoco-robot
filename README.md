# mujoco-robots

Reusable scaffolding for loading and viewing robot models from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie),
with automatic rendering-backend fallback for WSL2.

Lives on Windows at `C:\intern\mujoco-robots`, which is `/mnt/c/intern/mujoco-robots`
from inside WSL2 — run it from your Ubuntu terminal where `mujoco` is pip-installed.

## Layout

```
mujoco-robots/
├── src/mjrobots/
│   ├── gl.py        # MUJOCO_GL backend selection + software-render fallback
│   ├── models.py     # finds mujoco_menagerie, resolves model name -> scene.xml
│   └── viewer.py     # load()/view() helpers
│   └── pick_place.py  # damped-least-squares IK + scripted pick-and-place motion
├── scripts/
│   ├── view.py         # CLI entry point
│   └── pick_place.py   # CLI entry point for the pick-and-place demo
└── requirements.txt
```

## Setup (in WSL2 Ubuntu)

```bash
cd /mnt/c/intern/mujoco-robots
pip install -r requirements.txt
```

## Usage

```bash
python scripts/view.py --list                       # list available menagerie models
python scripts/view.py franka_emika_panda            # view a robot (prefers EGL, falls back to software)
python scripts/view.py franka_emika_panda --gl osmesa # try a different backend
MUJOCO_GL=glfw python scripts/view.py franka_emika_panda  # force a specific backend, bypassing fallback logic
```

Or from Python:

```python
from mjrobots import view
view("franka_emika_panda")
```

## Pick-and-place demo

```bash
python scripts/pick_place.py               # Panda picks up a cube and sets it on a table
python scripts/pick_place.py --gl osmesa    # try a different backend
```

The scene (`src/mjrobots/assets/pick_and_place_scene.xml`) adds a table, a
cube, and a target marker next to the Panda arm; `src/mjrobots/pick_place.py`
drives the arm with Jacobian inverse kinematics (`_solve_ik`) to reach a
sequence of Cartesian waypoints (approach, descend, grasp, lift, transport,
place, retreat), ramping the joint-position actuators (`data.ctrl`) toward
each solved pose so MuJoCo's own physics — not a scripted teleport — carries
the cube. It prints how far the cube ended up from the target when done.

## Rendering fallback

`configure_gl()` (in `src/mjrobots/gl.py`) tries your preferred backend
(default `egl`) by actually creating a tiny offscreen renderer. If that
fails, it sets `LIBGL_ALWAYS_SOFTWARE=1` so both offscreen rendering and the
interactive viewer window fall back to Mesa's software rasterizer — the same
fix you were applying by hand, just automatic now.

Note: `MUJOCO_GL` only affects *offscreen* contexts (`mujoco.Renderer`). The
interactive viewer window always uses GLFW and reads `LIBGL_ALWAYS_SOFTWARE`
directly, which is why the fallback sets both.

## Menagerie location

`models.py` looks for your existing `mujoco_menagerie` clone at (in order):

1. `$MENAGERIE_ROOT` if set
2. `/mnt/c/Users/pokem/mujoco_test/mujoco_menagerie` (WSL2 view of your existing Windows clone)
3. `C:\Users\pokem\mujoco_test\mujoco_menagerie` (native Windows view)
4. `~/mujoco_test/mujoco_menagerie` (native WSL clone, if any)

It isn't copied into this project (it's ~2.4GB) — set `MENAGERIE_ROOT` to
point elsewhere if you move or re-clone it.

## Adding a new robot

Nothing to change — any menagerie folder with a `scene.xml` is picked up
automatically by `--list` and `view()`.
