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
│   └── stationlite_pick_place.py  # two-arm block-stacking demo (stationlite robot)
│   └── camera_calibration.py      # aligns the 4 stationlite cameras into one shared frame
├── scripts/
│   ├── view.py                    # CLI entry point (menagerie robots by name)
│   ├── view_file.py                # CLI entry point (any .urdf/.xml by path)
│   ├── pick_place.py                # CLI entry point for the Panda pick-and-place demo
│   ├── stationlite_pick_place.py     # CLI entry point for the stationlite stacking demo
│   └── camera_calibration.py         # CLI entry point for the 4-camera calibration
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

## Stationlite two-arm block-stacking demo

The stationlite dual-arm robot's files live in `stationlite/` (URDF +
meshes, ~24MB, tracked in this repo). `stationlite/urdf/stationlite_pick_place.xml`
wraps the URDF with actuators, a table, two blocks, and camera viewpoints.

```bash
python scripts/stationlite_pick_place.py    # right arm places its block at the
                                             # middle, left arm stacks its own on top
python scripts/view_file.py stationlite/urdf/stationlite_pick_place.xml
                                             # just look around / try the cameras
                                             # (Tab for the side panel, or press [ / ]
                                             # to cycle: free cam, headcam, refcam,
                                             # left_handcam, right_handcam)
```

Waypoints are joint angles found by an offline forward-kinematics search
against the robot (not solved via IK at runtime — its kinematic conventions
weren't known going in). Grasping is a kinematic "carry" rather than pure
friction: once a gripper closes on its block, the block's position is
snapped to the fingertip midpoint each step until release — the mesh-only
finger geometry isn't reliable enough to hold an object through arm motion
on its own (confirmed experimentally).

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
