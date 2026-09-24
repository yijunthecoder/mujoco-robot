# mujoco-robot

Reusable scaffolding for loading and viewing robot models from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie),
with automatic rendering-backend fallback for WSL2 — plus the stationlite
dual-arm robot and the zebra DUPLO pick/place that talks to Victor's
`zebra_bt` behavior tree over ROS2.

Lives on Windows at `C:\intern\mujoco-robot`, which is `/mnt/c/intern/mujoco-robot`
from inside WSL2 — run everything from your Ubuntu terminal where `mujoco` is pip-installed.

## Setup (in WSL2 Ubuntu)

```bash
$ cd /mnt/c/intern/mujoco-robot
$ pip install -r requirements.txt
```

The zebra commands also need ROS2 Humble and Victor's `build_a_zebra`
package built in `~/ros2_ws` (from the `Victor` branch):

```bash
$ cd ~/ros2_ws && source /opt/ros/humble/setup.bash
$ colcon build --packages-select build_a_zebra
```

## How to run

All commands are run from `/mnt/c/intern/mujoco-robot` in WSL.

### Zebra pick/place with Victor's behavior tree (the main one)

Starts Victor's `zebra_bt` tree and our skill bridge together in one
terminal. The MuJoCo window opens, the right arm picks `zebra_legs` up and
places it in the green circle. Output lines are labelled `[tree]` /
`[bridge]`; close the MuJoCo window or press Ctrl+C to stop both.

```bash
$ bash scripts/run_zebra.sh
```

Use the left arm instead:

```bash
$ bash scripts/run_zebra.sh --arm left
```

Only `zebra_legs` is wired up so far — the tree waits ~30s each for body and
head, skips them, then finishes with `1 placed, 2 escalated`.

<details>
<summary>Running the two parts in separate terminals instead</summary>

Each terminal first needs:

```bash
$ source /opt/ros/humble/setup.bash
$ export ROS_DOMAIN_ID=42
$ export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

Terminal 1 — Victor's tree:

```bash
$ source ~/ros2_ws/install/setup.bash
$ ros2 run build_a_zebra zebra_bt_node
```

Terminal 2 — our bridge (MuJoCo viewer + IK + perception):

```bash
$ python3 scripts/zebra_skill_bridge.py
```

Do **not** also run `zebra_publisher.py`: the bridge already publishes
perception from the simulation the arm moves in (the publisher refuses to
start if it sees the bridge running).
</details>

### Zebra pick/place on its own (no ROS2)

Moves `zebra_legs` from its spawn spot to the green circle, without Victor's tree:

```bash
$ python3 scripts/zebra_pick_place.py
$ python3 scripts/zebra_pick_place.py --arm left
```

### Perception publisher on its own

Publishes the calibrated `zebra_legs` position on `/zebra/perception_updates`
(needs the ROS2 setup lines above). It simulates its own static copy of the
scene, so it only ever reports the spawn position — for testing perception,
not for a full run:

```bash
$ python3 scripts/zebra_publisher.py
```

Watch what's being published, in another terminal:

```bash
$ ros2 topic echo /zebra/perception_updates
```

### Camera calibration

Aligns the 4 stationlite cameras into one shared frame (headcam) and prints
how accurately each one locates the block:

```bash
$ python3 scripts/camera_calibration.py
$ python3 scripts/camera_calibration.py --loop      # keep reporting until Ctrl+C
$ python3 scripts/camera_calibration.py --view      # also open the MuJoCo viewer
```

### Move an arm to a point (Cartesian control)

Runtime IK: drives one stationlite arm's fingertips to any XYZ point (a red
sphere marks the target; prints the final error):

```bash
$ python3 scripts/cartesian_control.py --target 0.45 0.1 -0.05
$ python3 scripts/cartesian_control.py --arm right --target 0.4 -0.2 -0.1
```

### Stationlite two-arm block stacking

Right arm places its block in the middle, left arm stacks its own on top:

```bash
$ python3 scripts/stationlite_pick_place.py
```

> **Currently broken:** the scene's `block_right` / `block_left` cubes were
> replaced by the three zebra bricks, and this demo still looks them up by name.

### Panda pick-and-place

Menagerie's Panda picks up a cube and sets it on a table:

```bash
$ python3 scripts/pick_place.py
```

### Just look at a model

Any `.xml` / `.urdf` by path (Tab for the side panel, `[` / `]` to cycle
cameras: free cam, headcam, refcam, left_handcam, right_handcam):

```bash
$ python3 scripts/view_file.py stationlite/urdf/stationlite_pick_place.xml
```

A menagerie robot by name:

```bash
$ python3 scripts/view.py franka_emika_panda
$ python3 scripts/view.py --list
```

Every Python script also takes `--gl osmesa` to try a different rendering
backend if the default (`egl`) doesn't work.

## Layout

```
mujoco-robot/
├── src/mjrobots/
│   ├── gl.py                      # MUJOCO_GL backend selection + software-render fallback
│   ├── models.py                  # finds mujoco_menagerie, resolves model name -> scene.xml
│   ├── viewer.py                  # load()/view() helpers
│   ├── pick_place.py              # damped-least-squares IK + scripted pick-and-place motion
│   ├── stationlite_pick_place.py  # two-arm block-stacking demo (stationlite robot)
│   ├── cartesian_control.py       # runtime IK to an arbitrary XYZ point
│   ├── camera_calibration.py      # aligns the 4 stationlite cameras into one shared frame
│   ├── zebra_pick_place.py        # grasp/place one zebra brick via runtime IK
│   ├── zebra_publisher.py         # perception: calibrated brick position -> ROS2
│   └── zebra_skill_bridge.py      # executes zebra_bt pick/place commands + publishes perception
├── scripts/                       # CLI entry points for each module above, plus:
│   └── run_zebra.sh               # one-command launcher: zebra_bt + skill bridge
├── stationlite/                   # stationlite URDF, meshes, and the MuJoCo scene XML
└── requirements.txt
```

## How the pieces work

**Pick-and-place (Panda).** The scene (`src/mjrobots/assets/pick_and_place_scene.xml`)
adds a table, a cube, and a target marker next to the Panda arm;
`src/mjrobots/pick_place.py` drives the arm with Jacobian inverse kinematics
(`_solve_ik`) to reach a sequence of Cartesian waypoints (approach, descend,
grasp, lift, transport, place, retreat), ramping the joint-position actuators
(`data.ctrl`) toward each solved pose so MuJoCo's own physics — not a
scripted teleport — carries the cube.

**Stationlite stacking.** The stationlite robot's files live in `stationlite/`
(URDF + meshes, tracked in this repo). `stationlite/urdf/stationlite_pick_place.xml`
wraps the URDF with actuators, a table, the zebra bricks, and camera
viewpoints. Stacking waypoints are joint angles found by an offline
forward-kinematics search. Grasping is a kinematic "carry" rather than pure
friction: once a gripper closes on its block, the block's position is
snapped to the fingertip midpoint each step until release — the mesh-only
finger geometry isn't reliable enough to hold an object through arm motion
on its own (confirmed experimentally).

**Cartesian control.** Interpolates a straight line of Cartesian waypoints
from the hand's current position to the target and re-solves damped-least-squares
IK at each one (warm-started from the arm's current pose), then ramps
`data.ctrl` through each solution while stepping physics — so the hand's
path stays close to a straight line, not just the joints'.

## Menagerie location

`models.py` looks for your existing `mujoco_menagerie` clone at (in order):

1. `$MENAGERIE_ROOT` if set
2. `/mnt/c/Users/pokem/mujoco_test/mujoco_menagerie` (WSL2 view of your existing Windows clone)
3. `C:\Users\pokem\mujoco_test\mujoco_menagerie` (native Windows view)
4. `~/mujoco_test/mujoco_menagerie` (native WSL clone, if any)

It isn't copied into this project (it's ~2.4GB) — set `MENAGERIE_ROOT` to
point elsewhere if you move or re-clone it.
