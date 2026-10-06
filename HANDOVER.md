# Hand-over: zebra build in MuJoCo (state of 2026-10-05)

For whoever continues this work - on the simulation or on the real Station Lite.
Read this first (~10 min); [README.md](README.md) is the detailed reference behind it.

## 1. What this is

A MuJoCo simulation of the two-arm **Station Lite** robot building a LEGO "zebra"
(three bricks: **legs**, **body**, **head**, stacked in that order), driven by
**Victor's behaviour tree** (`zebra_bt`, ROS2 package `build_a_zebra`).

- **Victor's tree decides *what*** - locate the legs, flip them if needed, pick them,
  place them on the stack - and sends one command at a time over ROS2.
- **This repo decides *how*** and does it in the simulation: the bridge
  ([src/mjrobots/zebra_skill_bridge.py](src/mjrobots/zebra_skill_bridge.py)) receives
  each command, moves the arms (inverse kinematics, collision checks), and replies
  SUCCEEDED or FAILED. It also publishes "perception" (where each brick is, how it
  lies, which way its print faces) twice a second, from the simulated cameras.

```
Victor's tree  --/zebra/skill_commands-->  bridge (this repo)  --> MuJoCo arms
               <--/zebra/skill_status----
               <--/zebra/perception_updates--  (the bridge's simulated cameras)
```

Every brick ends up **upright** and with its **print facing FORWARD** (towards world
−Y, agreed with Victor). Bricks that start on their side, on their end or upside down
are **flipped** first - by one hand, or handed between both hands.

## 2. How to run it

Setup (WSL2 Ubuntu 22.04, ROS2 Humble, Victor's package built in `~/ros2_ws`): see
README "Setup". All commands from `/mnt/c/intern/mujoco-robot` in WSL.

```bash
bash scripts/run_zebra.sh --demo 3       # watch one: the MuJoCo window opens (demos 1-5)
bash scripts/run_zebra.sh                # a random one of the 5 demos
bash scripts/check_demos.sh              # check after ANY change: all 5 demos, headless, ~3 min
```

`run_zebra.sh` starts Victor's tree and the bridge together; it ends with
`Zebra fully assembled` / `3 placed, 0 escalated`. Close the window or press Ctrl+C
to stop. `check_demos.sh` runs the 5 demos without a window, 4x faster than real
time, and prints a report per demo (full zebra or not, flips, placement accuracy,
problems) - "5 of 5 built a full zebra" means nothing broke.

Other starts (not the demos): `--drop` (bricks dropped anywhere in reach, messy),
`--upright`, `--random`, `--scatter SEED` - see README.

## 3. What works (checked 2026-10-05)

| demo | legs | body | head |
|---|---|---|---|
| 1 | upside down (two hands) | upright | on its end (one hand) |
| 2 | on its side (one hand) | on its end (two hands) | upright |
| 3 | upside down (two hands) | on its end (one hand) | on its side (two hands) |
| 4 | on its side (one hand) | upright | upside down (two hands) |
| 5 | upright | on its end (one hand) | upside down (two hands) |

- All 5 build a full zebra, every run of the last checks (also with Victor's newest
  tree, `05b5d4c`).
- About **3 min** per zebra headless (5 min with the window open - see 4.4); flip
  planning 2-12 s; bricks placed 0.7-1.6 cm from their target.
- At the start of the week a run could take 10 min with a minute of thinking per
  flip and several retries; most of that came from awkward start positions, now
  avoided by placing the bricks in a box (4.1).

## 4. Known problems and limits

1. **Bricks must start inside a box, at an angle.** Brick centres within x 0.29-0.47 m,
   y −0.12..+0.12 m, at least 10 cm apart and from the stack; upside-down bricks turned
   70-80° or 100-110°, others 60-120° (0° = long side pointing away from the robot).
   Measured with the flip planner: outside this, flips often aren't possible at all
   (an upside-down brick can't be flipped at 0° or 90° anywhere). On the real table:
   tape the box, put the bricks in it diagonally. ([scatter.py](src/mjrobots/scatter.py), `PLACE_BOX`)
2. **A handover right beside the finished stack** can rub the stacked bricks (the brick
   sags ~1 cm in the second hand). That start (old demo seed 3) was taken out of the
   demos; idea for a fix: don't hand over right next to a built stack. (README Notes)
3. **Some upright bricks can't be turned to face FORWARD** with any grip of either arm,
   or picked at their angle (seeds 2 and 26 of the placed starts). They're refused
   cleanly (FACING_IMPOSSIBLE) and escalated to a human.
4. **The live window halves the speed** (~5 min instead of ~3): drawing the scene costs
   as much as the physics. Not fixed yet - idea: redraw 20-30 times a second instead
   of 60, and print the sim speed in the log.
5. **Bricks creep in the fingers** (1-2 cm when held tilted) - a MuJoCo soft-contact
   effect, not a weak grip (5 N per finger is ~30x what a 30 g brick needs). Fix found
   (two physics settings) but **not applied**, because the demos are tuned to today's
   physics. Details in README Notes, "A brick creeps in the fingers".
6. **The print-facing routine costs ~1 min per zebra**: every pick shows the brick to the
   head camera, puts it back and picks it again with the right grip. Skipping the
   put-back when the grip is already right would save ~20 s, but needs a camera-aimed
   place.

## 5. What is simulation-only (replace these for the real robot)

The robot code asks for what it "sees" in a few places; in the simulation the answers
come from the simulator's true state plus camera-like noise:

| what | where | real robot needs |
|---|---|---|
| how a brick lies, which way it's turned, where it sits in the hand, which side the print is on, where the other bricks are | [sim_sensing.py](src/mjrobots/sim_sensing.py) - **one class, `SimSensing`; replace it** | a brick detector on the camera images (head camera + hand cameras) |
| where each brick is (position) | [camera_calibration.py](src/mjrobots/camera_calibration.py) `SimulatedCameras` (images computed from the true pose + noise), used by [zebra_publisher.py](src/mjrobots/zebra_publisher.py) | the same detector; the camera calibration itself (`CameraCalibration`) is real code |
| which brick is which | simulation body names (`PART_BODIES`, [zebra_publisher.py:81](src/mjrobots/zebra_publisher.py#L81)) | recognise legs/body/head by colour or shape |
| the flip planner's world | it plans on a copy of the simulation state ([zebra_flip.py](src/mjrobots/zebra_flip.py), `PlanningProcess`) | build that state from perception's reports |
| moving the arms and grippers | MuJoCo position actuators (scene XML) driven by [cartesian_control.py](src/mjrobots/cartesian_control.py) | the Station Lite driver: send joint targets, read encoders, gripper open/close with ~5 N |

The IK, the collision/clearance checks, the flip strategies, the facing logic and the
bridge's protocol with Victor's tree are **real robot code** - they don't depend on
the simulation.

## 6. Numbers to measure on the real table

| value | now | where | change? |
|---|---|---|---|
| table height (brick origin resting on it) | −0.0842 m | [scatter.py:41](src/mjrobots/scatter.py#L41) `TABLE_Z` (Victor's tree has its own copy) | **yes** |
| where the zebra is built | (0.4148, 0) m | [scatter.py:42](src/mjrobots/scatter.py#L42) `STACK_XY` (Victor's place targets too) | **yes** |
| the work box and angles | see 4.1 | [scatter.py:61-62](src/mjrobots/scatter.py#L61) `PLACE_BOX`, `PLACE_YAWS` | **re-measure** with the real arm's reach |
| flip handover spots, distance kept from the stack | 0.35-0.45 m, 15 cm | [zebra_flip.py:73](src/mjrobots/zebra_flip.py#L73), [:220](src/mjrobots/zebra_flip.py#L220) | if the stack moves |
| safety margins: arms apart / arm to other bricks | 2 cm / 1 cm | [arm_clearance.py:29,35](src/mjrobots/arm_clearance.py#L29) | bigger for the first real tests |
| brick size | 6.4 x 3.2 x 3.84 cm | [zebra_pick_place.py:50-55](src/mjrobots/zebra_pick_place.py#L50) | no (same bricks) |
| gripper force | 5 N per finger | [stationlite_pick_place.xml:232](stationlite/urdf/stationlite_pick_place.xml#L232) | set the real gripper's current limit to match |
| FORWARD = print towards world −Y | | [zebra_facing.py:54](src/mjrobots/zebra_facing.py#L54) | no (agreed with Victor) |

All positions are in the MuJoCo world frame, brick origin (centre of the top face) -
Victor's `INTERFACE.md`, section 2b.

## 7. Next steps for the real robot (most important first)

1. **Arm driver interface**: one class "move joints / gripper / read joints" with the
   MuJoCo version and a Station Lite version, so the bridge runs on either.
2. **Start-up safety**: read the encoders before the first move, check each joint's
   direction, slow speeds and `--confirm-moves all` (asks before every move) for the
   first runs.
3. **Real perception**: a brick detector replacing `SimSensing` and `SimulatedCameras`
   (section 5); check the print can really be read from the head camera at 50-65 cm
   (~20-30 px tall in the image - assumed, not tested on the real camera).
4. **Measure the table** (section 6) and re-measure the work box with the real arm.
5. Then the sim-side items in section 4, if still useful.

## 8. Where things are

| file | what it does |
|---|---|
| [scripts/run_zebra.sh](scripts/run_zebra.sh) | starts Victor's tree + the bridge (the main command) |
| [scripts/check_demos.sh](scripts/check_demos.sh), [demo_report.py](scripts/demo_report.py) | the 5-demo check and its report |
| [src/mjrobots/zebra_skill_bridge.py](src/mjrobots/zebra_skill_bridge.py) | the bridge: commands from the tree -> pick / place / flip; replies; perception |
| [src/mjrobots/zebra_pick_place.py](src/mjrobots/zebra_pick_place.py) | grasp and place one brick (IK, grip checks) |
| [src/mjrobots/zebra_facing.py](src/mjrobots/zebra_facing.py) | show the brick to the camera, choose the grip so the print faces FORWARD |
| [src/mjrobots/zebra_flip.py](src/mjrobots/zebra_flip.py) | flip planning (in a separate process) and execution |
| [src/mjrobots/arm_clearance.py](src/mjrobots/arm_clearance.py) | refuses moves that bring the arms (or a held brick) too close to each other or to other bricks |
| [src/mjrobots/sim_sensing.py](src/mjrobots/sim_sensing.py) | the simulated "senses" (section 5) |
| [src/mjrobots/zebra_publisher.py](src/mjrobots/zebra_publisher.py), [camera_calibration.py](src/mjrobots/camera_calibration.py) | perception messages; camera calibration and simulated cameras |
| [src/mjrobots/scatter.py](src/mjrobots/scatter.py) | how the bricks start (box, drop, upright) |
| [stationlite/urdf/stationlite_pick_place.xml](stationlite/urdf/stationlite_pick_place.xml) | the MuJoCo scene: robot, table, bricks, cameras, gripper force |

**Victor's side**: branch `Victor` of this repo, folder `build_a_zebra/` (C++). It is
built from a plain copy in `~/ros2_ws/src/build_a_zebra`; to update it: back up that
folder, `git archive origin/Victor build_a_zebra | tar -x -C ~/ros2_ws/src/build_a_zebra
--strip-components=1`, then `colcon build --packages-select build_a_zebra` in
`~/ros2_ws`. Reply codes our bridge sends him on a FAILED: `FACING_IMPOSSIBLE`,
`NEEDS_FLIP`, `NO_FLIP_PLAN` (he escalates on the first and last at once).

Open with Victor (minor): a failed pick/place may still log as "timeout"; an incoming
PLACED can overwrite ESCALATED in his world model.
