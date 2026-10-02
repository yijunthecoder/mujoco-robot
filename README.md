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

This repo itself isn't built with colcon - its scripts run straight from
here. It can still sit inside a ROS2 workspace: `stationlite/` is the robot
maker's ROS1 (catkin) description package, which would stop a plain
`colcon build` with `find_package(catkin REQUIRED)`, so it has an empty
`COLCON_IGNORE` that tells colcon to skip it (MuJoCo reads its XML and
meshes directly).

## How to run

All commands are run from `/mnt/c/intern/mujoco-robot` in WSL.

### Zebra pick/place with Victor's behavior tree (the main one)

Starts Victor's `zebra_bt` tree and our skill bridge together in one
terminal. The MuJoCo window opens and the arms stack legs -> body -> head
on the green circle, finishing with `3 placed, 0 escalated` (~55-90s, more
when bricks have to be flipped first - see "Flipping" below). Each brick is
picked by the arm nearest to it (below).
Victor's tree picks every target: picks where perception last saw the brick,
places at his stack positions. All positions, perception included, are the
brick's origin in the MuJoCo world frame (his `INTERFACE.md`, section 2b).
Output lines are labelled `[tree]` / `[bridge]`; close the MuJoCo window or
press Ctrl+C to stop both.

```bash
$ bash scripts/run_zebra.sh
```

The bricks start **scattered and dropped**: each is dropped at a random spot,
so it lands any way up (see "Dropped bricks" below; `--upright` sets them down
upright instead). The spots are only where an arm can pick it up and place it on the stack at every angle -
the green area of the arms' reach maps (see Notes below; `src/mjrobots/scatter.py`,
at least 10 cm apart and 10 cm clear of the stack). Every run is a new
random scatter, and the first bridge line says which, e.g.
`scatter seed 7: legs (0.51, -0.38) 41 deg, body (0.48, -0.17) 84 deg, ...`.
Replay a scatter by its seed, or start from the old fixed square spots:

```bash
$ bash scripts/run_zebra.sh --scatter 7
$ bash scripts/run_zebra.sh --upright
$ bash scripts/run_zebra.sh --fixed-start
```

**Nearest arm per brick.** A brick in only one arm's green area is picked
by that arm; in both (the middle) or neither, by the arm whose base is closer
(`scatter.choose_arm`). The place always goes to the arm holding the brick.
Before an arm moves, the other arm is parked at its home joint angles
(`go_home`) - otherwise it's still hovering over the stack from its last
place, right where this arm is going (measured without parking: the arms'
planned paths overlapped by 7.7 cm). If a command names an arm
(`"arm": "left"`), that arm does it - Victor's tree doesn't send one yet, so
it can take the choice over later without changes here. One arm for
everything, scattered in that arm's own area only:

```bash
$ bash scripts/run_zebra.sh --arm left
```

Checked headless (ideal perception, real physics, upright bricks): 50 seeds,
149/150 bricks placed (150/150 before the fingers got twisting friction for
flipping - see Notes; the one miss, seed 1's head, lands 2.1 cm off with 2 cm
allowed, and 1.8 cm off even without it), 79 by the right arm and 70 by the
left. Place error 0.9 cm average; the arms never came closer than 8.1 cm
(median closest 19.5 cm). Right arm alone, in its own area: 150/150, max
1.6 cm (measured before the friction change). About half the bricks (72 of 150) were set down turned
180 deg (same footprint, printed face reversed): from most angles the square
grip isn't reachable at the stack - it never happened from the fixed square
spots.

**Dropped bricks** (the default; `--upright` for the old upright scatter,
`--drop` is still accepted): each brick is dropped from 15 cm above a random
spot with a random tumble and lands however physics lets it - so, like a
real messy table, usually not upright. Measured over 400 drops: 46% on a long
side, 21% upside down, 16% on an end, only **17% upright**; all three
upright in 3 of 400 seeds (129, 215, 333). They slide only 1-5 cm, and the
same seed always lands the same way.

Perception now also says how each brick lies and its angle on the table, in
three extra fields after z (Victor's tree reads LYING for its flip step and
shows FACING in its status table):

```
part,STATUS,x,y,z,LYING,yaw,FACING
```

LYING is `UPRIGHT`, `UPSIDE_DOWN`, `ON_SIDE`, `ON_END` or `TILTED`, yaw the angle in
degrees (-90..90), and FACING which way the printed face points (`FORWARD` =
towards world -Y / `BACKWARD`). From the table no camera can tell, so a brick
is `UNKNOWN` until a pick with `desired_facing` has seen its print (shown to the
headcam, or B's hand camera at a flip handover); from that moment it's known -
while held (arm joints + how it sits in the hand), put back on the table (also
when it can't be placed facing FORWARD) and placed (`_known_facing` in the
bridge). Pointing within 20 deg of sideways it stays `UNKNOWN` (in the hand the
brick can turn ~18 deg). Checked against the sim's truth on 9 bricks: every
value right. The bridge log shows it too, e.g. seed 1's body:
`body  LOCATED  ON_SIDE, facing UNKNOWN` (position and yaw are in the published
message, not the log). A pick
of a brick that isn't upright is refused before moving (only top-down grips
exist): his tree sends **`flip`** first (its `EnsureUpright` step).

**Flipping** (`src/mjrobots/zebra_flip.py`, called by the bridge's `flip`
skill). The bridge turns the brick upright and sets it down on the table;
his tree then picks and places it like any other. Which way it's done is
decided here, not in his tree - the tree says *what* (flip the body), the
bridge decides *how*:

| brick lies | turn needed | how |
|---|---|---|
| on its side | 90 deg about its long side | **one hand** if the arm can reach: grip the two ends, roll the wrist 90 deg, set it down. Else **two hands**: A rolls 45 deg, B takes it by the middle and turns 45 deg |
| on its end | 90 deg about its short side | **one hand** if the arm can reach: grip the two long faces, roll the wrist 90 deg, set it down. Else two hands, 45 + 45 deg |
| upside down | 180 deg | two hands, 90 + 90 deg (one wrist can't turn 180 deg) |

**No flip where it lies? Move it first** (`MovePlan`): most bricks that couldn't be
flipped lie far out, where an arm can grip them from above but can't reach with its
hand rolled. Then one arm picks it up as it lies, carries it to a free spot in the
middle of the area both arms reach (turned 0 or +-90 deg), sets it down the same way
up, and the flip is planned again from where it landed - all in the one `flip`
command. Seeds 1-30 dropped, with his tree: flippable bricks 48 -> 73 of 77 (plan
only), full zebras 7 of 28 -> 17 of 30.

A brick no safe plan fits is answered `FAILED` ("no way found to flip it
(arms not moved)") and stays where it is; his tree retries, then escalates
it to a human. Planned on a scratch copy first (nothing moves until a whole
flip is known to work), in a separate process so the viewer and perception
keep running. Over 30 dropped bricks (seeds 1-12): **18 can be flipped**
(on its side 13/17, on its end 1/4, upside down 4/9; over 40 seeds on its
end 14/19 since one hand can do it, upside down 8/26) - see Notes. Seeds to watch:

```bash
$ bash scripts/run_zebra.sh --scatter 7      # legs and body flipped by one hand each, full zebra
$ bash scripts/run_zebra.sh --scatter 3445   # legs (45+45) and head (on its end) by two hands, full zebra
$ bash scripts/run_zebra.sh --scatter 12     # legs one hand, body two hands; head upside down, escalated
$ bash scripts/run_zebra.sh --scatter 129    # all three land upright, nothing to flip
```

**No set-down after a two-hand flip of a brick on its side** (45 + 45 deg):
B ends holding it the normal way (top-down, by its long faces), so it keeps
it; perception reports it `PICKED`, the tree's next `pick` is answered at
once and B places it - ~25 s saved. Every other flip ends holding the ends
(or sideways) and still sets the brick down for a fresh pick. With a
`desired_facing`, which side the print is on comes from B's hand camera at
the handover (~8 cm away); if it didn't see it, or B's grip can't set it down
facing that way, B sets it down and it's picked as above. Tested: kept bricks
placed 1.0-1.4 cm off. Seed 3445's legs with a facing (print not seen at the
handover: set down, shown, picked again) used to fall out of the fingers after
the show; fixed by the deeper show grasp (see "Which way the print faces"),
now placed 0.8-0.9 cm off facing FORWARD with Victor's tree.

A flip takes 25-60 s live, longer than his tree's 30 s flip timeout: the
tree then marks the attempt `FAILED` and sends `flip` again, which finds the
brick upright and succeeds at once - it works, but logs a false failure
(asked Victor to raise the timeout to ~120 s).

**Which way the print faces** (`src/mjrobots/zebra_facing.py`). Each brick has
its print on one long face; on the stack it fits two ways round. A pick may
carry `"desired_facing": "FORWARD"` (agreed with Victor: FORWARD = the print
points to world -Y, the robot's right / the right of the headcam image;
`BACKWARD` the other way). Without it, picks work as before. With it:

1. pick the brick as usual, then **show it to the headcam**: hold it 50-65 cm
   in front of it with a long face turned to it (from the table no camera can
   tell which side the print is on: from above the long faces don't show, and
   the headcam sees them 3-10 px tall; shown, 20-30 px). This pick grips
   deeper (fingertips 0.5 cm above the table, not at mid-height): tilted, the
   brick slides ~1 cm down the fingers, and from the usual grip that left too
   little held - seed 3445's legs fell out upside down;
2. from which way the print points in the hand, work out how it's held;
3. **put it back and pick it up again** with the grip that sets it down facing
   that way - the same grip, the other one, or the other arm (tilted towards
   the camera it shifts in the fingers, up to 1.3 cm / 18 deg; placed from that
   grip, bricks landed off target or fell off the stack). If no grip works
   from where it lies, it's **put back turned** (30, 60 ... 150 deg, the
   wrist turning as it sets it down) so that one does - only a turn whose grip
   still works if it lands 5 deg / 1 cm off (`_regrip_turn`);
4. place it at exactly that yaw. If no grip of either arm can reach that yaw at
   the stack, the pick fails (`FAILED`, brick put back) with
   `"reason": "FACING_IMPOSSIBLE"`, so Victor's tree escalates at once instead
   of retrying. A brick that turned over in the fingers while shown gets
   `"reason": "NEEDS_FLIP"` (his retry re-checks upright and flips it); one
   whose print couldn't be seen gets no reason (a normal retry).

Headless, upright scatters (seeds 4-30): 53 bricks placed, **every one facing
FORWARD**, 1.0 cm off on average (max 1.8), all three in 12 of the 27 seeds;
14 failed as "can't face that way" (the wrist can't reach every angle at the
stack, not even after a turned put-back) and one slid 2.1 cm off the stack.
Without the turned put-back: 38 placed, 7 full zebras, 20 failed. A pick with a
facing takes ~30 s headless (show + put back + pick again), longer live:
asked Victor to raise the pick timeout to ~90 s. Seeing the print is a
stand-in (`_print_look` in the bridge): the true side, but only when the
headcam really has a usable view of a face (half of it unblocked, 20+ px) -
whether 20-30 px is enough for a real detector is still to be checked.

Confirm before moving (for the first real-robot runs): each arm move prints
every joint's current angle, target, and change (flagging changes over
0.5 rad), then waits - ENTER moves, `q` + ENTER refuses the move (the arm
stays put and the skill reports FAILED). `first` asks only before the first
move; `all` asks before every move. Answer within zebra_bt's 30 s skill
timeout.

```bash
$ bash scripts/run_zebra.sh --confirm-moves first
```

Every arm move - parking included - is also checked against the other arm
before it runs (the sim itself lets the arms pass through each other): a
move that would bring the arms - or a held brick - within 2 cm is refused
and the skill reports `FAILED` (`src/mjrobots/arm_clearance.py`).

### Fail tests (Victor's retry / escalate behaviour)

The results below were measured from the old fixed brick spots with the
right arm doing everything, so add `--fixed-start --arm right` to see the
same thing (from a scatter, which camera sees a brick - and so how a missed
pick is noticed - depends on where it lies and which arm is over it).

`--fail-part` sends every pick of that part 8 cm to the side of the real
brick, and the pick is reported `FAILED` one of two ways. Legs and head:
the arm looks again from above the (wrong) spot, sees the brick 8 cm away,
and gives up before descending ("brick is 7.9 cm from where the pick was
aimed"). Body: the arm hides it from the head camera at that point, so the
arm descends and the gripper closes on nothing (the fingers close to ~0 cm
instead of stopping at the brick's 3.2 cm). Either way the tree retries it. After the 4th failed pick the tree
gives up on that part ("retries exhausted, escalating to human"), and
anything stacked on top of it is skipped.

Body fails: legs placed, body escalated after 4 attempts, head skipped:

```bash
$ bash scripts/run_zebra.sh --fail-part body
```
→ `1 placed, 2 escalated`

Legs fail: nothing can be stacked, so all three escalate:

```bash
$ bash scripts/run_zebra.sh --fail-part legs
```
→ `0 placed, 3 escalated`

Head fails: legs and body placed, head escalated:

```bash
$ bash scripts/run_zebra.sh --fail-part head
```
→ `2 placed, 1 escalated`

Change how far off the pick goes (metres, default `0.08`). Small offsets
don't miss: up to 4 cm the look again from above corrects the aim (legs,
head), and the closing fingers push the brick into the middle of the jaws
anyway (measured on the body: 3 cm off still grasps, 5 cm misses), as a
real gripper would:

```bash
$ bash scripts/run_zebra.sh --fail-part body --fail-offset 0.05
```

Only fail the first N picks with `--fail-times N`. Here the body's first pick
misses, and the retry picks it normally:

```bash
$ bash scripts/run_zebra.sh --fail-part body --fail-times 1
```
→ `3 placed, 0 escalated`, body `attempt 2`

Relocate instead of retry: add `--fail-bump`. The missed pick also knocks the
brick 5 cm towards the stack, and perception reports it `LOST` for 2s. The
tree logs `will re-locate and retry`, waits for perception to find the brick
again, and picks it at its **new** position:

```bash
$ bash scripts/run_zebra.sh --fail-part body --fail-times 1 --fail-bump
```
→ `3 placed, 0 escalated` (~62s)

Knock a placed brick off the stack: `--knock-placed PART` knocks that part
8 cm sideways right after its first place. The bridge looks at every brick
it places before reporting success, so this place is reported `FAILED`
("placed brick didn't stay put: it's 8.2 cm to the side, -3.9 cm below the
target"), the tree re-locates it, and it is re-picked from where it fell and
placed again:

```bash
$ bash scripts/run_zebra.sh --knock-placed body
```
→ `3 placed, 0 escalated`, body picked and placed twice

Knock a brick off later: `--knock-later PART` knocks it off the stack after
its own landing check passed. Before every place the bridge looks at the
bricks already stacked, so the next part isn't stacked onto nothing: it's
put back where it was picked and the place fails ("won't place body: the
stack below it is broken (legs: 8.3 cm to the side)"). The second time, the
part is escalated to a human (the tree can't redo a part it already counts
as placed). Once every part is placed, the bridge also checks the whole
zebra ("zebra check: all 3 bricks in place"):

```bash
$ bash scripts/run_zebra.sh --knock-later legs
```
→ `1 placed, 2 escalated` (body refused twice and escalated, head skipped)

No perception at all: run only the tree, without the bridge. The legs are
never found; after 3 searches of 10s each they escalate, and body and head
are skipped:

```bash
$ source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
$ export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
$ ros2 run build_a_zebra zebra_bt_node
```
→ `0 placed, 3 escalated` (~34s)

If the MuJoCo window doesn't appear, restart WSL from Windows PowerShell
(`wsl --shutdown`), then open WSL and run again.

If a run finishes with `3 placed` after ~1s without the arm moving, a bridge
from an earlier run is still open and reporting every part as `PLACED`. Close
its MuJoCo window, or stop everything:

```bash
$ pkill -9 -f zebra_skill_bridge.py; pkill -f zebra_bt_node
```

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

The hand cameras ride on the arms, so their readings are first corrected for
the arm's current pose (forward kinematics), then converted into the headcam
frame.

**Checking the cameras.** A headless check (no ROS, no display, ~5 s): it
calibrates the cameras the same way zebra_publisher does, then tests every
camera against the brick's true position with the arms at home, hovering over
the stack, hovering over each brick, and in random arm poses, with the bricks
both spread out and stacked. It prints a summary per camera and ends with
`PASS` (every camera within 1 cm) or `FAIL` (exits non-zero). Run it after
changing the cameras, the scene XML, or the calibration:

```bash
$ python3 scripts/check_camera_agreement.py
$ python3 scripts/check_camera_agreement.py --tol 0.005                       # allowed error in metres (default 0.01)
$ python3 scripts/check_camera_agreement.py --pixel-noise 0 --depth-noise 0   # noise-free: should show 0.00 cm
$ python3 scripts/check_camera_agreement.py --seed 1                          # different random poses and noise
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
│   ├── scatter.py                 # random brick start spots / drops, inside the arm's reach zone
│   ├── zebra_facing.py            # picks a brick so its print faces a set way on the stack
│   ├── zebra_flip.py              # turns a brick that isn't upright onto its studs (one hand or two)
│   └── zebra_skill_bridge.py      # executes zebra_bt pick/place/flip commands + publishes perception
├── scripts/                       # CLI entry points for each module above, plus:
│   ├── check_camera_agreement.py  # headless check: every camera agrees with ground truth
│   ├── reach_map.py               # where each arm can pick+place; writes the scatter zones
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

## Notes

### Arm reach maps (2026-09-29)

Where on the table can each arm pick a brick up and put it on the stack?
Measured before scattering the bricks randomly, so random spots are only
drawn from where the pick can actually work.

![Right arm reach map](docs/reach_map_right.png)

How it was measured: the legs brick was put on a 3 cm grid over the table
(x 0.14–0.80, y ±0.45; 713 spots) at 6 angles each (every 30°; a brick looks
the same turned 180°). At each one, the bridge's real pick and place moves
were planned with the same IK, joint-jump and arm-clearance checks
(`grasp_part`, then `place_part` at stack levels 0 and 2), and the arm was
jumped to the end of each planned move instead of simulating it (planning is
where every refusal happens, so this is fast and gives the same yes/no). The
other arm stayed at home. Cameras were asked whether any of them sees a brick
there, with the arms at home.

What the right arm's map shows:

- **Green (172 spots): every angle works.** Roughly x 0.16–0.53 on the right
  arm's own side, narrowing towards the middle. Nothing works past y ≈ +0.15
  (the left arm's side).
- **Too close to the robot** (x < 0.15) fails: the joints would jump more
  than 0.2 rad between steps.
- **Far away** (x > 0.55) works at some angles only (yellow), then none: the
  grip orientation can't be reached there.
- **Today's legs spot (0.2, −0.15) sits on the edge:** a brick turned 30°
  there can be picked but not placed (IK can't reach the stack in the grip
  that fits it). It works today only because the bricks start square.
- **Every spot is seen by a camera** (headcam and refcam), so perception
  can find a brick anywhere on the table.

The left arm is almost an exact mirror image (also 172 green spots). Even
the edge case mirrors: at today's head spot (0.2, +0.15) the left arm can't
place a brick turned 150°.

![Left arm reach map](docs/reach_map_left.png)

**Either arm** (a spot and angle counts if at least one arm can do it) -
what using the nearest arm per brick would cover: **288 green spots, up from
172 with one arm (+67%)**. The two arms' green areas overlap in the middle
(71 spots either arm can do at every angle), which is where the choice of
arm is free.

![Either arm reach map](docs/reach_map_either.png)

### Flipping bricks (2026-09-30 / 10-01)

How `zebra_flip.py` got to where it is, with the measurements behind each
choice. "Plan-only" = the planner's yes/no on a scratch copy; "physics" = the
whole flip run headless with real physics and a noisy camera look.

**Order the planner tries** (first that works wins): one hand setting the
brick down near where it lay (6/10 cm) -> the two-hand strategies -> one hand
setting it down further away (anywhere in the green zone, one spot per 6 cm).
The far search is slow when nothing fits (~20 s), so it comes last. Every
set-down spot is inside the green zone, so the brick can be picked up again.

**Results**, 30 dropped bricks (seeds 1-12), plan-only, then physics:

| | plans | physics |
|---|---|---|
| on its side, one hand | 8 | 10/10 upright |
| on its side, two hands (45+45) | 5 | all tried upright (seeds 2, 3445, 5, 12) |
| on its end, two hands | 1 | upright (seed 3445's head) |
| on its end, one hand (added after; 40 seeds) | 7 of 19 (14/19 with two hands, was 8/19) | 7/7 upright |
| upside down, two hands | 4 | 4/4 upright (0/4 before the re-aim fix below) |
| no plan | 12 | - |

Planning: 7.1 s average, 15.6 s slowest (4 runs in parallel); a "no plan"
answer takes ~20 s in the bridge.

**Fixes found by watching flips fail:**

- **Twisting friction on the fingers** (`condim="4"` in
  `stationlite_converted.xml`): without it a brick held by its two ends spun
  between the fingertips - the hand rolled, the brick didn't. The strength is
  the bricks' torsional friction (MuJoCo takes the larger of the pair):

  | twisting friction | normal pick/place (150) | flips (8) |
  |---|---|---|
  | none | 150 | - (brick spins) |
  | 0.005 | 150 | 6 (an on-end brick slipped in A's hand) |
  | **0.01 (used)** | 149 | 7 |
  | 0.02 | 149 | 7 |
  | 0.05 | 146 | 7 |

  The one pick/place miss at 0.01 is 1.8 cm off even without the friction (a
  placement accuracy limit, 2 cm allowed). The flip that fails at every value
  is the upside-down seed 2 above.
- **B grips only the middle.** B used to be allowed to grip 0.75-1.25 cm
  above the middle (more poses fit past A's fingers). A brick held by its top
  strip slid out of B's fingers on the way down: 5 of 6 bricks in physics,
  and once live (seed 1's body landed upside down and was tossed around).
- **One hand alone** for bricks on their side: the brick is already upright
  in A's fingers after a 90 deg roll, so no handover is needed. It flips the
  bricks that used to need the slipping handover (4 of 4 in physics). It
  needs the rolled hand to reach the table without the gripper touching it
  (kept 0.5 cm clear).
- **Set-down aimed by a look at the held brick**, not by where the fingers
  are: a brick held off its middle, set down as if centred, was pushed into
  the table. If the planned spot is out of reach once the camera has seen how
  the brick really sits (seed 2's legs, tilted ~8 deg in the hand), the next
  free spot is used. The hand keeps its orientation on the way: a free move
  let the wrist swing and an on-end brick held by its ends swivelled.
- **B's re-aim measures the turn from the whole orientation.** It used the
  heading of the brick's long side, which in an upside-down flip points
  straight up - a meaningless heading that read 82-88 deg on all 4
  upside-down bricks, so B turned its hand that far and was out of reach.
  Upside down went from 0/4 to 4/4 in physics.
- **A steps clear before going home** after the handover: the plan checks A's
  way home with B where the plan put it, but B re-aims ~1 cm at where the
  brick really hangs (seed 1: refused at 1.8 cm from B, 2 cm allowed).
  Planning with extra margin instead lost plans (17 -> 12 of 30 at 0.3 cm).
- **IK gives up on out-of-reach targets** after 100 iterations without
  progress (`solve_ik_pose(give_up_after=...)`, planner only): ~80% of the
  targets the planner tries are unreachable. Planning 44.7 -> 8.2 s average
  over 30 bricks, the same plan every time.
- **Planning in its own process**: in a thread it shared one CPU core with
  the viewer, physics and cameras - seed 1's body took 42 s there, 18 s alone.

**Tried and dropped:** upside down in two steps (one hand turns it onto its
side, then the on-side flip): 2 more of 26 upside-down bricks, but a "no
plan" answer took up to 44-69 s, past the 30 s flip timeout.

**A flip that fails halfway** (e.g. a move refused) used to
open both hands, dropping the brick 11-15 cm, once 4 cm from the stack. Now a
hand that still holds it (the fingers stopped at one of the brick's widths)
first lowers it onto a free spot - the same rules as a set-down - then lets go:
seeds 9 body and 2 legs ended 14 and 22 cm from the stack, put down, not
dropped (`_put_down_after_failure`).

**Still open:** upside-down coverage (8/26 get a plan); an on-end brick with
no plan now takes longer to say so (the far one-hand search: up to ~39 s with
4 tests in parallel, past the 30 s timeout); which way the print
faces at the stack (perception only knows the angle up to 180 deg); the
brick's 3D pose comes from the simulation (a stand-in for a detector).

## Menagerie location

`models.py` looks for your existing `mujoco_menagerie` clone at (in order):

1. `$MENAGERIE_ROOT` if set
2. `/mnt/c/Users/pokem/mujoco_test/mujoco_menagerie` (WSL2 view of your existing Windows clone)
3. `C:\Users\pokem\mujoco_test\mujoco_menagerie` (native Windows view)
4. `~/mujoco_test/mujoco_menagerie` (native WSL clone, if any)

It isn't copied into this project (it's ~2.4GB) — set `MENAGERIE_ROOT` to
point elsewhere if you move or re-clone it.
