"""ROS2 bridge that actually executes Victor's zebra_bt pick/place commands
in this MuJoCo sim - the missing other half of zebra_publisher.py.

Victor's `SkillBridge` (build_a_zebra/src/main.cpp, checked out locally at
~/ros2_ws/src/build_a_zebra) publishes one JSON message per attempt on
`/zebra/skill_commands`:

    {"command_id", "skill": "pick"|"place"|"flip", "part_id", "target": {"x","y","z"}}

and blocks that part's `PickPart`/`PlacePart` BT node in RUNNING until a
matching reply arrives on `/zebra/skill_status`:

    {"command_id", "status": "SUCCEEDED"|"FAILED"[, "message"][, "reason"]}

`reason` (only on some FAILED replies) is a fixed code his tree acts on, so it
doesn't have to match words in `message`: "FACING_IMPOSSIBLE" = no grip of
either arm can place it facing as asked (he escalates at once), "NEEDS_FLIP" =
it turned over in the fingers (his retry re-checks upright and flips it).

This node is that reply: for all three zebra parts (legs/body/head, ids
31111p0e/f/g - see bom.json) and both arms, it keeps a MuJoCo viewer running,
executes `grasp_part`/`place_part` (zebra_pick_place.py), and reports the
result back. Both picks and places go to the command's target: picks to
where perception last saw the brick, places to Victor's stack positions
(`placeTargetFor` in his main.cpp).

Arm note: a command may name the arm (`"arm": "left"|"right"`) - then that
arm does it. Without one (Victor's tree doesn't send it yet) the bridge picks
the nearest arm for the brick (`scatter.choose_arm`: the arm whose reach zone
it's in, else the closer base), and a place always goes to the arm holding
the brick. Before an arm moves, the other arm is parked at home
(`go_home`) - otherwise it's still hovering over the stack from its last
place, right where this arm is going.

Flip note: "flip" (his EnsureUpright -> FlipPart, sent for a brick perception
reports isn't UPRIGHT) turns the brick upright - with one hand or both - and
sets it down on the table (zebra_flip.py); his tree then picks it normally.
Already upright -> SUCCEEDED at once; no safe way to flip it -> FAILED, arms not
moved. The plan is made in a separate process (zebra_flip.PlanningProcess), so
the viewer and perception keep running (his tree marks a part stale after 2 s
without perception) and the planner gets its own CPU core. Planning plus the
moves can run past his 30 s flip timeout: his tree then sends "flip" again, and
that one finds the brick upright and succeeds at once.

Coordinate note: per Victor's INTERFACE.md section 2b, every `target` - like
every perception update - is the brick's body *origin* in the MuJoCo world
frame (robot base). The only conversion before it's an IK goal is origin ->
geometric center (`_BRICK_CENTER_OFFSET_Z`), which `grasp_part`/`place_part`
expect. (Perception measures in headcam's frame and converts to world before
publishing - see zebra_publisher.py.)

Perception note: this process also publishes perception for all three
zebra parts (`/zebra/perception_updates`), from its OWN live simulation - a
`ZebraPerceptionPublisher` embedded on this same model/data. A separate
zebra_publisher.py process would load its own copy of the scene, where
nothing ever moves, and keep reporting the brick's spawn position forever.
Publishing is driven from every physics step (`_PerceivingSync`), not a ROS2
timer, so it keeps going during a blocking grasp/place too.

Threading note: the ROS2 subscription callback only enqueues commands
(`ZebraSkillBridge.pending`); the actual physics/IK work runs on the main
thread inside the viewer loop via non-blocking `spin_once` each idle tick -
not a background spin thread - so nothing ever touches MuJoCo's `data`
concurrently from two threads.

Requires ROS2 sourced first: `source /opt/ros/humble/setup.bash`.
"""

from __future__ import annotations

import json
import queue

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from . import sim_step
from .camera_calibration import CAMERAS, REFERENCE_CAMERA
from .pick_place import _RealtimeClock, _ThrottledSync
from .scatter import STACK_XY, TABLE_Z, UPRIGHT, Zone, arm_bases, choose_arm, describe, drop_bricks, lying, scatter_bricks
from .stationlite_pick_place import _GRIP_OPEN, _hold
from .zebra_facing import (
    FACING_YAW, SHOW_CAMERA, FacingError, can_place, full_yaw_near, held_from_print_in_hand, long_face_view,
    pick_facing,
)
from .zebra_flip import MovePlan, PlanningProcess, execute_flip, set_down_kept
from .zebra_publisher import ALL_PART_IDS, PART_BODIES, PART_LABELS, ZebraPerceptionPublisher
from .zebra_pick_place import (
    _BRICK_CENTER_OFFSET_Z,
    BRICK_HEIGHT,
    _DEFAULT_SCENE,
    ZebraArmContext,
    _wrap,
    go_home,
    grasp_part,
    place_part,
    placement_error,
    put_back,
)

COMMAND_TOPIC = "/zebra/skill_commands"
STATUS_TOPIC = "/zebra/skill_status"
# Looks averaged by the look again from hover (see `_relook` in run_bridge).
_RELOOK_SAMPLES = 10
# The print-side look (see `_print_look`): the headcam must see this share of a long
# face unblocked, this tall in its image, to tell printed from blank (assumed - to be
# checked on the real camera; shown 50-65 cm away a face is 20-30 px tall).
_PRINT_LOOK_MIN_SHARE = 0.5
_PRINT_LOOK_MIN_PX = 20.0
# Perception's FACING says UNKNOWN when the print points within this of sideways (world
# +-X) - see _known_facing.
_FACING_SIDEWAYS = np.radians(20.0)
# Noise of one simulated yaw look (see `_sim_brick_yaw`).
_YAW_SIGMA = np.radians(2.0)


def _sim_brick_yaw(data, brick_id: int, rng: np.random.Generator, samples: int) -> float:
    """Simulated yaw detector: the brick's rotation about vertical from square
    (radians, in [-pi/2, pi/2) - a brick looks the same turned 180 deg),
    averaged over `samples` noisy looks. Like SimulatedCameras.observe for
    position, it reads the true pose and adds noise - a stand-in for
    estimating the brick's long axis from a camera image. Only called once a
    camera has actually seen the brick."""
    R = data.xmat[brick_id].reshape(3, 3)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    looks = yaw + rng.normal(0.0, _YAW_SIGMA, samples)
    # Average on the doubled angle, where yaw and yaw + 180 deg coincide.
    mean = np.arctan2(np.sin(2 * looks).mean(), np.cos(2 * looks).mean()) / 2
    return float((mean + np.pi / 2) % np.pi - np.pi / 2)


class ZebraSkillBridge(Node):
    """Subscribes to Victor's skill commands for the zebra parts, queues them for
    the main (viewer) thread to execute, and publishes results back."""

    def __init__(self) -> None:
        super().__init__("zebra_skill_bridge")
        self.pending: queue.Queue[dict] = queue.Queue()
        self._status_pub = self.create_publisher(String, STATUS_TOPIC, 10)
        self.create_subscription(String, COMMAND_TOPIC, self._on_command, 10)

    def _on_command(self, msg: String) -> None:
        try:
            command = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"ignoring malformed command: {msg.data!r}")
            return

        part_id = command.get("part_id")
        self.get_logger().info(f"<- {command.get('skill')} {part_id} {command.get('target')}")

        if part_id not in ALL_PART_IDS:
            # Fail unknown parts right away instead of staying silent: silence
            # makes each attempt wait out zebra_bt's 30s skill timeout.
            self.report(command.get("command_id", ""), "FAILED", f"unknown part '{part_id}'")
            return

        self.pending.put(command)

    def report(self, command_id: str, status: str, message: str = "", reason: str | None = None) -> None:
        payload = {"command_id": command_id, "status": status}
        if message:
            payload["message"] = message
        if reason:
            payload["reason"] = reason
        msg = String()
        msg.data = json.dumps(payload)
        self._status_pub.publish(msg)
        self.get_logger().info(f"-> {status} {command_id}")


# fault_bump: where a missed grasp knocks the brick (world frame, towards the
# stack but clear of it and within the grasp workspace), and how long
# perception then loses it - longer than one zebra_bt tick (0.5s) so the
# retried pick sees LOST.
_BUMP = np.array([0.05, 0.0, 0.0])
_BUMP_LOST_S = 2.0
# knock_placed: how far a just-placed brick gets knocked (world frame) - off
# the stack, far enough from it (4.8 cm gap) for a finger to fit between them
# when the tree re-picks it.
_KNOCK = np.array([0.0, -0.08, 0.0])
# A part whose place is refused this many times because the stack below it
# is broken gets escalated (reported ESCALATED) instead of retried again:
# zebra_bt doesn't limit place retries, and it keeps treating the fallen
# part below as PLACED (its WorldModel ignores perception updates for placed
# parts), so it would otherwise re-pick and re-refuse forever.
_MAX_STACK_REFUSALS = 2


def _bump_brick(model, data, brick_id: int, delta: np.ndarray) -> None:
    """Teleport a free-jointed brick by `delta` and stop it - a stand-in for
    the gripper knocking it on a missed grasp."""
    joint = model.body_jntadr[brick_id]
    qpos, qvel = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    data.qpos[qpos : qpos + 3] += delta
    data.qvel[qvel : qvel + 6] = 0.0
    mujoco.mj_forward(model, data)


class _PerceivingSync:
    """`_ThrottledSync` that also gives perception a chance to publish on
    every physics step - grasp_part/place_part call `render.step()` each
    step, so this keeps perception going through a whole blocking move."""

    def __init__(self, render: _ThrottledSync, perception: ZebraPerceptionPublisher) -> None:
        self._render = render
        self._perception = perception

    def step(self) -> None:
        self._render.step()
        self._perception.maybe_publish()


def run_bridge(
    prefer_gl: str = "egl",
    scene_path: str | None = None,
    arm: str = "nearest",
    fault_part: str | None = None,
    fault_offset: float = 0.08,
    fault_times: int = 0,
    fault_bump: bool = False,
    knock_placed: str | None = None,
    knock_later: str | None = None,
    scatter_seed: int | None = None,
    drop: bool = False,
) -> None:
    """`arm` is "nearest" (each brick picked by the arm nearest to it, see
    the module docstring) or "left"/"right" (that arm does everything).

    `scatter_seed`, if given, starts the bricks at random spots and angles
    (upright) in the arms' measured reach zone instead of their fixed square
    spots - see scatter.py. Same seed, same scatter. With `drop`, they're
    dropped instead (random tumbles, landing any way up - scatter.drop_bricks);
    a pick of a brick that isn't upright fails before the arm moves, saying
    how it lies (only top-down grips exist), and his tree sends "flip" first
    (both arms needed: arm="nearest").

    `knock_later` (a part id) is a test hook: once that part has been placed
    and its landing checked, it's knocked `_KNOCK` off the stack - so the
    stack check before the next place finds it gone.

    `knock_placed` (a part id) is a test hook: the first time that part is
    placed, it's knocked `_KNOCK` off the stack right after the gripper lets
    go - so the place check sees it missing and reports FAILED, and zebra_bt
    re-locates, re-picks and re-places it.

    `fault_part` (a part id) is a test hook: picks of that part are sent
    `fault_offset` metres off to the side (world +y), so the gripper closes on
    air and the pick is reported FAILED - exercises zebra_bt's retry/escalate.
    `fault_times` > 0 limits it to that part's first N picks (0 = every pick).

    `fault_bump` also makes each missed pick knock the brick `_BUMP` away and
    perception lose track of it for `_BUMP_LOST_S` - so zebra_bt sees the part
    LOST (not just PICK_FAILED) and re-locates it instead of retrying the old
    position."""
    from .gl import configure_gl

    configure_gl(prefer_gl)

    path = scene_path or str(_DEFAULT_SCENE)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    arms = ("left", "right") if arm == "nearest" else (arm,)
    scattered = None
    if scatter_seed is not None:
        start = drop_bricks if drop else scatter_bricks
        scattered = start(
            model, data, [PART_BODIES[pid] for pid in ALL_PART_IDS], scatter_seed,
            Zone.load("either" if arm == "nearest" else arm),
        )

    # One pick/place context per arm and part (own brick).
    contexts = {(a, pid): ZebraArmContext(model, data, a, PART_BODIES[pid])
                for a in arms for pid in ALL_PART_IDS}
    brick_ids = {pid: contexts[(arms[0], pid)].brick_id for pid in ALL_PART_IDS}
    zones = {a: Zone.load(a) for a in arms}
    bases = arm_bases(model, data)
    held_by: dict[str, str] = {}  # part id -> arm holding it (picked, not yet placed)
    facing_yaw: dict[str, float] = {}  # part id -> yaw to place it at (picked with desired_facing)
    # Once a camera has seen which side a brick's print is on (the show to the headcam, or
    # B's hand camera at a flip handover) - for perception's FACING field (_known_facing):
    print_seen_by: dict[str, str] = {}  # part id -> arm holding it, its ctx.held_yaw the full one
    print_yaw: dict[str, float] = {}  # placed: its full yaw on the stack
    clock = _RealtimeClock(dt=sim_step.CONTROL_DT)

    fault_picks = 0  # picks of fault_part seen so far

    # Flips need both arms; their plans are made in this separate process (started once).
    planner = PlanningProcess(path) if len(arms) == 2 else None

    rclpy.init()
    node = ZebraSkillBridge()
    # 0.5s, not the standalone 1s: during a move, IK solves between physics
    # steps can delay a publish, and 1s left gaps up to ~1.98s - right at
    # zebra_bt's 2s staleness limit.
    perception = ZebraPerceptionPublisher(
        part_ids=ALL_PART_IDS, interval=0.5, model=model, data=data, use_timer=False
    )
    if scattered is not None:
        node.get_logger().info(f"{'drop' if drop else 'scatter'} seed {scatter_seed}: {describe(scattered)}")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(perception)

    relook_camera: dict[str, str] = {}  # part id -> camera its last look again used
    placed_at: dict[str, np.ndarray] = {}  # part id -> center it was placed (and checked) at
    stack_refusals: dict[str, int] = {pid: 0 for pid in ALL_PART_IDS}
    escalated: set[str] = set()  # parts this bridge escalated (see _MAX_STACK_REFUSALS)

    def _stack_problems(skip: str | None = None) -> list[str]:
        """Look at every brick placed so far (but `skip`) and describe any
        that's no longer where it was put, e.g. "legs: 8.2 cm to the side".
        A brick no camera sees can't be checked and isn't reported."""
        problems = []
        for pid, center in placed_at.items():
            if pid == skip:
                continue
            look = _relook(pid)
            if look is not None:
                _, problem = placement_error(look, center)
                if problem:
                    problems.append(f"{PART_LABELS[pid]}: {problem}")
        return problems

    def _relook(part_id: str) -> tuple[np.ndarray, float] | None:
        """grasp_part's look again from hover: `part_id`'s `(center, yaw)` -
        center in world frame, averaged over _RELOOK_SAMPLES looks from one
        camera, and its yaw (`_sim_brick_yaw`) - or None if no camera sees it
        in at least half its looks.

        Cameras are tried headcam first (its frame *is* the shared frame),
        then the others, each converted to the shared frame the same way
        perception does (`calibration.to_reference`). With the arm at hover
        over the middle brick (zebra_body) it blocks headcam and refcam, and
        the hovering arm's own hand camera, looking straight down, is the one
        that sees it (measured error 0.8-2.3 mm)."""
        cams = perception.cams[part_id]
        for camera in (REFERENCE_CAMERA, *(c for c in CAMERAS if c != REFERENCE_CAMERA)):
            looks = [p for p in (cams.observe(camera) for _ in range(_RELOOK_SAMPLES)) if p is not None]
            if len(looks) >= _RELOOK_SAMPLES / 2:
                shared = np.mean([perception.calibration.to_reference(camera, p, cams) for p in looks], axis=0)
                origin = perception.shared_to_world.apply(shared)
                relook_camera[part_id] = camera
                yaw = _sim_brick_yaw(data, brick_ids[part_id], perception.rng, _RELOOK_SAMPLES)
                return origin + np.array([0, 0, _BRICK_CENTER_OFFSET_Z]), yaw
        return None

    def _print_look(part_id: str, camera: str = SHOW_CAMERA):
        """zebra_facing's `look_print`: which way the printed face points, as `camera` sees
        the brick (the headcam: held up in front of it; a hand camera: at a flip's
        handover) - or None if it can't tell. Stand-in for a detector: the true direction
        plus ~3 deg noise, but only when that camera really has a usable view of a long
        face (zebra_facing.long_face_view: geometry, fingers and arms blocking included) -
        seeing either face tells the side."""
        def look(d):
            share, px = long_face_view(model, d, camera, brick_ids[part_id])
            if share < _PRINT_LOOK_MIN_SHARE or px < _PRINT_LOOK_MIN_PX:
                return None
            n = -d.xmat[brick_ids[part_id]].reshape(3, 3)[:, 1] + perception.rng.normal(0.0, 0.05, 3)
            return n / np.linalg.norm(n)  # the print is on the brick's -y face
        return look

    def _stack_target(part_id: str) -> np.ndarray:
        """Where this part will be placed (centre): Victor's stack spot and the part's level
        (legs, body, head) - known at the pick so the grip can be chosen for the facing; the
        place command's own target is what's used for the place."""
        k = list(ALL_PART_IDS).index(part_id)
        return np.array([STACK_XY[0], STACK_XY[1], TABLE_Z + _BRICK_CENTER_OFFSET_Z + k * BRICK_HEIGHT])

    def _place_from_hand(part_id: str, ctx, facing: str | None) -> tuple[bool, str, np.ndarray | None]:
        """The hand that ended a flip kept the brick (zebra_flip.keeps_holding): can it go
        straight to the stack, facing as asked? Returns (kept, why, None) - or after setting
        it back down where the flip would have, (False, why, the spot it's on now; the pick
        command's target is from before the flip) and it's picked as usual (with a facing,
        shown to the headcam and picked up again: zebra_facing.pick_facing). The print
        side comes from the flip's handover look (B's hand camera, ~8 cm away); not by
        showing the brick from this grip - tilted towards the headcam it shifted in the
        fingers and was placed off the stack."""
        target = _stack_target(part_id)
        if facing is not None:
            n_hand = getattr(ctx, "print_in_hand", None)
            if n_hand is None:
                why = "its print side wasn't seen at the handover"
            else:
                ctx.held_yaw = held_from_print_in_hand(ctx, n_hand)
                if can_place(ctx, target, FACING_YAW[facing]):
                    facing_yaw[part_id] = FACING_YAW[facing]
                    return True, f"its print side was seen at the handover: it can go on the stack facing {facing}", None
                why = f"this grip can't set it down facing {facing} at the stack"
        else:  # square, either way round
            ctx.held_yaw = _wrap(_sim_brick_yaw(data, brick_ids[part_id], perception.rng, _RELOOK_SAMPLES)
                                 - ctx.grip_yaw)
            if any(can_place(ctx, target, yaw) for yaw in (0.0, np.pi)):
                facing_yaw.pop(part_id, None)
                return True, "it can go on the stack", None
            why = "this grip can't reach the stack"
        spot = set_down_kept(ctx, render, clock, _flip_look(part_id),
                             [data.xpos[brick_ids[p]][:2].copy() for p in ALL_PART_IDS if p != part_id])
        held_by.pop(part_id, None)
        return False, why, spot

    def _known_facing(part_id: str) -> str | None:
        """Perception's FACING field: "FORWARD" (print towards world -Y, Victor's
        convention), "BACKWARD", or None (sent as UNKNOWN) - only for a brick whose
        print side a camera has seen. From the table nothing can tell (the print is on a
        side face, 3-10 px in the headcam), so until a pick shows it, it's unknown.
        Held (from the show on, also while shown and put back): the print's direction
        is the hand's rotation (arm joints) applied to where the print is in the hand
        (held_yaw, from the look) - right however the hand is tilted. On the table (put
        back after the show, or placed): its full yaw, kept up to date with the yaw look
        (`_sim_brick_yaw`, only known up to 180 deg - the remembered one decides which
        way round). Not upright any more: unknown. Pointing within _FACING_SIDEWAYS of
        sideways (world +-X) it's unknown too: in the hand the brick can turn up to ~18
        deg, and calling that side FORWARD/BACKWARD was wrong 3 times in 28 (seed 4 head)."""
        b = brick_ids[part_id]

        def side(n):  # the print's direction (world) -> FORWARD / BACKWARD / None
            horizontal = np.hypot(n[0], n[1])
            if horizontal < 1e-6 or abs(n[1]) < np.sin(_FACING_SIDEWAYS) * horizontal:
                return None
            return "FORWARD" if n[1] < 0 else "BACKWARD"

        arm_ = print_seen_by.get(part_id)
        if arm_ is not None:
            ctx = contexts[(arm_, part_id)]
            grip = np.empty(9)
            mujoco.mju_quat2Mat(grip, ctx.grip_quat)
            R_hand = data.xmat[ctx.body_id].reshape(3, 3)
            h = ctx.held_yaw  # the print is the brick's -y face; at held_yaw 0 it's the grip's -y
            if ctx.holding:
                return side(R_hand @ grip.reshape(3, 3).T @ np.array([np.sin(h), -np.cos(h), 0.0]))
            # it has just let go (put back / placed): on the table at the hand's turn about
            # vertical plus how it was held (the put-back / place then gives the exact yaw)
            turn = R_hand @ grip.reshape(3, 3).T
            print_yaw[part_id] = float(np.arctan2(turn[1, 0], turn[0, 0]) + h)
            print_seen_by.pop(part_id)
        if part_id in print_yaw:
            if lying(data.xmat[b].reshape(3, 3)) != UPRIGHT:
                print_yaw.pop(part_id)
                return None
            yaw = print_yaw[part_id] = full_yaw_near(_sim_brick_yaw(data, b, perception.rng, 1), print_yaw[part_id])
        else:
            return None
        return side([np.sin(yaw), -np.cos(yaw), 0.0])  # the print is on the brick's -y face

    perception.facing_of = _known_facing

    def _print_seen(part_id: str, ctx, yaw: float | None) -> None:
        """zebra_facing's on_seen: `part_id`'s print side is known - held by `ctx`
        (ctx.held_yaw the full one), or (ctx None) lying on the table at full yaw `yaw`."""
        if ctx is not None:
            print_seen_by[part_id] = ctx.arm
            print_yaw.pop(part_id, None)
        else:
            print_seen_by.pop(part_id, None)
            print_yaw[part_id] = yaw

    def _flip_look(part_id: str):
        """execute_flip's `look`: a hand camera looking at the held brick (B's at the
        handover, A's once B holds it) - here the true pose plus ~2 mm / ~2 deg noise,
        a stand-in for a real detector, like _sim_brick_yaw."""
        def look(d):
            R = d.xmat[brick_ids[part_id]].reshape(3, 3).copy()
            center = d.xpos[brick_ids[part_id]] + R @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
            th = perception.rng.normal(0.0, _YAW_SIGMA)
            Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
            return Rz @ R, center + perception.rng.normal(0.0, 0.002, 3)
        return look

    def _plan_flip_live(part_id: str, order: list[str], allow_move: bool = True):
        """plan_flip in the planning process, from the state right now, while the sim,
        viewer and perception keep running here (the arms are parked, nothing moves)."""
        others = [brick_ids[p] for p in ALL_PART_IDS if p != part_id]
        pending = planner.start(data, brick_ids[part_id], others, order, allow_move)
        while not pending.done():
            if not viewer.is_running():
                raise RuntimeError("viewer closed while planning the flip")
            executor.spin_once(timeout_sec=0.0)
            sim_step.step(model, data)
            clock.tick()
            render.step()
        return pending.result()  # re-raises the planner's error (or its process dying)

    def _let_go_and_park(part_id: str) -> None:
        """After a failed flip: open both hands (the brick drops where it is and
        perception reports how it landed) and park them - each arm's way home may
        be blocked until the other has gone, so try in both orders."""
        for a in arms:
            c = contexts[(a, part_id)]
            _hold(model, data, render, clock, c.this_arm, _GRIP_OPEN, 150)
            c.holding = False
        left_out = list(arms)
        for _ in range(2):
            for a in list(left_out):
                try:
                    go_home(contexts[(a, part_id)], render, clock)
                    left_out.remove(a)
                except Exception as exc:
                    problem = exc
        if left_out:
            node.get_logger().error(f"couldn't park the {'/'.join(left_out)} arm after the failed flip: {problem}")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            render = _PerceivingSync(_ThrottledSync(viewer, model, step_dt=sim_step.CONTROL_DT), perception)
            node.get_logger().info(
                f"Ready - watching {COMMAND_TOPIC} for legs/body/head "
                + ("(nearest arm per brick)." if arm == "nearest" else f"({arm} arm).")
            )

            while viewer.is_running() and rclpy.ok():
                executor.spin_once(timeout_sec=0.0)

                try:
                    command = node.pending.get_nowait()
                except queue.Empty:
                    sim_step.step(model, data)
                    clock.tick()
                    render.step()
                    continue

                # World-frame body origin -> geometric center (see module docstring).
                world_origin = np.array(
                    [command["target"]["x"], command["target"]["y"], command["target"]["z"]]
                )
                center_xyz = world_origin + np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
                skill = command["skill"]
                command_id = command["command_id"]
                part_id = command["part_id"]

                faulted = False
                if skill == "pick" and part_id == fault_part:
                    fault_picks += 1
                    faulted = fault_times == 0 or fault_picks <= fault_times
                if faulted:
                    center_xyz = center_xyz + np.array([0, fault_offset, 0])
                    node.get_logger().warn(
                        f"FAULT INJECTION: {part_id} pick shifted {fault_offset * 100:.0f} cm in +y"
                    )

                try:
                    # Which arm: a place goes to the arm holding the brick; else
                    # the one the command names, else the nearest one.
                    requested = command.get("arm")
                    if requested is not None and requested not in arms:
                        raise ValueError(f"asked for the {requested} arm, but only {'/'.join(arms)} is in use")
                    use = held_by.get(part_id)  # a brick in a hand stays with that hand (place; kept after a flip)
                    if requested is not None and use is not None and requested != use:
                        raise ValueError(f"asked for the {requested} arm, but the {use} arm is holding it")
                    use = use or requested or (
                        arms[0] if len(arms) == 1 else choose_arm(center_xyz[:2], zones, bases)
                    )
                    ctx = contexts[(use, part_id)]
                    for other in arms:
                        if other != use and go_home(contexts[(other, part_id)], render, clock):
                            node.get_logger().info(f"parked the {other} arm at home, out of the {use} arm's way")

                    kept = None
                    if skill == "pick" and part_id in held_by and ctx.holding:  # kept after a flip
                        facing = command.get("desired_facing")
                        if facing is not None and facing not in FACING_YAW:
                            raise ValueError(f"unknown desired_facing '{facing}' (use {'/'.join(FACING_YAW)})")
                        kept, why, spot = _place_from_hand(part_id, ctx, facing)
                        node.get_logger().info(
                            f"{use} arm already holds {PART_LABELS[part_id]} from the flip: {why}"
                            + ("" if kept else " - set it down to pick it up again"))
                        if kept:
                            perception.status_override[part_id] = "PICKED"
                            if facing is not None:  # held_yaw from B's hand camera at the handover
                                print_seen_by[part_id] = use
                        else:
                            center_xyz = spot  # where it is now (the command's target is from before the flip)
                    if skill == "pick" and kept:
                        pass  # already in the hand from the flip: nothing to move
                    elif skill == "pick":
                        lies = lying(data.xmat[brick_ids[part_id]].reshape(3, 3))
                        if lies != UPRIGHT:
                            raise RuntimeError(
                                f"{PART_LABELS[part_id]} is {lies} - can't grip it from the top "
                                f"(arm not moved; only upright bricks can be picked so far)"
                            )
                        facing = command.get("desired_facing")
                        if facing is not None:
                            if facing not in FACING_YAW:
                                raise ValueError(f"unknown desired_facing '{facing}' (use {'/'.join(FACING_YAW)})")
                            try:
                                ctx, how = pick_facing(
                                    {a: contexts[(a, part_id)] for a in arms}, use, render, clock, center_xyz,
                                    _stack_target(part_id), FACING_YAW[facing], lambda: _relook(part_id),
                                    _print_look(part_id),
                                    # still upright in the hand? (stand-in, like perception's LYING)
                                    still_upright=lambda d: lying(d.xmat[brick_ids[part_id]].reshape(3, 3)) == UPRIGHT,
                                    idle=perception.maybe_publish, look_held=_flip_look(part_id),
                                    other_xy=[data.xpos[brick_ids[p]][:2].copy() for p in ALL_PART_IDS if p != part_id],
                                    on_seen=lambda c, y: _print_seen(part_id, c, y))
                            except FacingError as exc:
                                print_seen_by.pop(part_id, None)  # not held any more
                                raise FacingError(f"can't place {PART_LABELS[part_id]} facing {facing}: {exc}",
                                                  exc.reason) from exc
                            use = ctx.arm
                            facing_yaw[part_id] = FACING_YAW[facing]
                            print_seen_by[part_id] = ctx.arm
                            print_yaw.pop(part_id, None)
                            node.get_logger().info(f"{use} arm: {PART_LABELS[part_id]} shown to the {SHOW_CAMERA}, "
                                                   f"picked up again ({how}) to face {facing} at the stack")
                        else:
                            facing_yaw.pop(part_id, None)
                            grasp_part(ctx, render, clock, center_xyz, relook=lambda: _relook(part_id))
                        relooked = (
                            "brick not visible from hover, kept the original aim"
                            if ctx.relook_shift is None
                            else f"looked again from hover ({relook_camera.get(part_id)}), aim moved {ctx.relook_shift * 100:.1f} cm"
                        )
                        held_by[part_id] = use
                        node.get_logger().info(
                            f"{use} arm grasped {part_id} ({relooked}; grip turned {np.degrees(ctx.grip_yaw):+.0f} deg; "
                            f"fingers stopped at {ctx.grasp_width * 100:.2f} cm; "
                            f"sim check: {ctx.grip_miss() * 100:.1f} cm off center)"
                        )
                        perception.status_override[part_id] = "PICKED"
                    elif skill == "place":
                        # Check A: never stack onto a brick that's no longer there. Looked
                        # at from over the pick spot, before carrying the brick over.
                        broken = _stack_problems(skip=part_id)
                        if broken:
                            put_back(ctx, render, clock)
                            stack_refusals[part_id] += 1
                            if stack_refusals[part_id] >= _MAX_STACK_REFUSALS:
                                escalated.add(part_id)
                            raise RuntimeError(
                                f"won't place {PART_LABELS[part_id]}: the stack below it is broken "
                                f"({'; '.join(broken)}) - put it back where it was picked"
                                + (" - escalating, needs a human" if part_id in escalated else "")
                            )

                        # Victor's stack position for this part; then look at where it landed
                        def _verify() -> tuple[np.ndarray, float] | None:
                            nonlocal knock_placed
                            if part_id == knock_placed:
                                knock_placed = None  # first place only
                                _bump_brick(model, data, ctx.brick_id, _KNOCK)
                                for _ in range(250):  # let it land
                                    sim_step.step(model, data)
                                    render.step()
                                node.get_logger().warn(
                                    f"FAULT INJECTION: knocked placed {part_id} "
                                    f"{np.linalg.norm(_KNOCK) * 100:.0f} cm off the stack"
                                )
                            return _relook(part_id)

                        place_part(ctx, render, clock, center_xyz, verify=_verify, facing_yaw=facing_yaw.get(part_id))
                        held_by.pop(part_id, None)
                        if part_id in print_seen_by and part_id in facing_yaw:
                            print_yaw[part_id] = facing_yaw[part_id]  # set down at exactly that yaw
                        print_seen_by.pop(part_id, None)
                        node.get_logger().info(
                            f"{use} arm placed {part_id} (" + (
                                "not visible from hover, landing unchecked" if ctx.place_error is None
                                else f"checked with {relook_camera.get(part_id)}: {ctx.place_error[0] * 100:.1f} cm "
                                     f"to the side, {ctx.place_error[1] * 100:+.1f} cm up/down, "
                                     f"{np.degrees(ctx.place_error[2]):.0f} deg from square"
                            ) + ")"
                        )
                        if ctx.placed_flipped:
                            node.get_logger().warn(
                                f"placed {part_id} turned 180 deg (square grip out of reach at the stack): "
                                f"same footprint, printed face reversed"
                            )
                        perception.status_override[part_id] = "PLACED"
                        placed_at[part_id] = center_xyz

                        if part_id == knock_later:  # test hook - see run_bridge
                            knock_later = None
                            _bump_brick(model, data, ctx.brick_id, _KNOCK)
                            node.get_logger().warn(
                                f"FAULT INJECTION: knocked {part_id} "
                                f"{np.linalg.norm(_KNOCK) * 100:.0f} cm off the stack after it was checked"
                            )

                        # Check B: once every part is placed, look at the whole zebra.
                        if set(placed_at) == set(ALL_PART_IDS):
                            final = _stack_problems()
                            if final:
                                node.get_logger().error(f"zebra check: NOT intact - {'; '.join(final)}")
                            else:
                                node.get_logger().info(
                                    f"zebra check: all {len(placed_at)} bricks in place"
                                )
                    elif skill == "flip":
                        label = PART_LABELS[part_id]
                        lies = lying(data.xmat[brick_ids[part_id]].reshape(3, 3))
                        if lies == UPRIGHT:
                            node.get_logger().info(f"{label} is already upright - nothing to flip")
                        else:
                            if len(arms) < 2:
                                raise RuntimeError(f"{label} is {lies}: flipping needs both arms (--arm nearest)")
                            go_home(ctx, render, clock)  # the other arm is parked above
                            order = [use, next(a for a in arms if a != use)]
                            node.get_logger().info(f"{label} is {lies}: planning a flip ...")
                            t0 = data.time  # sim time runs in real time on the viewer, planning included
                            plan, lies = _plan_flip_live(part_id, order)
                            if plan is None:
                                raise RuntimeError(f"{label} is {lies} - no way found to flip it (arms not moved)")
                            node.get_logger().info(f"flip plan ({data.time - t0:.1f} s): {plan.describe()}")
                            # where the other bricks are (sim positions, like the planner's)
                            others_xy = [data.xpos[brick_ids[p]][:2].copy() for p in ALL_PART_IDS if p != part_id]
                            if isinstance(plan, MovePlan):  # no flip where it lies: move it, then plan again
                                try:
                                    execute_flip(plan, {a: contexts[(a, part_id)] for a in arms}, render, clock,
                                                 _flip_look(part_id), others_xy)
                                except Exception:
                                    _let_go_and_park(part_id)
                                    raise
                                for _ in range(250):  # let it settle
                                    sim_step.step(model, data)
                                    clock.tick()
                                    render.step()
                                plan, lies = _plan_flip_live(part_id, order, allow_move=False)
                                if plan is None:
                                    raise RuntimeError(f"{label} is {lies} - moved it, but found no way to flip it there "
                                                       f"(arms not moved)")
                                node.get_logger().info(f"moved {label}; flip plan from there: {plan.describe()}")
                            try:
                                holder = execute_flip(plan, {a: contexts[(a, part_id)] for a in arms}, render,
                                                      clock, _flip_look(part_id), others_xy, keep=True,
                                                      look_print=lambda d, cam: _print_look(part_id, cam)(d))
                            except Exception:
                                _let_go_and_park(part_id)
                                raise
                            if holder is not None:  # kept, upright, for the place (zebra_flip.keeps_holding)
                                now = lying(data.xmat[brick_ids[part_id]].reshape(3, 3))
                                if now != UPRIGHT:
                                    set_down_kept(holder, render, clock, _flip_look(part_id), others_xy)
                                    raise RuntimeError(f"flip of {label} ended {now} in the {holder.arm} hand - set it down")
                                held_by[part_id] = holder.arm
                                perception.status_override[part_id] = "PICKED"
                                node.get_logger().info(f"flipped {label} upright, kept in the {holder.arm} hand for the "
                                                       f"place (no set-down) - {data.time - t0:.0f} s in all")
                            else:
                                for _ in range(250):  # let it settle
                                    sim_step.step(model, data)
                                    clock.tick()
                                    render.step()
                                now = lying(data.xmat[brick_ids[part_id]].reshape(3, 3))
                                if now != UPRIGHT:
                                    raise RuntimeError(f"flip of {label} ended {now}, not upright")
                                node.get_logger().info(
                                    f"flipped {label} upright, set down at ({plan.set_down[0]:.2f}, "
                                    f"{plan.set_down[1]:+.2f}) - {data.time - t0:.0f} s in all"
                                )
                    else:
                        raise ValueError(f"unknown skill '{skill}'")
                    # Publish the new status BEFORE replying, so no stale
                    # LOCATED can land after his tree has set PICKED/PLACED.
                    perception.maybe_publish(force=True)
                    node.report(command_id, "SUCCEEDED")
                except Exception as exc:  # report failure to the BT rather than crashing the bridge
                    node.get_logger().error(f"{skill} {command_id} failed: {exc}")
                    # Not held (failed pick) or let go somewhere wrong (failed place):
                    # either way report where perception actually sees it again, so
                    # the tree can re-locate and re-pick it.
                    perception.status_override[part_id] = "ESCALATED" if part_id in escalated else None
                    if part_id in held_by and not contexts[(held_by[part_id], part_id)].holding:
                        held_by.pop(part_id)  # missed, put back, or let go somewhere wrong
                        print_seen_by.pop(part_id, None)
                    if not isinstance(exc, FacingError):  # it may have tumbled: its facing is unknown again
                        print_seen_by.pop(part_id, None)
                        print_yaw.pop(part_id, None)
                    node.report(command_id, "FAILED", str(exc), getattr(exc, "reason", None))
                    if faulted and fault_bump:
                        _bump_brick(model, data, brick_ids[part_id], _BUMP)
                        perception.lose_track(part_id, _BUMP_LOST_S)
                        node.get_logger().warn(
                            f"FAULT INJECTION: missed grasp knocked {part_id} "
                            f"{np.linalg.norm(_BUMP) * 100:.0f} cm away; perception lost it "
                            f"for {_BUMP_LOST_S:.0f}s"
                        )
    finally:
        if planner is not None:
            planner.close()
        perception.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
