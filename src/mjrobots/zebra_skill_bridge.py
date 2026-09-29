"""ROS2 bridge that actually executes Victor's zebra_bt pick/place commands
in this MuJoCo sim - the missing other half of zebra_publisher.py.

Victor's `SkillBridge` (build_a_zebra/src/main.cpp, checked out locally at
~/ros2_ws/src/build_a_zebra) publishes one JSON message per attempt on
`/zebra/skill_commands`:

    {"command_id", "skill": "pick"|"place", "part_id", "target": {"x","y","z"}}

and blocks that part's `PickPart`/`PlacePart` BT node in RUNNING until a
matching reply arrives on `/zebra/skill_status`:

    {"command_id", "status": "SUCCEEDED"|"FAILED"[, "message"]}

This node is that reply: for all three zebra parts (legs/body/head, ids
31111p0e/f/g - see bom.json) and one arm, it keeps a MuJoCo viewer running,
executes `grasp_part`/`place_part` (zebra_pick_place.py), and reports the
result back. Both picks and places go to the command's target: picks to
where perception last saw the brick, places to Victor's stack positions
(`placeTargetFor` in his main.cpp).

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
from .zebra_publisher import ALL_PART_IDS, PART_BODIES, PART_LABELS, ZebraPerceptionPublisher
from .zebra_pick_place import (
    _BRICK_CENTER_OFFSET_Z,
    _DEFAULT_SCENE,
    ZebraArmContext,
    grasp_part,
    place_part,
    placement_error,
    put_back,
)

COMMAND_TOPIC = "/zebra/skill_commands"
STATUS_TOPIC = "/zebra/skill_status"
# Looks averaged by the look again from hover (see `_relook` in run_bridge).
_RELOOK_SAMPLES = 10
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

    def report(self, command_id: str, status: str, message: str = "") -> None:
        payload = {"command_id": command_id, "status": status}
        if message:
            payload["message"] = message
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
    arm: str = "right",
    fault_part: str | None = None,
    fault_offset: float = 0.08,
    fault_times: int = 0,
    fault_bump: bool = False,
    knock_placed: str | None = None,
    knock_later: str | None = None,
) -> None:
    """`knock_later` (a part id) is a test hook: once that part has been placed
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

    # One pick/place context per part (same arm, own brick).
    contexts = {pid: ZebraArmContext(model, data, arm, PART_BODIES[pid]) for pid in ALL_PART_IDS}
    clock = _RealtimeClock(dt=sim_step.CONTROL_DT)

    fault_picks = 0  # picks of fault_part seen so far

    rclpy.init()
    node = ZebraSkillBridge()
    # 0.5s, not the standalone 1s: during a move, IK solves between physics
    # steps can delay a publish, and 1s left gaps up to ~1.98s - right at
    # zebra_bt's 2s staleness limit.
    perception = ZebraPerceptionPublisher(
        part_ids=ALL_PART_IDS, interval=0.5, model=model, data=data, use_timer=False
    )
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
                yaw = _sim_brick_yaw(data, contexts[part_id].brick_id, perception.rng, _RELOOK_SAMPLES)
                return origin + np.array([0, 0, _BRICK_CENTER_OFFSET_Z]), yaw
        return None

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            render = _PerceivingSync(_ThrottledSync(viewer, model, step_dt=sim_step.CONTROL_DT), perception)
            node.get_logger().info(
                f"Ready - watching {COMMAND_TOPIC} for legs/body/head ({arm} arm)."
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
                ctx = contexts[part_id]

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
                    if skill == "pick":
                        grasp_part(ctx, render, clock, center_xyz, relook=lambda: _relook(part_id))
                        relooked = (
                            "brick not visible from hover, kept the original aim"
                            if ctx.relook_shift is None
                            else f"looked again from hover ({relook_camera.get(part_id)}), aim moved {ctx.relook_shift * 100:.1f} cm"
                        )
                        node.get_logger().info(
                            f"grasped {part_id} ({relooked}; grip turned {np.degrees(ctx.grip_yaw):+.0f} deg; "
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

                        place_part(ctx, render, clock, center_xyz, verify=_verify)
                        node.get_logger().info(
                            f"placed {part_id} (" + (
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
                    node.report(command_id, "FAILED", str(exc))
                    if faulted and fault_bump:
                        _bump_brick(model, data, ctx.brick_id, _BUMP)
                        perception.lose_track(part_id, _BUMP_LOST_S)
                        node.get_logger().warn(
                            f"FAULT INJECTION: missed grasp knocked {part_id} "
                            f"{np.linalg.norm(_BUMP) * 100:.0f} cm away; perception lost it "
                            f"for {_BUMP_LOST_S:.0f}s"
                        )
    finally:
        perception.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
