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
result back. Picks go to the command's target; places ignore it for now and
build the stack on the middle circle - see `_stack_center` in `run_bridge`.

Coordinate note (two conversions needed before `target` is usable as an IK
goal):

1. It's in headcam's own local camera frame, not MuJoCo's world frame -
   camera_calibration.py's calibration step explicitly targets "the shared
   frame (headcam)" as its output convention, and `zebra_publisher.py`
   forwards that unconverted (correctly - that's what Victor's WorldModel
   expects and just relays back verbatim in the command). Recovered via
   headcam's own known pose: `world = headcam_R @ shared + headcam_t`
   (the inverse of `SimulatedCameras.observe`'s `p_cam = R.T @ (world - t)`).
   Verified numerically against the true simulated position (~0.3cm error,
   within calibration noise).
2. It's the brick's body *origin* (see camera_calibration.py's
   `SimulatedCameras.observe`, which reads the freejoint qpos directly), not
   the geometric center that `grasp_part`/`place_part` expect (see
   `_BRICK_CENTER_OFFSET_Z`) - applied in world frame, after (1).

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

from .pick_place import _RealtimeClock, _ThrottledSync
from .zebra_publisher import ALL_PART_IDS, PART_BODIES, ZebraPerceptionPublisher
from .zebra_pick_place import (
    _BRICK_CENTER_OFFSET_Z,
    _DEFAULT_SCENE,
    _TABLE_PLACE_XYZ,
    BRICK_HEIGHT,
    ZebraArmContext,
    grasp_part,
    place_part,
)

COMMAND_TOPIC = "/zebra/skill_commands"
STATUS_TOPIC = "/zebra/skill_status"
# Stack level of each part (0 = on the table): the zebra is built bottom-up,
# legs -> body -> head, same order as bom.json's role_order.
STACK_LEVEL = {"31111p0e": 0, "31111p0f": 1, "31111p0g": 2}


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

        if part_id not in STACK_LEVEL:
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
) -> None:
    """`fault_part` (a part id) is a test hook: picks of that part are sent
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

    # One pick/place context per part (same arm, own brick + carry).
    contexts = {pid: ZebraArmContext(model, data, arm, PART_BODIES[pid]) for pid in ALL_PART_IDS}
    clock = _RealtimeClock(dt=model.opt.timestep)

    # Place targets are ours for now, not the command's: zebra_bt's
    # placeTargetFor() sends world-frame coordinates while everything else
    # (perception, pick targets, the conversion below) uses headcam frame -
    # until that's agreed, each part goes onto a stack on the middle circle
    # (target_site): legs on the table, body one brick height up, head two.
    # Table resting height = the bricks' spawn height (all three spawn on it).
    table_center_z = data.xpos[contexts["31111p0e"].brick_id][2] + _BRICK_CENTER_OFFSET_Z

    def _stack_center(part_id: str) -> np.ndarray:
        return np.array([
            _TABLE_PLACE_XYZ[0],
            _TABLE_PLACE_XYZ[1],
            table_center_z + STACK_LEVEL[part_id] * BRICK_HEIGHT,
        ])

    # headcam's own pose, to invert perception's "shared frame (headcam)"
    # coordinates back into MuJoCo world coordinates - see module docstring.
    headcam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "headcam")
    headcam_R = data.cam_xmat[headcam_id].reshape(3, 3).copy()
    headcam_t = data.cam_xpos[headcam_id].copy()

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

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            render = _PerceivingSync(_ThrottledSync(viewer, model), perception)
            node.get_logger().info(
                f"Ready - watching {COMMAND_TOPIC} for legs/body/head ({arm} arm)."
            )

            while viewer.is_running() and rclpy.ok():
                executor.spin_once(timeout_sec=0.0)

                try:
                    command = node.pending.get_nowait()
                except queue.Empty:
                    mujoco.mj_step(model, data)
                    clock.tick()
                    render.step()
                    continue

                shared_frame_origin = np.array(
                    [command["target"]["x"], command["target"]["y"], command["target"]["z"]]
                )
                # headcam-local shared frame -> world, then body origin -> geometric
                # center (see module docstring for both conversions)
                world_origin = headcam_R @ shared_frame_origin + headcam_t
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
                        grasp_part(ctx, render, clock, center_xyz)
                        node.get_logger().info(f"grasped {part_id} ({ctx.grip_miss() * 100:.1f} cm off center)")
                        perception.status_override[part_id] = "PICKED"
                    elif skill == "place":
                        place_part(ctx, render, clock, _stack_center(part_id))  # ignores command target - see above
                        perception.status_override[part_id] = "PLACED"
                    else:
                        raise ValueError(f"unknown skill '{skill}'")
                    # Publish the new status BEFORE replying, so no stale
                    # LOCATED can land after his tree has set PICKED/PLACED.
                    perception.maybe_publish(force=True)
                    node.report(command_id, "SUCCEEDED")
                except Exception as exc:  # report failure to the BT rather than crashing the bridge
                    node.get_logger().error(f"{skill} {command_id} failed: {exc}")
                    if skill == "pick":
                        perception.status_override[part_id] = None
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
