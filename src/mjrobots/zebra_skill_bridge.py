"""ROS2 bridge that actually executes Victor's zebra_bt pick/place commands
in this MuJoCo sim - the missing other half of zebra_publisher.py.

Victor's `SkillBridge` (build_a_zebra/src/main.cpp, checked out locally at
~/ros2_ws/src/build_a_zebra) publishes one JSON message per attempt on
`/zebra/skill_commands`:

    {"command_id", "skill": "pick"|"place", "part_id", "target": {"x","y","z"}}

and blocks that part's `PickPart`/`PlacePart` BT node in RUNNING until a
matching reply arrives on `/zebra/skill_status`:

    {"command_id", "status": "SUCCEEDED"|"FAILED"[, "message"]}

This node is that reply: for one part (zebra_legs, id "31111p0e" - see
bom.json; other parts are ignored for now) and one arm, it keeps a MuJoCo
viewer running, executes `grasp_part`/`place_part` (zebra_pick_place.py)
against whatever target the command carries, and reports the result back.

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

Perception note: this process also publishes perception
(`/zebra/perception_updates`), from its OWN live simulation - a
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
from .zebra_publisher import ZebraPerceptionPublisher
from .zebra_pick_place import (
    _BRICK_CENTER_OFFSET_Z,
    _DEFAULT_SCENE,
    _TABLE_PLACE_XYZ,
    ZebraArmContext,
    grasp_part,
    place_part,
)

COMMAND_TOPIC = "/zebra/skill_commands"
STATUS_TOPIC = "/zebra/skill_status"
OUR_PART_ID = "31111p0e"  # zebra_legs - see bom.json


class ZebraSkillBridge(Node):
    """Subscribes to Victor's skill commands for our part, queues them for
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

        if command.get("part_id") != OUR_PART_ID:
            return  # not ours - body/head aren't wired up yet

        self.get_logger().info(f"<- {command.get('skill')} {command.get('part_id')} {command.get('target')}")
        self.pending.put(command)

    def report(self, command_id: str, status: str, message: str = "") -> None:
        payload = {"command_id": command_id, "status": status}
        if message:
            payload["message"] = message
        msg = String()
        msg.data = json.dumps(payload)
        self._status_pub.publish(msg)
        self.get_logger().info(f"-> {status} {command_id}")


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


def run_bridge(prefer_gl: str = "egl", scene_path: str | None = None, arm: str = "right") -> None:
    from .gl import configure_gl

    configure_gl(prefer_gl)

    path = scene_path or str(_DEFAULT_SCENE)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    ctx = ZebraArmContext(model, data, arm)
    clock = _RealtimeClock(dt=model.opt.timestep)

    # Victor's PlacePart currently sends the part's own last-seen position as
    # the place target (main.cpp: `send("place", part_, state.position)`), so
    # honoring it just puts the brick back where it was picked from. Until his
    # tree sends a real assembly position, every place goes to the middle
    # circle (target_site) instead, at the same resting height the brick
    # spawns at on the table - same target as zebra_pick_place.run_demo.
    place_center_xyz = np.array([
        _TABLE_PLACE_XYZ[0],
        _TABLE_PLACE_XYZ[1],
        data.xpos[ctx.brick_id][2] + _BRICK_CENTER_OFFSET_Z,
    ])

    # headcam's own pose, to invert perception's "shared frame (headcam)"
    # coordinates back into MuJoCo world coordinates - see module docstring.
    headcam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "headcam")
    headcam_R = data.cam_xmat[headcam_id].reshape(3, 3).copy()
    headcam_t = data.cam_xpos[headcam_id].copy()

    rclpy.init()
    node = ZebraSkillBridge()
    perception = ZebraPerceptionPublisher(part_id=OUR_PART_ID, model=model, data=data, use_timer=False)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(perception)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            render = _PerceivingSync(_ThrottledSync(viewer, model), perception)
            node.get_logger().info(
                f"Ready - watching {COMMAND_TOPIC} for part '{OUR_PART_ID}' ({arm} arm)."
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

                try:
                    if skill == "pick":
                        grasp_part(ctx, render, clock, center_xyz)
                        perception.status_override = "PICKED"
                    elif skill == "place":
                        place_part(ctx, render, clock, place_center_xyz)  # ignores command target - see above
                        perception.status_override = "PLACED"
                    else:
                        raise ValueError(f"unknown skill '{skill}'")
                    # Publish the new status BEFORE replying, so no stale
                    # LOCATED can land after his tree has set PICKED/PLACED.
                    perception.maybe_publish(force=True)
                    node.report(command_id, "SUCCEEDED")
                except Exception as exc:  # report failure to the BT rather than crashing the bridge
                    node.get_logger().error(f"{skill} {command_id} failed: {exc}")
                    if skill == "pick":
                        perception.status_override = None
                    node.report(command_id, "FAILED", str(exc))
    finally:
        perception.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
