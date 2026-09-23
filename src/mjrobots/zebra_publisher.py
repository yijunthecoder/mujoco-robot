"""Bridge: publish this project's calibrated zebra-brick position to Victor's
zebra_bt Behavior Tree over ROS2.

Victor's `WorldModel` (build_a_zebra/src/world_model.cpp, checked out locally
at ~/ros2_ws/src/build_a_zebra) subscribes to a topic expecting one
comma-separated line per update:

    part,STATUS,x,y,z          (x,y,z only present when STATUS=LOCATED)

STATUS is one of LOCATED, LOST, PICKED, PICK_FAILED, PLACED, ESCALATED - this
publisher only ever sends LOCATED (it has a position) or LOST (no camera can
currently see it); the other statuses are set by his own Behavior Tree, not
by perception.

This file mirrors that format by hand - it is not generated from his code,
so if his `perceptionCallback` format changes, this needs updating to match.

PART NAMING: WorldModel is keyed by whatever part_ids main.cpp passes in from
TaskModel::buildOrder(), which reads them straight from bom.json - the real
LDraw part ids (31111p0e/f/g), not "head"/"body"/"feet" (checked directly
against the current world_model.cpp/main.cpp/bom.json - an earlier version
of this comment claimed a hardcoded {"head","body","feet"} loop in the
constructor that no longer exists in his code; sending "feet" made every
update get logged as "Unknown part" and silently dropped). This project's
`zebra_legs` body is his zebra-legs part, "31111p0e".

Requires ROS2 (this project was set up against Humble) sourced in the shell
before running: `source /opt/ros/humble/setup.bash`.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import mujoco

from .camera_calibration import (
    CAMERAS,
    REFERENCE_CAMERA,
    SimulatedCameras,
    _DEFAULT_SCENE,
    _collect_calibration_pairs,
    _observe_live_position,
    _setup_home_pose,
    calibrate,
)

PERCEPTION_TOPIC = "/zebra/perception_updates"
DEFAULT_PART_ID = "31111p0e"  # our zebra_legs body == his zebra-legs part id (bom.json)


class ZebraPerceptionPublisher(Node):
    """Calibrates once on startup, then republishes the block's position on a timer."""

    def __init__(
        self,
        part_id: str = DEFAULT_PART_ID,
        interval: float = 1.0,
        n_calib: int = 15,
        pixel_sigma: float = 0.5,
        depth_sigma: float = 0.002,
        seed: int = 0,
        scene_path: str | None = None,
    ) -> None:
        super().__init__("zebra_perception_publisher")
        self.part_id = part_id
        self.publisher = self.create_publisher(String, PERCEPTION_TOPIC, 10)

        model = mujoco.MjModel.from_xml_path(str(scene_path or _DEFAULT_SCENE))
        data = mujoco.MjData(model)
        _setup_home_pose(model, data)
        self.rng = np.random.default_rng(seed)
        self.cams = SimulatedCameras(model, data, pixel_sigma=pixel_sigma, depth_sigma=depth_sigma, rng=self.rng)

        # --- calibrate once, exactly like --loop does before its first tick ---
        self.get_logger().info(f"Calibrating from {n_calib} block positions per camera...")
        pairs = _collect_calibration_pairs(self.cams, n_calib, self.rng)
        self.transforms = calibrate(pairs)
        # _collect_calibration_pairs moves the block through random test
        # positions (via place_block) and leaves it at the last one - restore
        # its real position before we start reporting it as "live".
        _setup_home_pose(model, data)
        self.get_logger().info(
            f"Calibrated. Publishing '{part_id}' on {PERCEPTION_TOPIC} every {interval}s."
        )

        self.create_timer(interval, self._tick)

    def _tick(self) -> None:
        seen = _observe_live_position(self.cams)
        available = [name for name in CAMERAS if name in seen]

        if not available:
            self._publish(f"{self.part_id},LOST")
            return

        # headcam is the shared frame itself, so prefer it when it can see the
        # block; any other camera that can see it converts to the same answer.
        camera = REFERENCE_CAMERA if REFERENCE_CAMERA in available else available[0]
        x, y, z = self.transforms[camera].apply(seen[camera])
        self._publish(f"{self.part_id},LOCATED,{x:.4f},{y:.4f},{z:.4f}")

    def _publish(self, line: str) -> None:
        msg = String()
        msg.data = line
        self.publisher.publish(msg)
        self.get_logger().info(f"-> {line}")


def run_publisher(
    part_id: str = DEFAULT_PART_ID,
    interval: float = 1.0,
    n_calib: int = 15,
    pixel_sigma: float = 0.5,
    depth_sigma: float = 0.002,
    seed: int = 0,
    scene_path: str | None = None,
) -> None:
    rclpy.init()
    node = ZebraPerceptionPublisher(
        part_id=part_id,
        interval=interval,
        n_calib=n_calib,
        pixel_sigma=pixel_sigma,
        depth_sigma=depth_sigma,
        seed=seed,
        scene_path=scene_path,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        # Ctrl+C raises the former; a SIGTERM (e.g. from `timeout`, or a
        # process manager) is caught by ROS2's own handler and raises the
        # latter instead - both mean "stop", neither is a real error. Use a
        # plain print, not the node's logger: by now ROS2's own signal
        # handler may have already started tearing the context down, and
        # logging through it here can print a harmless-but-ugly
        # "publisher's context is invalid" warning.
        print("Stopped.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
