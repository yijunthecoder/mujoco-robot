"""Bridge: publish this project's calibrated zebra-brick position to Victor's
zebra_bt Behavior Tree over ROS2.

Victor's `WorldModel` (build_a_zebra/src/world_model.cpp, checked out locally
at ~/ros2_ws/src/build_a_zebra) subscribes to a topic expecting one
comma-separated line per update:

    part,STATUS,x,y,z          (x,y,z only present when STATUS=LOCATED)

STATUS is one of LOCATED, LOST, PICKED, PICK_FAILED, PLACED, ESCALATED - this
publisher sends LOCATED (it has a position) or LOST (no camera can currently
see it), unless whoever moves the brick sets `status_override` (the skill
bridge sends PICKED while holding it and PLACED once placed, matching what
his tree just set - see `ZebraPerceptionPublisher.status_override`).

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

import time
from datetime import datetime

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

# Display only - what goes over the topic must stay the LDraw id, since
# Victor's WorldModel looks parts up by it (his own prettyPart() does the same).
PART_LABELS = {"31111p0e": "legs", "31111p0f": "body", "31111p0g": "head"}


class ZebraPerceptionPublisher(Node):
    """Calibrates once on startup, then republishes the block's position.

    Two ways to run it:

    - Standalone (`model`/`data` omitted): loads its own copy of the scene and
      publishes on a ROS2 timer. Nothing moves in that copy, so this only
      ever reports the brick's spawn position - fine for testing perception
      on its own, but it can't see an arm move the brick.
    - Embedded (`model`/`data` given, `use_timer=False`): watches a live
      simulation someone else is stepping - zebra_skill_bridge.py passes its
      own, so perception sees the same brick the arm moves. The owner calls
      `maybe_publish()` from its own loop instead of a timer, since a
      blocking grasp/place never yields to rclpy.spin (and going quiet for
      that long trips Victor's 2s perception-staleness timeout).

    Calibration always runs on a scratch MjData: `_collect_calibration_pairs`
    teleports the brick through random positions (`place_block`), which must
    never happen to a live simulation.
    """

    def __init__(
        self,
        part_id: str = DEFAULT_PART_ID,
        interval: float = 1.0,
        n_calib: int = 15,
        pixel_sigma: float = 0.5,
        depth_sigma: float = 0.002,
        seed: int = 0,
        scene_path: str | None = None,
        model: mujoco.MjModel | None = None,
        data: mujoco.MjData | None = None,
        use_timer: bool = True,
    ) -> None:
        super().__init__("zebra_perception_publisher")
        self.part_id = part_id
        self.label = PART_LABELS.get(part_id, part_id)
        self.interval = interval
        self.publisher = self.create_publisher(String, PERCEPTION_TOPIC, 10)
        self.rng = np.random.default_rng(seed)

        # What to report instead of LOCATED, set by whoever is moving the
        # brick (e.g. "PICKED" while held, "PLACED" once assembled). Victor's
        # perceptionCallback overwrites the part's status on every message,
        # so a plain LOCATED after a successful place would undo his PLACED
        # and make the tree pick the part up again.
        self.status_override: str | None = None
        self._last_publish = float("-inf")

        if model is None:
            model = mujoco.MjModel.from_xml_path(str(scene_path or _DEFAULT_SCENE))
            data = mujoco.MjData(model)
            _setup_home_pose(model, data)

        # --- calibrate once, on a scratch copy (see class docstring) ---
        self._log(f"calibrating  ({n_calib} samples per camera)")
        scratch = mujoco.MjData(model)
        _setup_home_pose(model, scratch)
        calib_cams = SimulatedCameras(model, scratch, pixel_sigma=pixel_sigma, depth_sigma=depth_sigma, rng=self.rng)
        self.transforms = calibrate(_collect_calibration_pairs(calib_cams, n_calib, self.rng))

        self.cams = SimulatedCameras(model, data, pixel_sigma=pixel_sigma, depth_sigma=depth_sigma, rng=self.rng)
        self._log(f"ready        {self.label} ({part_id}) -> {PERCEPTION_TOPIC} every {interval}s")

        if use_timer:
            self.create_timer(interval, self._tick)

    def maybe_publish(self, force: bool = False) -> None:
        """Publish if `interval` seconds (wall clock) have passed - cheap
        enough to call every physics step."""
        now = time.monotonic()
        if force or now - self._last_publish >= self.interval:
            self._tick()

    def _tick(self) -> None:
        self._last_publish = time.monotonic()
        seen = _observe_live_position(self.cams)
        available = [name for name in CAMERAS if name in seen]
        status = self.status_override or "LOCATED"

        if not available:
            # A held/placed part the cameras can't see (e.g. the gripper is in
            # the way) is still held/placed - only report LOST otherwise.
            status = self.status_override or "LOST"
            self._publish(f"{self.part_id},{status}", f"{self.label}  {status}")
            return

        # headcam is the shared frame itself, so prefer it when it can see the
        # block; any other camera that can see it converts to the same answer.
        camera = REFERENCE_CAMERA if REFERENCE_CAMERA in available else available[0]
        x, y, z = self.transforms[camera].apply(seen[camera])
        self._publish(
            f"{self.part_id},{status},{x:.4f},{y:.4f},{z:.4f}",
            f"{self.label}  {status:<8} x={x:+.4f}  y={y:+.4f}  z={z:+.4f}",
        )

    def _publish(self, line: str, shown: str) -> None:
        msg = String()
        msg.data = line
        self.publisher.publish(msg)
        self._log(shown)

    @staticmethod
    def _log(text: str) -> None:
        print(f"[{datetime.now():%H:%M:%S}] {text}", flush=True)


def _other_perception_publisher(node: Node, wait: float = 3.0) -> bool:
    """True if some other node already publishes PERCEPTION_TOPIC. Polls for
    `wait` seconds so DDS discovery has time to see it (the count includes
    this node's own publisher, hence > 1)."""
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if node.count_publishers(PERCEPTION_TOPIC) > 1:
            return True
        time.sleep(0.2)
    return False


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
        if _other_perception_publisher(node):
            print(
                f"Another node is already publishing {PERCEPTION_TOPIC} - most likely "
                "zebra_skill_bridge.py, which publishes perception from the simulation "
                "the arm actually moves in. Not starting: this publisher only sees its "
                "own static copy of the scene, so it would keep reporting the brick at "
                "its spawn spot, overwrite PLACED, and make zebra_bt pick it again.",
                flush=True,
            )
            raise SystemExit(1)
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
