"""The head camera's picture on `/camera/image_raw` - what Victor's VLA planner looks at.

His `vla_planner_node.py` (build_a_zebra/scripts) subscribes to `/camera/image_raw`
(sensor_msgs/Image, read with cv_bridge as "rgb8"), and every 5 s sends the latest picture
plus the bricks' statuses to a vision-language model, which answers which skill to run next
("PICK 31111p0f", ...). Until a picture arrives it never answers, and his `VLADecide` BT
node times out - so in the simulation, this publishes one.

What is published: the fixed head camera ("headcam" - the real robot's head camera, the
same view perception's positions are measured in), IMAGE_WIDTH x IMAGE_HEIGHT, rgb8,
frame_id "headcam", once every `interval` s (wall clock). 640x480: the scene's own
offscreen size; a model reading the picture needs the bricks bigger than the 3-4 px they
are at 84x84. Without shadows: rendering stops the simulation while it runs, and in WSL
(software OpenGL, llvmpipe) a 640x480 picture takes 256 ms with shadows, 93 ms without -
and the arms' long shadows are only clutter in the picture.

Why every 1 s and not every 5 s like his planner: the two clocks don't line up, so the
picture his planner sends is up to one `interval` old. Right after a skill finishes his
tree asks for the next decision; at 5 s the picture could still show the brick before the
pick, and the model would decide on a scene that's gone. At 1 s it's at most ~1 s old.

Like ZebraPerceptionPublisher it runs on the bridge's own live simulation and is driven
from the bridge's loop (`maybe_publish` every physics step), not a ROS2 timer, so pictures
keep coming during a blocking move.

Real robot: not needed - the camera's own ROS driver publishes `/camera/image_raw`. This is
the simulation's stand-in for that driver, nothing else uses it.
"""

from __future__ import annotations

import time

import mujoco
from rclpy.node import Node
from sensor_msgs.msg import Image

IMAGE_TOPIC = "/camera/image_raw"
CAMERA = "headcam"
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480


class HeadCameraImagePublisher(Node):
    """Renders `CAMERA` from the live `model`/`data` and publishes it every `interval` s."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, interval: float = 1.0) -> None:
        super().__init__("head_camera_image_publisher")
        self.interval = interval
        self._data = data
        self._renderer = mujoco.Renderer(model, IMAGE_HEIGHT, IMAGE_WIDTH)
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self._publisher = self.create_publisher(Image, IMAGE_TOPIC, 1)
        self._last_publish = float("-inf")
        self.render_s = 0.0  # how long the last picture took to render (shown in the first log)
        self.get_logger().info(f"{CAMERA} {IMAGE_WIDTH}x{IMAGE_HEIGHT} -> {IMAGE_TOPIC} every {interval}s")

    def maybe_publish(self) -> None:
        """Publish if `interval` seconds (wall clock) have passed - cheap enough to call
        every physics step."""
        now = time.monotonic()
        if now - self._last_publish >= self.interval:
            self._last_publish = now
            self._publish()

    def _publish(self) -> None:
        t0 = time.monotonic()
        self._renderer.update_scene(self._data, CAMERA)
        pixels = self._renderer.render()  # (H, W, 3) uint8, top row first - rgb8's layout
        first = self.render_s == 0.0
        self.render_s = time.monotonic() - t0
        if not self.context.ok():  # stopped (Ctrl+C) during the render: ROS is gone, nothing to send on
            return

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = CAMERA
        msg.height, msg.width = pixels.shape[:2]
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = pixels.tobytes()
        self._publisher.publish(msg)
        if first:
            self.get_logger().info(f"first picture published ({self.render_s * 1000:.0f} ms to render)")

    def destroy_node(self) -> None:
        self._renderer.close()
        super().destroy_node()
