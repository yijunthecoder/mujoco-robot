import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
from google import genai
from apikeys import GOOGLE_API_KEY


class VLAPlanner(Node):
    def __init__(self):
        super().__init__("vla_planner")

        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.bridge = CvBridge()
        self.latest_image = None

        # Camera: keep the newest frame only.
        self.create_subscription(Image, "/camera/image_raw", self.on_image, 10)

        # One request -> one decision. VLADecide publishes JSON here.
        self.create_subscription(String, "/vla/request", self.on_request, 10)

        self.decision_pub = self.create_publisher(String, "/vla/decision", 10)

        self.get_logger().info(
            "VLAPlanner ready — waiting for /vla/request")

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------
    def on_image(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")

    def on_request(self, msg):
        if self.latest_image is None:
            # Don't answer with a guess; VLADecide will retry with a
            # fresh request_id, so we just drop this one.
            self.get_logger().warn(
                "request received before first camera frame — dropping")
            return

        try:
            req = json.loads(msg.data)
            request_id = req["request_id"]
            part_id    = req["part_id"]
            status     = req.get("status", "UNKNOWN")
            orientation = req.get("orientation", "UNKNOWN")
            facing     = req.get("facing", "UNKNOWN")
        except Exception as e:
            self.get_logger().error(f"bad request JSON: {e}")
            return

        prompt = f"""
You are controlling a robot arm building a LEGO zebra.

The robot is currently working on part: {part_id}
Current perception state for this part:
- status: {status}
- orientation: {orientation}
- facing: {facing}

Rules, in order — use the FIRST one that matches:
1. If status is PLACED, answer: DONE
2. If orientation is anything other than UPRIGHT, answer: FLIP {part_id}
3. If status is PICKED, answer: PLACE {part_id}
4. Otherwise, answer: PICK {part_id}

Answer with exactly one line, nothing else.
"""

        pil_image = PILImage.fromarray(self.latest_image)

        decision = None
        last_err = None
        for attempt in range(1, 4):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[pil_image, prompt],
                    config={
                        "max_output_tokens": 300,
                        "temperature": 0,
                        "thinking_config": {"thinking_budget": 0},
                    },
                )
                text = (response.text or "").strip()
                if text:
                    decision = text.splitlines()[0].strip()
                    break
                last_err = "empty response"
            except Exception as e:
                last_err = str(e)
                self.get_logger().warn(
                    f"Gemini call failed (attempt {attempt}/3): {e}")
                time.sleep(1.0)

        if decision is None:
            self.get_logger().error(
                f"Gemini failed for {request_id}: {last_err}")
            # No reply — VLADecide's wait_ticks_ timeout will fire and
            # the tree will retry with a new request_id.
            return

        self.get_logger().info(
            f"VLA decision for {request_id} ({part_id}): {decision}")

        out = String()
        out.data = json.dumps({
            "request_id": request_id,
            "decision":   decision,
        })
        self.decision_pub.publish(out)


def main():
    rclpy.init()
    node = VLAPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()