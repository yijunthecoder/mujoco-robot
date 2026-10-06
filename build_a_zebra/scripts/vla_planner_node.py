import json
import re
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
from google import genai
from google.genai import types
from apikeys import GOOGLE_API_KEY


def _sanitize_decision(raw, part_id):
    """Return 'DONE' or '<SKILL> <part_id>' if raw is a valid answer,
    else None.

    Tolerates: first-line-only, markdown code fences / backticks,
    surrounding quotes, trailing sentence punctuation, mixed case
    in the skill word. The part id must match part_id exactly.
    """
    if not raw:
        return None

    line = ""
    for candidate in raw.splitlines():
        candidate = candidate.strip()
        if candidate:
            line = candidate
            break
    if not line:
        return None

    # Strip markdown / quoting noise anywhere on the line.
    line = line.replace("`", "").strip()
    line = re.sub(r"\s+", " ", line)
    line = line.rstrip(" .,;:!?\"'")

    tokens = line.split(" ")
    if len(tokens) == 1:
        return "DONE" if tokens[0].upper() == "DONE" else None

    if len(tokens) == 2:
        skill = tokens[0].upper()
        part = tokens[1]
        if skill in ("PICK", "PLACE", "FLIP") and part == part_id:
            return f"{skill} {part_id}"

    return None


class VLAPlanner(Node):
    def __init__(self):
        super().__init__("vla_planner")

        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.bridge = CvBridge()
        self.latest_image = None

        self.create_subscription(Image, "/camera/image_raw", self.on_image, 10)
        self.create_subscription(String, "/vla/request", self.on_request, 10)
        self.decision_pub = self.create_publisher(String, "/vla/decision", 10)

        self.get_logger().info("VLAPlanner ready — waiting for /vla/request")

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
            self.get_logger().warn(
                "request received before first camera frame — dropping")
            return

        try:
            req = json.loads(msg.data)
            request_id = req["request_id"]
            part_id = req["part_id"]
            status = req.get("status", "UNKNOWN")
            orientation = req.get("orientation", "UNKNOWN")
            facing = req.get("facing", "UNKNOWN")
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

Answer with exactly one line, nothing else. Do not use backticks,
quotes, or a trailing period.
"""

        pil_image = PILImage.fromarray(self.latest_image)

        decision = None
        last_err = None
        for attempt in range(1, 4):
            try:
                response = self.client.models.generate_content(
                    model="gemma-4-26b-a4b-it",
                    contents=[pil_image, prompt],
                    config=types.GenerateContentConfig(
                        max_output_tokens=300,
                        temperature=0,
                        thinking_config=types.ThinkingConfig(
                            thinking_level="minimal",
                        ),
                    ),
                )
                raw = (response.text or "").strip()
                decision = _sanitize_decision(raw, part_id)
                if decision is not None:
                    break
                last_err = f"invalid answer: {raw!r}"
                self.get_logger().warn(
                    f"Gemma answer failed validation "
                    f"(attempt {attempt}/3): {raw!r}")
            except Exception as e:
                last_err = str(e)
                self.get_logger().warn(
                    f"Gemma call failed (attempt {attempt}/3): {e}")
            time.sleep(1.0)

        if decision is None:
            self.get_logger().error(
                f"Gemma failed for {request_id}: {last_err}")
            # No reply — VLADecide's wait_ticks_ timeout will fire and
            # the tree will retry with a new request_id.
            return

        self.get_logger().info(
            f"VLA decision for {request_id} ({part_id}): {decision}")

        out = String()
        out.data = json.dumps({
            "request_id": request_id,
            "decision": decision,
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