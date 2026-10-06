import json
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
        self.busy = False

        self.create_subscription(Image, "/camera/image_raw", self.on_image, 10)
        self.create_subscription(String, "/vla/request", self.on_request, 10)

        self.decision_pub = self.create_publisher(String, "/vla/decision", 10)

    def on_image(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            self.get_logger().warn(f"image decode failed: {e}")

    def _ask_gemini(self, part, status, orient, facing, prompt):
        pil_image = PILImage.fromarray(self.latest_image)
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil_image, prompt],
            config={
                "max_output_tokens": 300,
                "temperature": 0,
                "thinking_config": {"thinking_budget": 0},
            },
        )
        return (response.text or "").strip()

    def on_request(self, msg):
        if self.busy:
            self.get_logger().warn("request arrived while busy — dropping")
            return
        self.busy = True

        try:
            # ---- 1. Parse the JSON request from C++ ----
            try:
                req = json.loads(msg.data)
            except Exception as e:
                self.get_logger().warn(f"bad request json: {e}")
                return

            request_id = req.get("request_id", "")
            part       = req.get("part_id", "")
            status     = req.get("status", "UNKNOWN")
            orient     = req.get("orientation", "UNKNOWN")
            facing     = req.get("facing", "UNKNOWN")

            if not part or not request_id:
                self.get_logger().warn("request missing part_id / request_id")
                return

            if self.latest_image is None:
                self.get_logger().warn("request arrived but no camera frame yet")
                return

            # ---- 2. Build prompt using the WorldModel state from C++ ----
            prompt = f"""
You are controlling a robot arm building a LEGO zebra.
Allowed part IDs (copy exactly):
- 31111p0e
- 31111p0f
- 31111p0g

Available skills: PICK, PLACE, FLIP, ROTATE.

The robot is currently working on part: {part}
Its WorldModel state is:
  status      = {status}
  orientation = {orient}
  facing      = {facing}

Rules:
- If status is LOCATED, the next skill must be PICK.
- If status is PICKED, the next skill must be PLACE.
- If orientation is UPSIDE_DOWN or ON_SIDE, the next skill must be FLIP.
- If facing is BACKWARD, the next skill must be ROTATE.
- If status is PLACED, answer DONE.

Answer with EXACTLY one line, nothing else:
  PICK {part}
  PLACE {part}
  FLIP {part}
  ROTATE {part}
  DONE
"""

            valid = {
                f"PICK {part}",
                f"PLACE {part}",
                f"FLIP {part}",
                f"ROTATE {part}",
                "DONE",
            }

            # ---- 3. Call Gemini, retry once on empty / invalid ----
            decision = ""
            for attempt in (1, 2):
                try:
                    decision = self._ask_gemini(part, status, orient, facing, prompt)
                except Exception as e:
                    self.get_logger().error(
                        f"Gemini call failed (attempt {attempt}): {e}")
                    decision = ""

                if decision in valid:
                    break

                if decision:
                    self.get_logger().warn(
                        f"Gemini returned '{decision}' "
                        f"(attempt {attempt}), not in valid set")
                else:
                    self.get_logger().warn(
                        f"empty Gemini response (attempt {attempt})")

            if decision not in valid:
                self.get_logger().warn(
                    "no valid decision after retry — dropping request")
                return

            self.get_logger().info(
                f"VLA decision for {part} [{request_id}]: {decision}")

            # ---- 4. Publish the JSON reply back to C++ ----
            out = String()
            out.data = json.dumps({
                "request_id": request_id,
                "decision":   decision,
            })
            self.decision_pub.publish(out)

        finally:
            self.busy = False


def main():
    rclpy.init()
    node = VLAPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()