import os
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
        self.latest_perception = {}   # pid -> {"status": ..., "orientation": ...}
        self.last_state_key = None

        # Subscribers
        self.create_subscription(Image, "/camera/image_raw", self.on_image, 10)
        self.create_subscription(
            String, "/zebra/perception_updates", self.on_perception, 10)

        # Publisher
        self.decision_pub = self.create_publisher(String, "/vla/decision", 10)

        # Ask at most once every 10 s, and only when state changes
        self.create_timer(10.0, self.decide)

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------
    def on_image(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def on_perception(self, msg):
        # Format: part,STATUS,x,y,z,orientation,yaw,facing
        fields = msg.data.split(",")
        if len(fields) < 6:
            return
        self.latest_perception[fields[0]] = {
            "status": fields[1],
            "orientation": fields[5],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def current_part(self):
        """The part the tree is currently working on: first one that isn't
        PLACED or ESCALATED, in build order."""
        for pid in ["31111p0e", "31111p0f", "31111p0g"]:
            info = self.latest_perception.get(pid)
            if info is None:
                return pid
            if info["status"] not in ("PLACED", "ESCALATED"):
                return pid
        return "DONE"

    # ------------------------------------------------------------------
    # Main decision loop
    # ------------------------------------------------------------------
    def decide(self):
        if self.latest_image is None:
            return

        # Skip if nothing changed since the last decision
        state_key = tuple(sorted(
            (pid, info["status"], info["orientation"])
            for pid, info in self.latest_perception.items()
        ))
        if state_key == self.last_state_key:
            return
        self.last_state_key = state_key

        part = self.current_part()
        if part == "DONE":
            return

        # Build a readable state block
        state_lines = []
        for pid in ["31111p0e", "31111p0f", "31111p0g"]:
            info = self.latest_perception.get(pid, {})
            status = info.get("status", "UNKNOWN")
            orient = info.get("orientation", "UNKNOWN")
            state_lines.append(f"- {pid}: status={status}, orientation={orient}")
        state_text = "\n".join(state_lines)

        prompt = f"""
You are controlling a robot arm building a LEGO zebra.

Allowed part IDs (copy exactly):
- 31111p0e
- 31111p0f
- 31111p0g

The robot is currently working on: {part}
Your answer must use {part}, not a different part.

Current state of the table:
{state_text}

Rules, in order — use the FIRST one that matches:
1. If status is PLACED, answer: DONE
2. If orientation is anything other than UPRIGHT, answer: FLIP {part}
3. If status is PICKED, answer: PLACE {part}
4. Otherwise, answer: PICK {part}

Answer with exactly one line, nothing else.
"""

        pil_image = PILImage.fromarray(self.latest_image)

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[pil_image, prompt],
                config={"max_output_tokens": 10, "temperature": 0},
            )
            decision = response.text.strip()
        except Exception as e:
            self.get_logger().error(f"Gemini call failed: {e}")
            return

        self.get_logger().info(f"VLA decision: {decision}")

        out = String()
        out.data = decision
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