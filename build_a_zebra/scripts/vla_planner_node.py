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
        self.latest_perception = {}

        self.create_subscription(Image, "/camera/image_raw", self.on_image, 10)
        self.create_subscription(String, "/zebra/perception_updates",
                                 self.on_perception, 10)
        self.create_subscription(String, "/vla/request",
                                 self.on_request, 10)

        self.decision_pub = self.create_publisher(String, "/vla/decision", 10)

    def on_image(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def on_perception(self, msg):
        parts = msg.data.split(",")
        if len(parts) >= 2:
            self.latest_perception[parts[0]] = parts[1]

    def on_request(self, msg):
        part = msg.data.strip()
        if not part:
            return
        if self.latest_image is None:
            self.get_logger().warn("request arrived but no camera frame yet")
            return

        state_lines = []
        for pid in ["31111p0e", "31111p0f", "31111p0g"]:
            status = self.latest_perception.get(pid, "UNKNOWN")
            state_lines.append(f"- {pid}: {status}")
        state_text = "\n".join(state_lines)

        prompt = f"""
You must ONLY use the exact part IDs listed below.
You must NOT invent, modify, or abbreviate any ID.

Allowed part IDs (copy exactly):
- 31111p0e
- 31111p0f
- 31111p0g

Available skills: PICK, PLACE, FLIP.

The robot is currently working on: {part}
Your answer MUST use {part}. Any other part ID is WRONG.

Current state of the table:
{state_text}

Look at the image. Answer with EXACTLY one of these forms, nothing else:
  PICK {part}
  PLACE {part}
  FLIP {part}
  DONE
"""

        pil_image = PILImage.fromarray(self.latest_image)

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[pil_image, prompt],
                config={"max_output_tokens": 50, "temperature": 0},
            )
            decision = (response.text or "").strip()
        except Exception as e:
            self.get_logger().error(f"Gemini call failed: {e}")
            return

        if not decision:
            self.get_logger().warn("empty Gemini response, ignoring")
            return

        valid = {f"PICK {part}", f"PLACE {part}", f"FLIP {part}", "DONE"}
        if decision not in valid:
            self.get_logger().warn(
                f"Gemini returned '{decision}', not in valid set — ignoring")
            return

        self.get_logger().info(f"VLA decision for {part}: {decision}")

        out = String()
        out.data = decision
        self.decision_pub.publish(out)


def main():
    rclpy.init()
    node = VLAPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()