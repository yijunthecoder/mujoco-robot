import os, rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
from google import genai
from apikeys import GOOGLE_API_KEY
import json

class VLAPlanner(Node):
    def __init__(self):
        super().__init__("vla_planner")

        self.client = genai.Client(api_key=...)
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_perception = {}

        # Subscribers
        self.create_subscription(Image, "/camera/image_raw", self.on_image, 10)
        self.create_subscription(String, "/zebra/perception_updates",
                                 self.on_perception, 10)

        # Publisher
        self.decision_pub = self.create_publisher(String, "/vla/decision", 10)

        # Decide every 5 seconds
        self.create_timer(5.0, self.decide)

    def on_image(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def on_perception(self, msg):
        # Parse "part,STATUS,x,y,z,orient,yaw,facing"
        parts = msg.data.split(",")
        if len(parts) >= 2:
            self.latest_perception[parts[0]] = parts[1]

    def decide(self):
        if self.latest_image is None:
            return

        # Build state description from perception
        state_lines = []
        for pid in ["31111p0e", "31111p0f", "31111p0g"]:
            status = self.latest_perception.get(pid, "UNKNOWN")
            state_lines.append(f"- {pid}: {status}")

        prompt = f"""
You must ONLY use the exact part IDs listed below.
... (full prompt with the 10 valid answers)
Current state:
{chr(10).join(state_lines)}
"""

        pil_image = PILImage.fromarray(self.latest_image)

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil_image, prompt],
            config={"max_output_tokens": 10, "temperature": 0},
        )

        decision = response.text.strip()
        self.get_logger().info(f"VLA decision: {decision}")

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