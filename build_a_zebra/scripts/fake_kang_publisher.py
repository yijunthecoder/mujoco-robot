#!/usr/bin/env python3
"""Pretends to be Kang's camera bridge. Publishes fake coords to
/zebra/perception_updates so you can verify your BT reacts."""
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

FAKE_PARTS = {
    "31111p0e": (0.51, -0.32, -0.10),
    "31111p0f": (0.51,  0.05, -0.10),
    "31111p0g": (0.51,  0.32, -0.10),
}

class FakeKang(Node):
    def __init__(self):
        super().__init__("fake_kang")
        self.pub = self.create_publisher(String, "/zebra/perception_updates", 10)
        self.create_timer(0.2, self.tick)   # 5 Hz
        self.get_logger().info("Fake Kang publishing 3 parts at 5 Hz")

    def tick(self):
        for part_id, (x, y, z) in FAKE_PARTS.items():
            s = String()
            s.data = f"{part_id},LOCATED,{x:.3f},{y:.3f},{z:.3f}"
            self.pub.publish(s)

def main():
    rclpy.init()
    node = FakeKang()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()