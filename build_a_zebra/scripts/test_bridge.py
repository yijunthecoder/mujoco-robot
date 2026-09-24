#!/usr/bin/env python3
"""Scripted test bridge for BT verification.

Behavior:
  legs  (31111p0e): always SUCCEEDED          -> tests the happy path
  body  (31111p0f): FAIL first 2 attempts,    -> tests retry + recovery,
                    SUCCEED on attempt 3         then succeeds
  head  (31111p0g): always FAILED              -> tests escalation

Publishes fake coordinates for all three at 5 Hz so LocatePart works.
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# Fake coordinates -- any real values work; the BT just needs to see them.
FAKE_PARTS = {
    "31111p0e": (0.51, -0.32, -0.10),   # legs
    "31111p0f": (0.51,  0.05, -0.10),   # body
    "31111p0g": (0.51,  0.32, -0.10),   # head
}

# Per-part success policy:
#   (fail_first_n, then_succeed)
POLICY = {
    "31111p0e": (0, True),    # always succeed
    "31111p0f": (2, True),    # fail 2x, then succeed
    "31111p0g": (99, False),  # always fail -> escalates after 3 retries
}


class TestBridge(Node):
    def __init__(self):
        super().__init__("test_bridge")
        self.status_pub  = self.create_publisher(String, "/zebra/skill_status", 10)
        self.percept_pub = self.create_publisher(String, "/zebra/perception_updates", 10)
        self.create_subscription(String, "/zebra/skill_commands", self.on_command, 10)
        self.create_timer(0.2, self.publish_perception)

        # Counts how many times each part's PICK has been requested.
        self.pick_attempts = {}

        self.get_logger().info("Test bridge ready (legs=ok, body=2 fails then ok, head=always fail)")

    def publish_perception(self):
        for part_id, (x, y, z) in FAKE_PARTS.items():
            s = String()
            s.data = f"{part_id},LOCATED,{x:.3f},{y:.3f},{z:.3f}"
            self.percept_pub.publish(s)

    def on_command(self, msg):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        command_id = cmd["command_id"]
        skill      = cmd["skill"]
        part_id    = cmd["part_id"]

        # Place always succeeds -- only pick is scripted.
        if skill == "place":
            self.get_logger().info(f"{command_id}  place {part_id}  -> SUCCEEDED")
            self._reply(command_id, "SUCCEEDED")
            return

        # Count pick attempts per part.
        n = self.pick_attempts.get(part_id, 0) + 1
        self.pick_attempts[part_id] = n

        fail_first_n, then_succeed = POLICY.get(part_id, (0, True))
        should_fail = (n <= fail_first_n) or (not then_succeed)

        if should_fail:
            self.get_logger().warn(
                f"{command_id}  pick {part_id}  attempt {n}  -> FAILED")
            self._reply(command_id, "FAILED")
        else:
            self.get_logger().info(
                f"{command_id}  pick {part_id}  attempt {n}  -> SUCCEEDED")
            self._reply(command_id, "SUCCEEDED")

    def _reply(self, command_id, status):
        reply = String()
        reply.data = json.dumps({
            "command_id": command_id,
            "status": status,
        })
        self.status_pub.publish(reply)


def main():
    rclpy.init()
    node = TestBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    

if __name__ == "__main__":
    main()