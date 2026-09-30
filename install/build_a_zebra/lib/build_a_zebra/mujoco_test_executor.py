#!/usr/bin/env python3
"""
mujoco_test_executor.py

Listens on /zebra/skill_commands, replies on /zebra/skill_status,
and drives the Station Lite dual arms inside a live MuJoCo simulation.

HONEST LIMITATION: no real inverse kinematics. Each pick / place uses
fixed joint keyframes regardless of the target x/y/z sent on the
command. But it DOES verify grip success by counting physical contacts
between gripper fingers and bricks in the world -- so the BT now
receives a truthful SUCCEEDED / FAILED and the RecoveryManager can
actually fire.

Run with:
    cd ~/mujoco-project/build_a_zebra
    python3 scripts/mujoco_test_executor.py
"""

import json
import threading
import time

import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ---------------------------------------------------------------------------
# Actuator names, matching <actuator> in stationlite_mujoco/scene.xml
# ---------------------------------------------------------------------------
LEFT_ARM_ACTUATORS = [
    "left_joint1_act", "left_joint2_act", "left_joint3_act",
    "left_joint4_act", "left_joint5_act", "left_joint6_act",
]
LEFT_GRIPPER_ACTUATORS = ["left_gripper_joint1_act", "left_gripper_joint2_act"]

RIGHT_ARM_ACTUATORS = [
    "right_joint1_act", "right_joint2_act", "right_joint3_act",
    "right_joint4_act", "right_joint5_act", "right_joint6_act",
]
RIGHT_GRIPPER_ACTUATORS = ["right_gripper_joint1_act", "right_gripper_joint2_act"]


# ---------------------------------------------------------------------------
# Crude keyframe poses (joint1..joint6). NOT IK.
# ---------------------------------------------------------------------------
HOME      = [ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0]

# Left arm: reaches toward +Y side
L_ABOVE   = [ 0.4,   1.2,  -1.5,   0.0,   0.0,   0.0]   # HIGH  (approach from above)
L_TABLE   = [ 0.4,   1.0,  -1.1,   0.0,   0.0,   0.0]   # LOW   (fingertips reach block)
L_LIFT    = [ 0.4,   1.2,  -1.5,   0.0,   0.0,   0.0]   # HIGH  (retract same way)

# Right arm: reaches toward -Y side
R_ABOVE   = [-0.4,   1.0,  -1.1,   0.0,   0.0,   0.0]
R_PLACE   = [-0.4,   1.2,  -1.5,   0.0,   0.0,   0.0]
R_LIFT    = [-0.4,   0.6,  -1.0,   0.0,   0.0,   0.0]

GRIP_OPEN   = [-0.0425,  0.0425]
GRIP_CLOSED = [ 0.0,     0.0   ]


class MujocoExecutor(Node):
    def __init__(self, model, data, viewer):
        super().__init__("zebra_mujoco_executor")
        self.model = model
        self.data = data
        self.viewer = viewer

        self.arm_ids_l     = [model.actuator(n).id for n in LEFT_ARM_ACTUATORS]
        self.gripper_ids_l = [model.actuator(n).id for n in LEFT_GRIPPER_ACTUATORS]
        self.arm_ids_r     = [model.actuator(n).id for n in RIGHT_ARM_ACTUATORS]
        self.gripper_ids_r = [model.actuator(n).id for n in RIGHT_GRIPPER_ACTUATORS]

        # Cache geom ids once, so contact check doesn't re-scan every step
        self.gripper_geom_ids = set()
        self.brick_geom_ids   = set()
        for i in range(model.ngeom):
            body_id = model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if "griperlj" in body_name:
                self.gripper_geom_ids.add(i)
            if body_name.startswith("zebra_"):
                self.brick_geom_ids.add(i)

        self.busy = False
        self.lock = threading.Lock()

        self.status_pub = self.create_publisher(String, "/zebra/skill_status", 10)
        self.create_subscription(String, "/zebra/skill_commands", self.on_command, 10)

        self.get_logger().info(
            f"MuJoCo executor ready. "
            f"gripper geoms={len(self.gripper_geom_ids)}, "
            f"brick geoms={len(self.brick_geom_ids)}")

    # ------------------------------------------------------------------
    # Low-level: command a pose and hold it for `seconds`
    # ------------------------------------------------------------------
    def _drive(self, arm_ids, grip_ids, arm_pose, grip_pose, seconds=1.2):
        for aid, val in zip(arm_ids, arm_pose):
            self.data.ctrl[aid] = val
        for gid, val in zip(grip_ids, grip_pose):
            self.data.ctrl[gid] = val
        end = time.time() + seconds
        while time.time() < end:
            time.sleep(0.01)

        # --- DIAGNOSTIC: where did the gripper end up? ---
        # Prints the world position of the gripper body after each move.
        # Use these numbers to set the brick positions in scene.xml so
        # the fingers actually meet the brick.
        for name in ("left_linkgripper", "right_linkgripper",
                     "left_griperlj_link1", "left_griperlj_link2",
                     "right_griperlj_link1", "right_griperlj_link2"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                p = self.data.xpos[bid]
                self.get_logger().info(
                    f"  [{name}] at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
        # --- end diagnostic ---

    # ------------------------------------------------------------------
    # Reality check: how many contacts between gripper and any brick?
    # ------------------------------------------------------------------
    def _grip_contact_count(self):
        count = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            a_g = c.geom1 in self.gripper_geom_ids
            b_g = c.geom2 in self.gripper_geom_ids
            a_b = c.geom1 in self.brick_geom_ids
            b_b = c.geom2 in self.brick_geom_ids
            if (a_g and b_b) or (a_b and b_g):
                count += 1
        return count

    # ------------------------------------------------------------------
    # ROS: incoming commands
    # ------------------------------------------------------------------
    def on_command(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"Bad JSON on skill_commands: {e}")
            return

        command_id = cmd.get("command_id", "")
        skill      = cmd.get("skill", "")
        part_id    = cmd.get("part_id", "")

        with self.lock:
            if self.busy:
                self.get_logger().warn(
                    f"Received {command_id} while already busy -- ignoring.")
                return
            self.busy = True

        self.get_logger().info(f"Received {command_id}: {skill} {part_id}")
        threading.Thread(
            target=self._execute, args=(command_id, skill), daemon=True
        ).start()

    # ------------------------------------------------------------------
    # The actual pick / place sequences
    # ------------------------------------------------------------------
    def _execute(self, command_id, skill):
        try:
            if skill == "pick":
                # Approach from above with fingers open
                self._drive(self.arm_ids_l, self.gripper_ids_l, L_ABOVE, GRIP_OPEN)
                self._drive(self.arm_ids_l, self.gripper_ids_l, L_TABLE, GRIP_OPEN)

                # Close fingers, let contacts settle
                self._drive(self.arm_ids_l, self.gripper_ids_l,
                            L_TABLE, GRIP_CLOSED, seconds=3.0)

                # Reality check
                n = self._grip_contact_count()
                if n == 0:
                    self.get_logger().warn(
                        f"{command_id}: no grip contact -- reporting FAILED")
                    self._drive(self.arm_ids_l, self.gripper_ids_l,
                                L_ABOVE, GRIP_OPEN)
                    self._report(command_id, "FAILED",
                                 "no contact between gripper and any part")
                    return

                self.get_logger().info(f"{command_id}: grip contact x{n}")
                self._drive(self.arm_ids_l, self.gripper_ids_l,
                            L_LIFT, GRIP_CLOSED)

            elif skill == "place":
                self._drive(self.arm_ids_r, self.gripper_ids_r, R_ABOVE, GRIP_OPEN)
                self._drive(self.arm_ids_r, self.gripper_ids_r, R_PLACE, GRIP_OPEN)
                self._drive(self.arm_ids_r, self.gripper_ids_r,
                            R_PLACE, GRIP_CLOSED, seconds=0.6)
                self._drive(self.arm_ids_r, self.gripper_ids_r, R_LIFT, GRIP_CLOSED)
            else:
                self.get_logger().warn(f"Unknown skill '{skill}'")
                self._report(command_id, "FAILED", f"unknown skill {skill}")
                return

            self._report(command_id, "SUCCEEDED")
        finally:
            with self.lock:
                self.busy = False

    # ------------------------------------------------------------------
    # Reply to the BT
    # ------------------------------------------------------------------
    def _report(self, command_id, status, message=""):
        payload = {"command_id": command_id, "status": status}
        if message:
            payload["message"] = message
        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)
        self.get_logger().info(f"Reported {command_id} -> {status}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    model = mujoco.MjModel.from_xml_path("stationlite_mujoco/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rclpy.init()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        node = MujocoExecutor(model, data, viewer)

        try:
            while rclpy.ok() and viewer.is_running():
                step_start = time.time()

                mujoco.mj_step(model, data)
                viewer.sync()
                rclpy.spin_once(node, timeout_sec=0)

                time_until_next = model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()