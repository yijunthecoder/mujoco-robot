import json, math, threading, time
import mujoco, mujoco.viewer
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MujocoBridge(Node):
    def __init__(self):
        super().__init__('mujoco_bridge')
        self.model = mujoco.MjModel.from_xml_path('scene.xml')
        self.data  = mujoco.MjData(self.model)

        # Map JSON skill -> which arm handles it. Left arm = +Y side.
        self.pick_arm  = "left"     # or "right"; keep it simple at first
        self.place_arm = "left"

        self.cmd_sub = self.create_subscription(
            String, '/zebra/skill_commands', self.on_command, 10)
        self.status_pub  = self.create_publisher(String, '/zebra/skill_status', 10)
        self.percept_pub = self.create_publisher(String, '/zebra/perception_updates', 10)

        # physics thread (or run inside a timer callback)
        self.lock = threading.Lock()
        self.pending = None      # dict | None
        threading.Thread(target=self.sim_loop, daemon=True).start()

    # ---------- physics ---------------------------------------------------
    def sim_loop(self):
        dt = self.model.opt.timestep
        last_percept = 0.0
        while rclpy.ok():
            with self.lock:
                mujoco.mj_step(self.model, self.data)
            # 5 Hz perception
            if self.data.time - last_percept > 0.2:
                self.publish_perception()
                last_percept = self.data.time
            time.sleep(dt)

    # ---------- command handling -----------------------------------------
    def on_command(self, msg):
        payload = json.loads(msg.data)
        self.get_logger().info(f"recv {payload['command_id']} {payload['skill']}")
        with self.lock:
            self.pending = payload

        # Block the physics step from executing this synchronously here?
        # Simplest: run a short state machine on a worker thread.
        threading.Thread(target=self.execute, args=(payload,), daemon=True).start()

    def execute(self, cmd):
        try:
            if cmd["skill"] == "pick":
                ok = self.do_pick(cmd["part_id"], cmd["target"])
            elif cmd["skill"] == "place":
                ok = self.do_place(cmd["part_id"], cmd["target"])
            elif cmd["skill"] == "flip":
                ok = self.do_flip(cmd["part_id"], cmd["target"])
            else:
                ok = False
        except Exception as e:
            self.get_logger().error(f"executor error: {e}")
            ok = False

        out = String()
        out.data = json.dumps({
            "command_id": cmd["command_id"],
            "status": "SUCCEEDED" if ok else "FAILED",
            "message": "" if ok else "grasp failed",
        })
        self.status_pub.publish(out)

    # ---------- actual robot motion --------------------------------------
    def do_pick(self, part_id, target):
        """Move gripper to (target.x, target.y, target.z), close, lift.
        Returns True on success. Replace with real IK + trajectory."""
        arm = self.pick_arm
        self.set_gripper(arm, open=True)
        if not self.ik_move_to(arm, target["x"], target["y"], target["z"] + 0.05):
            return False
        self.ik_move_to(arm, target["x"], target["y"], target["z"])
        self.set_gripper(arm, open=False)
        # wait for fingers to close
        self.settle(0.3)
        # contact check: gripper joint positions reached closed value?
        return self.grasp_detected(arm)

    def do_place(self, part_id, target):
        arm = self.place_arm
        if not self.ik_move_to(arm, target["x"], target["y"], target["z"] + 0.15):
            return False
        self.ik_move_to(arm, target["x"], target["y"], target["z"])
        self.set_gripper(arm, open=True)
        self.settle(0.2)
        return True

    def do_flip(self, part_id, target):
        # Rotate the part 180 degrees about a horizontal axis so it sits upright.
        # Sim-only: directly set the part's quaternion. Real robot would need a
        # regrasp. Returns True on success.
        # Find the body id for this part in the model
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, part_id)
        except Exception:
            return False
        if body_id < 0:
            self.get_logger().error(f"flip: unknown body {part_id}")
            return False

        # Find the free joint for that body (if any) and flip its quaternion
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, part_id + "_free")
        if joint_id < 0:
            self.get_logger().error(f"flip: no free joint for {part_id}")
            return False

        qpos_adr = self.model.jnt_qposadr[joint_id]
        with self.lock:
            # qpos: [x, y, z, qw, qx, qy, qz] for a free joint
            # Flip 180 deg about the X axis: quaternion (0, 1, 0, 0)
            self.data.qpos[qpos_adr + 3] = 0.0   # qw
            self.data.qpos[qpos_adr + 4] = 1.0   # qx
            self.data.qpos[qpos_adr + 5] = 0.0   # qy
            self.data.qpos[qpos_adr + 6] = 0.0   # qz
            mujoco.mj_forward(self.model, self.data)

        self.settle(0.5)
        return True

    # ---------- low-level helpers (stubs you fill in) ---------------------
    def ik_move_to(self, arm, x, y, z):
        """Solve IK for the 6-DOF chain, set actuator targets, step until
        joint error < tol or timeout. Return True on success."""
        # You can use mujoco.mj_inverse / damped least squares, or plug in
        # a real planner (MoveIt is overkill for this scope).
        return True

    def set_gripper(self, arm, open):
        # drive left_gripper_joint1 / joint2 position actuators
        return

    def grasp_detected(self, arm):
        # read finger joint position + contact forces from self.data
        return True

    def settle(self, seconds):
        steps = int(seconds / self.model.opt.timestep)
        for _ in range(steps):
            with self.lock:
                mujoco.mj_step(self.model, self.data)

    # ---------- perception ------------------------------------------------
    def publish_perception(self):
        # Read orientation from the sim and report it
        parts = {
            "31111p0e": (0.40, 0.00, 0.32),
            "31111p0f": (0.40, 0.05, 0.32),
            "31111p0g": (0.40, -0.05, 0.32),
        }
        for pid, (x, y, z) in parts.items():
            orient = self.get_orientation(pid)   # returns "UPRIGHT" or "UPSIDE_DOWN"
            s = String()
            s.data = f"{pid},LOCATED,{x},{y},{z},{orient}"
            self.percept_pub.publish(s)

    def get_orientation(self, part_id):
        """Read the free joint quaternion and classify into one of four poses.
        Assumes the brick's local +Z axis is 'studs up' and the long axis is +X.
        """
        # Try body name as-given, then with the 'zebra_' prefix
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, part_id)
        if body_id < 0:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"zebra_{part_id}")
        if body_id < 0:
            return "UNKNOWN"

        # Find the free joint on that body
        joint_id = -1
        for j in range(self.model.njnt):
            if (self.model.jnt_bodyid[j] == body_id and
                    self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
                joint_id = j
                break
        if joint_id < 0:
            return "UNKNOWN"

        qpos_adr = self.model.jnt_qposadr[joint_id]
        qw = self.data.qpos[qpos_adr + 3]
        qx = self.data.qpos[qpos_adr + 4]
        qy = self.data.qpos[qpos_adr + 5]
        qz = self.data.qpos[qpos_adr + 6]

        # Rotate the brick's local axes into world frame.
        # We care about:
        #   local Z (studs) -> world Z component   | up/down
        #   local X (long)  -> world Z component   | on-end vs on-side
        # Using the rotation matrix of the quaternion:
        #   R = quat_to_matrix(qw, qx, qy, qz)
        # Row 2 of R gives the world-frame components of the local axes:
        #   world_z_of_local_x = R[2][0]
        #   world_z_of_local_z = R[2][2]
        r20 = 2.0 * (qx * qz - qw * qy)      # world Z component of local X
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)  # world Z component of local Z

        # Classify: which local axis points up in the world?
        if r22 > 0.7:
            return "UPRIGHT"       # studs point up
        if r22 < -0.7:
            return "UPSIDE_DOWN"   # studs point down
        if r20 > 0.7 or r20 < -0.7:
            return "ON_END"        # long axis vertical
        return "ON_SIDE"           # lying on the long side
    
def main():
    rclpy.init()
    MujocoBridge()
    rclpy.spin(MujocoBridge.__new__(MujocoBridge))  # spin an instance