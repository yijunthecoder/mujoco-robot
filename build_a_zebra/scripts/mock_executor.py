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
        # VERY IMPORTANT: this is what LocatePart / IsPartLocated react to.
        # For a first run, publish the three zebra parts as LOCATED at a
        # fixed table pose so the BT leaves LocatePart quickly.
        parts = {
            "31111p0e": (0.40, 0.00, 0.32),
            "31111p0f": (0.40, 0.05, 0.32),
            "31111p0g": (0.40, -0.05, 0.32),
        }
        for pid, (x, y, z) in parts.items():
            # If you want to simulate a "held" state, check contact between
            # gripper body and the part body here and swap in PICKED.
            s = String()
            s.data = f"{pid},LOCATED,{x},{y},{z}"
            self.percept_pub.publish(s)

def main():
    rclpy.init()
    MujocoBridge()
    rclpy.spin(MujocoBridge.__new__(MujocoBridge))  # spin an instance