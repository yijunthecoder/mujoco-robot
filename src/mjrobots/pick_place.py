"""Scripted pick-and-place demo: Panda arm moves a cube onto a table.

How it works, in three layers:

1. Scene (assets/pick_and_place_scene.xml): the Panda arm (from menagerie)
   plus a table, a cube to pick up, and a translucent marker showing where
   it should end up. `_deployed_scene_path` copies this template next to
   panda.xml so MuJoCo's relative mesh paths resolve correctly (see the
   comment at the top of that file).

2. Inverse kinematics (`_solve_ik`): the arm's motors are driven in *joint*
   space (`data.ctrl[i]` = target angle for joint i), but we plan in
   *Cartesian* space ("put the gripper at this XYZ, pointing straight
   down"). `_solve_ik` bridges the two with damped-least-squares Jacobian
   IK: nudge the joint angles a little, see how the gripper moved (via
   `mj_jac`), and repeat until the gripper reaches the target. This never
   touches the real simulation state — it works on a scratch copy of qpos.

3. Motion (`_move_to` / `_hold`): once we have a joint-angle target for a
   waypoint, we ramp `data.ctrl` toward it linearly over many physics
   steps, calling `mj_step` each time. The Panda's actuators are PD
   position servos, so this produces smooth, physically simulated motion
   rather than teleporting the arm.

The overall script is just a list of Cartesian waypoints (hover above the
cube, descend, close the gripper, lift, move over the table, descend,
open the gripper, retreat) run through steps 2 and 3 in sequence.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .gl import configure_gl
from .models import menagerie_root

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_SCENE_TEMPLATE = _ASSETS_DIR / "pick_and_place_scene.xml"

_ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
_GRIPPER_OPEN = 255.0
_GRIPPER_CLOSED = 0.0

# Vertical offset from the "hand" body's origin to the point midway between
# the fingertips, when the gripper points straight down. Found from the
# Panda's own geometry (finger body sits 0.0584 below the hand frame, and
# the fingertip pads sit another ~0.045 further along the finger).
_FINGERTIP_OFFSET = 0.103

# Orientation with the gripper pointing straight down (180 deg about world X).
_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def _deployed_scene_path(robot: str = "franka_emika_panda") -> Path:
    """Copy the scene template next to panda.xml and return its path.

    Must live alongside panda.xml: MuJoCo resolves an <include>d file's own
    relative asset paths (panda.xml's meshdir="assets") against the
    top-level scene file's directory, not the included file's directory.
    """
    dest_dir = menagerie_root() / robot
    dest = dest_dir / _SCENE_TEMPLATE.name
    if not dest.exists() or dest.read_text() != _SCENE_TEMPLATE.read_text():
        shutil.copyfile(_SCENE_TEMPLATE, dest)
    return dest


def _solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    arm_qpos_adr: np.ndarray,
    arm_dof_adr: np.ndarray,
    arm_joint_ids: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray = _DOWN_QUAT,
    iters: int = 200,
    damping: float = 0.1,
    tol: float = 1e-4,
    max_step: float = 0.2,
) -> np.ndarray:
    """Damped-least-squares IK for `body_id`'s pose, moving only the arm joints.

    Returns target joint angles (radians) for the 7 arm joints. Operates on
    a scratch copy of qpos/qvel — the real simulation state is untouched.
    """
    qpos_save = data.qpos.copy()
    qvel_save = data.qvel.copy()
    q = data.qpos[arm_qpos_adr].copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    lo = model.jnt_range[arm_joint_ids, 0]
    hi = model.jnt_range[arm_joint_ids, 1]

    try:
        for _ in range(iters):
            data.qpos[arm_qpos_adr] = q
            mujoco.mj_kinematics(model, data)

            pos_err = target_pos - data.xpos[body_id]
            neg_cur = np.empty(4)
            mujoco.mju_negQuat(neg_cur, data.xquat[body_id])
            err_quat = np.empty(4)
            mujoco.mju_mulQuat(err_quat, target_quat, neg_cur)
            rot_err = np.empty(3)
            mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)
            err = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(err) < tol:
                break

            mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body_id], body_id)
            J = np.vstack([jacp[:, arm_dof_adr], jacr[:, arm_dof_adr]])
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + damping**2 * np.eye(6), err)
            dq = np.clip(dq, -max_step, max_step)
            q = np.clip(q + dq, lo, hi)
    finally:
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        mujoco.mj_kinematics(model, data)

    return q


@dataclass
class _RealtimeClock:
    dt: float
    _next: float | None = None

    def tick(self) -> None:
        now = time.perf_counter()
        if self._next is None:
            self._next = now
        self._next += self.dt
        sleep_for = self._next - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            self._next = now


_TARGET_FPS = 60.0


class _ThrottledSync:
    """Sync the viewer at ~_TARGET_FPS instead of every physics step.

    mj_step runs at model.opt.timestep (often ~500Hz); syncing the viewer
    that often renders far more frames than any display can show and is
    the main source of perceived lag. Physics still steps every call —
    only the render is throttled.
    """

    def __init__(self, viewer, model, target_fps: float = _TARGET_FPS):
        self._viewer = viewer
        self._every = max(1, round(1.0 / target_fps / model.opt.timestep))
        self._count = 0

    def step(self) -> None:
        self._count += 1
        if self._count % self._every == 0:
            self._viewer.sync()


def _move_to(model, data, render, clock, ctrl_target, steps, grip=None):
    ctrl_start = data.ctrl[:7].copy()
    if grip is not None:
        data.ctrl[7] = grip
    for i in range(steps):
        alpha = (i + 1) / steps
        data.ctrl[:7] = ctrl_start + alpha * (ctrl_target - ctrl_start)
        mujoco.mj_step(model, data)
        if render is not None:
            clock.tick()
            render.step()


def _hold(model, data, render, clock, steps, grip=None):
    if grip is not None:
        data.ctrl[7] = grip
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if render is not None:
            clock.tick()
            render.step()


def run_demo(prefer_gl: str = "egl", robot: str = "franka_emika_panda") -> None:
    """Run the scripted pick-and-place demo in an interactive viewer."""
    configure_gl(prefer_gl)

    scene = _deployed_scene_path(robot)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pick_place_start")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_top_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube")

    arm_joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in _ARM_JOINTS]
    )
    arm_qpos_adr = model.jnt_qposadr[arm_joint_ids]
    arm_dof_adr = model.jnt_dofadr[arm_joint_ids]
    home_ctrl = data.qpos[arm_qpos_adr].copy()

    cube_half_z = model.geom_size[cube_geom][2]
    cube_xy = data.xpos[cube_id][:2].copy()
    cube_rest_z = data.xpos[cube_id][2]  # cube's own center height at pickup
    table_top_z = data.xpos[table_id][2] + model.geom_size[table_top_geom][2]
    place_xy = data.xpos[table_id][:2].copy()

    def ik(target_pos: np.ndarray) -> np.ndarray:
        return _solve_ik(
            model, data, hand_id, arm_qpos_adr, arm_dof_adr, arm_joint_ids, target_pos
        )

    above_cube = ik(np.array([*cube_xy, cube_rest_z + 0.15 + _FINGERTIP_OFFSET]))
    at_cube = ik(np.array([*cube_xy, cube_rest_z + _FINGERTIP_OFFSET]))
    lifted = ik(np.array([*cube_xy, cube_rest_z + 0.20 + _FINGERTIP_OFFSET]))
    place_z = table_top_z + cube_half_z
    above_place = ik(np.array([*place_xy, place_z + 0.20 + _FINGERTIP_OFFSET]))
    at_place = ik(np.array([*place_xy, place_z + _FINGERTIP_OFFSET]))
    retreat = ik(np.array([*place_xy, place_z + 0.25 + _FINGERTIP_OFFSET]))

    clock = _RealtimeClock(dt=model.opt.timestep)
    restart_requested = {"flag": False}

    def on_key(keycode: int) -> None:
        if keycode == ord("R"):
            restart_requested["flag"] = True

    def play_once(render) -> None:
        _move_to(model, data, render, clock, above_cube, 400, grip=_GRIPPER_OPEN)
        _move_to(model, data, render, clock, at_cube, 250, grip=_GRIPPER_OPEN)
        _hold(model, data, render, clock, 150, grip=_GRIPPER_CLOSED)
        _move_to(model, data, render, clock, lifted, 300, grip=_GRIPPER_CLOSED)
        _move_to(model, data, render, clock, above_place, 500, grip=_GRIPPER_CLOSED)
        _move_to(model, data, render, clock, at_place, 300, grip=_GRIPPER_CLOSED)
        _hold(model, data, render, clock, 150, grip=_GRIPPER_OPEN)
        _move_to(model, data, render, clock, retreat, 300, grip=_GRIPPER_OPEN)
        _move_to(model, data, render, clock, home_ctrl, 400, grip=_GRIPPER_OPEN)

        cube_final = data.xpos[cube_id].copy()
        horiz_error = np.linalg.norm(cube_final[:2] - place_xy)
        on_table = abs(cube_final[2] - place_z) < 0.02
        if horiz_error < 0.05 and on_table:
            print(f"[mjrobots] placement OK: cube at {cube_final}, {horiz_error*100:.1f} cm from target")
        else:
            print(
                f"[mjrobots] placement FAILED: cube at {cube_final}, "
                f"{horiz_error*100:.1f} cm from target, on_table={on_table}"
            )

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        render = _ThrottledSync(viewer, model)
        play_once(render)
        print(
            "[mjrobots] done - press R to run it again, use the viewer's own "
            "Run/Pause button to freeze physics and drag things around, or close the window to exit"
        )
        while viewer.is_running():
            if restart_requested["flag"]:
                restart_requested["flag"] = False
                mujoco.mj_resetDataKeyframe(model, data, key_id)
                mujoco.mj_forward(model, data)
                viewer.sync()
                play_once(render)
                print("[mjrobots] done - press R to run it again, or close the window to exit")
            else:
                # While the scripted sequence runs, this loop drives mj_step
                # unconditionally, so the viewer's own Run/Pause button has
                # nothing to control (that's why clicking it does nothing
                # mid-demo). Idling here, though, we do check its `run` flag
                # (the internal Simulate object read via the private
                # `_get_sim()` — no public API for this in launch_passive)
                # so pausing/resuming and dragging bodies with the mouse
                # work like a normal MuJoCo viewer once the task is done.
                sim = viewer._get_sim()
                if sim is None or sim.run:
                    mujoco.mj_step(model, data)
                    clock.tick()
                    render.step()
                else:
                    viewer.sync()
                    time.sleep(1.0 / _TARGET_FPS)
