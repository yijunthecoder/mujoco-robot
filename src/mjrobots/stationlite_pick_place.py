"""Scripted pick-and-place demo for the stationlite dual-arm robot.

Two things differ from pick_place.py (Panda):

1. No runtime IK. The arm's orientation conventions/reach were unknown going
   in, so the waypoints below are joint angles found by a forward-kinematics
   search (done once, offline) that verified they reach the ball's position
   and the target site defined in stationlite_pick_place.xml.

2. The grasp is a kinematic "carry", not a physical friction hold. The
   URDF's finger meshes are used as-is for collision (no real pad contact
   geometry), and a pure two-finger side-pinch on a smooth 4cm ball turned
   out to be right at the edge of what friction can hold through any
   sideways acceleration (confirmed experimentally - it slips out mid-swing
   regardless of grip force/speed). Instead, once the gripper closes at the
   grasp waypoint, `_carry` snaps the ball's position to the fingertip
   midpoint after every physics step until release - visually indistinguishable
   from a held object, without depending on uncharacterized contact geometry.

Only the LEFT arm moves. ctrl indices 0-7 are the left arm (6 joints + 2
gripper channels); 8-15 are the right arm, left at its keyframe "home" ctrl.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .gl import configure_gl
from .pick_place import _RealtimeClock, _ThrottledSync

_DEFAULT_SCENE = Path("/mnt/c/intern/stationlite/urdf/stationlite_pick_place.xml")

_ARM_CTRL = slice(0, 6)
_GRIP_CTRL = slice(6, 8)

_GRIP_OPEN = (-0.0425, 0.0425)
_GRIP_CLOSED = (-0.0425 * 0.4, 0.0425 * 0.4)  # visually close around the ball

# left_joint1..6, found via FK search against stationlite_mujoco.urdf.
_HOVER_PICK = [0.0, 1.98, -1.37, 0.0, 0.0, 0.0]
_GRASP = [0.0, 2.23, -1.215, 0.0, 0.0, 0.0]
_HOVER_PLACE = [-0.4, 1.98, -1.37, 0.0, 0.0, 0.0]
_PLACE = [-0.4, 2.23, -1.215, 0.0, 0.0, 0.0]

# Fingertip-midpoint position at _PLACE/_GRASP (also where the ball starts).
_TARGET_XY = np.array([0.481, 0.114])


def _move_to(model, data, render, clock, arm_target, grip_target, steps, carry=None) -> None:
    arm_start = data.ctrl[_ARM_CTRL].copy()
    grip_start = data.ctrl[_GRIP_CTRL].copy()
    arm_target = np.asarray(arm_target, dtype=float)
    grip_target = np.asarray(grip_target, dtype=float)
    for i in range(steps):
        alpha = (i + 1) / steps
        data.ctrl[_ARM_CTRL] = arm_start + alpha * (arm_target - arm_start)
        data.ctrl[_GRIP_CTRL] = grip_start + alpha * (grip_target - grip_start)
        mujoco.mj_step(model, data)
        if carry is not None:
            carry()
        clock.tick()
        render.step()


def _hold(model, data, render, clock, grip_target, steps, carry=None) -> None:
    grip_start = data.ctrl[_GRIP_CTRL].copy()
    grip_target = np.asarray(grip_target, dtype=float)
    for i in range(steps):
        alpha = (i + 1) / steps
        data.ctrl[_GRIP_CTRL] = grip_start + alpha * (grip_target - grip_start)
        mujoco.mj_step(model, data)
        if carry is not None:
            carry()
        clock.tick()
        render.step()


def _make_carry(model, data, ball_body_id, f1_id, f2_id):
    """Snap the ball's freejoint to the fingertip midpoint after each step."""
    joint_id = model.body_jntadr[ball_body_id]
    qpos_adr = model.jnt_qposadr[joint_id]
    dof_adr = model.jnt_dofadr[joint_id]

    def carry() -> None:
        mid = (data.xpos[f1_id] + data.xpos[f2_id]) / 2
        data.qpos[qpos_adr : qpos_adr + 3] = mid
        data.qvel[dof_adr : dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)

    return carry


def run_demo(prefer_gl: str = "egl", scene_path: str | None = None) -> None:
    """Run the stationlite pick-and-place demo in an interactive viewer."""
    configure_gl(prefer_gl)

    path = scene_path or str(_DEFAULT_SCENE)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    f1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_griperlj_link1")
    f2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_griperlj_link2")
    carry = _make_carry(model, data, ball_id, f1_id, f2_id)
    clock = _RealtimeClock(dt=model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        render = _ThrottledSync(viewer, model)

        _move_to(model, data, render, clock, _HOVER_PICK, _GRIP_OPEN, 400)
        _move_to(model, data, render, clock, _GRASP, _GRIP_OPEN, 300)
        _hold(model, data, render, clock, _GRIP_CLOSED, 150)
        # carry engages from here - ball rides the fingertip midpoint exactly
        _move_to(model, data, render, clock, _HOVER_PICK, _GRIP_CLOSED, 300, carry=carry)
        _move_to(model, data, render, clock, _HOVER_PLACE, _GRIP_CLOSED, 500, carry=carry)
        _move_to(model, data, render, clock, _PLACE, _GRIP_CLOSED, 300, carry=carry)
        _hold(model, data, render, clock, _GRIP_CLOSED, 100, carry=carry)
        # release: stop carrying, then open - the ball is now on its own again
        _hold(model, data, render, clock, _GRIP_OPEN, 300)
        _move_to(model, data, render, clock, _HOVER_PLACE, _GRIP_OPEN, 300)
        _move_to(model, data, render, clock, _HOVER_PICK, _GRIP_OPEN, 400)

        ball_final = data.xpos[ball_id].copy()
        err = np.linalg.norm(ball_final[:2] - _TARGET_XY)
        print(f"[mjrobots] ball final xyz={ball_final}, {err * 100:.1f} cm from target (xy)")
        print("[mjrobots] done - close the window to exit")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            clock.tick()
            render.step()
