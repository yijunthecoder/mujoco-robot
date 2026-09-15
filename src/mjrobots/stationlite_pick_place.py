"""Two-arm block-stacking demo for the stationlite dual-arm robot.

Hardcoded choreography (no coordination logic, order is fixed):
  1. The RIGHT arm picks up its own block and places it at the middle
     point, resting on the table - the bottom block.
  2. The LEFT arm picks up its own block and places it on top of the
     right arm's block, at the same XY, one block-height higher.

Waypoints are joint angles found by an offline forward-kinematics search
against stationlite_mujoco.urdf, not solved via runtime IK (see
pick_place.py's Panda demo for that approach) - this arm's kinematics/reach
weren't established going in, so hardcoded, pre-verified waypoints are used
instead. The two arms turned out to be exactly mirror-symmetric (confirmed
by search: the same joint values reach the mirrored (x,-y,z) point on either
side), so the right arm's numbers are just the left arm's with j1 negated.

Grasping is a kinematic "carry" (see `_make_carry`), not pure friction:
this gripper's mesh-only finger geometry (no real contact pads) can't
reliably hold an object through arm motion - confirmed experimentally on
the earlier single-block version, where it slipped/launched regardless of
grip force or speed. Instead, once a gripper closes on its block, that
block's position is snapped to the fingertip midpoint every step until
release, then handed back to real physics.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .gl import configure_gl
from .pick_place import _RealtimeClock, _ThrottledSync

_DEFAULT_SCENE = (
    Path(__file__).resolve().parent.parent.parent / "stationlite" / "urdf" / "stationlite_pick_place.xml"
)

_GRIP_OPEN = (-0.0425, 0.0425)
_GRIP_CLOSED = (-0.0425 * 0.4, 0.0425 * 0.4)  # visually close around the block

# left_joint1..6 / right_joint1..6, found by FK search against
# stationlite_mujoco.urdf.
_J1_MID = 0.653346  # left uses -_J1_MID, right uses +_J1_MID to reach the middle


def _wp(j1: float, j2: float, j3: float) -> list[float]:
    return [j1, j2, j3, 0.0, 0.0, 0.0]


_GRASP_START = _wp(0.0, 2.23, -1.215)  # each arm's own block, at its own side
_HOVER_START = _wp(0.0, 1.98, -1.37)

_HOVER_MID_RIGHT = _wp(_J1_MID, 1.98, -1.37)
_PLACE_MID_RIGHT = _wp(_J1_MID, 2.23, -1.215)  # bottom block: rests on the table

_HOVER_MID_LEFT = _wp(-_J1_MID, 1.98, -1.37)
_STACK_MID_LEFT = _wp(-_J1_MID, 2.1080808, -1.1545455)  # top block: one block-height up


class _Arm:
    """ctrl-index bookkeeping for one arm, given its actuator block offset."""

    def __init__(self, ctrl_offset: int):
        self.arm_ctrl = slice(ctrl_offset, ctrl_offset + 6)
        self.grip_ctrl = slice(ctrl_offset + 6, ctrl_offset + 8)


def _move_to(model, data, render, clock, arm, arm_target, grip_target, steps, carry=None) -> None:
    arm_start = data.ctrl[arm.arm_ctrl].copy()
    grip_start = data.ctrl[arm.grip_ctrl].copy()
    arm_target = np.asarray(arm_target, dtype=float)
    grip_target = np.asarray(grip_target, dtype=float)
    for i in range(steps):
        alpha = (i + 1) / steps
        data.ctrl[arm.arm_ctrl] = arm_start + alpha * (arm_target - arm_start)
        data.ctrl[arm.grip_ctrl] = grip_start + alpha * (grip_target - grip_start)
        mujoco.mj_step(model, data)
        if carry is not None:
            carry()
        clock.tick()
        render.step()


def _hold(model, data, render, clock, arm, grip_target, steps, carry=None) -> None:
    grip_start = data.ctrl[arm.grip_ctrl].copy()
    grip_target = np.asarray(grip_target, dtype=float)
    for i in range(steps):
        alpha = (i + 1) / steps
        data.ctrl[arm.grip_ctrl] = grip_start + alpha * (grip_target - grip_start)
        mujoco.mj_step(model, data)
        if carry is not None:
            carry()
        clock.tick()
        render.step()


def _make_carry(model, data, block_body_id, f1_name, f2_name):
    """Snap `block_body_id`'s freejoint to the named fingers' midpoint each step."""
    f1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f1_name)
    f2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f2_name)
    joint_id = model.body_jntadr[block_body_id]
    qpos_adr = model.jnt_qposadr[joint_id]
    dof_adr = model.jnt_dofadr[joint_id]

    def carry() -> None:
        mid = (data.xpos[f1_id] + data.xpos[f2_id]) / 2
        data.qpos[qpos_adr : qpos_adr + 3] = mid
        data.qvel[dof_adr : dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)

    return carry


def _pick_and_place(model, data, render, clock, arm, carry, hover_mid, place_mid) -> None:
    """One arm's full cycle: grasp its own block, carry it to `place_mid`, release."""
    _move_to(model, data, render, clock, arm, _GRASP_START, _GRIP_OPEN, 300)
    _hold(model, data, render, clock, arm, _GRIP_CLOSED, 150)
    _move_to(model, data, render, clock, arm, _HOVER_START, _GRIP_CLOSED, 300, carry=carry)
    _move_to(model, data, render, clock, arm, hover_mid, _GRIP_CLOSED, 500, carry=carry)
    _move_to(model, data, render, clock, arm, place_mid, _GRIP_CLOSED, 300, carry=carry)
    _hold(model, data, render, clock, arm, _GRIP_CLOSED, 100, carry=carry)  # settle before release
    _hold(model, data, render, clock, arm, _GRIP_OPEN, 300)
    _move_to(model, data, render, clock, arm, hover_mid, _GRIP_OPEN, 300)
    _move_to(model, data, render, clock, arm, _HOVER_START, _GRIP_OPEN, 300)


def run_demo(prefer_gl: str = "egl", scene_path: str | None = None) -> None:
    """Run the stationlite two-arm block-stacking demo in an interactive viewer."""
    configure_gl(prefer_gl)

    path = scene_path or str(_DEFAULT_SCENE)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    left = _Arm(ctrl_offset=0)
    right = _Arm(ctrl_offset=8)

    block_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block_right")
    block_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block_left")
    carry_right = _make_carry(model, data, block_right_id, "right_griperlj_link1", "right_griperlj_link2")
    carry_left = _make_carry(model, data, block_left_id, "left_griperlj_link1", "left_griperlj_link2")

    clock = _RealtimeClock(dt=model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        render = _ThrottledSync(viewer, model)

        _pick_and_place(model, data, render, clock, right, carry_right, _HOVER_MID_RIGHT, _PLACE_MID_RIGHT)
        bottom_final = data.xpos[block_right_id].copy()
        print(f"[mjrobots] bottom block (right arm) placed at {bottom_final}")

        _pick_and_place(model, data, render, clock, left, carry_left, _HOVER_MID_LEFT, _STACK_MID_LEFT)
        top_final = data.xpos[block_left_id].copy()

        lateral_err = np.linalg.norm(top_final[:2] - bottom_final[:2])
        rise = top_final[2] - bottom_final[2]
        print(
            f"[mjrobots] top block (left arm) at {top_final}, "
            f"{lateral_err * 100:.1f} cm lateral offset, {rise * 100:.1f} cm above bottom block "
            f"(stacked ok: {lateral_err < 0.02 and abs(rise - 0.04) < 0.015})"
        )
        print("[mjrobots] done - close the window to exit")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            clock.tick()
            render.step()
