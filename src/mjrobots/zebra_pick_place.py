"""Move one zebra Lego brick to the table's middle point, via runtime IK.

Uses cartesian_control's runtime IK (`move_to_point`, `move_to_pose`) for
reaching an arbitrary XYZ point, and holds the brick by finger friction
alone, as a real gripper must: the force-limited fingers squeeze it, and
nothing moves the brick but physics. (The older `_make_carry` /
`_make_anchor` in stationlite_pick_place.py teleported the brick to the
fingers every step instead; they are no longer used here. What made the
friction grip hold was finer physics - see sim_step.py.)

`zebra_legs` sits at x=0.2, y=-0.15 - a different table position than the
old block demo ever used, and one the old hand-found joint-angle waypoints
were never solved for - so this only works with the newer runtime-IK
approach, not the older hardcoded ones.
"""

from __future__ import annotations

import numpy as np
import mujoco
import mujoco.viewer

from . import sim_step
from .cartesian_control import ARM_JOINTS, HAND_LOCAL_OFFSET, move_to_point, move_to_pose, _hand_point_and_jac
from .pick_place import _RealtimeClock, _ThrottledSync
from .stationlite_pick_place import (
    _DEFAULT_SCENE,
    _GRIP_OPEN,
    _GRIP_CLOSED,
    _Arm,
    _hold,
)

# The brick body's geometric center sits 1.92cm below its body origin: the
# origin is the top surface (where the studs start) and the body is 3.84cm
# tall (collision box pos="0 0 -0.0192" in the XML - see
# stationlite_pick_place.xml's zebra-piece comment).
_BRICK_CENTER_OFFSET_Z = -0.0192
# Height of one DUPLO 2x4x2 brick body, studs excluded: stacked bricks' origins
# are exactly this far apart.
BRICK_HEIGHT = 0.0384
_TABLE_PLACE_XYZ = np.array([0.4148, 0.0, -0.1216])  # target_site, table height
_HOVER_DZ = 0.10  # scanned reachable across the whole grasp/place workspace
# The brick's narrow side - the jaws close across it (collision box half-size
# 0.016 in the XML; all three parts are the same 6.4 x 3.2 cm brick).
BRICK_WIDTH = 0.032
# The grasp check (see grasp_part) passes if the fingers stop within this of
# BRICK_WIDTH. Measured: fingers stop at 3.13-3.14 cm on a brick, and close
# to 0.00 cm on air - nothing lands in between.
_GRIP_WIDTH_TOL = 0.005


class ZebraArmContext:
    """Everything `grasp_part`/`place_part` need for one arm + one brick, set
    up once so repeated pick/place calls (e.g. from a ROS2 command bridge)
    don't redo the grip-orientation probe each time."""

    def __init__(self, model, data, arm: str, brick_body_name: str = "zebra_legs"):
        self.model = model
        self.data = data
        self.arm = arm
        self.joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS[arm]]
        )
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_linkgripper")
        ctrl_offset = 0 if arm == "left" else 8
        self.arm_ctrl = slice(ctrl_offset, ctrl_offset + 6)
        self.brick_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, brick_body_name)
        self.this_arm = _Arm(ctrl_offset=ctrl_offset, block_id=self.brick_id)
        self._finger_joints = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_gripper_joint{i}") for i in (1, 2)
        ]
        self.grasp_width: float | None = None  # finger gap at the last grasp check

        # See grasp_part()'s docstring for why this fixed orientation (not
        # position-only IK) is used for the grasp approach specifically.
        qpos_adr = model.jnt_qposadr[self.joint_ids]
        saved_qpos = data.qpos.copy()
        data.qpos[qpos_adr] = [0.0, 2.23, -1.215, 0.0, 0.0, 0.0]
        mujoco.mj_kinematics(model, data)
        self.grip_quat = data.xquat[self.body_id].copy()
        data.qpos[:] = saved_qpos
        mujoco.mj_forward(model, data)

    def grip_width(self) -> float:
        """Current gap between the two fingers, in metres - what a real
        gripper reports from its own encoder, so safe to act on."""
        q1 = self.data.qpos[self.model.jnt_qposadr[self._finger_joints[0]]]
        q2 = self.data.qpos[self.model.jnt_qposadr[self._finger_joints[1]]]
        return float(q2 - q1)

    def grip_miss(self) -> float:
        """Distance between the gripper's grasp point and the brick's center.

        Sim-only diagnostic (for logs): it reads the brick's true position,
        which a real robot can't know - grasp_part decides with grip_width.
        """
        xmat = self.data.xmat[self.body_id].reshape(3, 3)
        grip_point = self.data.xpos[self.body_id] + xmat @ HAND_LOCAL_OFFSET
        brick_center = self.data.xpos[self.brick_id] + np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
        return float(np.linalg.norm(grip_point - brick_center))

    def go(self, render, clock, target):
        move_to_point(
            self.model, self.data, render, clock, self.arm_ctrl, self.body_id,
            HAND_LOCAL_OFFSET, self.joint_ids, target,
        )

    def go_oriented(self, render, clock, target):
        move_to_pose(
            self.model, self.data, render, clock, self.arm_ctrl, self.body_id,
            HAND_LOCAL_OFFSET, self.joint_ids, target, self.grip_quat,
        )


def _close_and_settle(ctx: ZebraArmContext, render, clock, max_extra_steps: int = 200) -> None:
    """Close the gripper, then keep it closed until the fingers stop moving
    (under 0.2 mm in 10 control ticks) - like waiting for a real gripper to
    stall before reading its width. On air the fingers are still ~0.8 cm
    apart when the 150-tick close ramp ends and meet ~50 ticks later."""
    _hold(ctx.model, ctx.data, render, clock, ctx.this_arm, _GRIP_CLOSED, 150)
    last = ctx.grip_width()
    for _ in range(max_extra_steps // 10):
        _hold(ctx.model, ctx.data, render, clock, ctx.this_arm, _GRIP_CLOSED, 10)
        width = ctx.grip_width()
        if abs(width - last) < 0.0002:
            return
        last = width


def grasp_part(ctx: ZebraArmContext, render, clock, center_xyz) -> None:
    """Approach, descend onto, and grip the brick at `center_xyz` (its
    geometric center, not its body origin - see `_BRICK_CENTER_OFFSET_Z`),
    then lift it clear of the table.

    Uses `move_to_pose` (fixed grip orientation - `ctx.grip_quat`, found once
    at a known-good pose) for the approach/descend/lift, not position-only
    `move_to_point`: position-only IK leaves the wrist wherever it happens to
    converge, fine for transport but not for reliably closing the gripper
    around something. Forcing this same orientation over a *long* reach
    doesn't work either (tested: every joint pins to its limit, ~1m position
    error, since the reachable orientation rotates with the arm's own swing
    angle) - it's used only for this short grasp-approach range. So the arm
    travels to the hover point position-only, turns its wrist into the grip
    orientation there (a zero-length oriented move: all lead-in), and only
    then descends oriented.

    Whether the grasp worked is judged the way a real gripper can: by how
    far apart the fingers stopped (`grip_width`), not by where the sim says
    the brick is. The gripper closes with a force limit, so on a brick the
    fingers stall at its width; on air they close all the way.
    """
    hover_xyz = center_xyz + np.array([0, 0, _HOVER_DZ])
    ctx.go(render, clock, hover_xyz)
    ctx.go_oriented(render, clock, hover_xyz)
    ctx.go_oriented(render, clock, center_xyz)
    _close_and_settle(ctx, render, clock)

    width = ctx.grasp_width = ctx.grip_width()
    if abs(width - BRICK_WIDTH) > _GRIP_WIDTH_TOL:
        _hold(ctx.model, ctx.data, render, clock, ctx.this_arm, _GRIP_OPEN, 100)
        ctx.go(render, clock, hover_xyz)
        what = "nothing" if width < BRICK_WIDTH else "something too wide"
        raise RuntimeError(
            f"missed the brick: fingers closed to {width * 100:.1f} cm on {what} "
            f"(brick is {BRICK_WIDTH * 100:.1f} cm)"
        )

    # Lift - the brick comes along only because the fingers are gripping it.
    ctx.go(render, clock, hover_xyz)


def place_part(ctx: ZebraArmContext, render, clock, center_xyz) -> None:
    """Carry the gripped brick to `center_xyz` (geometric center), set it
    down, and let go - no teleport: where the brick ends up is where physics
    leaves it when the fingers open.

    Mirrors grasp_part's approach: travel to the hover point position-only,
    turn the wrist back into the grip orientation there, then descend
    oriented. The brick sits in the fingers the way it was picked up, so
    arriving in the grip orientation puts it down level and square (the grip
    orientation is reachable at all three stack levels, both arms) - a
    position-only descent would set it down at whatever angle IK left the
    wrist, measured ~20-27 deg off.
    """
    hover_xyz = center_xyz + np.array([0, 0, _HOVER_DZ])
    ctx.go(render, clock, hover_xyz)
    ctx.go_oriented(render, clock, hover_xyz)
    ctx.go_oriented(render, clock, center_xyz)
    _hold(ctx.model, ctx.data, render, clock, ctx.this_arm, _GRIP_OPEN, 300)
    ctx.go(render, clock, hover_xyz)


def run_demo(prefer_gl: str = "egl", scene_path: str | None = None, arm: str = "right") -> None:
    """Move `zebra_legs` from its spawn spot to the table's middle point."""
    from .gl import configure_gl

    configure_gl(prefer_gl)

    path = scene_path or str(_DEFAULT_SCENE)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    ctx = ZebraArmContext(model, data, arm)
    brick_center_z = data.xpos[ctx.brick_id][2] + _BRICK_CENTER_OFFSET_Z
    grasp_xyz = np.array([data.xpos[ctx.brick_id][0], data.xpos[ctx.brick_id][1], brick_center_z])
    place_xyz = np.array([_TABLE_PLACE_XYZ[0], _TABLE_PLACE_XYZ[1], brick_center_z])

    clock = _RealtimeClock(dt=sim_step.CONTROL_DT)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        render = _ThrottledSync(viewer, model, step_dt=sim_step.CONTROL_DT)

        grasp_part(ctx, render, clock, grasp_xyz)
        place_part(ctx, render, clock, place_xyz)

        final = data.xpos[ctx.brick_id].copy()
        err = np.linalg.norm(final[:2] - place_xyz[:2])
        print(f"[mjrobots] zebra_legs placed at {final}, {err * 100:.2f} cm lateral error from target")
        print("[mjrobots] done - close the window to exit")
        while viewer.is_running():
            sim_step.step(model, data)
            clock.tick()
            render.step()
