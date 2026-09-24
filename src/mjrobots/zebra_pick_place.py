"""Move one zebra Lego brick to the table's middle point, via runtime IK.

Combines two things that existed separately before this: cartesian_control's
runtime IK (`move_to_point`, `solve_ik`) for reaching an arbitrary XYZ point,
and stationlite_pick_place's kinematic grasp mechanism (`_make_carry`,
`_make_anchor`) for actually holding an object once the gripper closes -
this gripper's mesh-only fingers can't hold anything through friction alone
(see stationlite_pick_place.py's module docstring for why).

`zebra_legs` sits at x=0.2, y=-0.15 - a different table position than the
old block demo ever used, and one the old hand-found joint-angle waypoints
were never solved for - so this only works with the newer runtime-IK
approach, not the older hardcoded ones.
"""

from __future__ import annotations

import numpy as np
import mujoco
import mujoco.viewer

from .cartesian_control import ARM_JOINTS, HAND_LOCAL_OFFSET, move_to_point, move_to_pose, _hand_point_and_jac
from .pick_place import _RealtimeClock, _ThrottledSync
from .stationlite_pick_place import (
    _DEFAULT_SCENE,
    _GRIP_OPEN,
    _GRIP_CLOSED,
    _Arm,
    _hold,
    _make_anchor,
    _make_carry,
)

# The brick's geometric center sits 1.7cm below its body origin (the
# collision box's local pos="0 0 -0.017" in the XML) - see
# stationlite_pick_place.xml's zebra-piece comment for the full derivation.
_BRICK_CENTER_OFFSET_Z = -0.017
_TABLE_PLACE_XYZ = np.array([0.4148, 0.0, -0.1216])  # target_site, table height
_HOVER_DZ = 0.10  # scanned reachable across the whole grasp/place workspace


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
        self.carry = self.new_carry()

        # See grasp_part()'s docstring for why this fixed orientation (not
        # position-only IK) is used for the grasp approach specifically.
        qpos_adr = model.jnt_qposadr[self.joint_ids]
        saved_qpos = data.qpos.copy()
        data.qpos[qpos_adr] = [0.0, 2.23, -1.215, 0.0, 0.0, 0.0]
        mujoco.mj_kinematics(model, data)
        self.grip_quat = data.xquat[self.body_id].copy()
        data.qpos[:] = saved_qpos
        mujoco.mj_forward(model, data)

    def new_carry(self):
        """A carry that holds the brick where the fingers closed on it (see
        `_make_carry`'s keep_grasp_offset) - one per grasp, since the offset
        is captured on its first call."""
        return _make_carry(
            self.model, self.data, self.brick_id,
            f"{self.arm}_griperlj_link1", f"{self.arm}_griperlj_link2",
            keep_grasp_offset=True,
        )

    def go(self, render, clock, target, carry_fn=None):
        move_to_point(
            self.model, self.data, render, clock, self.arm_ctrl, self.body_id,
            HAND_LOCAL_OFFSET, self.joint_ids, target, carry=carry_fn,
        )

    def go_oriented(self, render, clock, target, carry_fn=None):
        move_to_pose(
            self.model, self.data, render, clock, self.arm_ctrl, self.body_id,
            HAND_LOCAL_OFFSET, self.joint_ids, target, self.grip_quat, carry=carry_fn,
        )


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
    angle) - it's used only for this short grasp-approach range.
    """
    hover_xyz = center_xyz + np.array([0, 0, _HOVER_DZ])
    ctx.go_oriented(render, clock, hover_xyz)
    ctx.go_oriented(render, clock, center_xyz)
    _hold(ctx.model, ctx.data, render, clock, ctx.this_arm, _GRIP_CLOSED, 150)
    ctx.carry = ctx.new_carry()
    ctx.go(render, clock, hover_xyz, carry_fn=ctx.carry)


def place_part(ctx: ZebraArmContext, render, clock, center_xyz) -> None:
    """Carry the already-grasped brick to `center_xyz` (geometric center)
    and release it there via anchor, not a raw handoff to physics - see
    stationlite_pick_place.py's module docstring for why the anchor step
    (pinning to an exact coordinate while the gripper opens) matters for
    placement accuracy over this gripper's friction-only grip.
    """
    hover_xyz = center_xyz + np.array([0, 0, _HOVER_DZ])
    ctx.go(render, clock, hover_xyz, carry_fn=ctx.carry)
    ctx.go(render, clock, center_xyz, carry_fn=ctx.carry)
    _hold(ctx.model, ctx.data, render, clock, ctx.this_arm, _GRIP_CLOSED, 100, carry=ctx.carry)

    # _make_anchor pins the body's ORIGIN (qpos), not its geometric center -
    # and this brick's origin sits 1.7cm above its center (see
    # _BRICK_CENTER_OFFSET_Z). Anchoring at center_xyz directly would wedge
    # the brick 1.7cm into the table - confirmed by testing: it held fine
    # during the anchor (which forces the position every step regardless of
    # penetration) then popped back out the moment the anchor released and
    # real contact physics took over. Anchor target must be the
    # origin-equivalent instead.
    anchor_origin_target = center_xyz - np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
    anchor = _make_anchor(ctx.model, ctx.data, ctx.brick_id, anchor_origin_target)
    _hold(ctx.model, ctx.data, render, clock, ctx.this_arm, _GRIP_OPEN, 300, carry=anchor)
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

    clock = _RealtimeClock(dt=model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        render = _ThrottledSync(viewer, model)

        grasp_part(ctx, render, clock, grasp_xyz)
        place_part(ctx, render, clock, place_xyz)

        final = data.xpos[ctx.brick_id].copy()
        err = np.linalg.norm(final[:2] - place_xyz[:2])
        print(f"[mjrobots] zebra_legs placed at {final}, {err * 100:.2f} cm lateral error from target")
        print("[mjrobots] done - close the window to exit")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            clock.tick()
            render.step()
