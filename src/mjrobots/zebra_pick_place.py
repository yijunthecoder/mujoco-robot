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


def run_demo(prefer_gl: str = "egl", scene_path: str | None = None, arm: str = "right") -> None:
    """Move `zebra_legs` from its spawn spot to the table's middle point."""
    from .gl import configure_gl

    configure_gl(prefer_gl)

    path = scene_path or str(_DEFAULT_SCENE)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS[arm]])
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_linkgripper")
    ctrl_offset = 0 if arm == "left" else 8
    arm_ctrl = slice(ctrl_offset, ctrl_offset + 6)
    grip_ctrl = slice(ctrl_offset + 6, ctrl_offset + 8)

    brick_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "zebra_legs")
    this_arm = _Arm(ctrl_offset=ctrl_offset, block_id=brick_id)
    brick_center_z = data.xpos[brick_id][2] + _BRICK_CENTER_OFFSET_Z
    grasp_xyz = np.array([data.xpos[brick_id][0], data.xpos[brick_id][1], brick_center_z])
    hover_grasp_xyz = grasp_xyz + np.array([0, 0, _HOVER_DZ])
    place_xyz = np.array([_TABLE_PLACE_XYZ[0], _TABLE_PLACE_XYZ[1], brick_center_z])
    hover_place_xyz = place_xyz + np.array([0, 0, _HOVER_DZ])

    # The grip orientation used at the ORIGINAL stationlite_pick_place.py
    # demo's own grasp waypoint (j1=0, j2=2.23, j3=-1.215, wrist joints 0) -
    # the only orientation confirmed, by direct testing, to both converge
    # cleanly AND actually grip well at a nearby table position. Position-
    # only IK (move_to_point, used for the rest of this sequence) leaves the
    # wrist wherever it happens to converge, which is fine for transport but
    # not for reliably closing the gripper - and forcing this SAME
    # orientation across the whole reach to the middle of the table doesn't
    # work either (tested: every joint pins to its limit, ~1m position
    # error) since the reachable orientation rotates with the arm's own
    # swing angle. So it's used only for the grasp approach itself.
    qpos_adr = model.jnt_qposadr[joint_ids]
    saved_qpos = data.qpos.copy()
    data.qpos[qpos_adr] = [0.0, 2.23, -1.215, 0.0, 0.0, 0.0]
    mujoco.mj_kinematics(model, data)
    grip_quat = data.xquat[body_id].copy()
    data.qpos[:] = saved_qpos
    mujoco.mj_forward(model, data)

    carry = _make_carry(model, data, brick_id, f"{arm}_griperlj_link1", f"{arm}_griperlj_link2")

    clock = _RealtimeClock(dt=model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        render = _ThrottledSync(viewer, model)

        def go(target, carry_fn=None):
            move_to_point(model, data, render, clock, arm_ctrl, body_id, HAND_LOCAL_OFFSET, joint_ids, target, carry=carry_fn)

        def go_oriented(target, carry_fn=None):
            move_to_pose(
                model, data, render, clock, arm_ctrl, body_id, HAND_LOCAL_OFFSET, joint_ids,
                target, grip_quat, carry=carry_fn,
            )

        # Approach and descend at the PROVEN grip orientation (not
        # position-only IK - see grip_quat's comment above), close, lift.
        go_oriented(hover_grasp_xyz)
        go_oriented(grasp_xyz)
        _hold(model, data, render, clock, this_arm, _GRIP_CLOSED, 150)
        go(hover_grasp_xyz, carry_fn=carry)

        # Carry to the middle, descend, release via anchor (not a raw
        # handoff to physics - see stationlite_pick_place.py's docstring for
        # why the anchor step matters for placement accuracy).
        go(hover_place_xyz, carry_fn=carry)
        go(place_xyz, carry_fn=carry)
        _hold(model, data, render, clock, this_arm, _GRIP_CLOSED, 100, carry=carry)

        # _make_anchor pins the body's ORIGIN (qpos), not its geometric
        # center - and this brick's origin sits 1.7cm above its center (see
        # _BRICK_CENTER_OFFSET_Z). Anchoring at place_xyz (a center-height
        # coordinate) directly would wedge the brick 1.7cm into the table -
        # confirmed by testing: it held fine during the anchor (which forces
        # the position every step regardless of penetration) then popped
        # back out the moment the anchor released and real contact physics
        # took over. Anchor target must be the origin-equivalent instead.
        anchor_origin_target = place_xyz - np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
        anchor = _make_anchor(model, data, brick_id, anchor_origin_target)
        _hold(model, data, render, clock, this_arm, _GRIP_OPEN, 300, carry=anchor)
        go(hover_place_xyz)

        final = data.xpos[brick_id].copy()
        err = np.linalg.norm(final[:2] - place_xyz[:2])
        print(f"[mjrobots] zebra_legs placed at {final}, {err * 100:.2f} cm lateral error from target")
        print("[mjrobots] done - close the window to exit")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            clock.tick()
            render.step()
