"""Cartesian control for the stationlite arms: given a target 3D point,
smoothly drive one arm's gripper there using inverse kinematics.

Two things separate this from stationlite_pick_place.py's choreography:

1. Runtime IK instead of hardcoded waypoints. stationlite_pick_place.py
   uses joint angles found by an offline FK search, because the arm's
   kinematics weren't established going in. They are now (this module *is*
   that IK), so any reachable point can be targeted directly.

2. Cartesian-space interpolation, not just joint-space. Ramping ctrl
   linearly from the current joint angles straight to one IK solution (as
   pick_place.py's `_move_to` does) gives smooth *joint* motion, but the
   hand's path through space is whatever curve that happens to produce.
   `move_to_point` instead interpolates a straight line of waypoints from
   the current hand position to the target and re-solves IK at each one
   (warm-started by the arm's own current pose), so the hand itself moves
   in something close to a straight line.

Each arm's "hand" point is the fixed attachment point where the two gripper
fingers meet (local offset (0.1733, 0, 0) from the `<side>_linkgripper`
body) - the same fingertip-midpoint convention stationlite_pick_place.py's
`_make_carry` computes at runtime from the finger bodies. Using the fixed
attachment point instead avoids needing the fingers' own (slide-jointed,
symmetric) Jacobians.

IK here solves position only (3 error dims against 6 joints per arm) - the
task is "reach this point," not "reach this point at this orientation," and
the arm has more freedom than a 3D constraint needs. Damped least squares
naturally returns a minimum-motion solution among the valid ones.
"""

from __future__ import annotations

import numpy as np
import mujoco

from .pick_place import _RealtimeClock, _ThrottledSync

# Local-frame offset from "<side>_linkgripper" to the fingertip midpoint,
# matching left_griperlj_link1/2's attachment point in stationlite_converted.xml.
HAND_LOCAL_OFFSET = np.array([0.1733, 0.0, 0.0])

ARM_JOINTS = {
    "left": [f"left_joint{i}" for i in range(1, 7)],
    "right": [f"right_joint{i}" for i in range(1, 7)],
}


def _hand_point_and_jac(model, data, body_id, local_offset):
    """World position of the point rigidly attached to body_id at
    local_offset, and its 3 x nv Jacobian."""
    xmat = data.xmat[body_id].reshape(3, 3)
    point = data.xpos[body_id] + xmat @ local_offset
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
    return point, jacp


def solve_ik(
    model,
    data,
    body_id: int,
    local_offset: np.ndarray,
    joint_ids: np.ndarray,
    target_pos: np.ndarray,
    iters: int = 200,
    damping: float = 0.1,
    tol: float = 1e-4,
    max_step: float = 0.2,
) -> np.ndarray:
    """Damped-least-squares IK for a point rigidly attached to body_id.

    Solves for the angles of `joint_ids` (each assumed 1-dof) that bring the
    point at `local_offset` in body_id's local frame to `target_pos` in
    world space. Starts from data's current joint angles (so calling this
    repeatedly with a slowly-moving target warm-starts each solve from the
    last one) and operates on a scratch copy of qpos/qvel - the real
    simulation state is restored before returning.
    """
    qpos_adr = model.jnt_qposadr[joint_ids]
    dof_adr = model.jnt_dofadr[joint_ids]
    lo = model.jnt_range[joint_ids, 0]
    hi = model.jnt_range[joint_ids, 1]

    qpos_save = data.qpos.copy()
    qvel_save = data.qvel.copy()
    q = data.qpos[qpos_adr].copy()

    try:
        for _ in range(iters):
            data.qpos[qpos_adr] = q
            mujoco.mj_kinematics(model, data)

            point, jacp = _hand_point_and_jac(model, data, body_id, local_offset)
            err = target_pos - point
            if np.linalg.norm(err) < tol:
                break

            J = jacp[:, dof_adr]
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + damping**2 * np.eye(3), err)
            dq = np.clip(dq, -max_step, max_step)
            q = np.clip(q + dq, lo, hi)
    finally:
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        mujoco.mj_kinematics(model, data)

    return q


def solve_ik_pose(
    model,
    data,
    body_id: int,
    local_offset: np.ndarray,
    joint_ids: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    iters: int = 400,
    damping: float = 0.1,
    tol: float = 1e-4,
    max_step: float = 0.15,
) -> np.ndarray:
    """Damped-least-squares IK for a point AND orientation, both attached to
    body_id - unlike `solve_ik`, which only constrains position.

    Needed specifically for grasping: `solve_ik`'s spare degrees of freedom
    (3 position constraints, 6 joints) let the wrist land in whatever
    orientation the solver happens to converge to, which is fine for merely
    moving the hand somewhere but not for reliably closing the gripper
    around an object - the approach angle matters. This arm's reachable
    orientation is narrow and rotates with the arm's own swing (confirmed:
    a top-down or clean-side target orientation fails to converge at all at
    the grasp's table position - every joint pins to its limit; the SAME
    orientation that grips well at the grasp point also fails completely at
    a distant swing angle like the place point). Use this only where the
    approach angle actually matters (the grasp itself), not throughout a
    whole transport move.
    """
    qpos_adr = model.jnt_qposadr[joint_ids]
    dof_adr = model.jnt_dofadr[joint_ids]
    lo = model.jnt_range[joint_ids, 0]
    hi = model.jnt_range[joint_ids, 1]

    q = data.qpos[qpos_adr].copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        data.qpos[qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        point, jacp = _hand_point_and_jac(model, data, body_id, local_offset)
        pos_err = target_pos - point

        cur_quat = data.xquat[body_id]
        neg_cur = np.empty(4)
        mujoco.mju_negQuat(neg_cur, cur_quat)
        err_quat = np.empty(4)
        mujoco.mju_mulQuat(err_quat, target_quat, neg_cur)
        rot_err = np.empty(3)
        mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)

        err = np.concatenate([pos_err, rot_err])
        if np.linalg.norm(err) < tol:
            break

        mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
        J = np.vstack([jacp[:, dof_adr], jacr[:, dof_adr]])
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
        dq = np.clip(dq, -max_step, max_step)
        q = np.clip(q + dq, lo, hi)

    return q


def move_to_pose(
    model,
    data,
    render,
    clock,
    arm_ctrl_slice: slice,
    body_id: int,
    local_offset: np.ndarray,
    joint_ids: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    steps: int = 600,
    waypoints: int = 30,
    settle_steps: int = 150,
    carry=None,
) -> None:
    """Like `move_to_point`, but also holds the gripper at `target_quat`
    throughout - see `solve_ik_pose`. The orientation target is held fixed
    across all waypoints (only position is interpolated); this is meant for
    short, single-purpose moves like the final grasp approach, not a long
    transport where forcing one fixed orientation the whole way may not
    even be reachable.
    """
    mujoco.mj_kinematics(model, data)
    start_point, _ = _hand_point_and_jac(model, data, body_id, local_offset)

    steps_per_wp = max(1, steps // waypoints)
    q = data.qpos[model.jnt_qposadr[joint_ids]].copy()
    for i in range(1, waypoints + 1):
        alpha = i / waypoints
        wp_target = start_point + alpha * (target_pos - start_point)
        q = solve_ik_pose(model, data, body_id, local_offset, joint_ids, wp_target, target_quat)

        ctrl_start = data.ctrl[arm_ctrl_slice].copy()
        for s in range(steps_per_wp):
            a = (s + 1) / steps_per_wp
            data.ctrl[arm_ctrl_slice] = ctrl_start + a * (q - ctrl_start)
            mujoco.mj_step(model, data)
            if carry is not None:
                carry()
            if render is not None:
                clock.tick()
                render.step()

    data.ctrl[arm_ctrl_slice] = q
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        if carry is not None:
            carry()
        if render is not None:
            clock.tick()
            render.step()


def move_to_point(
    model,
    data,
    render,
    clock,
    arm_ctrl_slice: slice,
    body_id: int,
    local_offset: np.ndarray,
    joint_ids: np.ndarray,
    target_pos: np.ndarray,
    steps: int = 600,
    waypoints: int = 30,
    settle_steps: int = 150,
    carry=None,
) -> None:
    """Smoothly drive the hand point to `target_pos`.

    Interpolates `waypoints` points on the straight line from the hand's
    current position to the target, solves IK at each (warm-started from
    wherever the arm physically is by that point), and ramps `data.ctrl`
    toward each solution in turn while stepping physics - so motion stays
    smooth (PD-servoed, not teleported) and the hand's own path stays close
    to a straight line rather than whatever a single joint-space ramp would
    trace out.

    The PD servos lag a fast-moving ctrl target, so a few cm of tracking
    error remains right after the last waypoint; `settle_steps` holds ctrl
    at the final solution so the arm actually catches up before returning.

    `carry`, if given, is called after every `mj_step` - same convention as
    stationlite_pick_place.py's `_move_to`/`_make_carry`: a no-argument
    callback that snaps a held object's freejoint to the gripper each step,
    since this arm can't hold anything through contact/friction alone.
    """
    mujoco.mj_kinematics(model, data)
    start_point, _ = _hand_point_and_jac(model, data, body_id, local_offset)

    steps_per_wp = max(1, steps // waypoints)
    q = data.qpos[model.jnt_qposadr[joint_ids]].copy()
    for i in range(1, waypoints + 1):
        alpha = i / waypoints
        wp_target = start_point + alpha * (target_pos - start_point)
        q = solve_ik(model, data, body_id, local_offset, joint_ids, wp_target)

        ctrl_start = data.ctrl[arm_ctrl_slice].copy()
        for s in range(steps_per_wp):
            a = (s + 1) / steps_per_wp
            data.ctrl[arm_ctrl_slice] = ctrl_start + a * (q - ctrl_start)
            mujoco.mj_step(model, data)
            if carry is not None:
                carry()
            if render is not None:
                clock.tick()
                render.step()

    data.ctrl[arm_ctrl_slice] = q
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        if carry is not None:
            carry()
        if render is not None:
            clock.tick()
            render.step()


def run_demo(
    prefer_gl: str = "egl",
    scene_path: str | None = None,
    arm: str = "left",
    target: tuple[float, float, float] = (0.4148, 0.0, -0.1216),
) -> None:
    """Move one stationlite arm's gripper to `target` and hold, in the viewer."""
    from .gl import configure_gl
    from .stationlite_pick_place import _DEFAULT_SCENE

    configure_gl(prefer_gl)

    path = scene_path or str(_DEFAULT_SCENE)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS[arm]]
    )
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_linkgripper")
    ctrl_offset = 0 if arm == "left" else 8
    arm_ctrl_slice = slice(ctrl_offset, ctrl_offset + 6)
    target_pos = np.array(target, dtype=float)

    clock = _RealtimeClock(dt=model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        render = _ThrottledSync(viewer, model)

        # Small red marker at the target point, for visual sanity-checking.
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0, 0],
            pos=target_pos,
            mat=np.eye(3).flatten(),
            rgba=[1, 0, 0, 0.6],
        )
        viewer.sync()

        move_to_point(model, data, render, clock, arm_ctrl_slice, body_id, HAND_LOCAL_OFFSET, joint_ids, target_pos)

        hand_final, _ = _hand_point_and_jac(model, data, body_id, HAND_LOCAL_OFFSET)
        error = np.linalg.norm(hand_final - target_pos)
        print(f"[mjrobots] {arm} hand at {hand_final}, {error * 100:.2f} cm from target {target_pos}")

        print("[mjrobots] done - close the window to exit")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            clock.tick()
            render.step()
