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

from . import move_check, sim_step
from .pick_place import _RealtimeClock, _ThrottledSync

# Local-frame offset from "<side>_linkgripper" to the fingertip midpoint,
# matching left_griperlj_link1/2's attachment point in stationlite_converted.xml.
HAND_LOCAL_OFFSET = np.array([0.1733, 0.0, 0.0])

ARM_JOINTS = {
    "left": [f"left_joint{i}" for i in range(1, 7)],
    "right": [f"right_joint{i}" for i in range(1, 7)],
}

# Safety limits on IK results, for moving to the real arm. A solve that ends
# further than these from its target did not converge (e.g. an unreachable
# orientation, where every joint pins to a limit), so the move is refused
# rather than sent. A zebra build's solves all land within 0.1 mm / 0.0001 rad.
IK_MAX_POS_ERR = 0.005  # m
IK_MAX_ROT_ERR = 0.05  # rad, ~3 deg
# IK solutions stay this far inside each joint's range, never on the hard
# limit itself (the zebra build's closest approach is 0.25 rad).
JOINT_LIMIT_MARGIN = 0.1  # rad
# Largest joint change allowed from one waypoint to the next, each of which
# gets ~0.04 s - so also a joint speed limit (~5 rad/s). Mid-path, a bigger
# jump means IK flipped to a different arm configuration and the move is
# refused; the lead-in from the arm's current pose is split into steps this
# size instead (see `_plan_path`). The zebra build's mid-path largest is ~0.11.
MAX_WAYPOINT_JUMP = 0.2  # rad


class IKError(RuntimeError):
    """IK could not produce a safe joint target; the arm was not moved."""


def _limits_with_margin(model, joint_ids):
    """Joint ranges shrunk by JOINT_LIMIT_MARGIN on each side (less for a
    range too narrow to spare the full margin)."""
    lo = model.jnt_range[joint_ids, 0]
    hi = model.jnt_range[joint_ids, 1]
    margin = np.minimum(JOINT_LIMIT_MARGIN, 0.25 * (hi - lo))
    return lo + margin, hi - margin


def _hand_point_and_jac(model, data, body_id, local_offset):
    """World position of the point rigidly attached to body_id at
    local_offset, and its 3 x nv Jacobian."""
    xmat = data.xmat[body_id].reshape(3, 3)
    point = data.xpos[body_id] + xmat @ local_offset
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
    return point, jacp


def _rot_err(model, data, body_id, target_quat):
    """World-frame rotation vector taking body_id's current orientation to
    target_quat."""
    neg_cur = np.empty(4)
    mujoco.mju_negQuat(neg_cur, data.xquat[body_id])
    err_quat = np.empty(4)
    mujoco.mju_mulQuat(err_quat, target_quat, neg_cur)
    rot_err = np.empty(3)
    mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)
    return rot_err


def _plan_path(q_start, solve, targets, what: str, lead_in_step: float = MAX_WAYPOINT_JUMP) -> list[np.ndarray]:
    """Solve every waypoint of a move up front and check it, so a bad
    waypoint raises IKError before the arm has moved at all.

    `solve(target, q_init)` returns one waypoint's joint angles. Each solve
    is seeded with the previous solution. Getting from the arm's current
    pose to the first waypoint may take a big joint change (e.g. turning
    the wrist into the grip orientation); that stretch is split into
    joint-space steps of at most `lead_in_step`, so it just takes longer -
    pass a smaller one to turn more gently (e.g. while holding something).
    Past the first waypoint, consecutive solutions must stay within
    MAX_WAYPOINT_JUMP of each other - a bigger jump means IK flipped to a
    different arm configuration mid-path, so the move is refused.
    """
    path = []
    q_prev = np.asarray(q_start, dtype=float)
    for i, target in enumerate(targets):
        q = solve(target, q_prev)
        jump = float(np.abs(q - q_prev).max())
        if i == 0:
            n = int(np.ceil(jump / min(lead_in_step, MAX_WAYPOINT_JUMP)))
            path.extend(q_prev + (k / n) * (q - q_prev) for k in range(1, n))
        elif jump > MAX_WAYPOINT_JUMP:
            raise IKError(
                f"{what}: IK solution jumps {jump:.2f} rad at waypoint {i + 1}/{len(targets)} "
                f"(limit {MAX_WAYPOINT_JUMP} rad) - arm not moved"
            )
        path.append(q)
        q_prev = q
    return path


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
    q_init: np.ndarray | None = None,
) -> np.ndarray:
    """Damped-least-squares IK for a point rigidly attached to body_id.

    Solves for the angles of `joint_ids` (each assumed 1-dof) that bring the
    point at `local_offset` in body_id's local frame to `target_pos` in
    world space. Starts from `q_init` if given, else data's current joint
    angles (so solving a path waypoint by waypoint, each seeded with the
    last solution, warm-starts every solve), and operates on a scratch copy
    of qpos/qvel - the real simulation state is restored before returning.

    Joints are kept JOINT_LIMIT_MARGIN inside their ranges, and IKError is
    raised if the result still ends more than IK_MAX_POS_ERR from the
    target (unreachable), so a failed solve is never returned as a target.
    """
    qpos_adr = model.jnt_qposadr[joint_ids]
    dof_adr = model.jnt_dofadr[joint_ids]
    lo, hi = _limits_with_margin(model, joint_ids)

    qpos_save = data.qpos.copy()
    qvel_save = data.qvel.copy()
    q = np.clip(data.qpos[qpos_adr] if q_init is None else q_init, lo, hi)

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

        data.qpos[qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        point, _ = _hand_point_and_jac(model, data, body_id, local_offset)
        pos_err = float(np.linalg.norm(target_pos - point))
    finally:
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        mujoco.mj_kinematics(model, data)

    if pos_err > IK_MAX_POS_ERR:
        raise IKError(
            f"IK could not reach {np.round(target_pos, 4)}: best solution is "
            f"{pos_err * 100:.1f} cm away - arm not moved"
        )
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
    q_init: np.ndarray | None = None,
    give_up_after: int | None = None,
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

    Like `solve_ik`, works on a scratch copy of qpos/qvel and restores the
    real simulation state before returning - otherwise the arm would be left
    teleported to the solution before the actuators ever drove it there -
    keeps joints off their limits, and raises IKError if the result misses
    the target by more than IK_MAX_POS_ERR / IK_MAX_ROT_ERR.

    `give_up_after`: stop early once the error hasn't improved by 1% for this
    many iterations while still out of tolerance - an unreachable target
    otherwise runs all `iters`. For planners that try many targets, most of
    them unreachable (zebra_flip: 100 lost no solve in testing).
    """
    qpos_adr = model.jnt_qposadr[joint_ids]
    dof_adr = model.jnt_dofadr[joint_ids]
    lo, hi = _limits_with_margin(model, joint_ids)

    qpos_save = data.qpos.copy()
    qvel_save = data.qvel.copy()
    q = np.clip(data.qpos[qpos_adr] if q_init is None else q_init, lo, hi)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    best, stuck = np.inf, 0
    try:
        for _ in range(iters):
            data.qpos[qpos_adr] = q
            mujoco.mj_kinematics(model, data)
            point, jacp = _hand_point_and_jac(model, data, body_id, local_offset)
            pos_err = target_pos - point
            rot_err = _rot_err(model, data, body_id, target_quat)

            err = np.concatenate([pos_err, rot_err])
            e = np.linalg.norm(err)
            if e < tol:
                break
            if give_up_after is not None:
                best, stuck = (e, 0) if e < 0.99 * best else (best, stuck + 1)
                if stuck >= give_up_after and (np.linalg.norm(pos_err) > IK_MAX_POS_ERR
                                               or np.linalg.norm(rot_err) > IK_MAX_ROT_ERR):
                    break

            mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
            J = np.vstack([jacp[:, dof_adr], jacr[:, dof_adr]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
            dq = np.clip(dq, -max_step, max_step)
            q = np.clip(q + dq, lo, hi)

        data.qpos[qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        point, _ = _hand_point_and_jac(model, data, body_id, local_offset)
        final_pos_err = float(np.linalg.norm(target_pos - point))
        final_rot_err = float(np.linalg.norm(_rot_err(model, data, body_id, target_quat)))
    finally:
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        mujoco.mj_kinematics(model, data)

    if final_pos_err > IK_MAX_POS_ERR or final_rot_err > IK_MAX_ROT_ERR:
        raise IKError(
            f"IK could not reach {np.round(target_pos, 4)} at the grip orientation: "
            f"best solution is {final_pos_err * 100:.1f} cm / "
            f"{np.degrees(final_rot_err):.1f} deg away - arm not moved"
        )
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
    lead_in_step: float = MAX_WAYPOINT_JUMP,
    path_check=None,
) -> None:
    """Like `move_to_point`, but also holds the gripper at `target_quat`
    throughout - see `solve_ik_pose`. The orientation target is held fixed
    across all waypoints (only position is interpolated): this arm's
    reachable orientation is narrow and moves with its position, so
    interpolating the orientation too leaves waypoints that can't be reached
    at all (tested: 13 cm / 12 deg off). Turning the wrist into
    `target_quat` from wherever it starts happens as `_plan_path`'s
    joint-space lead-in instead. This is meant for short, single-purpose
    moves like the final grasp approach, not a long transport where forcing
    one fixed orientation the whole way may not even be reachable.

    The whole path is solved and checked before the arm moves, so an
    IKError (unreachable waypoint, or an IK configuration flip) leaves the
    arm where it was.
    """
    mujoco.mj_kinematics(model, data)
    start_point, _ = _hand_point_and_jac(model, data, body_id, local_offset)
    targets = [start_point + (i / waypoints) * (target_pos - start_point) for i in range(1, waypoints + 1)]
    q_start = data.qpos[model.jnt_qposadr[joint_ids]].copy()
    what = f"oriented move to {np.round(target_pos, 4)}"
    path = _plan_path(
        q_start,
        lambda wp, q_init: solve_ik_pose(
            model, data, body_id, local_offset, joint_ids, wp, target_quat, q_init=q_init
        ),
        targets,
        what,
        lead_in_step,
    )
    if path_check is not None:
        path_check(path, what)
    _confirm_and_follow(model, data, render, clock, arm_ctrl_slice, q_start, path, what,
                        max(1, steps // waypoints), settle_steps, carry)


def _confirm_and_follow(model, data, render, clock, arm_ctrl_slice, q_start, path, what,
                        steps_per_wp, settle_steps, carry) -> None:
    """Pass a planned path through `move_check.confirm` (a no-op unless
    confirm-before-moving is switched on), then execute it."""
    duration_s = (len(path) * steps_per_wp + settle_steps) * sim_step.CONTROL_DT
    move_check.confirm(what, q_start, path, duration_s, render)
    _follow_path(model, data, render, clock, arm_ctrl_slice, path, steps_per_wp, settle_steps, carry)


def _follow_path(model, data, render, clock, arm_ctrl_slice, path, steps_per_wp, settle_steps, carry) -> None:
    """Ramp `data.ctrl` through each joint-space waypoint of `path` in turn,
    `steps_per_wp` physics steps each, then hold the last for `settle_steps`."""
    for q in path:
        ctrl_start = data.ctrl[arm_ctrl_slice].copy()
        for s in range(steps_per_wp):
            a = (s + 1) / steps_per_wp
            data.ctrl[arm_ctrl_slice] = ctrl_start + a * (q - ctrl_start)
            sim_step.step(model, data)
            if carry is not None:
                carry()
            if render is not None:
                clock.tick()
                render.step()

    data.ctrl[arm_ctrl_slice] = path[-1]
    for _ in range(settle_steps):
        sim_step.step(model, data)
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
    path_check=None,
) -> None:
    """Smoothly drive the hand point to `target_pos`.

    Interpolates `waypoints` points on the straight line from the hand's
    current position to the target, solves IK at each (warm-started from
    the previous waypoint's solution), and ramps `data.ctrl` toward each
    solution in turn while stepping physics - so motion stays smooth
    (PD-servoed, not teleported) and the hand's own path stays close to a
    straight line rather than whatever a single joint-space ramp would
    trace out.

    The whole path is solved and checked before the arm moves, so an
    IKError (unreachable waypoint, or a jump over MAX_WAYPOINT_JUMP) leaves
    the arm where it was. `path_check`, if given, is called with the
    planned joint-space path and a description of the move before anything
    moves too - e.g. arm_clearance.ArmClearance, raising to refuse it.

    The PD servos lag a fast-moving ctrl target, so a few cm of tracking
    error remains right after the last waypoint; `settle_steps` holds ctrl
    at the final solution so the arm actually catches up before returning.

    `carry`, if given, is called after every control tick (`sim_step.step`) - same convention as
    stationlite_pick_place.py's `_move_to`/`_make_carry`: a no-argument
    callback that snaps a held object's freejoint to the gripper each step,
    since this arm can't hold anything through contact/friction alone.
    """
    mujoco.mj_kinematics(model, data)
    start_point, _ = _hand_point_and_jac(model, data, body_id, local_offset)

    targets = [start_point + (i / waypoints) * (target_pos - start_point) for i in range(1, waypoints + 1)]
    q_start = data.qpos[model.jnt_qposadr[joint_ids]].copy()
    what = f"move to {np.round(target_pos, 4)}"
    path = _plan_path(
        q_start,
        lambda wp, q_init: solve_ik(model, data, body_id, local_offset, joint_ids, wp, q_init=q_init),
        targets,
        what,
    )
    if path_check is not None:
        path_check(path, what)
    _confirm_and_follow(model, data, render, clock, arm_ctrl_slice, q_start, path, what,
                        max(1, steps // waypoints), settle_steps, carry)


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

    clock = _RealtimeClock(dt=sim_step.CONTROL_DT)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        render = _ThrottledSync(viewer, model, step_dt=sim_step.CONTROL_DT)

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
            sim_step.step(model, data)
            clock.tick()
            render.step()
