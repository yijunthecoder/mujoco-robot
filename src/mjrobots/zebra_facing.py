"""Place a brick with its printed face pointing a set way ("desired_facing" in a pick).

Each zebra brick has its print on ONE long face (the brick's -y face). On the stack it
can sit two ways round with the same footprint; only one shows the print the way the
zebra should face. Agreed with Victor (2026-10-01): FORWARD = the print points towards
world -Y, the robot's right (the right of the headcam image) - brick yaw 0 at the stack.

Which side the print is on can't be seen while the brick lies on the table: from above
the long faces don't show, and from the headcam (~95 cm away) a face is 3-10 px tall
(scratch face_cams.py, 150 bricks). So after the pick the arm SHOWS it to the headcam:
it holds the brick 50-65 cm in front of it with a long face turned to it (nearer is out
of reach - the headcam looks 66 deg down, so "near" is 40-50 cm up in the air). Measured
over 150 bricks: 149 can be shown, the fingers hide ~12% of the face, which appears
20-30 px tall (scratch show_pose.py). Seeing either long face tells the side: blank
means the print is on the other one.

Knowing how the brick sits in the hand, the place grip that gives the wanted facing is
fixed. If the arm can't reach it at the stack: put the brick back, pick it up with the
other grip (turned 180 deg) and try again; else the other arm; else the pick fails.
Over 150 bricks: 78 direct, 19 after a regrip, 9 with the other arm, 44 can't
(scratch facing_regrip.py) - the wrist can't reach every angle at the stack.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from . import cartesian_control
from . import zebra_pick_place as zpp
from .camera_calibration import IMAGE_SIZE
from .cartesian_control import HAND_LOCAL_OFFSET, IKError, solve_ik_pose
from .zebra_flip import _plan_only
from .zebra_pick_place import ZebraArmContext, _BRICK_CENTER_OFFSET_Z, _HOVER_DZ, _wrap, go_home, grasp_part, put_back


class FacingError(RuntimeError):
    """The brick can't be placed facing the wanted way (it has been put back)."""

# The brick's full yaw at the stack for each facing (print = the brick's -y face:
# at yaw 0 it points to world -Y).
FACING_YAW = {"FORWARD": 0.0, "BACKWARD": np.pi}
SHOW_CAMERA = "headcam"
_SHOW_DISTANCES = (0.50, 0.55, 0.60, 0.65)  # m from the camera (nearer is out of reach)
_SHOW_OFFSETS = np.radians([(0, 0), (15, 0), (-15, 0), (0, 12), (0, -12), (15, 12), (-15, 12), (15, -12), (-15, -12)])
_SHOW_LEANS = np.radians([0, 20, 40])  # face turned this far from looking straight at the camera
_COLLISION_GROUPS = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)  # visual meshes are group 2


def _rot(axis, angle):
    q = np.empty(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float) / np.linalg.norm(axis), angle)
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def _mat_of(q):
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def _quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R).flatten())
    return q


def _brick_geom(model, brick_id):
    return next(g for g in range(model.ngeom) if model.geom_bodyid[g] == brick_id and model.geom_contype[g])


def long_face_view(model, data, camera: str, brick_id: int):
    """How well `camera` sees the brick's long faces right now: the best of the two as
    (share of a 7x5 grid on it the camera sees unblocked, its height in px). Geometry
    only - what a real camera there could see, not what it recognises."""
    W, H = IMAGE_SIZE
    ci = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    cpos, cR = data.cam_xpos[ci], data.cam_xmat[ci].reshape(3, 3)
    f = (H / 2) / np.tan(np.radians(model.cam_fovy[ci]) / 2)
    geom = _brick_geom(model, brick_id)
    R = data.xmat[brick_id].reshape(3, 3)
    sx, sy, sz = model.geom_size[geom]
    best = (0.0, 0.0)
    for sign in (-1, 1):
        n = R @ np.array([0, sign, 0])
        fc = data.geom_xpos[geom] + sy * n
        if n @ (cpos - fc) <= 0:
            continue  # turned away
        hits, total = 0, 0
        for x in np.linspace(-0.85, 0.85, 7):
            for z in np.linspace(-0.8, 0.8, 5):
                total += 1
                p = fc + R @ np.array([x * sx, 0, z * sz]) + 0.0005 * n
                pc = cR.T @ (p - cpos)
                if -pc[2] <= 0 or not (0 <= W / 2 + f * pc[0] / -pc[2] < W and 0 <= H / 2 - f * pc[1] / -pc[2] < H):
                    continue
                gid = np.array([-1], dtype=np.int32)
                vec = p - cpos
                dist = mujoco.mj_ray(model, data, cpos, vec / np.linalg.norm(vec), _COLLISION_GROUPS, 1,
                                     int(model.cam_bodyid[ci]), gid)  # the camera sits inside its mount
                if gid[0] == geom or dist < 0 or dist > np.linalg.norm(vec) - 0.002:
                    hits += 1
        cos = n @ (cpos - fc) / np.linalg.norm(cpos - fc)
        px = float(f * 2 * sz * cos / -(cR.T @ (fc - cpos))[2])
        if hits / total > best[0]:
            best = (hits / total, px)
    return best


def _show_pose(ctx: ZebraArmContext):
    """Joint angles that hold the gripped brick in front of SHOW_CAMERA with a long face
    turned to it (nearest distance first), or None. The brick's pose in the hand is taken
    from where it is now."""
    model, data = ctx.model, ctx.data
    ci = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, SHOW_CAMERA)
    cpos, cR = data.cam_xpos[ci].copy(), data.cam_xmat[ci].reshape(3, 3).copy()
    axis = -cR[:, 2]  # a MuJoCo camera looks along its -z
    R_hand = data.xmat[ctx.body_id].reshape(3, 3)
    R_brick = data.xmat[ctx.brick_id].reshape(3, 3)
    rel = R_hand.T @ R_brick
    centre_now = data.xpos[ctx.brick_id] + R_brick @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
    hand_from_brick = R_brick.T @ (data.xpos[ctx.body_id] + R_hand @ HAND_LOCAL_OFFSET - centre_now)
    q_now = data.qpos[model.jnt_qposadr[ctx.joint_ids]].copy()
    for dist in _SHOW_DISTANCES:
        for yaw_off, pitch_off in _SHOW_OFFSETS:
            view = _rot(cR[:, 1], yaw_off) @ _rot(cR[:, 0], pitch_off) @ axis
            side = np.cross(view, [0, 0, 1.0])
            side /= np.linalg.norm(side)
            for sign in (-1, 1):
                for lean in _SHOW_LEANS:
                    y_b = -(_rot(side, lean) @ view) * sign  # face sign*y turned to the camera
                    z_b = np.array([0, 0, 1.0]) - y_b[2] * y_b  # studs as near up as possible
                    z_b /= np.linalg.norm(z_b)
                    R_b = np.column_stack([np.cross(y_b, z_b), y_b, z_b])
                    target = cpos + dist * view + R_b @ hand_from_brick
                    try:
                        q = solve_ik_pose(model, data, ctx.body_id, HAND_LOCAL_OFFSET, ctx.joint_ids, target,
                                          _quat(R_b @ rel.T), q_init=q_now, iters=800, give_up_after=100)
                        ctx._check_clearance([q_now + t * (q - q_now) for t in np.linspace(0, 1, 30)], "show")
                    except IKError:
                        continue
                    return q
    return None


def show_and_look(ctx: ZebraArmContext, render, clock, look_print) -> float | None:
    """Hold the gripped brick up in front of SHOW_CAMERA and look at it, to learn how it's
    held: returns held_yaw, the brick's yaw relative to the grip (0 or pi - which way
    round the print is in the hand), or None if the camera can't tell or no show pose is
    reachable. `look_print(data)` returns the direction the printed face points in, as
    the camera sees it (unit vector, world), or None. Not the brick's yaw there: in the
    show pose it's turned and tilted - only its pose in the hand carries over to the
    table. The arm then goes back to where it started (above the pick spot)."""
    q = _show_pose(ctx)
    if q is None:
        return None
    q0 = ctx.data.qpos[ctx.model.jnt_qposadr[ctx.joint_ids]].copy()
    n = max(20, int(np.ceil(np.abs(q - q0).max() / zpp._HELD_LEAD_IN_STEP)))
    path = [q0 + (i / n) * (q - q0) for i in range(1, n + 1)]
    ctx._check_clearance(path, f"{ctx.arm} arm shows the brick to the {SHOW_CAMERA}")
    cartesian_control._confirm_and_follow(ctx.model, ctx.data, render, clock, ctx.arm_ctrl, q0, path,
                                          f"{ctx.arm} arm shows the brick", steps_per_wp=20, settle_steps=150,
                                          carry=None)
    n_print = look_print(ctx.data)
    held = None
    if n_print is not None:
        # where the print points in the hand, vs where it would with held_yaw 0 (brick yaw = grip
        # yaw: the print, the brick's -y face, along the grip orientation's -y turned like the hand)
        n_hand = ctx.data.xmat[ctx.body_id].reshape(3, 3).T @ np.asarray(n_print, float)
        held0 = _mat_of(ctx.grip_quat).T @ np.array([0, -1.0, 0])
        held = 0.0 if n_hand @ held0 > 0 else float(np.pi)
    # back the same way to where it lifted the brick: the stack and a regrip are planned from
    # there (from the show pose the ways to the stack or back to the table were often refused)
    back = [path[-1] + (i / n) * (q0 - path[-1]) for i in range(1, n + 1)]
    ctx._check_clearance(back, f"{ctx.arm} arm back from showing the brick")
    cartesian_control._confirm_and_follow(ctx.model, ctx.data, render, clock, ctx.arm_ctrl, path[-1], back,
                                          f"{ctx.arm} arm back from showing", steps_per_wp=20, settle_steps=150,
                                          carry=None)
    return held


def can_place(ctx: ZebraArmContext, place_xyz, facing_yaw: float) -> bool:
    """Plan-only: from where the arm is now, could it carry the brick to `place_xyz` and
    set it down at full yaw `facing_yaw` (ctx.held_yaw must be the full one)? Nothing moves."""
    d = copy.copy(ctx.data)
    c = ZebraArmContext(ctx.model, d, ctx.arm, mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_BODY, ctx.brick_id))
    c.held_yaw, c.holding = ctx.held_yaw, True
    hover = np.asarray(place_xyz, float) + [0, 0, _HOVER_DZ]
    with _plan_only():
        try:
            c.go(None, None, hover)
            c.grip_yaw = _wrap(facing_yaw - c.held_yaw)
            c.go_oriented(None, None, hover)
            c.go_oriented(None, None, place_xyz)
            return True
        except IKError:
            return False


def _can_pick_and_place(ctx: ZebraArmContext, center_xyz, grip: float, yaw_full: float, place_xyz,
                        facing_yaw: float) -> bool:
    """Plan-only: could this (empty-handed) arm pick the brick with grip yaw `grip` and then
    place it at `facing_yaw`? Nothing moves."""
    d = copy.copy(ctx.data)
    c = ZebraArmContext(ctx.model, d, ctx.arm, mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_BODY, ctx.brick_id))
    hover = np.asarray(center_xyz, float) + [0, 0, _HOVER_DZ]
    with _plan_only():
        try:
            c.grip_yaw, c.holding = 0.0, False
            c.go(None, None, hover)
            c.grip_yaw = grip
            c.go_oriented(None, None, hover)
            c.go_oriented(None, None, center_xyz)
            c.holding = True
            c.go(None, None, hover)
        except IKError:
            return False
    c.held_yaw = _wrap(yaw_full - grip)
    return can_place(c, place_xyz, facing_yaw)


def pick_facing(contexts: dict, arm: str, render, clock, center_xyz, place_xyz, facing_yaw: float,
                relook, look_print) -> tuple[ZebraArmContext, str]:
    """Pick the brick so it can be placed at `place_xyz` with full yaw `facing_yaw`.
    contexts: {arm: ZebraArmContext} for this brick (both arms); `arm` picks first.
    `relook()`: grasp_part's look from hover; `look_print(data)`: see show_and_look.
    Returns (the context now holding it, with held_yaw the full one; how: "same grip",
    "other grip" or "other arm"). Raises FacingError (brick put back) or RuntimeError."""
    c = contexts[arm]
    grasp_part(c, render, clock, center_xyz, relook=relook)
    held = show_and_look(c, render, clock, look_print)
    yaw = _wrap(c.grip_yaw + held) if held is not None else None  # the brick's full yaw where it lay
    # Always put it back and pick it up again: tilted towards the camera it shifts in the
    # fingers (up to 1.3 cm and 18 deg in testing) and was then placed off target or fell
    # off the stack - a fresh pick gives a centred grip, and the grip that gives the facing.
    put_back(c, render, clock)
    if held is None:
        raise FacingError(f"couldn't see which side its print is on (shown to the {SHOW_CAMERA}) - put it back")
    look = relook()
    yaw_now = full_yaw_near(look[1], yaw) if look is not None and look[1] is not None else yaw
    centre_now = look[0] if look is not None else c.picked_from
    same_grip = _wrap(yaw_now - held)
    options = [(c, same_grip, "same grip"), (c, _wrap(same_grip + np.pi), "other grip")]
    for a in contexts:  # the other arm too, if both are in use
        if a != arm:
            options += [(contexts[a], _wrap(yaw_now), "other arm"), (contexts[a], _wrap(yaw_now + np.pi), "other arm")]
    for holder, grip, how in options:
        if _can_pick_and_place(holder, centre_now, grip, yaw_now, place_xyz, facing_yaw):
            if holder is not c:
                go_home(c, render, clock)
            try:
                grasp_part(holder, render, clock, centre_now, relook=relook, grip_yaws=[grip])
            except IKError:
                continue  # just out of reach once it looked again from hover (refused before moving)
            holder.held_yaw = _wrap(yaw_now - holder.grip_yaw)
            return holder, how
    raise FacingError("neither grip of either arm can set it down facing that way at the stack - put it back")


def full_yaw_near(yaw_mod180: float, remembered: float) -> float:
    """A yaw seen only up to 180 deg (the look from hover), resolved by a full one
    remembered from the show look: whichever of the two is nearer it."""
    a, b = _wrap(yaw_mod180), _wrap(yaw_mod180 + np.pi)
    return a if abs(_wrap(a - remembered)) <= abs(_wrap(b - remembered)) else b
