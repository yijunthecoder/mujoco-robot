"""Turn a zebra brick that isn't upright onto its studs, with both arms.

A dropped brick lands upright only ~1 time in 6; on its side, on its end or
upside down it can't be gripped from the top and stacked. One hand can't stand
it up: whatever a hand does, the brick in it does too, and turning a hand that
far is out of this arm's reach. So the turn is shared in the air:

    1. hand A picks the brick as it lies (by the pair of faces that leaves the
       rest free), lifts it to a handover spot and rolls it part of the way;
    2. hand B comes in with its hand tilted back by the rest of the turn and
       grips another pair of faces;
    3. A opens, backs its hand straight out and goes home;
    4. B turns the rest of the way - now in its normal grip, brick upright -
       and sets the brick down upright on the table, where a normal pick can
       take it (Victor's tree picks and places it next).

Which pair each hand holds, how far A rolls and which arm is A were chosen by
measuring 192 brick poses per case (plan-only; see the README): on its side
86% can be flipped (split 45+45, else A alone 90), on its end 33%, upside down
20%. `plan_flip` tries those strategies in that order on a scratch copy of the
simulation - planning only, nothing moves - and returns the first that works;
`execute_flip` then does it for real, with physics.

While two hands are near one brick, the 2 cm arm-to-arm rule is relaxed for
the grippers only ("handover mode"): the arm links must still stay 2 cm apart,
and the grippers may not touch.

The brick's full 3D orientation comes from the simulation (a stand-in for a
detector, like the simulated yaw in zebra_skill_bridge.py); perception itself
only reports how it lies (scatter.lying).
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from . import cartesian_control
from . import zebra_pick_place as zpp
from .cartesian_control import HAND_LOCAL_OFFSET, IKError, move_to_pose, solve_ik_pose
from .scatter import MIN_BRICK_GAP, ON_END, ON_SIDE, STACK_CLEAR, STACK_XY, TABLE_Z, UPRIGHT, UPSIDE_DOWN, lying
from .stationlite_pick_place import _GRIP_OPEN, _hold
from .zebra_pick_place import (
    BRICK_HEIGHT, ZebraArmContext, _BRICK_CENTER_OFFSET_Z, _HOVER_DZ, _close_and_settle, _oriented_approach,
    _wrap, _yawed, go_home,
)

TABLE_TOP = TABLE_Z - BRICK_HEIGHT
_GRIP_Q = np.array([0.0, 2.23, -1.215, 0.0, 0.0, 0.0])  # the pose grip_quat was taken at (an IK seed)
# Where the handover happens: in the air, 12 and 16 cm above the table, beside
# the stack spot as well as over the middle (a stack may already stand there).
_HANDOVER_SPOTS = [np.array([x, y, z]) for z in (0.0, 0.04) for x in (0.35, 0.45) for y in (0.0, 0.12, -0.12)]
_YAW_TURNS = np.radians([0, 30, -30, 60, -60, 90, -90, 180])  # extra turn about vertical at the handover
_B_HEIGHTS = (0.0, 0.0075, 0.0125)  # B grips this much higher than A: keeps the fingers apart
_BACK_OUT = 0.06  # m A's hand backs out along its own fingers after letting go
_B_APPROACH = 0.08  # m B comes in straight from this far back along its own fingers
_MIN_LINKS = 0.02  # m arm links apart (as arm_clearance.MIN_ARM_CLEARANCE)
_MIN_OBSTACLE = 0.01  # m hands / held brick from other bricks
_HELD_JOINT_STEP = 0.03  # rad per waypoint for joint moves while holding (gentle, like zpp)
_WIDTH_TOL = 0.005  # m finger gap tolerance for "holding the brick"

# How each landing is flipped: A's finger axis (0 = long axis: A holds the ends,
# 1 = brick y: A holds the long faces), the total roll, the pair B ends up
# holding, finger widths, and A's share of the roll to try, in order.
_CASES = {
    ON_SIDE: dict(a_axis=0, total=np.pi / 2, b_pair="y", wA=0.064, wB=0.032, rolls=(np.pi / 4, np.pi / 2)),
    ON_END: dict(a_axis=1, total=np.pi / 2, b_pair="x", wA=0.032, wB=0.064, rolls=(np.pi / 4,)),
    UPSIDE_DOWN: dict(a_axis=1, total=np.pi, b_pair="x", wA=0.032, wB=0.064, rolls=(np.pi / 2,)),
}


def _mat(q):
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def _quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R).flatten())
    return q


def _rot(axis, angle):
    q = np.empty(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float) / np.linalg.norm(axis), angle)
    return _mat(q)


@contextlib.contextmanager
def _plan_only():
    """Moves only plan: the arm jumps to each planned move's end instead of
    being driven there (planning is where every refusal happens)."""
    def jump(model, data, render, clock, arm_ctrl, q_start, path, what, spw, settle, carry):
        joints = model.actuator_trnid[arm_ctrl, 0]
        data.qpos[model.jnt_qposadr[joints]] = path[-1]
        data.ctrl[arm_ctrl] = path[-1]
        mujoco.mj_forward(model, data)

    saved = cartesian_control._confirm_and_follow, zpp._confirm_and_follow
    cartesian_control._confirm_and_follow = zpp._confirm_and_follow = jump
    try:
        yield
    finally:
        cartesian_control._confirm_and_follow, zpp._confirm_and_follow = saved


@dataclass
class FlipPlan:
    lies: str
    a_side: str
    b_side: str
    roll_a: float
    grip_a: float  # A's grip yaw at the pick
    center: np.ndarray  # brick centre as it lies
    q_a_roll: np.ndarray  # A at the handover spot, rolled
    handover: np.ndarray  # A's grip point at the handover
    b_point: np.ndarray  # B's grip point (a little higher)
    b_quat_tilted: np.ndarray
    b_quat_final: np.ndarray
    grip_b: float  # B's normal-grip yaw once turned
    q_b_final: np.ndarray | None  # B turned at the handover spot (None: turn on the way down)
    set_down: np.ndarray  # upright brick centre on the table where B sets it down
    wA: float
    wB: float
    q_b_pre: np.ndarray  # B 8 cm back from its grip, hand already tilted
    q_b: np.ndarray  # B gripping at the handover
    q_a_out: np.ndarray  # A backed out after letting go
    brick_R: np.ndarray  # the brick's rotation A is expected to present at the handover

    def describe(self) -> str:
        rest = _CASES[self.lies]["total"] - self.roll_a
        how = (f"{self.a_side} arm rolls {np.degrees(self.roll_a):.0f} deg, "
               f"{self.b_side} arm takes it and turns {np.degrees(rest):.0f} deg" if rest > 1e-6
               else f"{self.a_side} arm rolls it 90 deg, {self.b_side} arm takes it")
        return f"{self.lies}: {how}, sets it down upright at ({self.set_down[0]:.2f}, {self.set_down[1]:+.2f})"


class _Geoms:
    """Collision geoms by arm part, for the handover distance checks."""

    def __init__(self, model):
        def geoms(names):
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
            return [g for g in range(model.ngeom) if model.geom_bodyid[g] in ids
                    and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self.model = model
        self.links = {s: geoms([f"{s}_link{i}" for i in range(1, 6)]) for s in ("left", "right")}
        self.grip = {s: geoms([f"{s}_linkgripper", f"{s}_griperlj_link1", f"{s}_griperlj_link2"])
                     for s in ("left", "right")}

    def body(self, body_id):
        return [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body_id
                and self.model.geom_contype[g]]

    def dist(self, data, a, b):
        best, ft = 1.0, np.zeros(6)
        for x in a:
            for y in b:
                best = min(best, mujoco.mj_geomDistance(self.model, data, x, y, best, ft))
        return best

    def links_gap(self, data, a, b):
        """Handover rule: each arm's links vs the whole other arm."""
        return min(self.dist(data, self.links[a], self.links[b] + self.grip[b]),
                   self.dist(data, self.links[b], self.grip[a]))


class _Planner:
    def __init__(self, model, data, brick_body: int, other_bricks: list[int], arms_order: list[str]):
        self.model, self.real = model, data
        self.brick = brick_body
        self.geo = _Geoms(model)
        self.obstacles = [g for b in other_bricks for g in self.geo.body(b)]
        self.brick_geoms = self.geo.body(brick_body)
        self.arms_order = arms_order
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, brick_body)
        self.name = name
        self.home = model.key("home").id
        self.other_xy = [data.xpos[b][:2].copy() for b in other_bricks]
        a_body = HAND_LOCAL_OFFSET / np.linalg.norm(HAND_LOCAL_OFFSET)
        self.a_body = a_body
        self.fbody = {}
        probe = copy.copy(data)
        for side in ("left", "right"):
            c = ZebraArmContext(model, probe, side, name)
            j = c._finger_joints[0]
            f = probe.xmat[c.body_id].reshape(3, 3).T @ (probe.xmat[model.jnt_bodyid[j]].reshape(3, 3) @ model.jnt_axis[j])
            f -= a_body * (a_body @ f)
            self.fbody[side] = f / np.linalg.norm(f)

    # --- scratch state ---
    def fresh(self):
        d = copy.copy(self.real)
        ctx = {s: ZebraArmContext(self.model, d, s, self.name) for s in ("left", "right")}
        return d, ctx

    def ik(self, d, c, P, R, seed=None):
        seeds = ([seed] if seed is not None else []) + [_GRIP_Q, self.model.key_qpos[self.home][self.model.jnt_qposadr[c.joint_ids]]]
        for s in seeds:
            try:
                return solve_ik_pose(self.model, d, c.body_id, HAND_LOCAL_OFFSET, c.joint_ids, P, _quat(R),
                                     q_init=s, iters=800)
            except IKError:
                pass
        return None

    @staticmethod
    def set_arm(d, c, q):
        d.qpos[c.model.jnt_qposadr[c.joint_ids]] = q
        d.ctrl[c.arm_ctrl] = q

    @staticmethod
    def set_fingers(d, c, w):
        j1, j2 = c._finger_joints
        d.qpos[c.model.jnt_qposadr[j1]], d.qpos[c.model.jnt_qposadr[j2]] = -w / 2, w / 2

    def carry_brick(self, d, c, rel):
        """Put the brick where hand c holds it (rel = brick pose in the hand frame)."""
        R = d.xmat[c.body_id].reshape(3, 3)
        adr = self.model.jnt_qposadr[self.model.body_jntadr[self.brick]]
        d.qpos[adr:adr + 3] = d.xpos[c.body_id] + R @ rel[0]
        d.qpos[adr + 3:adr + 7] = _quat(R @ rel[1])

    def joint_path_ok(self, d, c, other, q0, q1, rel=None, links_only=False, n=12):
        """Sampled joint-space path: arm-to-arm distance (2 cm, or links only in
        handover mode) and hand/held brick vs the other bricks."""
        for t in np.linspace(0, 1, n):
            self.set_arm(d, c, q0 + t * (q1 - q0))
            mujoco.mj_kinematics(self.model, d)
            if rel is not None:
                self.carry_brick(d, c, rel)
                mujoco.mj_kinematics(self.model, d)
            if links_only:
                if self.geo.links_gap(d, c.arm, other) < _MIN_LINKS:
                    return False
            else:
                mine = self.geo.links[c.arm] + self.geo.grip[c.arm] + (self.brick_geoms if rel is not None else [])
                if self.geo.dist(d, mine, self.geo.links[other] + self.geo.grip[other]) < _MIN_LINKS:
                    return False
            if self.obstacles:
                mine = self.geo.grip[c.arm] + (self.brick_geoms if rel is not None else [])
                if self.geo.dist(d, mine, self.obstacles) < _MIN_OBSTACLE:
                    return False
        self.set_arm(d, c, q1)
        mujoco.mj_forward(self.model, d)
        return True

    def set_down_spots(self, center, b_side):
        """Where B may set the brick down upright: where it lay, then spots near it,
        clear of the other bricks and the stack."""
        cands = [center[:2]] + [center[:2] + r * np.array([np.cos(a), np.sin(a)])
                                for r in (0.06, 0.10) for a in np.radians(np.arange(0, 360, 45))]
        out = []
        for xy in cands:
            if np.linalg.norm(xy - STACK_XY) < STACK_CLEAR:
                continue
            if any(np.linalg.norm(xy - o) < MIN_BRICK_GAP for o in self.other_xy):
                continue
            out.append(np.array([xy[0], xy[1], TABLE_TOP + 0.0192]))
        return out

    # --- one strategy ---
    def attempt(self, lies, Rb0, center, a_side, roll_a):
        cfg = _CASES[lies]
        b_side = "left" if a_side == "right" else "right"
        rest = cfg["total"] - roll_a
        axis_b = Rb0[:, cfg["a_axis"]]
        d, ctx = self.fresh()
        cA, cB = ctx[a_side], ctx[b_side]
        with _plan_only():
            # 1. A picks it by the chosen pair
            f0 = _mat(cA.grip_quat) @ self.fbody[a_side]
            g0 = np.arctan2(axis_b[1], axis_b[0]) - np.arctan2(f0[1], f0[0])
            grips = sorted({_wrap(g0), _wrap(g0 + np.pi)}, key=abs)
            hover = center + [0, 0, _HOVER_DZ]
            try:
                cA.grip_yaw, cA.holding = 0.0, False
                cA.go(None, None, hover); cA.go_oriented(None, None, hover)
                grip_a = _oriented_approach(cA, None, None, hover, center, grips)
            except IKError:
                return None
            Rgrasp = d.xmat[cA.body_id].reshape(3, 3).copy()
            q_grasp = d.qpos[self.model.jnt_qposadr[cA.joint_ids]].copy()
            rel = (Rgrasp.T @ (center - Rb0 @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z]) - d.xpos[cA.body_id]),
                   Rgrasp.T @ Rb0)
            try:
                cA.holding = True
                cA.go(None, None, hover)
            except IKError:
                return None
            q_lift = d.qpos[self.model.jnt_qposadr[cA.joint_ids]].copy()
            after_lift = d.qpos.copy(), d.ctrl.copy()
            fA = Rgrasp @ self.fbody[a_side]
            for sgn in (1, -1):
                if (_rot(fA, sgn * cfg["total"]) @ Rb0)[2, 2] < 0.99:
                    continue  # this way round ends upside down
                for S in _HANDOVER_SPOTS:
                    for gam in _YAW_TURNS:
                        Rz = _rot([0, 0, 1.0], gam)
                        d.qpos[:], d.ctrl[:] = after_lift
                        mujoco.mj_forward(self.model, d)
                        q_a = self.ik(d, cA, S, Rz @ _rot(fA, sgn * roll_a) @ Rgrasp, seed=q_grasp)
                        if q_a is None:
                            continue
                        if not self.joint_path_ok(d, cA, b_side, q_lift, q_a, rel=rel):
                            continue
                        Rb = Rz @ _rot(fA, sgn * roll_a) @ Rb0
                        Rb_up = _rot(Rz @ fA, sgn * rest) @ Rb
                        pair = Rb_up[:, 1] if cfg["b_pair"] == "y" else Rb_up[:, 0]
                        at_s = d.qpos.copy(), d.ctrl.copy()
                        for extra in (0.0, np.pi):
                            fB0 = _mat(cB.grip_quat) @ self.fbody[b_side]
                            grip_b = _wrap(np.arctan2(pair[1], pair[0]) - np.arctan2(fB0[1], fB0[0]) + extra)
                            RBfinal = _mat(_yawed(cB.grip_quat, grip_b))
                            RBtilt = _rot(Rz @ fA, -sgn * rest) @ RBfinal
                            for dz in _B_HEIGHTS:
                                plan = self._b_side(d, cA, cB, at_s, S, dz, RBtilt, RBfinal, grip_b, rest, rel, center,
                                                    cfg["wA"], cfg["wB"])
                                if plan is not None:
                                    q_turn, spot, q_pre, q_b, q_out = plan
                                    return FlipPlan(lies, a_side, b_side, roll_a, grip_a, center.copy(), q_a, S.copy(),
                                                    S + [0, 0, dz], _quat(RBtilt), _quat(RBfinal), grip_b,
                                                    q_turn, spot, cfg["wA"], cfg["wB"], q_pre, q_b, q_out, Rb.copy())
        return None

    def _b_side(self, d, cA, cB, at_s, S, dz, RBtilt, RBfinal, grip_b, rest, rel, center, wA, wB):
        """B's part, from A holding the rolled brick at S. Returns (q_b_final, set_down) or None."""
        d.qpos[:], d.ctrl[:] = at_s
        mujoco.mj_forward(self.model, d)
        PB = S + np.array([0, 0, dz])
        q_b = self.ik(d, cB, PB, RBtilt)
        if q_b is None:
            return None
        aB = RBtilt @ self.a_body
        q_pre = self.ik(d, cB, PB - _B_APPROACH * aB, RBtilt, seed=q_b)
        if q_pre is None:
            return None
        home_b = d.qpos[self.model.jnt_qposadr[cB.joint_ids]].copy()
        self.set_fingers(d, cB, 0.085)
        if not self.joint_path_ok(d, cB, cA.arm, home_b, q_pre):
            return None
        if not self.joint_path_ok(d, cB, cA.arm, q_pre, q_b, links_only=True, n=8):
            return None
        # both holding: links 2 cm apart, grippers not touching
        self.set_fingers(d, cA, wA)
        self.set_fingers(d, cB, wB)
        mujoco.mj_kinematics(self.model, d)
        if self.geo.links_gap(d, cA.arm, cB.arm) < _MIN_LINKS:
            return None
        if self.geo.dist(d, self.geo.grip[cA.arm], self.geo.grip[cB.arm]) <= 0.0:
            return None
        # A opens, backs out straight along its fingers, goes home
        self.set_fingers(d, cA, 0.085)
        q_a = d.qpos[self.model.jnt_qposadr[cA.joint_ids]].copy()
        RA = d.xmat[cA.body_id].reshape(3, 3).copy()
        pA = d.xpos[cA.body_id] + RA @ HAND_LOCAL_OFFSET
        q_out = self.ik(d, cA, pA - _BACK_OUT * (RA @ self.a_body), RA, seed=q_a)
        if q_out is None or not self.joint_path_ok(d, cA, cB.arm, q_a, q_out, links_only=True, n=8):
            return None
        home_a = self.model.key_qpos[self.home][self.model.jnt_qposadr[cA.joint_ids]]
        if not self.joint_path_ok(d, cA, cB.arm, q_out, home_a):
            return None
        # B now holds it (same pose relative to B's hand from here on)
        mujoco.mj_kinematics(self.model, d)
        adr = self.model.jnt_qposadr[self.model.body_jntadr[self.brick]]
        RBh = d.xmat[cB.body_id].reshape(3, 3).copy()
        # the brick's pose in B's hand: where A held it at the handover
        self.set_arm(d, cA, q_a); mujoco.mj_kinematics(self.model, d)
        self.carry_brick(d, cA, rel); mujoco.mj_kinematics(self.model, d)
        rel_b = (RBh.T @ (d.qpos[adr:adr + 3] - d.xpos[cB.body_id]), RBh.T @ _mat(d.qpos[adr + 3:adr + 7]))
        self.set_arm(d, cA, home_a); mujoco.mj_forward(self.model, d)
        # B turns the rest: at the handover spot if it can, else on the way down
        q_bf = self.ik(d, cB, PB, RBfinal, seed=q_b) if rest > 1e-6 else q_b
        start_turn = d.qpos.copy(), d.ctrl.copy()
        for q_turn in ([q_bf] if q_bf is not None else []) + [None]:
            d.qpos[:], d.ctrl[:] = start_turn
            mujoco.mj_forward(self.model, d)
            if q_turn is not None:
                if not self.joint_path_ok(d, cB, cA.arm, q_b, q_turn, rel=rel_b):
                    continue
            for spot in self.set_down_spots(center, cB.arm):
                d.qpos[:], d.ctrl[:] = start_turn
                self.set_arm(d, cB, q_turn if q_turn is not None else q_b)
                mujoco.mj_forward(self.model, d)
                if q_turn is None:  # one joint move straight to above the spot, already turned
                    q_h = self.ik(d, cB, spot + [0, 0, _HOVER_DZ], _mat(_yawed(cB.grip_quat, grip_b)), seed=q_b)
                    if q_h is None or not self.joint_path_ok(d, cB, cA.arm, q_b, q_h, rel=rel_b):
                        continue
                try:
                    cB.grip_yaw, cB.holding = grip_b, False
                    hover = spot + [0, 0, _HOVER_DZ]
                    cB.go(None, None, hover); cB.go_oriented(None, None, hover)
                    cB.go_oriented(None, None, spot)
                    cB.go(None, None, hover)
                except IKError:
                    continue
                return (q_turn, spot, q_pre, q_b, q_out)
        return None

    def plan(self, lies, Rb0, center):
        cfg = _CASES[lies]
        for a_side in self.arms_order:
            for roll_a in cfg["rolls"]:
                p = self.attempt(lies, Rb0, center, a_side, roll_a)
                if p is not None:
                    return p
        return None


def plan_flip(model, data, brick_body: int, other_bricks: list[int], arms_order: list[str]) -> tuple[FlipPlan | None, str]:
    """Plan a flip of `brick_body` from the current state (arms should be at
    home). Returns (plan, how it lies) - plan None if no strategy works."""
    R = data.xmat[brick_body].reshape(3, 3).copy()
    lies = lying(R)
    if lies not in _CASES:
        return None, lies
    center = data.xpos[brick_body] + R @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
    planner = _Planner(model, data, brick_body, other_bricks, arms_order)
    return planner.plan(lies, R, center), lies


def _handover_check(model, geo: _Geoms, me: str, other: str):
    """ArmClearance replacement for the moves near the other hand: links only."""
    def check(path, what, ctx):
        d = ctx.data
        q0 = d.qpos.copy()
        try:
            for q in path:
                d.qpos[model.jnt_qposadr[ctx.joint_ids]] = q
                mujoco.mj_kinematics(model, d)
                gap = geo.links_gap(d, me, other)
                if gap < _MIN_LINKS:
                    raise IKError(f"{what}: arm links {gap * 100:.1f} cm apart during the handover - arm not moved")
        finally:
            d.qpos[:] = q0
            mujoco.mj_kinematics(model, d)
    return check


def _joint_move(ctx: ZebraArmContext, render, clock, q_target, what, gentle=True):
    """Straight joint-space move (like go_home), checked against the other arm."""
    q0 = ctx.data.qpos[ctx.model.jnt_qposadr[ctx.joint_ids]].copy()
    n = max(20, int(np.ceil(np.abs(q_target - q0).max() / (_HELD_JOINT_STEP if gentle else 0.1))))
    path = [q0 + (i / n) * (q_target - q0) for i in range(1, n + 1)]
    ctx._check_clearance(path, what)
    cartesian_control._confirm_and_follow(ctx.model, ctx.data, render, clock, ctx.arm_ctrl, q0, path, what,
                                          steps_per_wp=20, settle_steps=150, carry=None)


def execute_flip(plan: FlipPlan, contexts: dict, render, clock, look) -> None:
    """Do `plan` for real (contexts: {arm: ZebraArmContext} on the live data).
    `look(data)` returns the held brick's (rotation, centre) as a camera sees it.
    Raises RuntimeError / IKError if a step fails (fingers miss, a move is refused)."""
    A, B = contexts[plan.a_side], contexts[plan.b_side]
    model, data = A.model, A.data
    geo = _Geoms(model)
    a_body = HAND_LOCAL_OFFSET / np.linalg.norm(HAND_LOCAL_OFFSET)
    _hold(model, data, render, clock, A.this_arm, _GRIP_OPEN, 60)
    _hold(model, data, render, clock, B.this_arm, _GRIP_OPEN, 60)

    # 1. A picks it as it lies
    hover = plan.center + [0, 0, _HOVER_DZ]
    A.grip_yaw, A.holding = 0.0, False
    A.go(render, clock, hover)
    A.go_oriented(render, clock, hover)
    _oriented_approach(A, render, clock, hover, plan.center, [plan.grip_a])
    _close_and_settle(A, render, clock)
    if abs(A.grip_width() - plan.wA) > _WIDTH_TOL:
        _hold(model, data, render, clock, A.this_arm, _GRIP_OPEN, 100)
        A.go(render, clock, hover)
        raise RuntimeError(f"flip: {A.arm} arm missed the brick (fingers at {A.grip_width() * 100:.1f} cm, "
                           f"expected {plan.wA * 100:.1f})")
    A.holding = True
    A.go(render, clock, hover)

    # 2. A carries it to the handover spot, rolling it
    _joint_move(A, render, clock, plan.q_a_roll, f"{A.arm} arm rolls the brick")

    # 3. B comes in, hand tilted, and grips it (handover mode near A's hand)
    B.holding = False
    _joint_move(B, render, clock, plan.q_b_pre, f"{B.arm} arm to the handover", gentle=False)
    handover = _handover_check(model, geo, B.arm, A.arm)
    # Look again (B's hand camera, 8 cm away): the brick never sits exactly at A's grip
    # point (it hung 1.4 cm low in testing), so B re-aims at where it really is -
    # position, and turn about vertical.
    Rb_seen, center_seen = look(data)
    offset = center_seen - plan.handover
    dyaw = _wrap(np.arctan2(Rb_seen[1, 0], Rb_seen[0, 0]) - np.arctan2(plan.brick_R[1, 0], plan.brick_R[0, 0]))
    if abs(abs(dyaw) - np.pi) < np.pi / 2:  # a brick looks the same turned 180 deg about its studs
        dyaw = _wrap(dyaw + np.pi)
    Rc = _rot([0, 0, 1.0], dyaw)
    b_point = plan.b_point + offset
    b_quat = _quat(Rc @ _mat(plan.b_quat_tilted))
    B_check, B._check_clearance = B._check_clearance, (lambda p, w: handover(p, w, B))
    try:
        move_to_pose(model, data, render, clock, B.arm_ctrl, B.body_id, HAND_LOCAL_OFFSET, B.joint_ids,
                     b_point, b_quat, lead_in_step=_HELD_JOINT_STEP, path_check=B._check_clearance)
    finally:
        B._check_clearance = B_check
    _close_and_settle(B, render, clock)
    if abs(B.grip_width() - plan.wB) > _WIDTH_TOL:
        raise RuntimeError(f"flip: {B.arm} arm missed the brick at the handover "
                           f"(fingers at {B.grip_width() * 100:.1f} cm, expected {plan.wB * 100:.1f})")

    # 4. A lets go, backs straight out, goes home
    _hold(model, data, render, clock, A.this_arm, _GRIP_OPEN, 150)
    A.holding, B.holding = False, True
    back = _handover_check(model, geo, A.arm, B.arm)
    A_check, A._check_clearance = A._check_clearance, (lambda p, w: back(p, w, A))
    try:
        _joint_move(A, render, clock, plan.q_a_out, f"{A.arm} arm backs out", gentle=True)
    finally:
        A._check_clearance = A_check
    go_home(A, render, clock)

    # 5. B turns the rest of the way (at the spot, or on the way down) and sets it down upright
    B.grip_yaw = plan.grip_b
    spot_hover = plan.set_down + [0, 0, _HOVER_DZ]
    if plan.q_b_final is not None:
        turn = plan.q_b_final
        try:
            here = data.xpos[B.body_id] + data.xmat[B.body_id].reshape(3, 3) @ HAND_LOCAL_OFFSET
            turn = solve_ik_pose(model, data, B.body_id, HAND_LOCAL_OFFSET, B.joint_ids, here,
                                 _quat(Rc @ _mat(plan.b_quat_final)), q_init=plan.q_b_final, iters=800)
        except IKError:
            pass
        _joint_move(B, render, clock, turn, f"{B.arm} arm turns the brick upright")
    else:
        q_h = solve_ik_pose(model, data, B.body_id, HAND_LOCAL_OFFSET, B.joint_ids, spot_hover,
                            _yawed(B.grip_quat, plan.grip_b), q_init=data.qpos[model.jnt_qposadr[B.joint_ids]].copy(),
                            iters=800)
        _joint_move(B, render, clock, q_h, f"{B.arm} arm turns the brick upright on the way")
    B.go(render, clock, spot_hover)
    B.go_oriented(render, clock, spot_hover)
    B.go_oriented(render, clock, plan.set_down)
    _hold(model, data, render, clock, B.this_arm, _GRIP_OPEN, 300)
    B.holding = False
    B.go(render, clock, spot_hover)
