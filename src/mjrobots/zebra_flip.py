"""Turn a zebra brick that isn't upright onto its studs, with one arm or both.

A dropped brick lands upright only ~1 time in 6; on its side, on its end or
upside down it can't be gripped from the top and stacked.

On its side, one hand can often do it alone (`SoloPlan`): A picks it by its two
ends, rolls its hand 90 deg - the brick is then upright in its fingers, still
held at its middle - lowers it onto the table and lets go. That needs the
rolled hand to reach down to the table without the gripper touching it, so it
isn't always possible.

Otherwise - and always on its end or upside down, where the turn is bigger than
one wrist can make - the turn is shared in the air:

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
20%. `plan_flip` tries one hand alone setting it down nearby first (on its
side), then those strategies in that order, then one hand setting it down
further away, on a scratch copy of the simulation - planning only, nothing
moves - and returns the first that works; `execute_flip` then does it for real,
with physics.

B only ever grips the brick's middle (`_B_HEIGHTS`): gripping higher lets B's
fingers fit past A's in more poses (A rolls 90, B takes it), but a brick held by
its top strip slid out of B's fingers on the way down in 5 of 6 physics tests
and once live. One hand alone flips most of those instead.

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
from .scatter import (
    MIN_BRICK_GAP, ON_END, ON_SIDE, STACK_CLEAR, STACK_XY, TABLE_Z, UPRIGHT, UPSIDE_DOWN, Zone, lying,
)
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
# How much higher than A's grip B may grip. Only the middle: 0.75 and 1.25 cm used to be
# allowed (they keep the four fingers apart), but a brick held by its top strip slid
# out of B's fingers on the way down - 5 of 6 bricks in physics tests, and seed 1's
# body live. A brick no plan can grip in the middle isn't flipped (FAILED, left as is).
_B_HEIGHTS = (0.0,)
# Set-down spots beyond the ones near the brick: the green zone (where an arm can pick
# the brick up again and stack it - scatter.Zone), one candidate every this many metres.
_FAR_SPOT_SPACING = 0.06
_BACK_OUT = 0.06  # m A's hand backs out along its own fingers after letting go
_B_APPROACH = 0.08  # m B comes in straight from this far back along its own fingers
_MIN_LINKS = 0.02  # m arm links apart (as arm_clearance.MIN_ARM_CLEARANCE)
_MIN_OBSTACLE = 0.01  # m hands / held brick from other bricks
# m gripper from the table when one hand sets the brick down with its hand rolled
# sideways: the gripper then sits beside the brick, only ~2 cm up (its middle).
_MIN_TABLE_GAP = 0.005
_HELD_JOINT_STEP = 0.03  # rad per waypoint for joint moves while holding (gentle, like zpp)
_WIDTH_TOL = 0.005  # m finger gap tolerance for "holding the brick"
# Most spots/turns the planner tries are out of reach; an IK solve stops once it
# has made no progress for this many iterations instead of running all 800
# (5x faster planning; no solve that would have succeeded was lost in testing).
_IK_GIVE_UP = 100

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
    def jump(model, data, render, clock, arm_ctrl, q_start, path, what, *timing, **named):
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
    # B's grip point minus the brick centre once B holds it turned upright (world, m). Not
    # zero: B may grip up to 1.25 cm above A (_B_HEIGHTS), and a set-down aimed as if the
    # brick were centred pushed it into the table and tipped it over (seed 12's legs).
    hold_offset: np.ndarray

    def describe(self) -> str:
        rest = _CASES[self.lies]["total"] - self.roll_a
        how = (f"{self.a_side} arm rolls {np.degrees(self.roll_a):.0f} deg, "
               f"{self.b_side} arm takes it and turns {np.degrees(rest):.0f} deg" if rest > 1e-6
               else f"{self.a_side} arm rolls it 90 deg, {self.b_side} arm takes it")
        return f"{self.lies}: {how}, sets it down upright at ({self.set_down[0]:.2f}, {self.set_down[1]:+.2f})"


@dataclass
class SoloPlan:
    """One hand alone: pick it by the ends, roll 90 deg, set it down upright."""
    lies: str
    a_side: str
    grip_a: float  # A's grip yaw at the pick
    center: np.ndarray  # brick centre as it lies
    wA: float
    q_a_hover: np.ndarray  # A above the set-down spot, rolled (brick upright in its fingers)
    set_down: np.ndarray  # upright brick centre on the table where A sets it down
    a_quat: np.ndarray  # A's hand orientation, rolled
    hold_offset: np.ndarray  # A's grip point minus the brick centre, rolled (world, m)

    def describe(self) -> str:
        return (f"{self.lies}: {self.a_side} arm rolls it 90 deg and sets it down upright itself "
                f"at ({self.set_down[0]:.2f}, {self.set_down[1]:+.2f})")


def free_spots(zone: Zone, center_xy, other_xy, far=False) -> list[np.ndarray]:
    """Table spots (xy) to set a brick down on: where it lay and spots 6 and 10 cm from
    it; or with `far`, the rest of the green zone (nearest first, one every
    _FAR_SPOT_SPACING). Always inside the zone, so an arm can pick it up again and stack
    it, and clear of the other bricks and the stack. The far spots gave one hand alone 3
    more bricks of 6 tried (seeds 3 head, 7 and 8 legs), but searching them all takes
    ~20 s when none works - so the planner tries them last (see `_Planner.plan`)."""
    center_xy = np.asarray(center_xy)[:2]

    def free(xy):
        return (zone.covers(xy) and np.linalg.norm(xy - STACK_XY) >= STACK_CLEAR
                and all(np.linalg.norm(xy - o) >= MIN_BRICK_GAP for o in other_xy))

    near = [xy for xy in [center_xy] + [center_xy + r * np.array([np.cos(a), np.sin(a)])
                                        for r in (0.06, 0.10) for a in np.radians(np.arange(0, 360, 45))]
            if free(xy)]
    if not far:
        return near
    spots = []
    for xy in sorted(zone.cells, key=lambda xy: np.linalg.norm(xy - center_xy)):
        if free(xy) and all(np.linalg.norm(xy - s) >= _FAR_SPOT_SPACING for s in near + spots):
            spots.append(xy)
    return spots


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
        self.table = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")]

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
        self.zone = Zone.load("either")  # where an arm can pick a brick and stack it
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
                                     q_init=s, iters=800, give_up_after=_IK_GIVE_UP)
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

    def set_down_spots(self, center, far=False):
        """Where the brick may be set down upright (`free_spots`), at its upright centre height."""
        return [np.array([x, y, TABLE_TOP + BRICK_HEIGHT / 2])
                for x, y in free_spots(self.zone, center[:2], self.other_xy, far)]

    # --- A's pick, shared by the strategies ---
    def pick(self, d, cA, Rb0, center, a_axis):
        """A picks the brick as it lies, fingers across brick axis `a_axis`, and lifts it
        (plan-only). Returns (grip_a, Rgrasp, q_grasp, rel, q_lift) - rel is the brick's
        pose in A's hand - or None if it can't."""
        axis_b = Rb0[:, a_axis]
        f0 = _mat(cA.grip_quat) @ self.fbody[cA.arm]
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
        return grip_a, Rgrasp, q_grasp, rel, d.qpos[self.model.jnt_qposadr[cA.joint_ids]].copy()

    # --- one hand alone (on its side) ---
    def attempt_solo(self, lies, Rb0, center, a_side, far=False):
        """A picks it by the ends, rolls its hand the whole 90 deg (the brick is then
        upright in its fingers), lowers it onto a set-down spot (`far`: see
        set_down_spots) and lets go."""
        wA = _CASES[ON_SIDE]["wA"]  # fingers across the long side: the ends
        other = "left" if a_side == "right" else "right"
        d, ctx = self.fresh()
        cA = ctx[a_side]
        with _plan_only():
            picked = self.pick(d, cA, Rb0, center, 0)
            if picked is None:
                return None
            grip_a, Rgrasp, _, rel, q_lift = picked
            after_lift = d.qpos.copy(), d.ctrl.copy()
            fA = Rgrasp @ self.fbody[a_side]
            # the brick centre relative to A's grip point is fixed in A's hand
            off_hand = HAND_LOCAL_OFFSET - rel[0] - rel[1] @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
            for sgn in (1, -1):
                if lying(_rot(fA, sgn * np.pi / 2) @ Rb0) != UPRIGHT:
                    continue  # this way round ends upside down
                for gam in _YAW_TURNS:
                    RA = _rot([0, 0, 1.0], gam) @ _rot(fA, sgn * np.pi / 2) @ Rgrasp
                    hold = RA @ off_hand
                    # the brick centre's height above the table once set down (from its box size)
                    height = float(np.abs((RA @ rel[1])[2]) @ self.model.geom_size[self.brick_geoms[0]])
                    for spot in self.set_down_spots(center, far):
                        spot = np.array([spot[0], spot[1], TABLE_TOP + height])
                        d.qpos[:], d.ctrl[:] = after_lift
                        mujoco.mj_forward(self.model, d)
                        hover = spot + hold + [0, 0, _HOVER_DZ]
                        q_h = self.ik(d, cA, hover, RA, seed=q_lift)
                        if q_h is None or not self.joint_path_ok(d, cA, other, q_lift, q_h, rel=rel):
                            continue
                        try:
                            # straight down (the held brick isn't carried in plan-only mode, so the
                            # moves are checked against the other arm only - it's parked at home)
                            cA.holding = False
                            move_to_pose(self.model, d, None, None, cA.arm_ctrl, cA.body_id, HAND_LOCAL_OFFSET,
                                         cA.joint_ids, spot + hold, _quat(RA), lead_in_step=_HELD_JOINT_STEP,
                                         path_check=cA._check_clearance)
                        except IKError:
                            continue
                        self.set_fingers(d, cA, wA)
                        mujoco.mj_kinematics(self.model, d)
                        if self.geo.dist(d, self.geo.grip[a_side], self.geo.table) < _MIN_TABLE_GAP:
                            continue  # the sideways gripper would hit the table
                        try:
                            self.set_fingers(d, cA, 0.085)
                            move_to_pose(self.model, d, None, None, cA.arm_ctrl, cA.body_id, HAND_LOCAL_OFFSET,
                                         cA.joint_ids, hover, _quat(RA), path_check=cA._check_clearance)
                            go_home(cA, None, None)
                        except IKError:
                            continue
                        return SoloPlan(lies, a_side, grip_a, center.copy(), wA, q_h, spot, _quat(RA), hold)
        return None

    # --- one strategy, two hands ---
    def attempt(self, lies, Rb0, center, a_side, roll_a):
        cfg = _CASES[lies]
        b_side = "left" if a_side == "right" else "right"
        rest = cfg["total"] - roll_a
        d, ctx = self.fresh()
        cA, cB = ctx[a_side], ctx[b_side]
        with _plan_only():
            # 1. A picks it by the chosen pair
            picked = self.pick(d, cA, Rb0, center, cfg["a_axis"])
            if picked is None:
                return None
            grip_a, Rgrasp, q_grasp, rel, q_lift = picked
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
                                    q_turn, spot, q_pre, q_b, q_out, hold = plan
                                    return FlipPlan(lies, a_side, b_side, roll_a, grip_a, center.copy(), q_a, S.copy(),
                                                    S + [0, 0, dz], _quat(RBtilt), _quat(RBfinal), grip_b,
                                                    q_turn, spot, cfg["wA"], cfg["wB"], q_pre, q_b, q_out, Rb.copy(),
                                                    hold)
        return None

    def _b_side(self, d, cA, cB, at_s, S, dz, RBtilt, RBfinal, grip_b, rest, rel, center, wA, wB):
        """B's part, from A holding the rolled brick at S. Returns
        (q_b_final, set_down, q_pre, q_b, q_out, hold_offset) or None."""
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
        if not self.joint_path_ok(d, cA, cB.arm, q_out, home_a):  # if B's re-aim tightens it: _leave_for_home
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
        # Where B's grip point is relative to the brick centre once turned upright: the set-down
        # aims B's grip point there, so the brick's centre (not B's grip point) lands on the spot.
        hold = RBfinal @ (HAND_LOCAL_OFFSET - rel_b[0] - rel_b[1] @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z]))
        # B turns the rest: at the handover spot if it can, else on the way down
        q_bf = self.ik(d, cB, PB, RBfinal, seed=q_b) if rest > 1e-6 else q_b
        start_turn = d.qpos.copy(), d.ctrl.copy()
        for q_turn in ([q_bf] if q_bf is not None else []) + [None]:
            d.qpos[:], d.ctrl[:] = start_turn
            mujoco.mj_forward(self.model, d)
            if q_turn is not None:
                if not self.joint_path_ok(d, cB, cA.arm, q_b, q_turn, rel=rel_b):
                    continue
            for spot in self.set_down_spots(center):
                d.qpos[:], d.ctrl[:] = start_turn
                self.set_arm(d, cB, q_turn if q_turn is not None else q_b)
                mujoco.mj_forward(self.model, d)
                hover = spot + hold + [0, 0, _HOVER_DZ]
                if q_turn is None:  # one joint move straight to above the spot, already turned
                    q_h = self.ik(d, cB, hover, RBfinal, seed=q_b)
                    if q_h is None or not self.joint_path_ok(d, cB, cA.arm, q_b, q_h, rel=rel_b):
                        continue
                try:
                    cB.grip_yaw, cB.holding = grip_b, False
                    cB.go(None, None, hover); cB.go_oriented(None, None, hover)
                    cB.go_oriented(None, None, spot + hold)
                    cB.go(None, None, hover)
                except IKError:
                    continue
                return (q_turn, spot, q_pre, q_b, q_out, hold)
        return None

    def plan(self, lies, Rb0, center):
        """Quick and likely first: one hand setting it down near where it lay (on its
        side), then the two-hand strategies, then one hand with the far spots (slow when
        nothing fits: tried last, so a two-hand brick isn't kept waiting ~20 s).
        Tried and dropped (2026-10-01): upside down in two steps (one hand onto its side,
        then the on-side flip) planned 2 more of 26 upside-down bricks, but made a
        "no plan" answer take up to 44-69 s, past zebra_bt's 30 s flip timeout."""
        cfg = _CASES[lies]
        solo = [lambda side: self.attempt_solo(lies, Rb0, center, side)] if lies == ON_SIDE else []
        two = [lambda side, r=r: self.attempt(lies, Rb0, center, side, r) for r in cfg["rolls"]]
        slow = [lambda side: self.attempt_solo(lies, Rb0, center, side, far=True)] if lies == ON_SIDE else []
        for strategies in (solo, two, slow):
            for a_side in self.arms_order:
                for strategy in strategies:
                    p = strategy(a_side)
                    if p is not None:
                        return p
        return None


def plan_flip(model, data, brick_body: int, other_bricks: list[int],
              arms_order: list[str]) -> tuple[FlipPlan | SoloPlan | None, str]:
    """Plan a flip of `brick_body` from the current state (arms should be at
    home). Returns (plan, how it lies) - plan None if no strategy works."""
    R = data.xmat[brick_body].reshape(3, 3).copy()
    lies = lying(R)
    if lies not in _CASES:
        return None, lies
    center = data.xpos[brick_body] + R @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
    planner = _Planner(model, data, brick_body, other_bricks, arms_order)
    return planner.plan(lies, R, center), lies


_worker_model = _worker_data = None  # in the planning process (see PlanningProcess)


def _worker_init(scene_path: str) -> None:
    global _worker_model, _worker_data
    _worker_model = mujoco.MjModel.from_xml_path(scene_path)
    _worker_data = mujoco.MjData(_worker_model)


def _worker_plan(state, brick_body, other_bricks, arms_order):
    qpos, qvel, ctrl = state
    _worker_data.qpos[:], _worker_data.qvel[:], _worker_data.ctrl[:] = qpos, qvel, ctrl
    mujoco.mj_forward(_worker_model, _worker_data)
    return plan_flip(_worker_model, _worker_data, brick_body, other_bricks, arms_order)


class PlanningProcess:
    """plan_flip in a separate process, so it gets its own CPU core. In a thread of a
    process that also runs the viewer, physics and cameras it shares Python's one
    running thread with them: seed 1's body took 42 s there against 18 s alone - past
    zebra_bt's 30 s flip timeout. The process loads the same scene once; each plan
    sends it only the current state (joint positions/velocities, controls), not the
    ~90 MB model."""

    def __init__(self, scene_path: str) -> None:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        # spawn, not fork: the parent runs ROS and viewer threads, which a fork can't copy
        # safely. An executor, not a Pool: if the process dies, the plan raises
        # (BrokenProcessPool) instead of never finishing.
        self._pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                                         initializer=_worker_init, initargs=(str(scene_path),))

    def start(self, data, brick_body: int, other_bricks: list[int], arms_order: list[str]):
        """Start planning from `data`'s current state; returns a Future whose .result()
        is plan_flip's (plan, how it lies)."""
        state = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
        return self._pool.submit(_worker_plan, state, brick_body, list(other_bricks), list(arms_order))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


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


_CLEAR_STEPS = (np.array([0, 0, 0.05]), None, np.array([0, 0, 0.10]))  # up 5 cm, 4 cm further back, up 10 cm


def _leave_for_home(A: ZebraArmContext, render, clock, handover_check, a_body) -> None:
    """A's way home once it has let go and backed out. The plan checked it with
    B where the plan put B, but B re-aims at where the brick really hangs
    (~1 cm off), so the straight way home can come just under 2 cm (seed 1's
    body: 1.8 cm). Then A first moves a little further clear - up, or further
    back along its fingers, checked links-only like the back-out - and tries
    home again. Planning with slack instead cost too many plans (17 -> 12 of 30)."""
    try:
        go_home(A, render, clock)
        return
    except IKError as first:
        refused = first
    model, data = A.model, A.data
    q0 = data.qpos[model.jnt_qposadr[A.joint_ids]].copy()
    R = data.xmat[A.body_id].reshape(3, 3).copy()
    here = data.xpos[A.body_id] + R @ HAND_LOCAL_OFFSET
    normal_check = A._check_clearance
    for step in _CLEAR_STEPS:
        target = here + (step if step is not None else -0.04 * (R @ a_body))
        try:
            q = solve_ik_pose(model, data, A.body_id, HAND_LOCAL_OFFSET, A.joint_ids, target, _quat(R),
                              q_init=q0, iters=800)
            A._check_clearance = lambda p, w: handover_check(p, w, A)  # the small step: links only
            try:
                _joint_move(A, render, clock, q, f"{A.arm} arm moves clear of the other hand", gentle=False)
            finally:
                A._check_clearance = normal_check  # home: the full 2 cm rule again
            go_home(A, render, clock)
            return
        except IKError:
            continue
    raise refused


def _pick_as_it_lies(A: ZebraArmContext, plan, render, clock) -> None:
    """A picks the brick as it lies (plan.grip_a) and lifts it to hover."""
    model, data = A.model, A.data
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


def _set_down_seen(ctx: ZebraArmContext, render, clock, set_down, quat, look) -> None:
    """Lower the held brick straight down onto `set_down` (its centre, upright) and let
    go. Aimed by where a camera sees the brick in the fingers (`look`), not where the
    fingers are: a brick held 1.25 cm off its middle, set down as if centred, was
    pushed into the table and tipped over. The hand keeps its orientation all the way
    (a position-only move to above the spot let the wrist swing, and an on-end brick
    held by its ends swivelled out of upright - seed 3445's head)."""
    model, data = ctx.model, ctx.data
    _, center_seen = look(data)
    grip_point = data.xpos[ctx.body_id] + data.xmat[ctx.body_id].reshape(3, 3) @ HAND_LOCAL_OFFSET
    target = set_down + (grip_point - center_seen)
    hover = target + [0, 0, _HOVER_DZ]
    move_to_pose(model, data, render, clock, ctx.arm_ctrl, ctx.body_id, HAND_LOCAL_OFFSET, ctx.joint_ids,
                 hover, quat, lead_in_step=_HELD_JOINT_STEP, path_check=ctx._check_clearance)
    move_to_pose(model, data, render, clock, ctx.arm_ctrl, ctx.body_id, HAND_LOCAL_OFFSET, ctx.joint_ids,
                 target, quat, lead_in_step=_HELD_JOINT_STEP, path_check=ctx._check_clearance)
    _hold(model, data, render, clock, ctx.this_arm, _GRIP_OPEN, 300)
    ctx.holding = False
    move_to_pose(model, data, render, clock, ctx.arm_ctrl, ctx.body_id, HAND_LOCAL_OFFSET, ctx.joint_ids,
                 hover, quat, path_check=ctx._check_clearance)


def _spots_around(center, z, other_xy) -> list[np.ndarray]:
    """free_spots around `center` (near ones, then the far ones), at centre height `z`."""
    zone = Zone.load("either")
    xys = free_spots(zone, center, other_xy) + free_spots(zone, center, other_xy, far=True)
    return [np.array([x, y, z]) for x, y in xys]


def _set_down_somewhere(ctx: ZebraArmContext, render, clock, spots, quat, look):
    """_set_down_seen on the first of `spots` the arm can reach; returns the spot used.
    The plan's spot comes first, but it was checked with where the plan expected the brick
    in the fingers: once the camera has seen where it really sits (seed 2's legs: tilted
    ~8 deg), the plan's spot can be just out of reach - then the next free one is used.
    Raises IKError if none is reachable (refused moves don't move the arm)."""
    last = None
    for spot in spots:
        try:
            _set_down_seen(ctx, render, clock, spot, quat, look)
            return spot
        except IKError as error:
            last = error
    raise IKError(f"no reachable spot to set the brick down: {last}")


def _execute_solo(plan: SoloPlan, A: ZebraArmContext, render, clock, look, other_xy) -> None:
    """One hand alone: pick it by the ends, roll 90 deg over the spot, set it down, go home."""
    _hold(A.model, A.data, render, clock, A.this_arm, _GRIP_OPEN, 60)
    _pick_as_it_lies(A, plan, render, clock)
    _joint_move(A, render, clock, plan.q_a_hover, f"{A.arm} arm rolls the brick upright")
    spots = [plan.set_down] + _spots_around(plan.set_down, plan.set_down[2], other_xy)
    _set_down_somewhere(A, render, clock, spots, plan.a_quat, look)
    go_home(A, render, clock)


def _turn_about_vertical(R_seen, R_planned) -> float:
    """How far the brick is turned about the vertical from where the plan expected it
    (rad), from the whole rotation between the two. Not from the heading of the brick's
    long side: in an upside-down flip A holds that side pointing straight up, where its
    heading is noise - it read 82-88 deg off on all 4 upside-down bricks tested, B's
    "corrected" grip was then out of reach and every one of those flips failed.
    A brick looks the same turned 180 deg about its studs, so with the studs about
    vertical the smaller of the two turns is taken."""
    rel = R_seen @ R_planned.T
    turn = float(np.arctan2(rel[1, 0] - rel[0, 1], rel[0, 0] + rel[1, 1]))
    if abs(R_seen[2, 2]) > np.cos(np.radians(30)) and abs(turn) > np.pi / 2:
        turn = _wrap(turn + np.pi)
    return turn


def _put_down_after_failure(plan, contexts: dict, render, clock, look, other_xy) -> str:
    """After a flip failed halfway: if a hand still holds the brick, lower it onto a free
    table spot (free_spots: in the green zone, clear of the stack and the other bricks)
    and only then let go. Opening in mid-air dropped it ~10 cm, often right next to the
    stack (the handover spots are beside it). With both hands on it (mid-handover) A lets
    go first. Returns what was done, for the failure message."""
    A = contexts[plan.a_side]
    B = contexts[plan.b_side] if isinstance(plan, FlipPlan) else None
    model, data = A.model, A.data
    size = model.geom_size[next(g for g in range(model.ngeom)
                                if model.geom_bodyid[g] == A.brick_id and model.geom_contype[g])]

    def holding(c):  # the fingers stopped at one of the brick's widths (read from the gripper)
        return c is not None and min(abs(c.grip_width() - 2 * s) for s in size) <= _WIDTH_TOL

    holders = [c for c in (B, A) if holding(c)]
    if not holders:
        return "no hand was holding the brick"
    if len(holders) == 2:  # mid-handover: B keeps it
        _hold(model, data, render, clock, A.this_arm, _GRIP_OPEN, 150)
        A.holding = False
    c = holders[0]
    c.holding = True
    quat = data.xquat[c.body_id].copy()  # keep the hand as it is: the brick stays the way it lies
    R_seen, _ = look(data)
    height = float(np.abs(R_seen[2]) @ size)  # centre height above the table, lying as it is
    try:
        spot = _set_down_somewhere(c, render, clock, _spots_around(plan.center, TABLE_TOP + height, other_xy),
                                   quat, look)
        return f"the {c.arm} arm set the brick down on the table at ({spot[0]:.2f}, {spot[1]:+.2f})"
    except IKError:
        pass
    _hold(model, data, render, clock, c.this_arm, _GRIP_OPEN, 150)
    c.holding = False
    return f"no free spot was reachable - the {c.arm} arm let go where it was"


def execute_flip(plan: FlipPlan | SoloPlan, contexts: dict, render, clock, look, other_xy=()) -> None:
    """Do `plan` for real (contexts: {arm: ZebraArmContext} on the live data).
    `look(data)` returns the held brick's (rotation, centre) as a camera sees it - at the
    handover (B re-aims its grip) and before the set-down (aimed by where the brick really
    sits in the fingers). `other_xy`: where the other bricks are (kept clear if a failed
    flip has to put the brick down somewhere).
    Raises RuntimeError if a step fails (fingers miss, a move is refused) - after putting
    a still-held brick down on the table (`_put_down_after_failure`)."""
    try:
        _execute_flip(plan, contexts, render, clock, look, other_xy)
    except Exception as exc:
        try:
            what = _put_down_after_failure(plan, contexts, render, clock, look, other_xy)
        except Exception as again:
            what = f"putting it down failed too: {again}"
        raise RuntimeError(f"{exc} ({what})") from exc


def _execute_flip(plan: FlipPlan | SoloPlan, contexts: dict, render, clock, look, other_xy) -> None:
    if isinstance(plan, SoloPlan):
        _execute_solo(plan, contexts[plan.a_side], render, clock, look, other_xy)
        return
    A, B = contexts[plan.a_side], contexts[plan.b_side]
    model, data = A.model, A.data
    geo = _Geoms(model)
    a_body = HAND_LOCAL_OFFSET / np.linalg.norm(HAND_LOCAL_OFFSET)
    _hold(model, data, render, clock, A.this_arm, _GRIP_OPEN, 60)
    _hold(model, data, render, clock, B.this_arm, _GRIP_OPEN, 60)

    # 1. A picks it as it lies
    _pick_as_it_lies(A, plan, render, clock)

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
    dyaw = _turn_about_vertical(Rb_seen, plan.brick_R)
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
    _leave_for_home(A, render, clock, back, a_body)

    # 5. B turns the rest of the way (at the spot, or on the way down) and sets it down upright
    B.grip_yaw = plan.grip_b
    spot_hover = plan.set_down + plan.hold_offset + [0, 0, _HOVER_DZ]
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
    # (the camera looking at the brick in B's hand: A's hand camera, backed out, faces it)
    spots = [plan.set_down] + _spots_around(plan.set_down, plan.set_down[2], other_xy)
    _set_down_somewhere(B, render, clock, spots, _yawed(B.grip_quat, plan.grip_b), look)
