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
from .arm_clearance import ArmClearance
from .cartesian_control import (
    ARM_JOINTS,
    HAND_LOCAL_OFFSET,
    MAX_WAYPOINT_JUMP,
    IKError,
    move_to_point,
    move_to_pose,
    _confirm_and_follow,
    _hand_point_and_jac,
)
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
# The look again from hover (grasp_part's `relook`) corrects the aim by up to
# this much (horizontally). Further than that, the first estimate was badly
# wrong or the brick moved: the pick fails so the caller re-locates it. The
# closing jaws already center a brick up to ~3 cm off by themselves.
_MAX_RELOOK_SHIFT = 0.04
# Wrist turns in place (the lead-in of an oriented move) are split into
# joint steps of at most this per 0.04 s waypoint while a brick is held,
# ~0.8 rad/s - gentler than the empty-hand 0.2 rad (5 rad/s), which flung a
# held brick out of the fingers on the up-to-90 deg turn back to square at
# the stack (4 of 5 test cases dropped at 5 rad/s, 1 at 2.5, none at 1.2).
_HELD_LEAD_IN_STEP = 0.03
# place_part's check of where the brick ended up (its `verify`): further than
# this from the target sideways, up/down (e.g. it fell off the stack), or
# turned from square, and the place fails. Good placements land 0.2-1.3 cm
# off, level, within ~1 deg.
_PLACE_TOL_XY = 0.02  # m
_PLACE_TOL_Z = 0.015  # m
_PLACE_TOL_YAW = np.radians(10.0)


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
        # How far the last look again from hover moved the aim (m), or None
        # if there was no look or it couldn't see the brick.
        self.relook_shift: float | None = None
        # Extra rotation of the grip about vertical (rad) on top of grip_quat,
        # set by grasp_part/place_part to match the brick's yaw.
        self.grip_yaw = 0.0
        # The held brick's yaw relative to grip_yaw: 0 if grasped square, pi if
        # grasped the other way round (a brick looks the same turned 180 deg,
        # so either grip works) - place_part turns the brick square from it.
        self.held_yaw = 0.0
        # Whether the last place set the brick down turned 180 deg (same
        # footprint, studs still line up; only the printed face is reversed).
        self.placed_flipped = False
        # Whether the fingers are holding the brick (wrist turns go gentler).
        self.holding = False
        # Every move is checked against the other arm before it runs.
        # the other bricks in the scene (sim: the zebra_* bodies - on the robot, the bricks
        # perception reports) must not be touched by this arm while it handles its own
        others = [b for b in range(model.nbody) if b != self.brick_id
                  and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("zebra_")]
        self.clearance = ArmClearance(model, arm, others)
        # How far the last placed brick was seen from its target: (xy, z, yaw)
        # in m/m/rad, or None if there was no check or nothing saw it.
        self.place_error: tuple[float, float, float] | None = None
        # Brick center the last grasp closed on (after any relook correction)
        # - where put_back sets it down again.
        self.picked_from: np.ndarray | None = None

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

    def _check_clearance(self, path, what) -> None:
        """Refuse a planned move that comes too close to the other arm (see
        arm_clearance.py) - with the held brick carried along, if any."""
        self.clearance.check(
            self.data, self.joint_ids, path, self.body_id,
            self.brick_id if self.holding else None, what,
        )

    def go(self, render, clock, target):
        move_to_point(
            self.model, self.data, render, clock, self.arm_ctrl, self.body_id,
            HAND_LOCAL_OFFSET, self.joint_ids, target, path_check=self._check_clearance,
        )

    def go_oriented(self, render, clock, target):
        """Move keeping the gripper in the grip orientation, turned by grip_yaw
        (turning into it gently while holding a brick)."""
        move_to_pose(
            self.model, self.data, render, clock, self.arm_ctrl, self.body_id,
            HAND_LOCAL_OFFSET, self.joint_ids, target, _yawed(self.grip_quat, self.grip_yaw),
            lead_in_step=_HELD_LEAD_IN_STEP if self.holding else MAX_WAYPOINT_JUMP,
            path_check=self._check_clearance,
        )


def _yawed(quat: np.ndarray, yaw: float) -> np.ndarray:
    """`quat` turned by `yaw` radians about the world vertical."""
    turn = np.empty(4)
    mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), yaw)
    out = np.empty(4)
    mujoco.mju_mulQuat(out, turn, quat)
    return out


def _wrap(angle: float) -> float:
    """`angle` wrapped into [-pi, pi)."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _oriented_approach(ctx: ZebraArmContext, render, clock, hover_xyz, target_xyz, yaws) -> float:
    """Turn the wrist at `hover_xyz` to the first grip yaw in `yaws` that IK
    can reach, then descend to `target_xyz` - trying the next yaw if either
    move is refused. Refusals happen before the arm moves (IKError, see
    cartesian_control._plan_path), so trying is safe. Returns the yaw used."""
    last_error = None
    for yaw in yaws:
        ctx.grip_yaw = yaw
        try:
            ctx.go_oriented(render, clock, hover_xyz)
            ctx.go_oriented(render, clock, target_xyz)
            return yaw
        except IKError as error:
            last_error = error
    raise IKError(
        f"no reachable grip angle among {[round(float(np.degrees(y))) for y in yaws]} deg: {last_error}"
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


def grasp_part(ctx: ZebraArmContext, render, clock, center_xyz, relook=None, grip_yaws=None,
               depth: float = 0.0, lowest: float | None = None) -> None:
    """Approach, descend onto, and grip the brick at `center_xyz` (its
    geometric center, not its body origin - see `_BRICK_CENTER_OFFSET_Z`),
    then lift it clear of the table.

    `depth`: how far below the centre the grip point (= the fingertips: the
    finger mesh ends there) goes. 0, the default, grips the brick's top half;
    deeper holds more of it (zebra_facing's show grasp). `lowest`, if given: the
    grip point never goes below this height, whatever the look says (a look 7 mm low
    put a deep grasp's fingertips on the table, which then stalled them open).

    `relook`, if given, is called once the arm is hovering over the brick,
    before it descends: a no-argument function returning a fresh look at the
    brick as `(center, yaw)` - center in world frame, yaw its rotation about
    vertical from square (radians, or None if not measured) - or None if it
    can't see the brick. The descent then aims at the fresh center - a
    closer, later look than the one `center_xyz` came from, so it catches
    perception error and a brick that has moved since. A shift over
    `_MAX_RELOOK_SHIFT` fails the pick instead (the arm stays at hover); None
    keeps the original aim.

    With a yaw, the grip turns to match it, so the jaws close across the
    brick's narrow side whatever angle it lies at. A brick looks the same
    turned 180 deg, so there are two grips that fit (yaw and yaw + 180): the
    one needing less wrist turn is tried first, the other if IK can't reach
    it (every brick angle has at least one reachable grip, both arms, at all
    three spawn spots). Without a yaw the brick is assumed square.

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
    ctx.grip_yaw = 0.0
    ctx.holding = False
    ctx.go(render, clock, hover_xyz)
    ctx.go_oriented(render, clock, hover_xyz)

    ctx.relook_shift = None
    brick_yaw = 0.0  # assumed square unless the look again measures it
    yaw_seen = False
    look = relook() if relook is not None else None
    if look is not None:
        seen, seen_yaw = look
        shift = np.asarray(seen, dtype=float) - center_xyz
        ctx.relook_shift = float(np.linalg.norm(shift))
        if np.linalg.norm(shift[:2]) > _MAX_RELOOK_SHIFT:
            raise RuntimeError(
                f"brick is {np.linalg.norm(shift[:2]) * 100:.1f} cm from where the pick was aimed "
                f"(looked again from hover) - re-locate it"
            )
        center_xyz = center_xyz + shift
        hover_xyz = center_xyz + np.array([0, 0, _HOVER_DZ])
        if seen_yaw is not None:
            brick_yaw = float(seen_yaw)
            yaw_seen = True

    grips = sorted({_wrap(brick_yaw), _wrap(brick_yaw + np.pi)}, key=abs)
    if grip_yaws is not None:  # a chosen grip only (zebra_facing: the one that gives the wanted facing)
        # a function of the yaw just seen from hover (None if not seen), or fixed grip yaws
        grips = list(grip_yaws(brick_yaw if yaw_seen else None) if callable(grip_yaws) else grip_yaws)
    grip_xyz = center_xyz - np.array([0, 0, depth])
    if lowest is not None:
        grip_xyz[2] = max(grip_xyz[2], lowest)
    grip_yaw = _oriented_approach(ctx, render, clock, hover_xyz, grip_xyz, grips)
    ctx.held_yaw = _wrap(brick_yaw - grip_yaw)
    ctx.picked_from = center_xyz.copy()
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
    ctx.holding = True
    ctx.go(render, clock, hover_xyz)


def place_part(ctx: ZebraArmContext, render, clock, center_xyz, verify=None, facing_yaw=None) -> None:
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

    A brick grasped at an angle (see grasp_part) is turned back square: the
    grip yaw that undoes `ctx.held_yaw` is tried first. If IK can't reach it
    (the arm's grip turn is narrower at the stack), the brick goes down
    turned 180 deg instead - same footprint, studs still line up, only the
    printed face reversed (`ctx.placed_flipped`). With `facing_yaw` (the brick's
    full yaw wanted at the stack, print side included - zebra_facing) only that
    one yaw is tried: turned 180 deg would put the print the wrong way round.

    `verify`, if given, looks at the brick once the arm has let go and backed
    up to hover - the same kind of function as grasp_part's `relook`,
    returning `(center, yaw)` or None. If the brick isn't where it should be
    (see `_PLACE_TOL_*`: slid, fell off the stack, knocked round), the place
    fails with what's wrong, so the caller can find and re-pick it. None
    (nothing saw it) passes unchecked (`ctx.place_error` stays None).
    """
    hover_xyz = center_xyz + np.array([0, 0, _HOVER_DZ])
    ctx.go(render, clock, hover_xyz)
    if facing_yaw is not None:  # an exact yaw (print facing a set way: zebra_facing), no 180 deg fallback
        _oriented_approach(ctx, render, clock, hover_xyz, center_xyz, [_wrap(facing_yaw - ctx.held_yaw)])
        ctx.placed_flipped = False
    else:
        square, flipped = _wrap(-ctx.held_yaw), _wrap(np.pi - ctx.held_yaw)
        ctx.placed_flipped = _oriented_approach(ctx, render, clock, hover_xyz, center_xyz, [square, flipped]) != square
    _hold(ctx.model, ctx.data, render, clock, ctx.this_arm, _GRIP_OPEN, 300)
    ctx.holding = False
    ctx.go(render, clock, hover_xyz)

    # Check where the brick actually ended up, from the hover point.
    ctx.place_error = None
    look = verify() if verify is not None else None
    if look is not None:
        ctx.place_error, problem = placement_error(look, center_xyz)
        if problem:
            raise RuntimeError(f"placed brick didn't stay put: it's {problem}")


def put_back(ctx: ZebraArmContext, render, clock) -> None:
    """Set the held brick back down where grasp_part picked it up (the arm is
    still in the grip it used there), let go, and back up to hover - for when
    it turns out it can't be placed after all."""
    hover_xyz = ctx.picked_from + np.array([0, 0, _HOVER_DZ])
    ctx.go(render, clock, hover_xyz)
    ctx.go_oriented(render, clock, hover_xyz)
    ctx.go_oriented(render, clock, ctx.picked_from)
    _hold(ctx.model, ctx.data, render, clock, ctx.this_arm, _GRIP_OPEN, 300)
    ctx.holding = False
    ctx.go(render, clock, hover_xyz)


def go_home(ctx: ZebraArmContext, render, clock, waypoints: int = 30) -> bool:
    """Park the arm at its home joint angles (the scene's "home" keyframe) -
    a straight move in joint space, no IK: home is known to be clear of the
    table, the other arm and the stack. Checked against the other arm like
    every move. Returns False if it was already there (within 0.02 rad)."""
    qpos_adr = ctx.model.jnt_qposadr[ctx.joint_ids]
    home = ctx.model.key("home").qpos[qpos_adr]
    start = ctx.data.qpos[qpos_adr].copy()
    if np.max(np.abs(home - start)) < 0.02:
        return False
    path = [start + (i / waypoints) * (home - start) for i in range(1, waypoints + 1)]
    what = f"{ctx.arm} arm home"
    ctx._check_clearance(path, what)
    _confirm_and_follow(ctx.model, ctx.data, render, clock, ctx.arm_ctrl, start, path, what,
                        steps_per_wp=20, settle_steps=150, carry=None)
    return True


def placement_error(look, center_xyz) -> tuple[tuple[float, float, float], str]:
    """How far a looked-at brick (`look` = `(center, yaw)`, as from a relook)
    is from where it should be: `((xy, z, yaw) error in m/m/rad, problem)`,
    where `problem` describes what's past the `_PLACE_TOL_*` tolerances -
    e.g. "8.2 cm to the side, -3.9 cm below the target" - or is "" if the
    brick is in place. Yaw is measured from square, mod 180 deg."""
    seen, seen_yaw = look
    off = np.asarray(seen, dtype=float) - np.asarray(center_xyz, dtype=float)
    yaw_off = abs(_wrap(2 * seen_yaw) / 2) if seen_yaw is not None else 0.0
    error = (float(np.linalg.norm(off[:2])), float(off[2]), float(yaw_off))
    problems = []
    if error[0] > _PLACE_TOL_XY:
        problems.append(f"{error[0] * 100:.1f} cm to the side")
    if abs(off[2]) > _PLACE_TOL_Z:
        problems.append(f"{off[2] * 100:+.1f} cm {'above' if off[2] > 0 else 'below'} the target")
    if yaw_off > _PLACE_TOL_YAW:
        problems.append(f"turned {np.degrees(yaw_off):.0f} deg")
    return error, ", ".join(problems)


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
