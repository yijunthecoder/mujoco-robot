#!/usr/bin/env python3
"""The robot builds the zebra by CLICKING the DUPLO bricks together on the table - an experiment.

Like two_hand_click.py (same clicking bricks, same line-up), but the stack stands on the
table, which takes the push. The push goes in like a person does it: set the brick on the
studs, let go, lift clear, then press down on its top with the closed gripper - solid
contact, nothing to slip in the fingers. And like a person, the other hand holds the bottom
brick while the bricks are clicked on.

  1. All three bricks upright on the table, prints forward. The LEFT hand grips the legs by
     their two ends FROM THE SIDE (hand rolled 90 deg, top face free) and keeps holding them
     on the table, so the stack can't slide or tip while it is pressed (without it the legs
     slid 2-3 mm when the body was set on them). --no-hold: the left arm stays parked.
  2. Then for the body onto the legs, and the head onto the body (--bricks 2: body only):
     a. right hand picks the brick, lifts it, goes across high above the stack;
     b. line up 3 mm above the top brick's studs: a close look at both bricks and a
        correction, until it looks lined up (position, turn and tilt), adding the arm's sag
        to the aim (it sags ~6.5 mm under load: re-aiming at the same spot left 2.7 mm);
        always aimed at the brick it goes on AS SEEN NOW, so errors don't pile up;
     c. lower until it sits on the studs (in their rounded tops), open, lift CLEAR (the
        fingers are 7.8 cm long: lifted only 3 cm, closing them grabbed the brick again,
        seated it off-centre and shoved both bricks 4 mm across the table);
     d. close the gripper and press down on its top, a little at a time, looking at the gap
        in between: stop when seated, or at MAX_PUSH_N.
  3. Check: the left hand lets go; the right picks up the top brick - does the whole stack
     come along, flat, prints all on the same side?
Pictures go to logs/table_click/ (not with --watch). Times are robot (sim) time.

usage (WSL, repo folder):
  MUJOCO_GL=egl python3 experiments/table_click.py [--seed N] [--bricks 2] [--no-hold]   # headless, pictures
  python3 experiments/table_click.py --watch                                           # watch it in a window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_hand_click import (  # noqa: E402  (also puts src/ on the path)
    H, MM, ROOT, STUD_H, Look, _C, _R, _clicking_brick, _grip_point, _set_pose, _yaw,
)
from brick_click import contact_force  # noqa: E402
from mjrobots import sim_step  # noqa: E402
from mjrobots.cartesian_control import HAND_LOCAL_OFFSET, IKError, move_to_pose  # noqa: E402
from mjrobots.pick_place import _RealtimeClock, _ThrottledSync  # noqa: E402
from mjrobots.scatter import _axis_rot  # noqa: E402
from mjrobots.stationlite_pick_place import _DEFAULT_SCENE, _GRIP_CLOSED, _GRIP_OPEN, _hold  # noqa: E402
from mjrobots.zebra_flip import (  # noqa: E402
    _MIN_TABLE_GAP, _WIDTH_TOL, _Geoms, _handover_check, _mat, _Planner, _quat, _rot,
)
from mjrobots.zebra_pick_place import (  # noqa: E402
    ZebraArmContext, _BRICK_CENTER_OFFSET_Z, _HOVER_DZ, _close_and_settle, _oriented_approach, _wrap, go_home, grasp_part,
)

ALIGN_STEPS = 6
# line up this far above the stud tops (tried 8 mm: the longer last move down let go of the
# body before it was down on the studs and it fell over, 8 of 8 runs)
ABOVE = 3 * MM
LINED_UP = (0.6 * MM, 1.0)  # seen within this (sideways, deg): stop lining up
# start spots, in the work box (scatter.PLACE_BOX), each reachable by the arm that takes it
# and >= 10 cm from the others
LEGS_XY, BODY_XY, HEAD_XY = (0.40, 0.06), (0.36, -0.08), (0.46, -0.06)
# all start upright turned so the print (each brick's -y face) faces FORWARD = world -Y, the
# zebra build's convention (zebra_facing.FACING_YAW) - the bricks then hardly have to turn
START_YAW = 0.0
SAG_PROBE = 10 * MM  # first move this far above the line-up height, to measure the arm's sag
SET_DOWN_INTO = 1.5 * MM  # lower the brick this far onto the studs' rounded tops before letting go
LIFT_CLEAR = 0.09  # m up after letting go, before closing: more than the fingers' 7.8 cm
PRESS_STEP = 0.5 * MM  # each press step goes this much deeper
PRESS_STEPS = 24  # from 1 mm above the stud tops; closed fingertips may go between the studs to the top face
SEATED_MM = 0.3
MAX_PUSH_N = 15.0  # never push harder (brick_click: 5-7 N clicks 1 mm off)
# the left hand's side grip (see side_hold): hand rolls tried, how far down it may point (as
# the sine), grip heights around the brick's middle, room kept to the brick going on top
_SIDE_ROLLS = np.radians(np.arange(-180, 180, 3))
_SIDE_MAX_DOWN = np.sin(np.radians(25))
_SIDE_HEIGHTS = np.arange(-4, 9, 2) * MM
_SIDE_MIN_UPPER = 2 * MM
OUT = ROOT / "logs" / "table_click"


def build():
    """The Station Lite scene with all three bricks given the clicking studs and tubes."""
    spec = mujoco.MjSpec.from_file(str(_DEFAULT_SCENE))
    for name in ("zebra_legs", "zebra_body", "zebra_head"):
        _clicking_brick(spec, name)
    spec.option.noslip_iterations = 10  # no creep in the fingers (README, "A brick creeps")
    return spec.compile()


def _snap(model, data, name, target):
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = target
    cam.distance, cam.azimuth, cam.elevation = 0.35, 180, -10
    renderer.update_scene(data, cam)
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(renderer.render()).save(OUT / name)
    renderer.close()
    print(f"   picture: logs/table_click/{name}")


def side_hold(model, data, render, clock, ctx, brick, upper, others, look) -> str:
    """`ctx`'s hand grips the upright `brick` on the table FROM THE SIDE and keeps holding it,
    its top face free for `upper` (the brick that goes on it).

    The hand is the usual grip rolled about its finger axis so it comes in from the side:
    fingers on the two ends (the gripper opens 8.5 cm, the brick is 6.4 cm long), or on the
    two long faces. Room is tight: the brick above sits flush with the faces the fingers
    hold, so the fingers must stay below its top edge, and above the table. Measured on
    copies of the scene (`_SIDE_ROLLS` x `_SIDE_HEIGHTS`): the arm's grip points 32 deg off
    vertical, so rolled 90 deg the fingers point down into the table (-3 mm); level, the
    gripper is too thick (-12 mm); ~12 deg down there is ~15 mm of room in all. So every
    roll that points the hand level to 25 deg down, at grip heights around the brick's
    middle, is checked (reachable, fingers clear of the table with the brick on top, closed
    on it) and the one with the most room both ways is used. The hand goes above it, turns,
    comes straight down around the brick and closes; whether it holds is read from the
    finger gap, like a real gripper. Returns which faces it holds and the room."""
    Rl, top, _ = look(brick)
    centre = top - Rl[:, 2] * H / 2
    planner = _Planner(model, data, brick, others, ["left", "right"])
    f_hand = planner.fbody[ctx.arm]  # finger closing axis, hand frame
    R_grip = _mat(ctx.grip_quat)  # the usual grip from above
    upper_geoms = planner.geo.body(upper)
    uadr = model.jnt_qposadr[model.body_jntadr[upper]]
    best = None  # (room, faces, width, target, R)
    for axis, faces, width in ((0, "ends", 0.064), (1, "long faces", 0.032)):
        across = Rl[:, axis]  # the fingers close along this
        f0 = R_grip @ f_hand
        turn = np.arctan2(across[1], across[0]) - np.arctan2(f0[1], f0[0])
        for t in (turn, turn + np.pi):
            R_top = _rot([0, 0, 1.0], t) @ R_grip
            for roll in _SIDE_ROLLS:
                R = _rot(R_top @ f_hand, roll) @ R_top
                if not -_SIDE_MAX_DOWN <= (R @ planner.a_body)[2] <= 0.0:
                    continue  # pointing up, or too far down
                for dz in _SIDE_HEIGHTS:
                    d, cs = planner.fresh()
                    q = planner.ik(d, cs[ctx.arm], centre + [0, 0, dz], R)
                    if q is None:
                        break  # (the other heights won't reach either)
                    planner.set_arm(d, cs[ctx.arm], q)
                    planner.set_fingers(d, cs[ctx.arm], 0.085)  # open, coming down
                    mujoco.mj_kinematics(model, d)
                    to_table = planner.geo.dist(d, planner.geo.grip[ctx.arm], planner.geo.table)
                    planner.set_fingers(d, cs[ctx.arm], width)  # closed on it, the next brick seated on top
                    d.qpos[uadr:uadr + 3] = top + Rl[:, 2] * H
                    d.qpos[uadr + 3:uadr + 7] = _quat(Rl)
                    mujoco.mj_kinematics(model, d)
                    to_table = min(to_table, planner.geo.dist(d, planner.geo.grip[ctx.arm], planner.geo.table))
                    to_upper = planner.geo.dist(d, planner.geo.grip[ctx.arm], upper_geoms)
                    if to_table < _MIN_TABLE_GAP or to_upper < _SIDE_MIN_UPPER:
                        continue
                    room = min(to_table, to_upper)
                    if best is None or room > best[0]:
                        best = (room, faces, width, centre + [0, 0, dz], R, to_table, to_upper)
    if best is None:
        raise RuntimeError(f"no side grip for the {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, brick)} "
                           f"with its fingers clear of the table and of the brick going on top")
    _, faces, width, target, R, to_table, to_upper = best
    ctx.go(render, clock, target + [0, 0, _HOVER_DZ])
    move_to_pose(model, data, render, clock, ctx.arm_ctrl, ctx.body_id, HAND_LOCAL_OFFSET,
                 ctx.joint_ids, target + [0, 0, _HOVER_DZ], _quat(R), path_check=ctx._check_clearance)
    move_to_pose(model, data, render, clock, ctx.arm_ctrl, ctx.body_id, HAND_LOCAL_OFFSET,
                 ctx.joint_ids, target, _quat(R), path_check=ctx._check_clearance)
    _hold(model, data, render, clock, ctx.this_arm, _GRIP_CLOSED, 150)
    if abs(ctx.grip_width() - width) > _WIDTH_TOL:
        raise RuntimeError(f"side grip on the {faces} closed to {ctx.grip_width() * 100:.1f} cm, "
                           f"not {width * 100:.1f} cm")
    ctx.holding = True
    tilt = np.degrees(np.arcsin(-(R @ planner.a_body)[2]))
    return (f"{faces} (hand {tilt:.0f} deg down, fingers {to_table * 1000:.0f} mm above the table, "
            f"{to_upper * 1000:.0f} mm below the next brick)")


def let_go_from_side(model, data, render, clock, ctx) -> None:
    """Open, back the fingers straight out along the hand, up clear of the stack, home."""
    _hold(model, data, render, clock, ctx.this_arm, _GRIP_OPEN, 150)
    ctx.holding = False
    here = _grip_point(ctx)
    R = data.xmat[ctx.body_id].reshape(3, 3).copy()
    out = R @ (HAND_LOCAL_OFFSET / np.linalg.norm(HAND_LOCAL_OFFSET))
    try:
        move_to_pose(model, data, render, clock, ctx.arm_ctrl, ctx.body_id, HAND_LOCAL_OFFSET,
                     ctx.joint_ids, here - 0.06 * out, _quat(R))
        move_to_pose(model, data, render, clock, ctx.arm_ctrl, ctx.body_id, HAND_LOCAL_OFFSET,
                     ctx.joint_ids, here - 0.06 * out + [0, 0, 0.12], _quat(R))
        go_home(ctx, render, clock)
    except IKError as e:
        print(f"   ({ctx.arm} hand couldn't back away: {e})")


def run(seed: int = 0, hold: bool = True, watch: bool = False, bricks: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    model = build()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    legs, body, head = (model.body(n).id for n in ("zebra_legs", "zebra_body", "zebra_head"))
    stack = [legs, body, head][:bricks]
    names = {legs: "legs", body: "body", head: "head"}
    _set_pose(model, data, legs, LEGS_XY, _axis_rot(2, START_YAW))
    _set_pose(model, data, body, BODY_XY, _axis_rot(2, START_YAW))
    if bricks == 3:
        _set_pose(model, data, head, HEAD_XY, _axis_rot(2, START_YAW))
    else:
        hadr = model.jnt_qposadr[model.body_jntadr[head]]
        data.qpos[hadr:hadr + 3] = (0.1, 0.45, -0.08)  # out of the way
    mujoco.mj_forward(model, data)

    viewer = None
    if watch:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        render, clock = _ThrottledSync(viewer, model, step_dt=sim_step.CONTROL_DT), _RealtimeClock(dt=sim_step.CONTROL_DT)
    else:
        render, clock = _R(), _C()

    def wait(seconds):
        for _ in range(int(seconds / sim_step.CONTROL_DT)):
            sim_step.step(model, data)
            clock.tick()
            render.step()

    def snap(name, target):
        if not watch:
            _snap(model, data, f"{name}_seed{seed}{'' if hold else '_nohold'}.png", target)

    def t():
        return f"[{data.time:5.1f} s]"

    wait(1.0)  # bricks settle
    t0 = data.time
    left = ZebraArmContext(model, data, "left", "zebra_legs")
    look = Look(data, rng)
    legs_start = data.xpos[legs].copy()

    def slid():
        return np.linalg.norm(data.xpos[legs][:2] - legs_start[:2]) / MM

    # 1. the left hand holds the legs from the side
    if hold:
        faces = side_hold(model, data, render, clock, left, legs, body, [body, head], look)
        print(f"{t()} 0. left hand holds the legs from the side by their {faces}; legs moved {slid():.1f} mm")
        near_right = _handover_check(model, _Geoms(model), "right", "left")
    else:
        go_home(left, render, clock)

    def truth(lower, upper):
        """(along, across, turn deg, tilt deg, gap mm) of `upper` vs `lower` - true pose."""
        Rl, Ru = data.xmat[lower].reshape(3, 3), data.xmat[upper].reshape(3, 3)
        off = Rl.T @ ((data.xpos[upper] - Ru[:, 2] * H) - data.xpos[lower])
        turn = np.degrees((_yaw(Rl) - _yaw(Ru) + np.pi / 2) % np.pi - np.pi / 2)
        tilt = np.degrees(np.arccos(np.clip(Rl[:, 2] @ Ru[:, 2], -1, 1)))
        return off[0] / MM, off[1] / MM, turn, tilt, off[2] / MM

    def right_hand_for(brick):
        """The right arm, handling `brick` (its checks keep it clear of the other bricks)."""
        ctx = ZebraArmContext(model, data, "right", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, brick))
        if hold:  # links only, near the other hand (which holds the legs, right where this one works)
            ctx._check_clearance = lambda p, w: near_right(p, w, ctx)
        return ctx

    def click(lower, upper, n):
        """2a-d: the right hand clicks `upper` onto `lower` (the top of the stack so far)."""
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, upper).replace("zebra_", "")
        right = right_hand_for(upper)
        Rb = data.xmat[upper].reshape(3, 3).copy()
        bc = data.xpos[upper] + Rb @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
        grasp_part(right, render, clock, bc.copy(), relook=lambda: (bc.copy(), _yaw(Rb)))
        right.go(render, clock, _grip_point(right) + [0, 0, 0.10])
        print(f"{t()} {n}a. right hand holds the {name} (fingers at {right.grip_width() * 100:.2f} cm)")
        sag = None  # where the hand ends minus where it was sent (none measured yet)

        def move_upper_to(bottom_target, R_lower, learn=True):
            """Right hand so the brick's bottom centre lands on `bottom_target`, turned and
            tilted like the brick below (R_lower, as seen), from a look at it; the last
            move's sag added."""
            nonlocal sag
            Ru_seen, _, bottom_seen = look(upper)
            # exactly like the brick below, print on the same side - it would also fit turned
            # 180 deg, and taking whichever needed less wrist turn put the prints on opposite sides
            dR = R_lower @ Ru_seen.T
            R_hand = data.xmat[right.body_id].reshape(3, 3).copy()
            sent = bottom_target - dR @ (bottom_seen - _grip_point(right)) - (0 if sag is None else sag)
            move_to_pose(model, data, render, clock, right.arm_ctrl, right.body_id, HAND_LOCAL_OFFSET,
                         right.joint_ids, sent, _quat(dR @ R_hand), path_check=right._check_clearance)
            wait(0.2)
            if learn:  # half of it: the sag isn't the same every move, all of it swung +-4 mm
                # (tried ignoring readings 3 mm off the last one: fewer bricks clicked, 4/8 vs 4/4)
                # The first reading is taken in full: averaged with "no sag" it left the body
                # 2.7 mm too low - on the studs, rubbing across them through the line-up, which
                # slid it 6-13 mm in the fingers and it fell over when let go.
                now = _grip_point(right) - sent
                sag = now if sag is None else 0.5 * sag + 0.5 * now

        Rl, top_l, _ = look(lower)
        move_upper_to(top_l + Rl[:, 2] * (STUD_H + ABOVE + 0.08), Rl, learn=False)  # a long move: its sag differs
        # measure the sag close by but clear of the studs (it is ~5.5 mm down here: lining up
        # without knowing it put the brick on the studs)
        Rl, top_l, _ = look(lower)
        move_upper_to(top_l + Rl[:, 2] * (STUD_H + ABOVE + SAG_PROBE), Rl)

        # b. line up just above the studs - until it looks lined up (within LINED_UP), at most ALIGN_STEPS
        for k in range(ALIGN_STEPS):
            Rl, top_l, _ = look(lower)
            move_upper_to(top_l + Rl[:, 2] * (STUD_H + ABOVE), Rl)
            a, c, tu, ti, _ = truth(lower, upper)
            print(f"{t()} {n}b.{k + 1} lined up: {a:+.2f} mm along, {c:+.2f} mm across, {tu:+.1f} deg turned, "
                  f"{ti:.1f} deg tilted; legs moved {slid():.1f} mm, left fingers {left.grip_width() * 100:.2f} cm")
            Rl, top_l, _ = look(lower)
            Ru_seen, _, bottom_u = look(upper)
            off = Rl.T @ (bottom_u - (top_l + Rl[:, 2] * (STUD_H + ABOVE)))
            turn = abs(np.degrees((_yaw(Rl) - _yaw(Ru_seen) + np.pi / 2) % np.pi - np.pi / 2))
            if np.linalg.norm(off[:2]) < LINED_UP[0] and turn < LINED_UP[1]:
                break
        snap(f"{n}_1_lined_up", data.xpos[lower])

        # c. set it on the studs, let go, lift clear
        Rl, top_l, _ = look(lower)
        move_upper_to(top_l + Rl[:, 2] * (STUD_H - SET_DOWN_INTO), Rl)
        _hold(model, data, render, clock, right.this_arm, _GRIP_OPEN, 150)
        right.holding = False
        right.go(render, clock, _grip_point(right) + [0, 0, LIFT_CLEAR])
        wait(0.5)
        a, c, tu, ti, gap = truth(lower, upper)
        print(f"{t()} {n}c. set on the studs, let go, lifted clear: {gap:+.2f} mm above seated, {ti:.1f} deg "
              f"tilted, {a:+.2f} / {c:+.2f} mm off; legs moved {slid():.1f} mm")
        snap(f"{n}_2_set_on_studs", data.xpos[lower])

        # d. close the gripper, press down on its top a little at a time
        _hold(model, data, render, clock, right.this_arm, _GRIP_CLOSED, 150)
        R_hand = data.xmat[right.body_id].reshape(3, 3).copy()
        depth = -1.0 * MM  # start 1 mm above the stud tops
        push, k = 0.0, 0

        def seated_now():
            """Seated, by LOOKS: the gap (average of 5 looks - one look's noise once said
            'seated' at +0.9 mm and no press was made) under SEATED_MM, and flat (tilt under 1 deg)."""
            gaps, tilts = [], []
            for _ in range(5):
                Ru_s, _, bottom_s = look(upper)
                Rl_s, top_s, _ = look(lower)
                gaps.append(Rl_s[:, 2] @ (bottom_s - top_s))
                tilts.append(np.degrees(np.arccos(np.clip(Rl_s[:, 2] @ Ru_s[:, 2], -1, 1))))
            return np.mean(gaps) < SEATED_MM * MM and np.mean(tilts) < 1.0

        for k in range(PRESS_STEPS):
            Ru_seen, top_u, _ = look(upper)
            push = contact_force(model, data, lower, upper)
            if seated_now() or push > MAX_PUSH_N:
                break
            depth += PRESS_STEP
            try:
                move_to_pose(model, data, render, clock, right.arm_ctrl, right.body_id, HAND_LOCAL_OFFSET,
                             right.joint_ids, top_u + Ru_seen[:, 2] * (STUD_H - depth), _quat(R_hand),
                             path_check=right._check_clearance)
            except IKError as e:
                print(f"   press move refused: {e}")
                break
            wait(0.2)
        a, c, tu, ti, gap = truth(lower, upper)
        seated = abs(gap) < 0.5 and ti < 1.0
        print(f"{t()} {n}d. pressed ({k} press steps, push {push:.1f} N): {gap:+.2f} mm from seated, {ti:.1f} deg "
              f"tilted -> {'SEATED' if seated else 'not seated'}; legs moved {slid():.1f} mm in all")
        snap(f"{n}_3_pressed", data.xpos[lower])
        right.go(render, clock, _grip_point(right) + [0, 0, LIFT_CLEAR])
        _hold(model, data, render, clock, right.this_arm, _GRIP_OPEN, 150)
        return dict(gap_mm=gap, tilt_deg=ti, push_N=push, seated=seated)

    results = [click(lower, upper, n) for n, (lower, upper) in enumerate(zip(stack, stack[1:]), start=1)]
    t_built = data.time - t0

    # 3. (left hand lets go,) pick up the top brick: does the whole stack come along?
    top = stack[-1]
    if hold:  # the right hand out of the way first - it's right above where the left one leaves
        go_home(right_hand_for(top), render, clock)
        let_go_from_side(model, data, render, clock, left)
    right = ZebraArmContext(model, data, "right", f"zebra_{names[top]}")  # the normal checks again
    Rb = data.xmat[top].reshape(3, 3).copy()
    bc = data.xpos[top] + Rb @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
    z0 = {b: data.xpos[b][2] for b in stack}
    try:  # grasp_part's pick, but lifted straight up with the hand kept as it is: its own lift
        # (position only) let the wrist turn and peeled the head off the body sideways (7.8 deg,
        # 10 mm), though it was seated flat until the lift
        hover = bc + [0, 0, _HOVER_DZ]
        right.grip_yaw, right.holding = 0.0, False
        right.go(render, clock, hover)
        _oriented_approach(right, render, clock, hover, bc, sorted({_wrap(_yaw(Rb)), _wrap(_yaw(Rb) + np.pi)}, key=abs))
        _close_and_settle(right, render, clock)
        right.holding = True
        right.go_oriented(render, clock, hover)
        lifted = {b: (data.xpos[b][2] - z0[b]) / MM for b in stack}
    except (RuntimeError, IKError) as e:
        print(f"   couldn't pick the top brick up again: {e}")
        lifted = {b: 0.0 for b in stack}
    clicked = all(v > 50 for v in lifted.values())
    print(f"{t()} {len(stack) + 1}. picked the {names[top]} up: " +
          ", ".join(f"{names[b]} came up {lifted[b]:.0f} mm" for b in stack) +
          f" -> {'ALL CLICKED TOGETHER' if clicked else 'NOT all clicked'}")
    prints = [-data.xmat[b].reshape(3, 3)[:, 1] for b in stack]  # print = -y face
    same_side = all(float(prints[0] @ p) > 0.9 for p in prints[1:])
    print(f"   prints: {'SAME side' if same_side else 'DIFFERENT sides'} (" +
          ", ".join(f"{names[b]} ({p[0]:+.2f}, {p[1]:+.2f})" for b, p in zip(stack, prints)) + ")")
    print(f"   robot time: built in {t_built:.0f} s, {data.time - t0:.0f} s with the check")
    snap("9_lifted", data.xpos[top])
    if viewer is not None:
        print("   (close the window to end)")
        while viewer.is_running():
            wait(0.05)
    return dict(clicks=results, slid_mm=slid(), lifted_mm=list(lifted.values()), clicked=clicked,
                same_side=same_side, built_s=t_built)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="noise of the looks")
    p.add_argument("--bricks", type=int, default=3, choices=(2, 3), help="2: legs + body only")
    p.add_argument("--no-hold", action="store_true", help="the left hand doesn't hold the legs")
    p.add_argument("--watch", action="store_true", help="show it in the MuJoCo window (real speed, no pictures)")
    a = p.parse_args()
    run(a.seed, not a.no_hold, a.watch, a.bricks)
