#!/usr/bin/env python3
"""A robot hand clicks a DUPLO brick onto another standing on the table - an experiment.

Like two_hand_click.py (same clicking bricks, same line-up), but the lower brick (legs)
stands on the table, which takes the push. The push goes in like a person does it: set the
brick on the studs, let go, lift clear, then press down on its top with the closed gripper
- solid contact, nothing to slip in the fingers.

  1. Legs upright on the table; body upright nearby. With --hold the legs start on their
     side and the LEFT hand does the one-hand flip move - grip the ends, roll upright, set
     them on the table - but keeps holding them from the side (top free), so nothing can
     slide, like a person holding the bottom brick. Without --hold the left arm stays parked.
  2. Right hand picks the body, lifts it, goes across high above the legs.
  3. Line up 3 mm above the legs' studs: a close look at both bricks and a correction,
     ALIGN_STEPS times (position, turn and tilt), adding the arm's sag to the aim (it
     sags ~6.5 mm under load: re-aiming at the same spot left 2.7 mm across).
  4. Lower until the body sits on the studs (in their rounded tops), open, lift CLEAR (the
     fingers are 7.8 cm long: lifted only 3 cm, closing them grabbed the body again,
     seated it off-centre and shoved both bricks 4 mm across the table).
  5. Close the gripper and press down on the body's top, a little at a time, looking at the
     gap in between: stop when seated, or at MAX_PUSH_N.
  6. Check: (left hand lets go,) pick the body up again - do the legs come along?
Pictures go to logs/table_click/ (not with --watch).

usage (WSL, repo folder):
  MUJOCO_GL=egl python3 experiments/table_click.py [--hold] [--seed N]   # headless, pictures
  python3 experiments/table_click.py --watch [--hold]                    # watch it in a window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_hand_click import (  # noqa: E402  (also puts src/ on the path)
    H, MM, ROOT, STUD_H, Look, _C, _R, _grip_point, _set_pose, _yaw, build,
)
from brick_click import contact_force  # noqa: E402
from mjrobots import sim_step  # noqa: E402
from mjrobots.cartesian_control import HAND_LOCAL_OFFSET, IKError, move_to_pose  # noqa: E402
from mjrobots.pick_place import _RealtimeClock, _ThrottledSync  # noqa: E402
from mjrobots.scatter import ON_SIDE, _PLACE_POSES, _axis_rot, lying  # noqa: E402
from mjrobots.stationlite_pick_place import _GRIP_CLOSED, _GRIP_OPEN, _hold  # noqa: E402
from mjrobots.zebra_flip import (  # noqa: E402
    _HELD_JOINT_STEP, _Geoms, _handover_check, _joint_move, _pick_as_it_lies, _Planner, _quat,
)
from mjrobots.zebra_pick_place import ZebraArmContext, _BRICK_CENTER_OFFSET_Z, go_home, grasp_part  # noqa: E402

ALIGN_STEPS = 6
# line up this far above the stud tops (tried 8 mm: the longer last move down let go of the
# body before it was down on the studs and it fell over, 8 of 8 runs)
ABOVE = 3 * MM
LINED_UP = (0.6 * MM, 1.0)  # seen within this (sideways, deg): stop lining up
LEGS_XY, BODY_XY = (0.40, 0.06), (0.36, -0.08)  # in the work box, both reachable by the right arm
# both start upright turned so the print (each brick's -y face) faces FORWARD = world -Y, the
# zebra build's convention (zebra_facing.FACING_YAW) - the body then hardly has to turn
START_YAW = 0.0
LEGS_SIDE_XY = (0.36, 0.08)  # --hold: legs on their side here (the left hand's roll works from it)
SET_DOWN_INTO = 1.5 * MM  # lower the body this far onto the studs' rounded tops before letting go
LIFT_CLEAR = 0.09  # m up after letting go, before closing: more than the fingers' 7.8 cm
PRESS_STEP = 0.5 * MM  # each press step goes this much deeper
PRESS_STEPS = 24  # from 1 mm above the stud tops; closed fingertips may go between the studs to the top face
SEATED_MM = 0.3
MAX_PUSH_N = 15.0  # never push harder (brick_click: 5-7 N clicks 1 mm off)
OUT = ROOT / "logs" / "table_click"


def _snap(model, data, name, target):
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = target
    cam.distance, cam.azimuth, cam.elevation = 0.30, 180, -10
    renderer.update_scene(data, cam)
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(renderer.render()).save(OUT / name)
    renderer.close()
    print(f"   picture: logs/table_click/{name}")


def run(seed: int = 0, hold: bool = False, watch: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    model = build()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    legs, body, head = (model.body(n).id for n in ("zebra_legs", "zebra_body", "zebra_head"))
    hadr = model.jnt_qposadr[model.body_jntadr[head]]
    data.qpos[hadr:hadr + 3] = (0.1, 0.45, -0.08)  # out of the way
    if hold:  # on its side, this side down: the left hand's roll ends upright
        _set_pose(model, data, legs, LEGS_SIDE_XY, _axis_rot(2, 90) @ _PLACE_POSES[ON_SIDE][1])
    else:
        _set_pose(model, data, legs, LEGS_XY, _axis_rot(2, START_YAW))
    _set_pose(model, data, body, BODY_XY, _axis_rot(2, START_YAW))
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
            _snap(model, data, f"{name}_seed{seed}{'_hold' if hold else ''}.png", target)

    wait(1.0)  # bricks settle
    left = ZebraArmContext(model, data, "left", "zebra_legs")
    right = ZebraArmContext(model, data, "right", "zebra_body")
    look = Look(data, rng)

    # 1. legs: on the table (--hold: the left hand rolls them upright and keeps holding)
    if hold:
        R0 = data.xmat[legs].reshape(3, 3).copy()
        centre = data.xpos[legs] + R0 @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
        plan = _Planner(model, data, legs, [body, head], ["left", "right"]).attempt_solo(
            ON_SIDE, R0, centre, "left", "middle")
        if plan is None:
            raise RuntimeError("the left hand can't do the one-hand roll for the legs from here")
        _hold(model, data, render, clock, left.this_arm, _GRIP_OPEN, 60)
        _pick_as_it_lies(left, plan, render, clock)
        _joint_move(left, render, clock, plan.q_a_hover, "left arm rolls the legs upright")
        move_to_pose(model, data, render, clock, left.arm_ctrl, left.body_id, HAND_LOCAL_OFFSET, left.joint_ids,
                     plan.set_down + plan.hold_offset, plan.a_quat, lead_in_step=_HELD_JOINT_STEP,
                     path_check=left._check_clearance)
        wait(0.3)
        print(f"0. left hand set the legs {lying(data.xmat[legs].reshape(3, 3))} on the table at "
              f"({data.xpos[legs][0]:.3f}, {data.xpos[legs][1]:+.3f}) and keeps holding them from the side")
        near_right = _handover_check(model, _Geoms(model), "right", "left")
        right._check_clearance = lambda p, w: near_right(p, w, right)  # links only, near the other hand
    else:
        go_home(left, render, clock)
    legs_start = data.xpos[legs].copy()

    # 2. pick the body, lift, across high above the legs
    Rb = data.xmat[body].reshape(3, 3).copy()
    bc = data.xpos[body] + Rb @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
    grasp_part(right, render, clock, bc.copy(), relook=lambda: (bc.copy(), _yaw(Rb)))
    right.go(render, clock, _grip_point(right) + [0, 0, 0.10])
    print(f"1. right hand holds the body (fingers at {right.grip_width() * 100:.2f} cm)")

    sag = np.zeros(3)  # where the hand ends minus where it was sent

    def move_body_to(bottom_target, R_legs, learn=True):
        """Right hand so the body's bottom centre lands on `bottom_target`, turned and tilted
        like the legs (R_legs, as seen), from a look at the body; the last move's sag added."""
        nonlocal sag
        Rb_seen, _, bottom_seen = look(body)
        # exactly like the legs, print on the same side - it would also fit turned 180 deg, and
        # taking whichever needed less wrist turn put the prints on opposite sides
        dR = R_legs @ Rb_seen.T
        R_hand = data.xmat[right.body_id].reshape(3, 3).copy()
        sent = bottom_target - dR @ (bottom_seen - _grip_point(right)) - sag
        move_to_pose(model, data, render, clock, right.arm_ctrl, right.body_id, HAND_LOCAL_OFFSET,
                     right.joint_ids, sent, _quat(dR @ R_hand), path_check=right._check_clearance)
        wait(0.2)
        if learn:  # half of it: the sag isn't the same every move, all of it swung +-4 mm
            # (tried ignoring readings 3 mm off the last one: fewer bricks clicked, 4/8 vs 4/4)
            sag = 0.5 * sag + 0.5 * (_grip_point(right) - sent)

    def truth():
        """(along, across, turn deg, tilt deg, gap mm) of the body vs the legs - true pose."""
        Rl, Rb_ = data.xmat[legs].reshape(3, 3), data.xmat[body].reshape(3, 3)
        off = Rl.T @ ((data.xpos[body] - Rb_[:, 2] * H) - data.xpos[legs])
        turn = np.degrees((_yaw(Rl) - _yaw(Rb_) + np.pi / 2) % np.pi - np.pi / 2)
        tilt = np.degrees(np.arccos(np.clip(Rl[:, 2] @ Rb_[:, 2], -1, 1)))
        return off[0] / MM, off[1] / MM, turn, tilt, off[2] / MM

    def slid():
        return np.linalg.norm(data.xpos[legs][:2] - legs_start[:2]) / MM

    Rl, top_l, _ = look(legs)
    move_body_to(top_l + Rl[:, 2] * (STUD_H + ABOVE + 0.08), Rl, learn=False)  # a long move: its sag differs

    # 3. line up just above the studs - until it looks lined up (within LINED_UP), at most ALIGN_STEPS
    for k in range(ALIGN_STEPS):
        Rl, top_l, _ = look(legs)
        move_body_to(top_l + Rl[:, 2] * (STUD_H + ABOVE), Rl)
        a, c, t, ti, _ = truth()
        print(f"2.{k + 1} lined up: {a:+.2f} mm along, {c:+.2f} mm across, {t:+.1f} deg turned, {ti:.1f} deg tilted")
        Rl, top_l, _ = look(legs)
        Rb_seen, _, bottom_b = look(body)
        off = Rl.T @ (bottom_b - (top_l + Rl[:, 2] * (STUD_H + ABOVE)))
        turn = abs(np.degrees((_yaw(Rl) - _yaw(Rb_seen) + np.pi / 2) % np.pi - np.pi / 2))
        if np.linalg.norm(off[:2]) < LINED_UP[0] and turn < LINED_UP[1]:
            break
    snap("1_lined_up", data.xpos[legs])

    # 4. set it on the studs, let go, lift clear
    Rl, top_l, _ = look(legs)
    move_body_to(top_l + Rl[:, 2] * (STUD_H - SET_DOWN_INTO), Rl)
    _hold(model, data, render, clock, right.this_arm, _GRIP_OPEN, 150)
    right.holding = False
    right.go(render, clock, _grip_point(right) + [0, 0, LIFT_CLEAR])
    wait(0.5)
    a, c, t, ti, gap = truth()
    print(f"3. set on the studs, let go, lifted clear: {gap:+.2f} mm above seated, {ti:.1f} deg tilted, "
          f"{a:+.2f} / {c:+.2f} mm off; legs slid {slid():.1f} mm")
    snap("2_set_on_studs", data.xpos[legs])

    # 5. close the gripper, press down on the body's top a little at a time
    _hold(model, data, render, clock, right.this_arm, _GRIP_CLOSED, 150)
    R_hand = data.xmat[right.body_id].reshape(3, 3).copy()
    depth = -1.0 * MM  # start 1 mm above the stud tops
    push, k = 0.0, 0
    def seated_now():
        """Seated, by LOOKS: the gap (average of 5 looks - one look's noise once said 'seated' at
        +0.9 mm and no press was made) under SEATED_MM, and flat (tilt under 1 deg)."""
        gaps, tilts = [], []
        for _ in range(5):
            Rb_s, _, bottom_s = look(body)
            Rl_s, top_s, _ = look(legs)
            gaps.append(Rl_s[:, 2] @ (bottom_s - top_s))
            tilts.append(np.degrees(np.arccos(np.clip(Rl_s[:, 2] @ Rb_s[:, 2], -1, 1))))
        return np.mean(gaps) < SEATED_MM * MM and np.mean(tilts) < 1.0

    for k in range(PRESS_STEPS):
        Rb_seen, top_b, bottom_b = look(body)
        push = contact_force(model, data, legs, body)
        if seated_now() or push > MAX_PUSH_N:
            break
        depth += PRESS_STEP
        try:
            move_to_pose(model, data, render, clock, right.arm_ctrl, right.body_id, HAND_LOCAL_OFFSET,
                         right.joint_ids, top_b + Rb_seen[:, 2] * (STUD_H - depth), _quat(R_hand),
                         path_check=right._check_clearance)
        except IKError as e:
            print(f"   press move refused: {e}")
            break
        wait(0.2)
    a, c, t, ti, gap = truth()
    seated = abs(gap) < 0.5 and ti < 1.0
    print(f"4. pressed ({k} press steps, push {push:.1f} N): {gap:+.2f} mm from seated, {ti:.1f} deg tilted "
          f"-> {'SEATED' if seated else 'not seated'}; legs slid {slid():.1f} mm in all")
    snap("3_pressed", data.xpos[legs])

    # 6. (left hand lets go,) pick the body up again: do the legs come along?
    right.go(render, clock, _grip_point(right) + [0, 0, LIFT_CLEAR])
    _hold(model, data, render, clock, right.this_arm, _GRIP_OPEN, 150)
    if hold:  # the right hand out of the way first - it's right above where the left one leaves
        go_home(right, render, clock)
        _hold(model, data, render, clock, left.this_arm, _GRIP_OPEN, 150)
        left.holding = False
        here = _grip_point(left)
        R_left = data.xmat[left.body_id].reshape(3, 3).copy()
        a_dir = R_left @ (HAND_LOCAL_OFFSET / np.linalg.norm(HAND_LOCAL_OFFSET))
        try:  # back the left fingers straight out along the hand, up clear of the bricks, then home
            move_to_pose(model, data, render, clock, left.arm_ctrl, left.body_id, HAND_LOCAL_OFFSET,
                         left.joint_ids, here - 0.06 * a_dir, _quat(R_left))
            move_to_pose(model, data, render, clock, left.arm_ctrl, left.body_id, HAND_LOCAL_OFFSET,
                         left.joint_ids, here - 0.06 * a_dir + [0, 0, 0.10], _quat(R_left))
            go_home(left, render, clock)
        except IKError as e:
            print(f"   (left hand couldn't back away: {e})")
        del right._check_clearance  # back to the normal check (2 cm from the other arm, 1 cm from bricks)
    Rb = data.xmat[body].reshape(3, 3).copy()
    bc = data.xpos[body] + Rb @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
    legs_z0 = data.xpos[legs][2]
    try:
        grasp_part(right, render, clock, bc.copy(), relook=lambda: (bc.copy(), _yaw(Rb)))
        lifted = (data.xpos[legs][2] - legs_z0) / MM
    except (RuntimeError, IKError) as e:
        print(f"   couldn't pick the body up again: {e}")
        lifted = 0.0
    clicked = lifted > 50
    print(f"5. picked the body up: the legs came up {lifted:.0f} mm -> {'CLICKED TOGETHER' if clicked else 'NOT clicked'}")
    p_legs, p_body = -data.xmat[legs].reshape(3, 3)[:, 1], -data.xmat[body].reshape(3, 3)[:, 1]  # print = -y face
    same_side = float(p_legs @ p_body) > 0.9
    print(f"   prints: {'SAME side' if same_side else 'DIFFERENT sides'} (legs' print points "
          f"({p_legs[0]:+.2f}, {p_legs[1]:+.2f}), body's ({p_body[0]:+.2f}, {p_body[1]:+.2f}))")
    snap("4_lifted", data.xpos[body])
    if viewer is not None:
        print("   (close the window to end)")
        while viewer.is_running():
            wait(0.05)
    return dict(gap_mm=gap, tilt_deg=ti, push_N=push, seated=seated, slid_mm=slid(), lifted_mm=lifted,
                clicked=clicked, same_side=same_side)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold", action="store_true", help="the left hand holds the legs from the side")
    p.add_argument("--watch", action="store_true", help="show it in the MuJoCo window (real speed, no pictures)")
    a = p.parse_args()
    run(a.seed, a.hold, a.watch)
