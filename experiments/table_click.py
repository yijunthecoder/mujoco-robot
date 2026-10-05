#!/usr/bin/env python3
"""One robot hand clicks a DUPLO brick onto another standing on the table - an experiment.

Like two_hand_click.py (same clicking bricks, same line-up), but the lower brick (legs)
stands on the table, which holds it still - so no second hand, and no swivelling in a
two-point grip. The push goes in like a person does it: set the brick on the studs, let
go, then press down on its top with the closed gripper - solid contact, nothing to slip.

  1. Legs upright on the table; body upright nearby; the left arm stays parked.
  2. Right hand picks the body, lifts it, goes across high above the legs.
  3. Line up 3 mm above the legs' studs: a close look at both bricks and a correction, a
     few times (position, turn and tilt).
  4. Lower until the body sits on the studs (in their rounded tops), open, move up.
  5. Close the gripper and press down on the body's top, a little at a time, looking at
     the gap in between: stop when seated, or at MAX_PUSH_N.
  6. Check: pick the body up again - do the legs come along (= clicked)?
Pictures go to logs/table_click/.

usage (WSL, repo folder):  MUJOCO_GL=egl python3 experiments/table_click.py [--seed N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_hand_click import (  # noqa: E402  (also puts src/ on the path)
    ABOVE, H, MM, ROOT, STUD_H, Look, _C, _R, _grip_point, _set_pose, _yaw, build,
)
from brick_click import contact_force  # noqa: E402
from mjrobots.cartesian_control import HAND_LOCAL_OFFSET, IKError, move_to_pose  # noqa: E402
from mjrobots.scatter import _axis_rot, lying  # noqa: E402
from mjrobots.stationlite_pick_place import _GRIP_CLOSED, _GRIP_OPEN, _hold  # noqa: E402
from mjrobots.zebra_flip import _quat, _rot  # noqa: E402
from mjrobots.zebra_pick_place import ZebraArmContext, _BRICK_CENTER_OFFSET_Z, go_home, grasp_part  # noqa: E402

ALIGN_STEPS = 5
LEGS_XY, BODY_XY = (0.40, 0.06), (0.36, -0.08)  # in the work box, both reachable by the right arm
SET_DOWN_INTO = 1.5 * MM  # lower the body this far onto the studs' rounded tops before letting go
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


def run(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    model = build()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    legs, body, head = (model.body(n).id for n in ("zebra_legs", "zebra_body", "zebra_head"))
    hadr = model.jnt_qposadr[model.body_jntadr[head]]
    data.qpos[hadr:hadr + 3] = (0.1, 0.45, -0.08)  # out of the way
    _set_pose(model, data, legs, LEGS_XY, _axis_rot(2, 90))
    _set_pose(model, data, body, BODY_XY, _axis_rot(2, 90))
    mujoco.mj_forward(model, data)
    for _ in range(600):
        mujoco.mj_step(model, data)
    render, clock = _R(), _C()
    right = ZebraArmContext(model, data, "right", "zebra_body")
    go_home(ZebraArmContext(model, data, "left", "zebra_legs"), render, clock)
    look = Look(data, rng)
    legs_start = data.xpos[legs].copy()
    print(f"legs {lying(data.xmat[legs].reshape(3, 3))} on the table, body {lying(data.xmat[body].reshape(3, 3))}")

    # 2. pick the body, lift, across high above the legs
    Rb = data.xmat[body].reshape(3, 3)
    bc = data.xpos[body] + Rb @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
    grasp_part(right, render, clock, bc.copy(), relook=lambda: (bc.copy(), _yaw(Rb)))
    right.go(render, clock, _grip_point(right) + [0, 0, 0.10])
    print(f"1. right hand holds the body (fingers at {right.grip_width() * 100:.2f} cm)")

    sag = np.zeros(3)  # where the hand ends minus where it was sent (the arm sags under load)

    def move_body_to(bottom_target, R_legs, learn=True):
        """Right hand so the body's bottom centre lands on `bottom_target`, turned and tilted
        like the legs (R_legs, as seen) - from a look at the body (as two_hand_click.py).
        The last move's sag is added to the aim: re-aiming at the same spot left a steady
        2.7 mm across (each move sagged the same way)."""
        nonlocal sag
        Rb_seen, _, bottom_seen = look(body)
        options = [R_legs, R_legs @ _rot([0, 0, 1.0], np.pi)]
        R_goal = min(options, key=lambda R: np.arccos(np.clip((np.trace(R @ Rb_seen.T) - 1) / 2, -1, 1)))
        dR = R_goal @ Rb_seen.T
        R_hand = data.xmat[right.body_id].reshape(3, 3).copy()
        target = bottom_target - dR @ (bottom_seen - _grip_point(right))
        sent = target - sag
        move_to_pose(model, data, render, clock, right.arm_ctrl, right.body_id, HAND_LOCAL_OFFSET,
                     right.joint_ids, sent, _quat(dR @ R_hand), path_check=right._check_clearance)
        for _ in range(100):  # let it settle before measuring the sag
            mujoco.mj_step(model, data)
        if learn:
            sag = _grip_point(right) - sent

    def truth():
        """(along, across, turn deg, tilt deg, gap mm) of the body vs the legs - true pose."""
        Rl, Rb_ = data.xmat[legs].reshape(3, 3), data.xmat[body].reshape(3, 3)
        off = Rl.T @ ((data.xpos[body] - Rb_[:, 2] * H) - data.xpos[legs])
        turn = np.degrees((_yaw(Rl) - _yaw(Rb_) + np.pi / 2) % np.pi - np.pi / 2)
        tilt = np.degrees(np.arccos(np.clip(Rl[:, 2] @ Rb_[:, 2], -1, 1)))
        return off[0] / MM, off[1] / MM, turn, tilt, off[2] / MM

    Rl, top_l, _ = look(legs)
    move_body_to(top_l + Rl[:, 2] * (STUD_H + ABOVE + 0.08), Rl, learn=False)  # a long move: its sag differs

    # 3. line up just above the studs
    for k in range(ALIGN_STEPS):
        Rl, top_l, _ = look(legs)
        move_body_to(top_l + Rl[:, 2] * (STUD_H + ABOVE), Rl)
        a, c, t, ti, _ = truth()
        print(f"2.{k + 1} lined up: {a:+.2f} mm along, {c:+.2f} mm across, {t:+.1f} deg turned, {ti:.1f} deg tilted "
              f"(arm sag {np.linalg.norm(sag) / MM:.1f} mm, compensated next)")
    _snap(model, data, f"1_lined_up_seed{seed}.png", data.xpos[legs])

    # 4. set it on the studs, let go, move up
    Rl, top_l, _ = look(legs)
    move_body_to(top_l + Rl[:, 2] * (STUD_H - SET_DOWN_INTO), Rl)
    _hold(model, data, render, clock, right.this_arm, _GRIP_OPEN, 150)
    right.holding = False
    right.go(render, clock, _grip_point(right) + [0, 0, 0.03])
    for _ in range(300):
        mujoco.mj_step(model, data)
    a, c, t, ti, gap = truth()
    print(f"3. set on the studs and let go: {gap:+.2f} mm above seated, {ti:.1f} deg tilted, "
          f"{a:+.2f} / {c:+.2f} mm off")
    _snap(model, data, f"2_set_on_studs_seed{seed}.png", data.xpos[legs])

    # 5. close the gripper, press down on the body's top a little at a time
    _hold(model, data, render, clock, right.this_arm, _GRIP_CLOSED, 150)
    R_hand = data.xmat[right.body_id].reshape(3, 3).copy()
    _, top_b, _ = look(body)
    depth = -1.0 * MM  # start 1 mm above the top (studs): fingertips come down onto them
    push = 0.0
    for k in range(PRESS_STEPS):
        Rb_seen, top_b, bottom_b = look(body)
        Rl, top_l, _ = look(legs)
        seen_gap = Rl[:, 2] @ (bottom_b - top_l)
        push = contact_force(model, data, legs, body)
        if seen_gap < SEATED_MM * MM or push > MAX_PUSH_N:
            break
        depth += PRESS_STEP
        target = top_b + Rb_seen[:, 2] * (STUD_H - depth)  # fingertips on the stud tops, then deeper
        try:
            move_to_pose(model, data, render, clock, right.arm_ctrl, right.body_id, HAND_LOCAL_OFFSET,
                         right.joint_ids, target, _quat(R_hand), path_check=right._check_clearance)
        except IKError as e:
            print(f"   press move refused: {e}")
            break
        for _ in range(100):
            mujoco.mj_step(model, data)
    a, c, t, ti, gap = truth()
    seated = abs(gap) < 0.5 and ti < 1.0
    print(f"4. pressed ({k + 1} steps, last push {push:.1f} N): {gap:+.2f} mm from seated, {ti:.1f} deg tilted "
          f"-> {'SEATED' if seated else 'not seated'}; legs moved {np.linalg.norm(data.xpos[legs] - legs_start) / MM:.1f} mm on the table")
    _snap(model, data, f"3_pressed_seed{seed}.png", data.xpos[legs])

    # 6. pick the body up again: do the legs come along?
    right.go(render, clock, _grip_point(right) + [0, 0, 0.03])
    _hold(model, data, render, clock, right.this_arm, _GRIP_OPEN, 150)  # it was closed for the press
    Rb = data.xmat[body].reshape(3, 3)
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
    _snap(model, data, f"4_lifted_seed{seed}.png", data.xpos[body])
    return dict(gap_mm=gap, tilt_deg=ti, push_N=push, seated=seated, lifted_mm=lifted, clicked=clicked)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args().seed)
