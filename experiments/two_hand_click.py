#!/usr/bin/env python3
"""Both robot hands click two DUPLO bricks together in the air - an experiment.

Uses the robot code (IK, grasping, the one-hand flip move) but changes nothing in it. The
scene is the real Station Lite scene, with the legs and body given the clicking stud/tube
geometry found in brick_click.py (0.01 mm squeeze, 0.05 mm give); the head is moved away.

  1. The legs start ON THEIR SIDE, the body UPRIGHT, both in the work box.
  2. Left hand: the one-hand flip move - grip the legs by their ends, roll 90 deg - so it
     holds them upright FROM THE SIDE, top face (studs) free, in the air. (A grip from
     above would cover the top face.)
  3. Right hand: picks the body from above, as usual.
  4. Line up: the right hand brings the body 3 mm above the legs' studs; a close look at
     both bricks (CLOSE_LOOK_SIGMA: assumed hand-camera accuracy at a few cm - to check on
     the real camera) and a correction, a few times.
  5. Press: down to PRESS_MM past seated (brick_click: ~5-7 N clicks 1 mm off; each hand
     holds ~10 N before the brick slips in its fingers).
  6. Check: seated flat? The right hand lets go and moves away; the left hand moves the
     pair - does the body come along (= clicked)?
Pictures of each step go to logs/two_hand_click/.

usage (WSL, repo folder):  MUJOCO_GL=egl python3 experiments/two_hand_click.py [--seed N]
  --seed: noise of the looks (different runs = different small look errors)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brick_click import CHAMFER, H, MM, STUD_H, STUD_R, STUDS, TUBE_BOTTOM, TUBES, WALL  # noqa: E402
from mjrobots.cartesian_control import HAND_LOCAL_OFFSET, IKError, move_to_pose  # noqa: E402
from mjrobots.scatter import ON_SIDE, _PLACE_POSES, _axis_rot, lying  # noqa: E402
from mjrobots.stationlite_pick_place import _DEFAULT_SCENE, _GRIP_OPEN, _hold  # noqa: E402
from mjrobots.zebra_flip import (  # noqa: E402
    _Geoms, _handover_check, _joint_move, _pick_as_it_lies, _Planner, _quat, _rot,
)
from mjrobots.zebra_pick_place import ZebraArmContext, _BRICK_CENTER_OFFSET_Z, go_home, grasp_part  # noqa: E402

FIT, FLEX = 0.01 * MM, 0.05 * MM  # the click found in brick_click.py
CLOSE_LOOK_SIGMA = (0.3 * MM, np.radians(0.3))  # assumed: a hand camera a few cm away
ALIGN_STEPS = 3
ABOVE = 3 * MM  # line up this far above the stud tops
PRESS_MM = 1.0  # the press aims this far past seated
LEGS_XY, BODY_XY = (0.36, 0.08), (0.36, -0.08)  # start spots, in the work box (scatter.PLACE_BOX)
OUT = ROOT / "logs" / "two_hand_click"


class _R:
    def step(self):
        pass


class _C:
    def tick(self):
        pass


def _ring_mesh(spec, name, rings, n=32):
    verts = []
    for r, z in rings:
        for k in range(n):
            a = 2 * np.pi * k / n
            verts += [r * np.cos(a), r * np.sin(a), z]
    spec.add_mesh(name=name, uservert=verts)


def _clicking_brick(spec, body_name: str) -> None:
    """Give a scene brick (origin = centre of its TOP face) the stud/tube geometry of
    brick_click.py instead of its plain collision box (that one stays, only as a picture)."""
    body = spec.body(body_name)
    for g in body.geoms:
        if g.contype:  # the plain box: touches nothing and weighs nothing any more, but stays
            # this body's first collision geom (contype 2, conaffinity 0: no other geom has
            # conaffinity 2) - the flip planner reads the brick's size from it (zebra_flip.py:457)
            g.contype, g.conaffinity = 2, 0
            g.mass = 0.0
    z0 = -H  # the brick's bottom, in its frame
    touch = np.hypot(8 * MM, 8 * MM) - STUD_R
    r_tube = touch + FIT
    if spec.mesh("click_stud") is None:
        _ring_mesh(spec, "click_stud", [(STUD_R, 0.0), (STUD_R, STUD_H - CHAMFER), (STUD_R - CHAMFER, STUD_H)])
        _ring_mesh(spec, "click_tube", [(r_tube - CHAMFER, TUBE_BOTTOM), (r_tube, TUBE_BOTTOM + CHAMFER),
                                        (r_tube, H - WALL)])
    m = 0.03 / 10
    soft = dict(solimp=[0.5, 0.99, FLEX, 0.5, 2], solref=[0.005, 1])
    hard = dict(solref=[0.002, 1], solimp=[0.95, 0.99, 0.0002, 0.5, 2])
    # contype/conaffinity 1 like the original box: the scene's default class would give new
    # geoms contype 2, which the fingers (contype 2, conaffinity 1) pass straight through
    # friction 0.4 (plastic on plastic), as in brick_click.py: with 1.0 the studs wedged at the
    # tube entry and stopped ~1.9 mm short at any push. Fingers keep their own 1.0 (MuJoCo
    # takes the larger of the two geoms' friction).
    common = dict(condim=4, friction=[0.4, 0.01, 0.001], rgba=[0, 0, 0, 0], contype=1, conaffinity=1)
    box = mujoco.mjtGeom.mjGEOM_BOX
    hw = (H - WALL) / 2
    for size, pos, mass in [([32 * MM, 16 * MM, WALL / 2], [0, 0, z0 + H - WALL / 2], 3 * m),
                            ([32 * MM, WALL / 2, hw], [0, 16 * MM - WALL / 2, z0 + hw], m),
                            ([32 * MM, WALL / 2, hw], [0, -16 * MM + WALL / 2, z0 + hw], m),
                            ([WALL / 2, 16 * MM - WALL, hw], [32 * MM - WALL / 2, 0, z0 + hw], m),
                            ([WALL / 2, 16 * MM - WALL, hw], [-32 * MM + WALL / 2, 0, z0 + hw], m)]:
        body.add_geom(type=box, size=size, pos=pos, mass=mass, **common, **hard)
    mesh = mujoco.mjtGeom.mjGEOM_MESH
    for tx, ty in TUBES:
        body.add_geom(type=mesh, meshname="click_tube", pos=[tx, ty, z0], mass=m / 3, **common, **soft)
    for sx, sy in STUDS:
        body.add_geom(type=mesh, meshname="click_stud", pos=[sx, sy, 0.0], mass=m / 8, **common, **soft)


def build(grip_n: float = 5.0):
    """`grip_n`: the grippers' squeeze per finger (N; the scene's is 5)."""
    spec = mujoco.MjSpec.from_file(str(_DEFAULT_SCENE))
    for act in spec.actuators:
        if "gripper" in act.name:
            act.forcerange = [-grip_n, grip_n]
    _clicking_brick(spec, "zebra_legs")
    _clicking_brick(spec, "zebra_body")
    spec.option.noslip_iterations = 10  # no creep in the fingers (README, "A brick creeps")
    return spec.compile()


def _set_pose(model, data, body, xy, R):
    """Put a brick (origin: top-face centre) resting on the table at `xy` with rotation R."""
    adr = model.jnt_qposadr[model.body_jntadr[body]]
    half = np.abs(R[2]) @ np.array([0.032, 0.016, 0.0192])
    table_top = -0.0842 - 0.0384
    centre = np.array([xy[0], xy[1], table_top + half + 0.0005])
    data.qpos[adr:adr + 3] = centre - R @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    data.qpos[adr + 3:adr + 7] = q


class Look:
    """A close look at a brick: (rotation, top-face centre, bottom-face centre) + noise."""

    def __init__(self, data, rng):
        self.data, self.rng = data, rng

    def __call__(self, body):
        R = self.data.xmat[body].reshape(3, 3).copy()
        axis = self.rng.normal(0.0, 1.0, 3)
        R = _rot(axis / np.linalg.norm(axis), self.rng.normal(0.0, CLOSE_LOOK_SIGMA[1])) @ R
        top = self.data.xpos[body] + self.rng.normal(0.0, CLOSE_LOOK_SIGMA[0], 3)
        return R, top, top - R[:, 2] * H


def _yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _grip_point(ctx):
    d = ctx.data
    return d.xpos[ctx.body_id] + d.xmat[ctx.body_id].reshape(3, 3) @ HAND_LOCAL_OFFSET


def _snap(model, data, path, cam_target):
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = cam_target
    cam.distance, cam.azimuth, cam.elevation = 0.30, 180, -5  # side-on, close
    opt = mujoco.MjvOption()
    renderer.update_scene(data, cam, opt)
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(renderer.render()).save(OUT / path)
    renderer.close()
    print(f"   picture: logs/two_hand_click/{path}")


def run(seed: int = 0, grip_n: float = 5.0) -> dict:
    rng = np.random.default_rng(seed)
    model = build(grip_n)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    legs, body, head = (model.body(n).id for n in ("zebra_legs", "zebra_body", "zebra_head"))
    # head out of the way; legs on their side, body upright, both turned 90 deg (long side across)
    hadr = model.jnt_qposadr[model.body_jntadr[head]]
    data.qpos[hadr:hadr + 3] = (0.1, 0.45, -0.08)
    _set_pose(model, data, legs, LEGS_XY, _axis_rot(2, 90) @ _PLACE_POSES[ON_SIDE][1])  # this side down: the left hand's roll ends upright
    _set_pose(model, data, body, BODY_XY, _axis_rot(2, 90))
    mujoco.mj_forward(model, data)
    for _ in range(600):
        mujoco.mj_step(model, data)
    render, clock = _R(), _C()
    left = ZebraArmContext(model, data, "left", "zebra_legs")
    right = ZebraArmContext(model, data, "right", "zebra_body")
    look = Look(data, rng)
    out = {}
    print(f"legs lie {lying(data.xmat[legs].reshape(3, 3))}, body {lying(data.xmat[body].reshape(3, 3))}")

    # 2. left hand: grip the legs by their ends and roll them upright, held from the side
    R0 = data.xmat[legs].reshape(3, 3).copy()
    centre = data.xpos[legs] + R0 @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
    plan = _Planner(model, data, legs, [body, head], ["left", "right"]).attempt_solo(
        ON_SIDE, R0, centre, "left", "middle")
    if plan is None:
        raise RuntimeError("the left hand can't do the one-hand roll for the legs from here")
    _hold(model, data, render, clock, left.this_arm, _GRIP_OPEN, 60)
    _pick_as_it_lies(left, plan, render, clock)
    _joint_move(left, render, clock, plan.q_a_hover, "left arm rolls the legs upright")
    print(f"1. left hand holds the legs {lying(data.xmat[legs].reshape(3, 3))} from the side "
          f"at ({data.xpos[legs][0]:.3f}, {data.xpos[legs][1]:+.3f}, {data.xpos[legs][2]:.3f})")

    # 3. right hand picks the body from above
    go_home(right, render, clock)
    Rb = data.xmat[body].reshape(3, 3)
    bc = data.xpos[body] + Rb @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
    grasp_part(right, render, clock, bc.copy(), relook=lambda: (bc.copy(), _yaw(Rb)))
    up = lambda R: float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    print(f"2. right hand holds the body (fingers at {right.grip_width() * 100:.2f} cm); tilted from vertical: "
          f"legs {up(data.xmat[legs].reshape(3, 3)):.1f} deg, body {up(data.xmat[body].reshape(3, 3)):.1f} deg")
    right.go(render, clock, _grip_point(right) + [0, 0, 0.10])  # straight up, clear of everything

    # links-only check near the other hand (like the flip's handover), both ways
    geo = _Geoms(model)
    near_right = _handover_check(model, geo, "right", "left")
    right._check_clearance = lambda p, w: near_right(p, w, right)

    def move_body_to(bottom_target, R_legs, what):
        """Move the right hand so the body's bottom centre lands on `bottom_target` with the
        body turned AND tilted like the legs (R_legs, as seen) - both bricks sit a few degrees
        tilted in their hands (legs 6.6, body 9.9 deg from vertical in the first try), and
        matching only the turn brought the body down at an angle. From a look at the body."""
        Rb_seen, _, bottom_seen = look(body)
        # a brick looks the same turned 180 deg about its studs' axis: the nearer of the two
        options = [R_legs, R_legs @ _rot([0, 0, 1.0], np.pi)]
        R_goal = min(options, key=lambda R: np.arccos(np.clip((np.trace(R @ Rb_seen.T) - 1) / 2, -1, 1)))
        dR = R_goal @ Rb_seen.T  # turn the hand by this (about the grip point)
        R_hand = data.xmat[right.body_id].reshape(3, 3).copy()
        grip = _grip_point(right)
        target = bottom_target - dR @ (bottom_seen - grip)
        move_to_pose(model, data, render, clock, right.arm_ctrl, right.body_id, HAND_LOCAL_OFFSET,
                     right.joint_ids, target, _quat(dR @ R_hand), path_check=right._check_clearance)

    # 4. over to the legs: up first, then across high above them (a straight line from the
    # pick passed below the legs' top and ran into the left hand), then line up just above
    Rl, top_l, _ = look(legs)
    move_body_to(top_l + Rl[:, 2] * (STUD_H + ABOVE + 0.08), Rl, "over the legs")
    for k in range(ALIGN_STEPS):
        Rl, top_l, _ = look(legs)
        move_body_to(top_l + Rl[:, 2] * (STUD_H + ABOVE), Rl, f"line up {k + 1}")
        Rl, top_l, _ = (lambda r: r)(look(legs))
        Rb_seen, _, bottom_b = look(body)
        err = bottom_b - (top_l + Rl[:, 2] * (STUD_H + ABOVE))
        dyaw = np.degrees((_yaw(Rl) - _yaw(Rb_seen) + np.pi / 2) % np.pi - np.pi / 2)
        # truth, for the report
        Rl_t, Rb_t = data.xmat[legs].reshape(3, 3), data.xmat[body].reshape(3, 3)
        true_off = (data.xpos[body] - Rb_t[:, 2] * H) - data.xpos[legs]
        true_off_l = Rl_t.T @ true_off
        true_yaw = np.degrees((_yaw(Rl_t) - _yaw(Rb_t) + np.pi / 2) % np.pi - np.pi / 2)
        print(f"3.{k + 1} lined up: seen {np.linalg.norm(err[:2]) / MM:.2f} mm / {abs(dyaw):.1f} deg off; "
              f"truly {true_off_l[0] / MM:+.2f} mm along, {true_off_l[1] / MM:+.2f} mm across, "
              f"{true_yaw:+.1f} deg turned, "
              f"{np.degrees(np.arccos(np.clip(Rl_t[:, 2] @ Rb_t[:, 2], -1, 1))):.1f} deg tilted vs the legs")
    out["aligned_mm"] = (float(true_off_l[0] / MM), float(true_off_l[1] / MM))
    out["aligned_deg"] = float(true_yaw)
    _snap(model, data, f"1_lined_up_seed{seed}.png", data.xpos[legs])

    # 5. press: one move aiming the body's bottom PRESS_MM past the legs' top. (Tried: pressing
    # in small steps, re-aiming the whole hand after each look - it re-tilted the hand while
    # the body swivelled in the fingers and pushed both bricks out of the hands, 31 and 17 mm.)
    from brick_click import contact_force
    Rl, top_l, _ = look(legs)
    in_hand = lambda ctx, b: data.xmat[ctx.body_id].reshape(3, 3).T @ (data.xpos[b] - _grip_point(ctx))
    body_in_hand0, legs_in_hand0 = in_hand(right, body), in_hand(left, legs)
    move_body_to(top_l - Rl[:, 2] * PRESS_MM * MM, Rl, "press")
    aimed = _grip_point(right).copy()
    for _ in range(300):
        mujoco.mj_step(model, data)
    slid_b = np.linalg.norm(in_hand(right, body) - body_in_hand0) / MM
    slid_l = np.linalg.norm(in_hand(left, legs) - legs_in_hand0) / MM
    print(f"   during the press the body slid {slid_b:.2f} mm in the right hand, the legs {slid_l:.2f} mm "
          f"in the left; right hand {np.linalg.norm(_grip_point(right) - aimed) / MM:.2f} mm from where "
          f"the move ended")
    push = contact_force(model, data, legs, body)
    Rl_t, Rb_t = data.xmat[legs].reshape(3, 3), data.xmat[body].reshape(3, 3)
    gap = float(Rl_t[:, 2] @ ((data.xpos[body] - Rb_t[:, 2] * H) - data.xpos[legs]) / MM)
    tilt = float(np.degrees(np.arccos(np.clip(Rl_t[:, 2] @ Rb_t[:, 2], -1, 1))))
    print(f"4. pressed: body {gap:+.2f} mm from fully seated, tilted {tilt:.1f} deg vs the legs, "
          f"pushing {push:.1f} N (from vertical: legs {up(Rl_t):.1f} deg, body {up(Rb_t):.1f} deg)")
    _snap(model, data, f"2_pressed_seed{seed}.png", data.xpos[legs])

    # 6. right hand lets go and moves up; left hand moves the pair: does the body come along?
    _hold(model, data, render, clock, right.this_arm, _GRIP_OPEN, 150)
    right.holding = False
    try:
        right.go(render, clock, _grip_point(right) + [0, 0, 0.06])
    except IKError as e:
        print(f"   (right hand couldn't move up: {e})")
    rel0 = data.xmat[legs].reshape(3, 3).T @ (data.xpos[body] - data.xpos[legs])
    here = _grip_point(left)
    R_left = data.xmat[left.body_id].reshape(3, 3).copy()
    try:
        move_to_pose(model, data, render, clock, left.arm_ctrl, left.body_id, HAND_LOCAL_OFFSET, left.joint_ids,
                     here + [0.0, 0.04, 0.04], _quat(R_left))
    except IKError as e:
        print(f"   (left hand couldn't move the pair: {e})")
    for _ in range(300):
        mujoco.mj_step(model, data)
    rel1 = data.xmat[legs].reshape(3, 3).T @ (data.xpos[body] - data.xpos[legs])
    moved = float(np.linalg.norm(rel1 - rel0) / MM)
    clicked = moved < 1.0 and abs(gap) < 0.5
    print(f"5. left hand moved the pair 4 cm sideways and 4 cm up: the body moved {moved:.2f} mm "
          f"relative to the legs -> {'CLICKED TOGETHER' if clicked else 'NOT clicked'}")
    _snap(model, data, f"3_moved_seed{seed}.png", data.xpos[legs])
    out.update(gap_mm=gap, tilt_deg=tilt, push_N=push, moved_mm=moved, clicked=clicked)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grip", type=float, default=5.0, help="gripper squeeze per finger (N; the scene's is 5)")
    a = p.parse_args()
    run(a.seed, a.grip)
