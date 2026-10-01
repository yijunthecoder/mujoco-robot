#!/usr/bin/env python3
"""Reach map: where on the table can an arm pick a zebra brick up and place
it on the stack? Also writes the scatter zone (src/mjrobots/scatter.py) and a
top-down picture of the table.

For each spot of a grid over the table, at several brick angles, the legs
brick is put there and the bridge's real pick and place moves are PLANNED
(grasp_part's steps, then place_part's at stack levels 0 and 2) with the same
IK, joint-jump and arm-clearance checks. Instead of simulating each planned
move, the arm is jumped to its end - planning is where every refusal happens,
so this gives the same yes/no, fast. The other arm stays at home. Each spot
also records which cameras see a brick there (arms at home).

Usage (from the repo root, in WSL):
    python3 scripts/reach_map.py --arm right --csv /tmp/right.csv --zone --plot docs/reach_map_right.png
    python3 scripts/reach_map.py --from /tmp/right.csv /tmp/left.csv --plot docs/reach_map_either.png

Takes a few minutes with the defaults (3 cm grid, every 30 deg, 3 processes).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjrobots import cartesian_control
from mjrobots.camera_calibration import CAMERAS, SimulatedCameras
from mjrobots.cartesian_control import IKError
from mjrobots.scatter import STACK_CLEAR, STACK_XY, TABLE_Z, ZONES_DIR
from mjrobots.stationlite_pick_place import _DEFAULT_SCENE
from mjrobots.zebra_pick_place import (
    BRICK_HEIGHT, ZebraArmContext, _BRICK_CENTER_OFFSET_Z, _HOVER_DZ, _oriented_approach, _wrap,
)

X_RANGE = (0.14, 0.80)
Y_RANGE = (-0.45, 0.45)
STACK_LEVELS = (0, 2)  # bottom and top: the place is checked at both


def _jump_to_end(model, data, render, clock, arm_ctrl_slice, q_start, path, what,
                 steps_per_wp, settle_steps, carry):
    """Plan-only stand-in for executing a move: put the arm at the path's end."""
    joints = model.actuator_trnid[arm_ctrl_slice, 0]
    data.qpos[model.jnt_qposadr[joints]] = path[-1]
    data.ctrl[arm_ctrl_slice] = path[-1]
    mujoco.mj_forward(model, data)


def _check_row(job) -> list[dict]:
    """Every spot and angle along one grid row (fixed x)."""
    arm, x, ys, angles = job
    cartesian_control._confirm_and_follow = _jump_to_end
    model = mujoco.MjModel.from_xml_path(str(_DEFAULT_SCENE))
    data = mujoco.MjData(model)
    home = model.key("home").id
    legs = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "zebra_legs")
    legs_adr = model.jnt_qposadr[model.body_jntadr[legs]]

    def reset(y: float, yaw: float) -> None:
        mujoco.mj_resetDataKeyframe(model, data, home)
        for other in ("zebra_body", "zebra_head"):  # out of the way, under the table
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, other)
            data.qpos[model.jnt_qposadr[model.body_jntadr[b]] + 2] = -0.6
        data.qpos[legs_adr:legs_adr + 3] = (x, y, TABLE_Z)
        data.qpos[legs_adr + 3:legs_adr + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        mujoco.mj_forward(model, data)

    reset(0.0, 0.0)
    ctx = ZebraArmContext(model, data, arm, "zebra_legs")
    cams = SimulatedCameras(model, data, block_body="zebra_legs")
    rows = []
    for y in ys:
        reset(y, 0.0)
        seen = "+".join(c for c in CAMERAS if cams.observe(c, noisy=False) is not None)
        for angle in angles:
            yaw = np.radians(angle)
            reset(y, yaw)
            center = np.array([x, y, TABLE_Z + _BRICK_CENTER_OFFSET_Z])
            hover = center + [0, 0, _HOVER_DZ]
            result, reason = "ok", ""
            try:  # grasp_part's moves
                ctx.grip_yaw, ctx.holding = 0.0, False
                ctx.go(None, None, hover)
                ctx.go_oriented(None, None, hover)
                grips = sorted({_wrap(yaw), _wrap(yaw + np.pi)}, key=abs)
                ctx.held_yaw = _wrap(yaw - _oriented_approach(ctx, None, None, hover, center, grips))
                ctx.holding = True
                ctx.go(None, None, hover)
            except IKError as e:
                result, reason = "pick", str(e)
            if result == "ok":  # place_part's moves, from the lifted pose
                lifted = data.qpos.copy(), data.ctrl.copy()
                for level in STACK_LEVELS:
                    data.qpos[:], data.ctrl[:] = lifted
                    mujoco.mj_forward(model, data)
                    target = np.array([*STACK_XY, TABLE_Z + _BRICK_CENTER_OFFSET_Z + level * BRICK_HEIGHT])
                    s_hover = target + [0, 0, _HOVER_DZ]
                    try:
                        ctx.go(None, None, s_hover)
                        _oriented_approach(ctx, None, None, s_hover, target,
                                           [_wrap(-ctx.held_yaw), _wrap(np.pi - ctx.held_yaw)])
                    except IKError as e:
                        result, reason = f"place{level}", str(e)
                        break
            rows.append(dict(x=round(x, 4), y=round(y, 4), yaw=angle, result=result,
                             cams=seen, reason=reason[:160]))
    print(f"  x={x:.2f} done", flush=True)
    return rows


def measure(arm: str, step: float, angle_step: int, workers: int) -> list[dict]:
    xs = np.arange(X_RANGE[0], X_RANGE[1] + 1e-9, step)
    ys = np.arange(Y_RANGE[0], Y_RANGE[1] + 1e-9, step)
    angles = list(range(0, 180, angle_step))  # a brick looks the same turned 180 deg
    jobs = [(arm, float(x), ys, angles) for x in xs]
    print(f"{arm} arm: {len(xs)} x {len(ys)} spots, {len(angles)} angles each, {workers} processes")
    with Pool(workers) as pool:
        return [row for rows in pool.map(_check_row, jobs) for row in rows]


def load_csvs(paths) -> tuple[dict, dict]:
    """{(x, y): [ok per angle]} - ok if ANY of the files (arms) managed that
    spot and angle - and {(x, y): cameras}."""
    ok = defaultdict(bool)
    cams = {}
    for path in paths:
        for r in csv.DictReader(open(path)):
            spot = (float(r["x"]), float(r["y"]))
            ok[(spot, r["yaw"])] |= r["result"] == "ok"
            cams[spot] = r["cams"]
    cells = defaultdict(list)
    for (spot, _), v in ok.items():
        cells[spot].append(v)
    return cells, cams


def write_zone(cells: dict, arm: str, path: Path) -> None:
    """The green squares (every angle works), for scatter.Zone."""
    xs = sorted({x for x, _ in cells})
    green = [[x, y] for (x, y), oks in sorted(cells.items()) if all(oks)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"arm": arm, "cell": round(xs[1] - xs[0], 4), "cells": green}) + "\n")
    print(f"wrote {path} ({len(green)} green squares)")


def plot(cells: dict, cams: dict, title: str, path: str) -> None:
    """Top-down picture: robot at the bottom, +y (left arm's side) on the left."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    model = mujoco.MjModel.from_xml_path(str(_DEFAULT_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    bases = {s: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_link1")][:2]
             for s in ("left", "right")}

    xs = sorted({x for x, _ in cells})
    ys = sorted({y for _, y in cells})
    step = xs[1] - xs[0]
    colors = {"all": "#2e9e44", "some": "#f2c230", "none": "#d9534f", "blind": "0.6"}
    counts = dict.fromkeys(colors, 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    for (x, y), oks in cells.items():
        kind = "blind" if not cams[(x, y)] else "all" if all(oks) else "some" if any(oks) else "none"
        counts[kind] += 1
        ax.add_patch(Rectangle((-y - step / 2, x - step / 2), step, step, color=colors[kind], lw=0))
    ax.add_patch(Circle((-STACK_XY[1], STACK_XY[0]), STACK_CLEAR, fill=False, ls="--", lw=2, color="k"))
    ax.text(-STACK_XY[1], STACK_XY[0], "stack\n(keep clear)", ha="center", va="center", fontsize=9)
    for y, name in ((-0.15, "legs"), (0.0, "body"), (0.15, "head")):
        ax.plot(-y, 0.2, "kx", ms=10, mew=2)
        ax.text(-y, 0.17, f"{name} today", ha="center", va="top", fontsize=8)
    for side, (bx, by) in bases.items():
        ax.plot(-by, bx, "ks", ms=12)
        ax.text(-by, bx - 0.03, f"{side} arm base", ha="center", va="top", fontsize=9)
    ax.set_xlim(-max(ys) - 0.05, -min(ys) + 0.05)
    ax.set_ylim(min(bx for bx, _ in bases.values()) - 0.08, max(xs) + 0.05)
    ax.set_aspect("equal")
    ticks = np.arange(-0.4, 0.41, 0.1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{-t:+.1f}" for t in ticks])
    ax.set_xlabel("y (m)   <- left arm's side      right arm's side ->")
    ax.set_ylabel("x (m), away from the robot")
    n_angles = len(next(iter(cells.values())))
    ax.set_title(f"{title}: can it pick here and place on the stack? ({n_angles} brick angles per spot)")
    labels = {"all": "every angle works", "some": "some angles only",
              "none": "no angle works", "blind": "no camera sees it"}
    ax.legend([Rectangle((0, 0), 1, 1, color=colors[k]) for k in colors],
              [f"{labels[k]} ({counts[k]})" for k in colors], loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"wrote {path} {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--step", type=float, default=0.03, help="grid spacing, m (default 0.03)")
    parser.add_argument("--angle-step", type=int, default=30, help="brick angle spacing, deg (default 30)")
    parser.add_argument("--workers", type=int, default=3, help="processes to use (default 3)")
    parser.add_argument("--csv", help="write every spot and angle's result here")
    parser.add_argument("--from", dest="from_csv", nargs="+",
                        help="don't measure, read these CSVs (several = either arm)")
    parser.add_argument("--zone", action="store_true",
                        help="write the green squares to src/mjrobots/assets/scatter_zones/<arm>.json")
    parser.add_argument("--plot", help="write the map picture here (.png)")
    parser.add_argument("--title", help="picture title (default: '<Arm> arm', or 'Either arm')")
    args = parser.parse_args()

    if args.from_csv:
        paths = args.from_csv
    else:
        rows = measure(args.arm, args.step, args.angle_step, args.workers)
        path = args.csv or f"/tmp/reach_map_{args.arm}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")
        paths = [path]

    cells, cams = load_csvs(paths)
    if args.zone:
        write_zone(cells, args.arm, ZONES_DIR / f"{args.arm}.json")
    if args.plot:
        title = args.title or ("Either arm" if len(paths) > 1 else f"{args.arm.capitalize()} arm")
        plot(cells, cams, title, args.plot)


if __name__ == "__main__":
    main()
