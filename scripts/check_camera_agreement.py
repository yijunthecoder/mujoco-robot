#!/usr/bin/env python3
"""Headless check: every camera that can see a zebra brick must report the
same headcam-frame position as ground truth, to within --tol (default 1 cm).

It calibrates exactly as the perception publisher does, then puts the arms
and bricks into the situations that broke before:
  - arms at home, bricks at their spawn spots;
  - the three bricks stacked at the place target (the headcam line-of-sight
    test used to hide the lower bricks of a stack);
  - each arm hovering over the stack and over each spawn spot, both
    position-only and in the grip orientation (the hand cameras' transforms
    used to be frozen at the home pose, so this read ~40 cm off);
  - random arm joint angles (kinematics only, no physics).

For each state, each brick and each camera that can see it, it compares
`CameraCalibration.to_reference` (what zebra_publisher reports) against ground
truth: the brick's data.xpos converted into the headcam frame with headcam's
data.cam_xpos/cam_xmat. It exits non-zero if any camera is off by more than
--tol, or if headcam can't see every brick of an unobstructed stack.

No ROS2 or display needed:
    python3 scripts/check_camera_agreement.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mjrobots.camera_calibration import (  # noqa: E402
    CAMERAS,
    REFERENCE_CAMERA,
    SimulatedCameras,
    _DEFAULT_SCENE,
    _observe_live_position,
    calibrate_cameras,
)
from mjrobots.cartesian_control import ARM_JOINTS  # noqa: E402
from mjrobots.zebra_pick_place import (  # noqa: E402
    BRICK_HEIGHT,
    _BRICK_CENTER_OFFSET_Z,
    _HOVER_DZ,
    _TABLE_PLACE_XYZ,
    ZebraArmContext,
)


# zebra_publisher.PART_BODIES (not imported here: that module needs ROS2)
BRICKS = ("zebra_legs", "zebra_body", "zebra_head")


class _Headless:
    """Stands in for both the viewer sync and the realtime clock."""

    def step(self) -> None:
        pass

    def tick(self) -> None:
        pass


class Checker:
    def __init__(self, model, data, calibration, rng, tol: float, pixel_sigma: float, depth_sigma: float) -> None:
        self.model, self.data, self.calibration, self.tol = model, data, calibration, tol
        self.cams = {
            b: SimulatedCameras(model, data, block_body=b, pixel_sigma=pixel_sigma, depth_sigma=depth_sigma, rng=rng)
            for b in BRICKS
        }
        self.headcam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, REFERENCE_CAMERA)
        self.errors = {name: [] for name in CAMERAS}  # cm
        self.failures: list[str] = []

    def truth(self, brick: str) -> np.ndarray:
        R = self.data.cam_xmat[self.headcam].reshape(3, 3)
        return R.T @ (self.data.xpos[self.model.body(brick).id] - self.data.cam_xpos[self.headcam])

    def check(self, label: str, headcam_must_see: tuple[str, ...] = (), quiet: bool = False) -> None:
        lines = [f"== {label}"]
        for brick in BRICKS:
            truth = self.truth(brick)
            seen = _observe_live_position(self.cams[brick])
            lines.append(f"  {brick:<11} truth {_fmt(truth)}   seen by: {', '.join(seen) or 'none'}")
            for name, p_cam in seen.items():
                est = self.calibration.to_reference(name, p_cam, self.cams[brick])
                err = np.linalg.norm(est - truth) * 100
                self.errors[name].append(err)
                flag = "" if err <= self.tol * 100 else "   <-- FAIL"
                lines.append(f"    {name:<14} {_fmt(est)}  err {err:5.2f} cm{flag}")
                if flag:
                    self.failures.append(f"{label}: {brick} via {name} off by {err:.1f} cm")
            if brick in headcam_must_see and REFERENCE_CAMERA not in seen:
                self.failures.append(f"{label}: headcam cannot see {brick}")
                lines.append(f"    headcam cannot see {brick}   <-- FAIL")
        if not quiet:
            print("\n".join(lines))


def _fmt(p: np.ndarray) -> str:
    return f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})"


def _set_brick(model, data, brick: str, xyz) -> None:
    joint = model.body_jntadr[model.body(brick).id]
    adr, dof = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    data.qpos[adr : adr + 3] = xyz
    data.qpos[adr + 3 : adr + 7] = (1, 0, 0, 0)
    data.qvel[dof : dof + 6] = 0


def _reset(model, data) -> None:
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)


def _stack(model, data) -> float:
    """Stack all three bricks at the place target; returns the top brick's centre z.

    The bottom brick sits on the table like a freshly spawned one: its origin
    at zebra_legs' keyframe height. (_TABLE_PLACE_XYZ's z is the table
    surface, not a brick height.)
    """
    home = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, home, model.key("home").id)
    mujoco.mj_kinematics(model, home)
    bottom_origin_z = home.xpos[model.body(BRICKS[0]).id][2]
    for k, brick in enumerate(BRICKS):
        _set_brick(model, data, brick, (_TABLE_PLACE_XYZ[0], _TABLE_PLACE_XYZ[1], bottom_origin_z + k * BRICK_HEIGHT))
    mujoco.mj_forward(model, data)
    return bottom_origin_z + (len(BRICKS) - 1) * BRICK_HEIGHT + _BRICK_CENTER_OFFSET_Z


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--tol", type=float, default=0.01, help="max allowed error, metres (default 0.01)")
    parser.add_argument("--n-calib", type=int, default=30, help="calibration samples per camera (default 30, as zebra_publisher)")
    parser.add_argument("--pixel-noise", type=float, default=0.5, help="detector noise, pixels (default 0.5; 0 = exact)")
    parser.add_argument("--depth-noise", type=float, default=0.002, help="depth noise, metres (default 0.002; 0 = exact)")
    parser.add_argument("--random-poses", type=int, default=200, help="random arm configurations per arm")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene or _DEFAULT_SCENE))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)
    # Same call and defaults as zebra_publisher.
    calibration = calibrate_cameras(model, args.n_calib, rng, args.pixel_noise, args.depth_noise)
    checker = Checker(model, data, calibration, rng, args.tol, args.pixel_noise, args.depth_noise)
    headless = _Headless()

    _reset(model, data)
    spawn = {b: data.xpos[model.body(b).id].copy() for b in BRICKS}
    checker.check("home, bricks at spawn")

    _reset(model, data)
    top_z = _stack(model, data)
    checker.check("home, bricks stacked", headcam_must_see=BRICKS)

    for arm in ("left", "right"):
        # Hover over the stack, like place_part does before lowering.
        _reset(model, data)
        top_z = _stack(model, data)
        ctx = ZebraArmContext(model, data, arm)
        hover = np.array([_TABLE_PLACE_XYZ[0], _TABLE_PLACE_XYZ[1], top_z + _HOVER_DZ])
        ctx.go(headless, headless, hover)
        checker.check(f"{arm} arm hovering over stack (position-only)")
        ctx.go_oriented(headless, headless, hover)
        checker.check(f"{arm} arm hovering over stack (grip orientation)")

        # Hover over each spawn spot, like grasp_part does.
        for brick in BRICKS:
            _reset(model, data)
            ctx = ZebraArmContext(model, data, arm)
            hover = spawn[brick] + np.array([0, 0, _BRICK_CENTER_OFFSET_Z + _HOVER_DZ])
            ctx.go(headless, headless, hover)
            ctx.go_oriented(headless, headless, hover)
            checker.check(f"{arm} arm hovering over {brick} spawn (grip orientation)")

    # Random joint angles, kinematics only: covers hand-camera poses the
    # scripted motions never reach.
    n_before = {name: len(e) for name, e in checker.errors.items()}
    for arm in ("left", "right"):
        joints = [model.joint(j).id for j in ARM_JOINTS[arm]]
        lo, hi = model.jnt_range[joints].T
        for k in range(args.random_poses):
            _reset(model, data)
            if k % 2:
                _stack(model, data)
            data.qpos[model.jnt_qposadr[joints]] = rng.uniform(lo, hi)
            mujoco.mj_forward(model, data)
            checker.check(f"{arm} arm random pose {k}", quiet=True)
    extra = {name: len(e) - n_before[name] for name, e in checker.errors.items()}
    print(f"\n== random arm poses ({args.random_poses} per arm): sightings per camera {extra}")

    print("\nSummary (error vs ground truth, headcam frame):")
    for name in CAMERAS:
        e = np.array(checker.errors[name])
        if len(e):
            print(f"  {name:<14} {len(e):4d} sightings   mean {e.mean():.2f} cm   max {e.max():.2f} cm")
        else:
            print(f"  {name:<14}    0 sightings")
            checker.failures.append(f"{name} never saw a brick - check not exercised")
    if checker.failures:
        print(f"\nFAIL ({len(checker.failures)}):")
        for f in checker.failures[:30]:
            print("  " + f)
        sys.exit(1)
    print(f"\nPASS: every camera within {args.tol * 100:.1f} cm of ground truth")


if __name__ == "__main__":
    main()
