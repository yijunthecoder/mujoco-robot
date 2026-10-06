#!/usr/bin/env python3
"""Step 1 of the learned policy: record demos of the IK controller reaching above a brick.

The teacher is the project's own IK (cartesian_control.solve_ik), used as a feedback
controller: every DECISION_DT it looks at where the hand is, picks a waypoint at most
MAX_REACH_STEP closer to the target, solves IK for it from the arm's current joints, and
sends those joint angles. The student (train.py) later learns to make the same choice
from the same inputs, with no IK.

One episode = one brick at a random spot in the work box (scatter.PLACE_BOX), turned a
random way, and the right arm starting near home (each joint jiggled by up to
START_JIGGLE, so the demos cover more than one start). The task: bring the hand point
(fingertip midpoint) to HOVER above the brick's top face, and stop there.

Saved per decision (logs/learned_policy/demos.npz):
  obs    (N, 9)  sensed hover target xyz (3) + right arm joint angles (6)
  act    (N, 6)  joint change the IK teacher chose (commanded angles - current angles)
  episode (N,)   which episode the row came from (so train.py can hold some out)
Episodes where IK can't reach the target are skipped and counted.

usage (repo folder):
  python learned_policy/record_demos.py                  # 300 episodes, seeds 0..299
  python learned_policy/record_demos.py --episodes 1     # quick check on seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mjrobots import sim_step  # noqa: E402
from mjrobots.cartesian_control import (  # noqa: E402
    ARM_JOINTS, HAND_LOCAL_OFFSET, IKError, _hand_point_and_jac, solve_ik,
)
from mjrobots.scatter import PLACE_BOX  # noqa: E402
from mjrobots.stationlite_pick_place import _DEFAULT_SCENE  # noqa: E402
from mjrobots.zebra_pick_place import _HOVER_DZ  # noqa: E402

ARM = "right"
ARM_CTRL = slice(8, 14)  # the right arm's 6 joint actuators (cartesian_control.run_demo)
BRICK = "zebra_body"
PARKED = {"zebra_legs": (0.40, 0.30), "zebra_head": (0.46, 0.30)}  # out of the box, out of the way
HOVER = _HOVER_DZ  # m above the brick's top face, reachable across the whole box
DECISION_DT = 0.02  # s: one decision per 10 control ticks (50 Hz, a usual policy rate)
TICKS_PER_DECISION = round(DECISION_DT / sim_step.CONTROL_DT)
MAX_REACH_STEP = 0.03  # m the waypoint may lead the hand (the servos lag it: 1 cm gave ~0.12 m/s)
MAX_DECISIONS = 300  # 6 s per episode, then give up
REACHED = 0.01  # m: hand this close to the true target...
STILL = 0.02  # m/s: ...and moving slower than this = reached
START_JIGGLE = 0.15  # rad, each joint's start offset from home, at most
SENSE_SIGMA = 0.001  # m, noise of a camera's brick-position look (see sense_brick_top)
OUT = ROOT / "logs" / "learned_policy"


class Arm:
    """The scene, its right arm and the brick - what record, evaluate and train share."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(_DEFAULT_SCENE))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joint_ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS[ARM]])
        self.qpos_adr = m.jnt_qposadr[self.joint_ids]
        self.hand_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{ARM}_linkgripper")
        self.brick = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BRICK)
        self.home = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")

    def reset(self, seed: int) -> np.random.Generator:
        """Home pose, arm jiggled, brick upright at a random spot and turn. Returns the rng."""
        rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self.home)
        q = d.qpos[self.qpos_adr] + rng.uniform(-START_JIGGLE, START_JIGGLE, 6)
        d.qpos[self.qpos_adr] = q
        d.ctrl[ARM_CTRL] = q
        (x0, x1), (y0, y1) = PLACE_BOX
        _place_upright(m, d, self.brick, (rng.uniform(x0, x1), rng.uniform(y0, y1)), rng.uniform(-np.pi, np.pi))
        for name, xy in PARKED.items():
            _place_upright(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name), xy, 0.0)
        mujoco.mj_forward(m, d)
        return rng

    def joints(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def hand(self) -> np.ndarray:
        point, _ = _hand_point_and_jac(self.model, self.data, self.hand_body, HAND_LOCAL_OFFSET)
        return point

    def true_target(self) -> np.ndarray:
        return self.data.xpos[self.brick] + [0.0, 0.0, HOVER]

    def apply(self, q_cmd: np.ndarray) -> None:
        """Send joint angles to the arm's servos and run one decision's worth of physics."""
        self.data.ctrl[ARM_CTRL] = q_cmd
        for _ in range(TICKS_PER_DECISION):
            sim_step.step(self.model, self.data)


def _place_upright(model, data, body, xy, yaw) -> None:
    """Brick `body` upright on the table at `xy`, turned `yaw` (as two_hand_click._set_pose)."""
    adr = model.jnt_qposadr[model.body_jntadr[body]]
    table_top = -0.0842 - 0.0384
    # the brick's origin is its top-face centre; upright it is 3.84 cm tall
    data.qpos[adr:adr + 3] = (xy[0], xy[1], table_top + 0.0384 + 0.0005)
    data.qpos[adr + 3:adr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    data.qvel[model.body_dofadr[body]:model.body_dofadr[body] + 6] = 0.0


def sense_brick_top(arm: Arm, rng: np.random.Generator) -> np.ndarray:
    """Where the camera sees the brick's top-face centre (x, y, z) - the one sensor stand-in.
    Sim: the true position plus SENSE_SIGMA noise. Real robot: perception's report."""
    return arm.data.xpos[arm.brick] + rng.normal(0.0, SENSE_SIGMA, 3)


def observation(target: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """What the policy is given: the sensed hover target and the arm's joint angles."""
    return np.concatenate([target, joints]).astype(np.float32)


def ik_teacher(arm: Arm, target: np.ndarray) -> np.ndarray:
    """The IK controller's choice of joint angles for this decision (raises IKError)."""
    hand = arm.hand()
    gap = target - hand
    dist = np.linalg.norm(gap)
    waypoint = target if dist <= MAX_REACH_STEP else hand + gap * (MAX_REACH_STEP / dist)
    return solve_ik(arm.model, arm.data, arm.hand_body, HAND_LOCAL_OFFSET, arm.joint_ids, waypoint)


def run_episode(arm: Arm, seed: int, choose) -> dict:
    """Reach from seed `seed`'s start; `choose(obs) -> joint change` picks each move.
    Returns the rows seen and whether/when it reached the true target."""
    rng = arm.reset(seed)
    for _ in range(50):  # let the brick settle on the table before looking
        sim_step.step(arm.model, arm.data)
    target = sense_brick_top(arm, rng) + [0.0, 0.0, HOVER]
    obs_rows, act_rows = [], []
    prev_hand = arm.hand()
    for i in range(MAX_DECISIONS):
        obs = observation(target, arm.joints())
        act = choose(obs)
        obs_rows.append(obs)
        act_rows.append(act.astype(np.float32))
        arm.apply(arm.joints() + act)
        hand = arm.hand()
        speed = np.linalg.norm(hand - prev_hand) / DECISION_DT
        prev_hand = hand
        if np.linalg.norm(hand - arm.true_target()) < REACHED and speed < STILL:
            return dict(obs=obs_rows, act=act_rows, reached=True, decisions=i + 1,
                        miss=float(np.linalg.norm(hand - arm.true_target())))
    return dict(obs=obs_rows, act=act_rows, reached=False, decisions=MAX_DECISIONS,
                miss=float(np.linalg.norm(arm.hand() - arm.true_target())))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=300, help="seeds 0..N-1")
    args = p.parse_args()

    arm = Arm()
    obs, act, episode = [], [], []
    kept, unreachable, timed_out = 0, 0, 0
    t0 = time.time()
    for seed in range(args.episodes):
        try:
            ep = run_episode(arm, seed, lambda o: ik_teacher(arm, o[:3]) - o[3:])
        except IKError:
            unreachable += 1
            continue
        if not ep["reached"]:
            timed_out += 1
            continue
        obs += ep["obs"]
        act += ep["act"]
        episode += [seed] * len(ep["obs"])
        kept += 1
        if args.episodes <= 5 or seed % 50 == 0:
            print(f"seed {seed}: reached in {ep['decisions']} decisions "
                  f"({ep['decisions'] * DECISION_DT:.2f} s), {ep['miss'] * 1000:.1f} mm off")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "demos.npz", obs=np.array(obs), act=np.array(act), episode=np.array(episode))
    print(f"\n{kept}/{args.episodes} episodes kept ({unreachable} unreachable by IK, "
          f"{timed_out} didn't settle in {MAX_DECISIONS * DECISION_DT:.0f} s), "
          f"{len(obs)} rows -> logs/learned_policy/demos.npz  [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
