#!/usr/bin/env python3
"""Step 3 of the learned policy: the trained network drives the arm alone - no IK.

On brick spots it has never seen (seeds from 1000 on; the demos were seeds 0..299), the
network (train.py's LearnedPolicy) picks every joint change from the observation alone.
The IK teacher runs on the same seeds for comparison. Success = record_demos' test: the
hand within REACHED of the true hover point above the brick, and nearly still, inside
MAX_DECISIONS.

With --vision, the network is train_vision's VisionPolicy: it gets the head camera's
picture and the joint angles, never the brick's position.

Writes a summary to logs/learned_policy/eval_results.txt (--vision: eval_results_vision.txt).

usage (repo folder):
  python learned_policy/evaluate.py               # 50 new spots
  python learned_policy/evaluate.py --episodes 1  # seed 1000 only
  python learned_policy/evaluate.py --vision      # the camera-picture network
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from record_demos import (
    DECISION_DT, MAX_DECISIONS, OUT, REACHED, Arm, HeadCamera, IKError, ik_teacher, run_episode,
)
from train import LearnedPolicy

FIRST_SEED = 1000  # well clear of the demo seeds


def evaluate(name: str, arm: Arm, choose, seeds) -> list[dict]:
    results = []
    for seed in seeds:
        try:
            ep = run_episode(arm, seed, choose)
        except IKError:  # only the teacher can raise this
            ep = dict(reached=False, decisions=MAX_DECISIONS, miss=float("nan"))
        results.append(ep)
        if len(seeds) <= 5:
            print(f"  {name} seed {seed}: {'reached' if ep['reached'] else 'MISSED'} "
                  f"in {ep['decisions'] * DECISION_DT:.2f} s, {ep['miss'] * 1000:.1f} mm off")
    return results


def summary(name: str, results: list[dict]) -> str:
    ok = [r for r in results if r["reached"]]
    misses = np.array([r["miss"] for r in results]) * 1000
    line = f"{name:8s} reached {len(ok)}/{len(results)}"
    if ok:
        line += (f", time {np.mean([r['decisions'] for r in ok]) * DECISION_DT:.2f} s avg, "
                 f"end {np.mean([r['miss'] for r in ok]) * 1000:.1f} mm off avg")
    failed = misses[[not r["reached"] for r in results]]
    if len(failed):
        line += f"; failures ended {np.nanmin(failed):.0f}-{np.nanmax(failed):.0f} mm off"
    return line


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--vision", action="store_true", help="the camera-picture network (train_vision.py)")
    args = p.parse_args()
    seeds = list(range(FIRST_SEED, FIRST_SEED + args.episodes))

    arm = Arm()
    if args.vision:
        from train_vision import VisionPolicy
        camera, vision = HeadCamera(arm), VisionPolicy()
        policy = lambda o: vision(camera(), o[3:])  # noqa: E731  (picture + joints; o[:3], the brick, unused)
    else:
        policy = LearnedPolicy()
    t0 = time.time()
    net = evaluate("network", arm, policy, seeds)
    ik = evaluate("IK", arm, lambda o: ik_teacher(arm, o[:3]) - o[3:], seeds)

    lines = [f"{'vision network (camera picture + joints)' if args.vision else 'network (brick xyz + joints)'}",
             f"{args.episodes} new brick spots (seeds {seeds[0]}-{seeds[-1]}), "
             f"success = within {REACHED * 1000:.0f} mm and still, in {MAX_DECISIONS * DECISION_DT:.0f} s",
             summary("network", net), summary("IK", ik)]
    text = "\n".join(lines)
    print(f"\n{text}\n[{time.time() - t0:.0f} s]")
    (OUT / ("eval_results_vision.txt" if args.vision else "eval_results.txt")).write_text(text + "\n")


if __name__ == "__main__":
    main()
