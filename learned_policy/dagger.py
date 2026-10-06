#!/usr/bin/env python3
"""DAgger for the vision policy: the network drives, the IK teacher says what it should have done.

Plain copying (record_demos -> train_vision) only ever shows the network the teacher's own
paths. When the network is a little off, the arm ends up in a pose no demo passed through,
its next guess is worse, and it drifts away (evaluate --vision: 27/50, misses up to 68 cm).
DAgger (Ross et al. 2011, "Dataset Aggregation") fixes that by collecting data where the
NETWORK goes:
  1. the current vision network drives, from the head-camera picture, as in evaluate;
  2. at every decision the IK teacher also works out what it would do from there - that is
     the label, the network's own move is only used to drive;
  3. those rows are added to the demos and the network is trained again (train_vision.py
     reads every dagger_round*.npz next to demos_vision.npz).

One run = one round, on new brick spots: round r uses seeds 2000 + 100 (r - 1) onward, clear
of the demos (0..299) and the test (1000..1049). Saves logs/learned_policy/dagger_round<r>.npz
(same fields as demos_vision.npz). A decision where IK can't solve from the network's pose
gets no label and ends that episode.

usage (repo folder), one round:
  python learned_policy/dagger.py --round 1          # ~15-20 min
  python learned_policy/train_vision.py              # retrain on demos + all rounds
  python learned_policy/evaluate.py --vision         # test on the same 50 spots
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from record_demos import OUT, Arm, HeadCamera, IKError, ik_teacher, observation, run_episode
from train_vision import VisionPolicy

FIRST_SEED = 2000
SEEDS_PER_ROUND = 100


class _NoLabel(Exception):
    """IK can't solve from where the network has taken the arm: stop this episode."""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", type=int, required=True, help="1, 2, ... (picks the seeds and the file name)")
    p.add_argument("--episodes", type=int, default=SEEDS_PER_ROUND)
    args = p.parse_args()
    first = FIRST_SEED + SEEDS_PER_ROUND * (args.round - 1)
    seeds = range(first, first + args.episodes)

    arm = Arm()
    camera, vision = HeadCamera(arm), VisionPolicy()
    obs, act, episode, images = [], [], [], []
    reached, unlabelled = 0, 0
    t0 = time.time()
    for seed in seeds:
        def drive(o):
            """Label this decision with the teacher's move, then let the network make its own."""
            try:
                label = ik_teacher(arm, o[:3]) - o[3:]
            except IKError:
                raise _NoLabel
            image = camera()
            obs.append(observation(o[:3], o[3:]))
            act.append(label.astype(np.float32))
            images.append(image)
            episode.append(seed)
            return vision(image, o[3:])

        try:
            ep = run_episode(arm, seed, drive)
            reached += ep["reached"]
        except _NoLabel:
            unlabelled += 1
        done = seed - first + 1
        if done % 10 == 0:
            print(f"{done}/{args.episodes} episodes, network reached {reached}, "
                  f"{len(obs)} rows  [{time.time() - t0:.0f} s]", flush=True)

    name = f"dagger_round{args.round}.npz"
    np.savez(OUT / name, obs=np.array(obs), act=np.array(act), episode=np.array(episode),
             image=np.array(images, dtype=np.uint8))
    print(f"\nround {args.round}: seeds {first}-{first + args.episodes - 1}, network reached "
          f"{reached}/{args.episodes} while collecting ({unlabelled} ended where IK couldn't solve), "
          f"{len(obs)} labelled rows -> logs/learned_policy/{name}  [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
