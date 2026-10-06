#!/usr/bin/env python3
"""Step 2 of the learned policy: train a small network to copy the IK teacher.

Reads logs/learned_policy/demos.npz (record_demos.py) and fits an MLP
  9 numbers in (hover target xyz + 6 joint angles) -> 6 out (joint change this decision)
by behavior cloning: show it each recorded observation, compare its guess with the IK
teacher's action (mean squared error), nudge its weights to be less wrong, repeat.

Inputs and outputs are scaled to mean 0, spread 1 with the demos' own statistics (saved
with the network, so evaluate.py scales the same way). The last HOLD_OUT episodes are
never trained on: their error says whether the network handles brick spots it hasn't seen,
not just the ones it memorised.

Saves logs/learned_policy/policy.pt (weights + scaling).

usage (repo folder):
  python learned_policy/train.py              # 100 epochs, ~1-2 min on a laptop CPU
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parent.parent / "logs" / "learned_policy"
HOLD_OUT = 30  # episodes kept out of training, to check on unseen brick spots
HIDDEN = 256
BATCH = 256
LR = 1e-3


def make_net(n_in: int = 9, n_out: int = 6) -> nn.Module:
    """The policy network: an MLP, ~70 thousand weights."""
    return nn.Sequential(
        nn.Linear(n_in, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, n_out),
    )


class LearnedPolicy:
    """The trained network as a policy: observation (9) -> joint change (6), in real units."""

    def __init__(self, path: Path = OUT / "policy.pt"):
        saved = torch.load(path, weights_only=False)
        self.net = make_net()
        self.net.load_state_dict(saved["net"])
        self.net.eval()
        self.obs_mean, self.obs_std = saved["obs_mean"], saved["obs_std"]
        self.act_mean, self.act_std = saved["act_mean"], saved["act_std"]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(((obs - self.obs_mean) / self.obs_std).astype(np.float32))
        with torch.no_grad():
            y = self.net(x).numpy()
        return y * self.act_std + self.act_mean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    torch.manual_seed(0)

    demos = np.load(OUT / "demos.npz")
    obs, act, episode = demos["obs"], demos["act"], demos["episode"]
    held = np.isin(episode, np.unique(episode)[-HOLD_OUT:])
    obs_mean, obs_std = obs[~held].mean(0), obs[~held].std(0) + 1e-6
    act_mean, act_std = act[~held].mean(0), act[~held].std(0) + 1e-6

    def tensors(rows):
        return (torch.from_numpy((obs[rows] - obs_mean) / obs_std).float(),
                torch.from_numpy((act[rows] - act_mean) / act_std).float())

    x_train, y_train = tensors(~held)
    x_val, y_val = tensors(held)
    print(f"{len(x_train)} training rows ({len(np.unique(episode[~held]))} episodes), "
          f"{len(x_val)} held-out rows ({HOLD_OUT} episodes)")

    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f"network: {sum(w.numel() for w in net.parameters())} weights")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = torch.randperm(len(x_train))
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            loss = ((net(x_train[b]) - y_train[b]) ** 2).mean()  # how wrong is it?
            opt.zero_grad()
            loss.backward()  # which way to nudge each weight
            opt.step()  # nudge
        sched.step()
        if epoch == 1 or epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                train_err = ((net(x_train) - y_train) ** 2).mean().item()
                val_err = ((net(x_val) - y_val) ** 2).mean().item()
                # the same error in real units: how far its joint change is from the teacher's
                val_rad = ((net(x_val) - y_val) * torch.from_numpy(act_std)).abs().mean().item()
            print(f"epoch {epoch:3d}: error train {train_err:.4f}  held-out {val_err:.4f} "
                  f"(off the teacher by {np.degrees(val_rad):.3f} deg per joint per decision)  "
                  f"[{time.time() - t0:.0f} s]")

    torch.save(dict(net=net.state_dict(), obs_mean=obs_mean, obs_std=obs_std,
                    act_mean=act_mean, act_std=act_std), OUT / "policy.pt")
    print("saved logs/learned_policy/policy.pt")


if __name__ == "__main__":
    main()
