#!/usr/bin/env python3
"""Vision version of step 2: train a network that reaches from the head camera's picture.

Like train.py, but the network is NOT told where the brick is: it gets the head-camera
picture (84x84, record_demos.HeadCamera) and the arm's 6 joint angles, and must find the
brick in the picture itself. Same teacher (the IK controller's joint changes), same
held-out episodes, same error printout - so the two can be compared directly.

Network (a CNN in front of train.py's MLP):
  picture -> 3 convolution layers (find edges, then blobs, then "brick here") -> 3136 numbers
  + 6 joint angles -> 256 -> 256 -> 6 joint changes

Reads logs/learned_policy/demos_vision.npz (record_demos.py --camera) and every
dagger_round*.npz there (dagger.py: rows where the network drove, labelled by IK); saves
logs/learned_policy/policy_vision.pt.

usage (repo folder):
  python learned_policy/train_vision.py --epochs 2   # quick check (~1-2 min)
  python learned_policy/train_vision.py              # 30 epochs (my estimate ~15-30 min on CPU)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn

from train import BATCH, HIDDEN, HOLD_OUT, LR, OUT


class VisionNet(nn.Module):
    """Picture + joint angles -> joint change. The convolution layers are the classic
    84x84 Atari-game stack (Mnih et al. 2015)."""

    def __init__(self, n_joints: int = 6):
        super().__init__()
        self.see = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),   # 84 -> 20
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),  # 20 -> 9
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),  # 9 -> 7
            nn.Flatten(),                                # 64 * 7 * 7 = 3136
        )
        self.decide = nn.Sequential(
            nn.Linear(3136 + n_joints, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, n_joints),
        )

    def forward(self, image: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        return self.decide(torch.cat([self.see(image), joints], dim=1))


def to_input(images: np.ndarray) -> torch.Tensor:
    """uint8 pictures (N, 84, 84, 3) -> floats 0..1 in the (N, 3, 84, 84) order torch wants."""
    return torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0


class VisionPolicy:
    """The trained network as a policy: (picture, joint angles) -> joint change, real units."""

    def __init__(self, path=OUT / "policy_vision.pt"):
        saved = torch.load(path, weights_only=False)
        self.net = VisionNet()
        self.net.load_state_dict(saved["net"])
        self.net.eval()
        self.j_mean, self.j_std = saved["j_mean"], saved["j_std"]
        self.act_mean, self.act_std = saved["act_mean"], saved["act_std"]

    def __call__(self, image: np.ndarray, joints: np.ndarray) -> np.ndarray:
        j = torch.from_numpy(((joints - self.j_mean) / self.j_std).astype(np.float32))[None]
        with torch.no_grad():
            y = self.net(to_input(image[None]), j)[0].numpy()
        return y * self.act_std + self.act_mean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    torch.manual_seed(0)

    files = [OUT / "demos_vision.npz"] + sorted(OUT.glob("dagger_round*.npz"))
    parts = [np.load(f) for f in files]
    images = np.concatenate([d["image"] for d in parts])
    joints = np.concatenate([d["obs"][:, 3:] for d in parts])
    act = np.concatenate([d["act"] for d in parts])
    episode = np.concatenate([d["episode"] for d in parts])
    print("data: " + ", ".join(f"{f.name} {len(d['act'])} rows" for f, d in zip(files, parts)))
    # held out: the last demo episodes only, the same rows with or without DAgger data
    held = np.isin(episode, np.unique(parts[0]["episode"])[-HOLD_OUT:])
    j_mean, j_std = joints[~held].mean(0), joints[~held].std(0) + 1e-6
    act_mean, act_std = act[~held].mean(0), act[~held].std(0) + 1e-6
    j_all = torch.from_numpy((joints - j_mean) / j_std).float()
    y_all = torch.from_numpy((act - act_mean) / act_std).float()
    train_idx, val_idx = np.flatnonzero(~held), np.flatnonzero(held)
    print(f"{len(train_idx)} training rows, {len(val_idx)} held-out rows ({HOLD_OUT} episodes)")

    net = VisionNet()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f"network: {sum(w.numel() for w in net.parameters())} weights")

    def error(idx) -> tuple[float, float]:
        """Mean squared error (scaled units) and mean joint miss in degrees, over rows idx."""
        net.eval()
        sq, deg = 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(idx), 1024):
                b = idx[i:i + 1024]
                diff = net(to_input(images[b]), j_all[b]) - y_all[b]
                sq += (diff ** 2).mean(1).sum().item()
                deg += np.degrees((diff * torch.from_numpy(act_std)).abs().mean(1).sum().item())
        return sq / len(idx), deg / len(idx)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = np.random.default_rng(epoch).permutation(train_idx)
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            loss = ((net(to_input(images[b]), j_all[b]) - y_all[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch <= 2 or epoch % 5 == 0 or epoch == args.epochs:
            val_err, val_deg = error(val_idx)
            train_err, _ = error(train_idx[::10])  # a tenth of it: enough to see the trend
            print(f"epoch {epoch:3d}: error train {train_err:.4f}  held-out {val_err:.4f} "
                  f"(off the teacher by {val_deg:.3f} deg per joint per decision)  "
                  f"[{time.time() - t0:.0f} s]", flush=True)

    torch.save(dict(net=net.state_dict(), j_mean=j_mean, j_std=j_std,
                    act_mean=act_mean, act_std=act_std), OUT / "policy_vision.pt")
    print("saved logs/learned_policy/policy_vision.pt")


if __name__ == "__main__":
    main()
