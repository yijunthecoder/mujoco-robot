"""Confirm-before-moving safety check, for the first runs on the real arm.

Before an arm move is executed, print each joint's current angle, its final
target, and how far it will travel (flagging big ones), then wait for the
operator: ENTER executes the move, "q" + ENTER refuses it (MoveRefused, the
arm stays put). Modelled on ABC's first-action check
(amazon-far/abc: deploy/robot/gym/policy_safety.py).

The wait never blocks outright: it keeps calling `render.step()` (the
viewer, and in the skill bridge also the perception publisher - zebra_bt
treats perception older than 2s as stale) while polling stdin. zebra_bt
still times a skill out after 30s, so answer within that.

Modes (`set_mode`): "off" (default - moves run unprompted, as before),
"first" (only the first move of the run), "all" (every move).
"""

from __future__ import annotations

import select
import sys
import time

import numpy as np

# A joint travelling further than this in one move gets flagged, as does
# a first move - ABC warns on a first-step delta over 0.5 rad.
LARGE_CHANGE = 0.5  # rad

MODES = ("off", "first", "all")

_mode = "off"
_asked = 0


class MoveRefused(RuntimeError):
    """The operator refused a move at the confirmation prompt; the arm was not moved."""


def set_mode(mode: str) -> None:
    global _mode, _asked
    if mode not in MODES:
        raise ValueError(f"confirm mode must be one of {MODES}, got {mode!r}")
    _mode = mode
    _asked = 0


def describe(what: str, q_start: np.ndarray, path: list[np.ndarray], duration_s: float) -> str:
    """The per-joint table for one planned move."""
    q_start = np.asarray(q_start, dtype=float)
    q_end = path[-1]
    steps = np.abs(np.diff(np.vstack([q_start, *path]), axis=0)).max(axis=1)
    lines = [
        "=" * 64,
        f"MOVE CHECK: {what}",
        f"{len(path)} waypoints, {duration_s:.1f}s",
        f"{'joint':<6} {'now':>8} {'target':>8} {'change':>8}",
        "-" * 64,
    ]
    for j, (a, b) in enumerate(zip(q_start, q_end)):
        d = b - a
        flag = "  <<< large" if abs(d) > LARGE_CHANGE else ""
        lines.append(f"j{j + 1:<5} {a:>+8.3f} {b:>+8.3f} {d:>+8.3f}  ({np.degrees(d):+6.1f} deg){flag}")
    biggest = float(np.abs(q_end - q_start).max())
    lines += [
        "-" * 64,
        f"largest joint change {biggest:.3f} rad ({np.degrees(biggest):.1f} deg), "
        f"largest single waypoint step {steps.max():.3f} rad",
        "=" * 64,
    ]
    return "\n".join(lines)


def confirm(what: str, q_start: np.ndarray, path: list[np.ndarray], duration_s: float, render=None) -> None:
    """Ask the operator before executing a planned move, per the current mode.

    Returns to execute the move; raises MoveRefused on "q" or if stdin is
    closed (no operator to confirm means don't move).
    """
    global _asked
    if _mode == "off" or (_mode == "first" and _asked > 0):
        return
    _asked += 1

    # Ends in a newline: run_zebra.sh pipes output through a line-buffered sed.
    print(describe(what, q_start, path, duration_s), flush=True)
    print("Press ENTER to move, or q + ENTER to refuse:", flush=True)

    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.02)
        if ready:
            line = sys.stdin.readline()
            if line == "":
                raise MoveRefused(f"{what}: no operator input (stdin closed) - arm not moved")
            if line.strip().lower() == "q":
                raise MoveRefused(f"{what}: refused by operator - arm not moved")
            print("-> moving", flush=True)
            return
        if render is not None:
            render.step()
        else:
            time.sleep(0.02)
