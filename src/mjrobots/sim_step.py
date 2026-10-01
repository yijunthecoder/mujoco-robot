"""One control tick = several finer physics substeps.

Control (ctrl ramps, IK waypoints, gripper holds, the realtime clock) runs
every CONTROL_DT = 2 ms, as it always has - every step count in the motion
code is in these ticks. Physics runs underneath at the scene's own
`timestep` (0.5 ms in stationlite_pick_place.xml), `substeps` of them per
tick - the same split as ABC's sim env (amazon-far/abc: abc_sim/env.py,
`control_decimation`).

Why: a 30 g brick squeezed at 5 N between the fingers can't be resolved at
2 ms physics - the grip wobbles and shakes the brick loose within a move
(measured: 2 of 6 zebra carries dropped at 2 ms, 0 of 6 at 0.5 ms, with no
other change). The finer physics costs ~0.17 ms per 2 ms tick.
"""

from __future__ import annotations

import mujoco

CONTROL_DT = 0.002  # s


def substeps(model) -> int:
    """Physics steps per control tick for this model's timestep."""
    return max(1, round(CONTROL_DT / model.opt.timestep))


def step(model, data) -> None:
    """Advance one control tick."""
    for _ in range(substeps(model)):
        mujoco.mj_step(model, data)
