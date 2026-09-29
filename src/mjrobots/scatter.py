"""Scatter the zebra bricks at random spots and angles on the table.

Every brick used to start in the same square spot, so nothing was ever
tested anywhere else. `scatter_bricks` puts each brick at a random spot and
angle (about vertical - it stays upright) inside a *zone*: the table spots
where the arm was measured able to pick a brick up and place it on the stack
at every angle (scripts/reach_map.py, the "green" squares of its map).

The zone is a JSON file of those grid squares. Only squares whose four
neighbours are green too are used, and a spot is drawn anywhere inside one -
so a brick is always between measured-green points, never half in an area
where some angles fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from . import sim_step

ZONES_DIR = Path(__file__).resolve().parent / "assets" / "scatter_zones"

TABLE_Z = -0.0842  # brick origin resting on the table (Victor's TABLE_Z)
STACK_XY = np.array([0.4148, 0.0])  # where the zebra gets built
MIN_BRICK_GAP = 0.10  # m between brick centres: room for the open fingers
STACK_CLEAR = 0.10  # m kept free around the stack spot
_SETTLE_STEPS = 300  # physics steps to let the bricks come to rest


class Zone:
    """Grid squares (centres, `cell` m wide) a brick may be scattered into."""

    def __init__(self, cells: np.ndarray, cell: float, arm: str) -> None:
        self.cells, self.cell, self.arm = cells, cell, arm

    @classmethod
    def load(cls, arm: str = "right", path: str | Path | None = None) -> "Zone":
        spec = json.loads(Path(path or ZONES_DIR / f"{arm}.json").read_text())
        green = {(round(x, 4), round(y, 4)) for x, y in spec["cells"]}
        c = spec["cell"]

        def inner(x: float, y: float) -> bool:
            return all((round(x + dx, 4), round(y + dy, 4)) in green
                       for dx, dy in ((c, 0), (-c, 0), (0, c), (0, -c)))

        cells = np.array(sorted(p for p in green if inner(*p)))
        return cls(cells, c, spec["arm"])

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        x, y = self.cells[rng.integers(len(self.cells))]
        return np.array([x, y]) + rng.uniform(-self.cell / 2, self.cell / 2, size=2)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def scatter_bricks(model, data, bodies, seed: int, zone: Zone, max_tries: int = 10_000) -> dict:
    """Put each brick in `bodies` at a random spot and angle in `zone`, at
    least MIN_BRICK_GAP from the others and STACK_CLEAR from the stack, then
    let physics settle them. Same `seed`, same scatter. Returns
    {body: (x, y, yaw_deg)} as placed."""
    rng = np.random.default_rng(seed)
    placed: dict[str, tuple[float, float, float]] = {}
    for body in bodies:
        for _ in range(max_tries):
            xy = zone.sample(rng)
            if np.linalg.norm(xy - STACK_XY) < STACK_CLEAR:
                continue
            if any(np.hypot(xy[0] - px, xy[1] - py) < MIN_BRICK_GAP for px, py, _ in placed.values()):
                continue
            break
        else:
            raise RuntimeError(f"couldn't find a free spot for {body} in the {zone.arm} arm's zone")
        yaw = rng.uniform(0, np.pi)  # a brick looks the same turned 180 deg
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        adr = model.jnt_qposadr[model.body_jntadr[b]]
        data.qpos[adr:adr + 3] = (xy[0], xy[1], TABLE_Z)
        data.qpos[adr + 3:adr + 7] = _yaw_quat(yaw)
        dof = model.jnt_dofadr[model.body_jntadr[b]]
        data.qvel[dof:dof + 6] = 0
        placed[body] = (float(xy[0]), float(xy[1]), float(np.degrees(yaw)))
    mujoco.mj_forward(model, data)
    for _ in range(_SETTLE_STEPS):
        sim_step.step(model, data)
    return placed


def describe(placed: dict) -> str:
    """"legs (0.31, -0.22) 37 deg, ..." for the log."""
    return ", ".join(f"{body.removeprefix('zebra_')} ({x:.2f}, {y:+.2f}) {yaw:.0f} deg"
                     for body, (x, y, yaw) in placed.items())
