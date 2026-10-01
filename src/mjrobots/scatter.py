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

With both arms (`Zone.load("either")`) the zone is both arms' squares
together, and `choose_arm` says which arm should pick a brick at a spot.

`drop_bricks` is the messier start: each brick is dropped from above a
random zone spot, tumbled to a random 3D rotation, and lands however physics
lets it - measured over 100 drops from 15 cm: 43% on a long side, 24%
upside down, 17% on an end, only 16% upright. `lying` names how a brick
lies, from its rotation.
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
DROP_HEIGHT = 0.15  # m above the table a dropped brick starts
_DROP_SETTLE_STEPS = 1500  # 3 s: fall, bounce, come to rest

# How a brick can lie (`lying`). Body axes: z = the studs' direction (up when
# UPRIGHT), x = the long side (6.4 cm), y = the short side (3.2 cm).
UPRIGHT, UPSIDE_DOWN, ON_SIDE, ON_END, TILTED = "UPRIGHT", "UPSIDE_DOWN", "ON_SIDE", "ON_END", "TILTED"


class Zone:
    """Grid squares (centres, `cell` m wide) a brick may be scattered into."""

    def __init__(self, cells: np.ndarray, cell: float, arm: str) -> None:
        self.cells, self.cell, self.arm = cells, cell, arm

    @classmethod
    def load(cls, arm: str = "right", path: str | Path | None = None) -> "Zone":
        """`arm` is "right", "left", or "either" (both arms' squares together)."""
        if arm == "either":
            right, left = cls.load("right"), cls.load("left")
            cells = np.unique(np.round(np.vstack([right.cells, left.cells]), 4), axis=0)
            return cls(cells, right.cell, "either")
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

    def covers(self, xy) -> bool:
        """Whether spot `xy` lies inside one of the zone's squares."""
        return bool(np.any(np.all(np.abs(self.cells - np.asarray(xy)[:2]) <= self.cell / 2 + 1e-9, axis=1)))


def arm_bases(model, data) -> dict[str, np.ndarray]:
    """Each arm's base (link1) position on the table plane, in world frame."""
    return {side: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_link1")][:2].copy()
            for side in ("left", "right")}


def choose_arm(xy, zones: dict[str, Zone], bases: dict[str, np.ndarray]) -> str:
    """The arm to pick a brick at `xy` with: the one whose zone it's in, or -
    in both zones (the middle) or neither (e.g. knocked out of reach) - the
    one whose base is closer."""
    fits = [arm for arm, zone in zones.items() if zone.covers(xy)]
    if len(fits) == 1:
        return fits[0]
    return min(fits or zones, key=lambda arm: float(np.linalg.norm(np.asarray(xy)[:2] - bases[arm])))


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _random_quat(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random 3D rotation (w, x, y, z)."""
    u1, u2, u3 = rng.random(3)
    a, b = np.sqrt(1 - u1), np.sqrt(u1)
    return np.array([b * np.cos(2 * np.pi * u3), a * np.sin(2 * np.pi * u2),
                     a * np.cos(2 * np.pi * u2), b * np.sin(2 * np.pi * u3)])


def lying(R: np.ndarray) -> str:
    """How a brick with rotation matrix `R` (body -> world) lies: which of its
    axes points up. TILTED if none is within ~18 deg of vertical (e.g.
    leaning on another brick)."""
    up = R[2]  # world vertical in body axes
    axis = int(np.argmax(np.abs(up)))
    if abs(up[axis]) < 0.95:
        return TILTED
    if axis == 2:
        return UPRIGHT if up[2] > 0 else UPSIDE_DOWN
    return ON_SIDE if axis == 1 else ON_END


def table_yaw(R: np.ndarray) -> float:
    """The brick's angle on the table (radians, in [-pi/2, pi/2) - it looks
    the same turned 180 deg): the direction its long side points, or - stood
    on its end, long side vertical - its short side."""
    axis = R[:, 0] if lying(R) != ON_END else R[:, 1]
    yaw = np.arctan2(axis[1], axis[0])
    return float((yaw + np.pi / 2) % np.pi - np.pi / 2)


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


def drop_bricks(model, data, bodies, seed: int, zone: Zone, height: float = DROP_HEIGHT,
                max_tries: int = 10_000) -> dict:
    """Drop each brick in `bodies` from `height` above a random spot in
    `zone` (spots MIN_BRICK_GAP apart and STACK_CLEAR from the stack, as in
    scatter_bricks), turned to a random 3D rotation, and let it land. Same
    `seed`, same drop. Returns {body: (x, y, yaw_deg, lying)} where each came
    to rest (they slide ~1-5 cm from the drop spot)."""
    rng = np.random.default_rng(seed)
    spots: list[np.ndarray] = []
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) for body in bodies]
    for body, b in zip(bodies, ids):
        for _ in range(max_tries):
            xy = zone.sample(rng)
            if np.linalg.norm(xy - STACK_XY) >= STACK_CLEAR and all(
                    np.linalg.norm(xy - s) >= MIN_BRICK_GAP for s in spots):
                break
        else:
            raise RuntimeError(f"couldn't find a free spot for {body} in the {zone.arm} zone")
        spots.append(xy)
        adr = model.jnt_qposadr[model.body_jntadr[b]]
        data.qpos[adr:adr + 3] = (xy[0], xy[1], TABLE_Z + height)
        data.qpos[adr + 3:adr + 7] = _random_quat(rng)
        dof = model.jnt_dofadr[model.body_jntadr[b]]
        data.qvel[dof:dof + 6] = 0
    mujoco.mj_forward(model, data)
    for _ in range(_DROP_SETTLE_STEPS):
        sim_step.step(model, data)
    landed = {}
    for body, b in zip(bodies, ids):
        R = data.xmat[b].reshape(3, 3)
        landed[body] = (float(data.xpos[b][0]), float(data.xpos[b][1]),
                        float(np.degrees(table_yaw(R))), lying(R))
    return landed


def describe(placed: dict) -> str:
    """"legs (0.31, -0.22) 37 deg, ..." (plus how it lies, for a drop) for the log."""
    return ", ".join(f"{body.removeprefix('zebra_')} ({x:.2f}, {y:+.2f}) {yaw:.0f} deg"
                     + (f" {rest[0]}" if rest else "")
                     for body, (x, y, yaw, *rest) in placed.items())
