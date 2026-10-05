"""The robot's senses in the simulation - in one place.

Every time the zebra bridge needs to know something about a brick that on the real
robot only a camera could tell it (how it lies, which way it's turned, where it sits in
the hand, which side the print is on, where the other bricks are), it asks `SimSensing`.
Here the simulation answers from its true state, with noise like a camera's; nothing in
this module moves anything.

On the real robot this is the module to replace: a class with the same methods,
answered by the real cameras and a brick detector. The rest of the code doesn't need to
change for that.

Not in here yet (see README, "Real robot"):
- where each brick is: perception's camera images are rendered from the true pose
  with pixel and depth noise (camera_calibration.SimulatedCameras, used by
  zebra_publisher and the bridge's look again from hover);
- which brick is which: by the simulation's body names (zebra_publisher.PART_BODIES,
  and the arms' obstacle bricks in zebra_pick_place: every "zebra_*" body);
- the flip planner plans on a copy of the simulation state (zebra_flip.PlanningProcess):
  on the robot it needs that state built from what perception reports;
- the grasp's "sim check: x cm off center" in the log (ZebraArmContext.grip_miss) is a
  test aid only - the real check is the finger width, already used.
"""

from __future__ import annotations

import numpy as np

from .scatter import lying
from .zebra_facing import SHOW_CAMERA, long_face_view
from .zebra_pick_place import _BRICK_CENTER_OFFSET_Z

# Noise of one simulated yaw look (brick_yaw) and of a look at a held brick (held_brick_look).
YAW_SIGMA = np.radians(2.0)
_HELD_POS_SIGMA = 0.002  # m
# The print-side look (print_look): the camera must see this share of a long face
# unblocked, this tall in its image, to tell printed from blank (assumed - to be checked
# on the real camera; shown 50-65 cm away a face is 20-30 px tall).
PRINT_LOOK_MIN_SHARE = 0.5
PRINT_LOOK_MIN_PX = 20.0
_PRINT_DIR_SIGMA = 0.05  # noise on the print direction (unit vector, ~3 deg)


class SimSensing:
    """What the cameras would tell the robot about the bricks, answered by the simulation.
    `brick_ids`: {part id: MuJoCo body id}; `rng`: the noise source (the bridge passes
    perception's, so the noise is drawn in the same order as before this module existed)."""

    def __init__(self, model, data, brick_ids: dict[str, int], rng: np.random.Generator) -> None:
        self.model, self.data = model, data
        self.brick_ids = dict(brick_ids)
        self.rng = rng

    def brick_yaw(self, part_id: str, samples: int = 1) -> float:
        """The brick's rotation about vertical from square (radians, in [-pi/2, pi/2) - a
        brick looks the same turned 180 deg), averaged over `samples` noisy looks. Real
        robot: estimate the long axis from the camera image. Only asked once a camera has
        actually seen the brick."""
        R = self.data.xmat[self.brick_ids[part_id]].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        looks = yaw + self.rng.normal(0.0, YAW_SIGMA, samples)
        # Average on the doubled angle, where yaw and yaw + 180 deg coincide.
        mean = np.arctan2(np.sin(2 * looks).mean(), np.cos(2 * looks).mean()) / 2
        return float((mean + np.pi / 2) % np.pi - np.pi / 2)

    def how_it_lies(self, part_id: str, data=None) -> str:
        """UPRIGHT / UPSIDE_DOWN / ON_SIDE / ON_END / TILTED (scatter.lying) - on the table
        or in the hand. `data`: the state to look at (default the live one). Real robot:
        from the detector (which face is up), like perception's LYING field."""
        d = self.data if data is None else data
        return lying(d.xmat[self.brick_ids[part_id]].reshape(3, 3))

    def brick_xy(self, part_id: str) -> np.ndarray:
        """Where the brick is on the table (x, y of its origin, like perception reports it).
        Real robot: perception's latest report."""
        return self.data.xpos[self.brick_ids[part_id]][:2].copy()

    def other_bricks_xy(self, part_id: str) -> list[np.ndarray]:
        """Where every other brick is on the table (x, y), to keep set-downs clear of them.
        Real robot: from perception's latest reports."""
        return [self.data.xpos[b][:2].copy() for p, b in self.brick_ids.items() if p != part_id]

    def held_brick_look(self, part_id: str):
        """A look at the brick in the hand (a hand camera: B's at a flip's handover, A's once
        B holds it): returns look(data) -> (rotation, centre) - here the true pose plus
        ~2 mm / ~2 deg noise. Real robot: the hand camera's detector."""
        b = self.brick_ids[part_id]

        def look(d):
            R = d.xmat[b].reshape(3, 3).copy()
            center = d.xpos[b] + R @ np.array([0, 0, _BRICK_CENTER_OFFSET_Z])
            th = self.rng.normal(0.0, YAW_SIGMA)
            Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
            return Rz @ R, center + self.rng.normal(0.0, _HELD_POS_SIGMA, 3)
        return look

    def print_look(self, part_id: str, camera: str = SHOW_CAMERA):
        """Which way the printed face points, as `camera` sees the brick (the headcam: held up
        in front of it; a hand camera: at a flip's handover): returns look(data) -> unit
        vector (world), or None if that camera can't tell. Here the true direction plus ~3
        deg noise, but only when the camera really has a usable view of a long face
        (zebra_facing.long_face_view: geometry, fingers and arms blocking included) - seeing
        either long face tells the side. Real robot: read the print in the camera image."""
        b = self.brick_ids[part_id]

        def look(d):
            share, px = long_face_view(self.model, d, camera, b)
            if share < PRINT_LOOK_MIN_SHARE or px < PRINT_LOOK_MIN_PX:
                return None
            n = -d.xmat[b].reshape(3, 3)[:, 1] + self.rng.normal(0.0, _PRINT_DIR_SIGMA, 3)
            return n / np.linalg.norm(n)  # the print is on the brick's -y face
        return look
