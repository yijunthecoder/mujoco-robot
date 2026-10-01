"""Refuse arm moves that would bring one arm too close to the other.

`ArmClearance.check` runs over a move's planned joint-space path 
(every waypoint, lead-in included) before the arm moves:it puts 
the moving arm - and the brick it's holding, carried along with the
gripper - at each waypoint on a scratch copy of the state, and measures the
closest distance to the other arm (which stays put: moves run one at a time).
Closer than MIN_ARM_CLEARANCE anywhere raises ClearanceError, a kind of
IKError, so callers that try alternatives on IKError (e.g. another grip
angle) try them here too.

Distances are between the links' collision meshes, which MuJoCo treats as
their convex hulls - a little bigger than the real links, so the true gap is
at least this.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .cartesian_control import IKError

# Closest the arms (and a held brick) may come. Measured: a normal zebra
# build with the other arm idle at home gets to 4.0 cm (right arm reaching
# the head brick on the left arm's side: right_link4 vs left_link2); every
# other move stays 19-34 cm apart.
MIN_ARM_CLEARANCE = 0.02  # m


class ClearanceError(IKError):
    """A planned move would bring the arms closer than MIN_ARM_CLEARANCE; the arm was not moved."""


def _arm_geoms(model, side: str) -> list[int]:
    """Collision geoms of the bodies named `<side>_...` (links and fingers)."""
    geoms = []
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if body.startswith(side + "_") and (model.geom_contype[g] or model.geom_conaffinity[g]):
            geoms.append(g)
    return geoms


class ArmClearance:
    """Clearance check for one arm's moves against the other arm."""

    def __init__(self, model, arm: str) -> None:
        self.model = model
        self.moving = _arm_geoms(model, arm)
        self.other = _arm_geoms(model, "left" if arm == "right" else "right")

    def closest(self, data, extra: list[int] = ()) -> float:
        """Current closest distance between this arm (plus `extra` geoms) and
        the other arm, capped at 1 m."""
        fromto = np.zeros(6)
        best = 1.0
        for g1 in (*self.moving, *extra):
            for g2 in self.other:
                best = min(best, mujoco.mj_geomDistance(self.model, data, g1, g2, best, fromto))
        return best

    def check(self, data, joint_ids, path, gripper_body: int, held_body: int | None, what: str) -> None:
        """Raise ClearanceError if any waypoint of `path` (joint angles for
        `joint_ids`) comes closer than MIN_ARM_CLEARANCE. `held_body`, if
        given, is a free-jointed brick in the gripper: it's carried along at
        its current pose relative to `gripper_body`. Works on a scratch copy
        of qpos - the real state is restored before returning."""
        model = self.model
        qpos_adr = model.jnt_qposadr[joint_ids]
        held_geoms: list[int] = []
        if held_body is not None:
            held_geoms = [g for g in range(model.ngeom)
                          if model.geom_bodyid[g] == held_body and model.geom_contype[g]]
            held_adr = model.jnt_qposadr[model.body_jntadr[held_body]]
            # brick pose in the gripper's frame, kept fixed along the path
            grip_R = data.xmat[gripper_body].reshape(3, 3).copy()
            grip_t = data.xpos[gripper_body].copy()
            rel_t = grip_R.T @ (data.xpos[held_body] - grip_t)
            rel_R = grip_R.T @ data.xmat[held_body].reshape(3, 3)

        qpos_save = data.qpos.copy()
        try:
            for i, q in enumerate(path):
                data.qpos[qpos_adr] = q
                mujoco.mj_kinematics(model, data)
                if held_body is not None:
                    R = data.xmat[gripper_body].reshape(3, 3)
                    data.qpos[held_adr:held_adr + 3] = data.xpos[gripper_body] + R @ rel_t
                    quat = np.empty(4)
                    mujoco.mju_mat2Quat(quat, (R @ rel_R).flatten())
                    data.qpos[held_adr + 3:held_adr + 7] = quat
                    mujoco.mj_kinematics(model, data)
                gap = self.closest(data, held_geoms)
                if gap < MIN_ARM_CLEARANCE:
                    raise ClearanceError(
                        f"{what}: would pass {gap * 100:.1f} cm from the other arm at waypoint "
                        f"{i + 1}/{len(path)} (minimum {MIN_ARM_CLEARANCE * 100:.0f} cm) - arm not moved"
                    )
        finally:
            data.qpos[:] = qpos_save
            mujoco.mj_kinematics(model, data)
