"""Compute a MuJoCo camera orientation that points at a target.

A ``<camera>`` element's ``quat`` attribute is an orientation, not a target -
MuJoCo has no "look at this point" option in the XML itself, so pointing a
camera somewhere specific means computing that orientation yourself.

"""

from __future__ import annotations

import mujoco
import numpy as np


def look_at_quat(eye, target, up_ref=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Return the quat that makes a camera at `eye` look toward `target`.

    `up_ref` is a reference "up" direction, used only to fix the camera's
    roll (rotation about its own viewing axis) - it does not need to be
    exactly perpendicular to the view direction, just not parallel to it.
    Use world-vertical (the default) for a camera fixed in the world; use
    a body's own local axis instead for a camera attached to a moving body
    (pass eye/target in that body's local frame too, in that case) - see
    stationlite_converted.xml's left_handcam/right_handcam for an example,
    where up_ref=(0,1,0) (the gripper's own local Y) was used since the
    look-at direction there has no Y component to conflict with it.

    MuJoCo convention: a camera looks down its own local -Z axis, with
    local +Y as "up" in the rendered image. Building the rotation is just
    three mutually-perpendicular axes, worked out via cross products:
        z_axis = -forward                        (points backward from view dir)
        x_axis = up_ref x z_axis                  ("right", perpendicular to both)
        y_axis = z_axis x x_axis                  ("up", perpendicular to both of those)
    Stacking those three as columns of a 3x3 matrix *is* the camera's
    orientation; mju_mat2Quat just converts that matrix into the quat
    format the XML wants.
    """
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward = forward / np.linalg.norm(forward)

    z_axis = -forward
    x_axis = np.cross(up_ref, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rot_matrix = np.column_stack([x_axis, y_axis, z_axis]).flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot_matrix)
    return quat
