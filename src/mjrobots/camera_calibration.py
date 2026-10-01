"""Four-camera calibration for the stationlite scene, at the robot's home pose.

Problem: each camera reports where a reference object is *relative to itself*,
so the four reports can't be compared. Calibration finds, for every camera, the
rigid transform (rotation R + translation t) that converts its private
coordinates into one shared frame: ``p_shared = R @ p_cam + t``. If the
transforms are right, all four cameras report the same coordinates for the same
object.

The four cameras: ``headcam`` and ``refcam`` are fixed on the mast; the two
hand cameras ride on the arms. Calibration is done with the arms held at their
original (home) pose, so the hand cameras' transforms are fixed too.

At the home pose each hand camera looks down at its own side of the table, so
no block position is visible to all four cameras at once. That is fine:
``headcam`` (which sees the whole table) is the shared frame, and each other
camera is tied to it separately using block positions that *both* can see.

A hand camera's calibrated transform only holds while its arm is at home.
Once the arm moves, the camera moves with it, so `CameraCalibration.to_reference`
first undoes that motion. It uses the arm's forward kinematics (joint angles
plus the camera's fixed mount on the gripper link, both from the robot model),
which a real robot also knows from its joint encoders. Without this step, a
hand camera hovering over the table reports positions tens of cm off.

Method:
  1. Move a reference object (the zebra ``zebra_legs`` brick, Victor's "feet"
     part) around the table.
  2. At each position, every camera that can see it measures its 3D position in
     the camera's own frame (pixel of the block's centre + depth -> back-project).
  3. For each camera other than headcam, take the positions both it and headcam
     saw: the same points in two frames. Solve for the rotation and translation
     that best map one onto the other (Kabsch / Procrustes, an SVD).
  4. Check on *new* block positions that were not used for fitting: transform
     every camera's estimate into the shared frame and measure how much the
     cameras that saw the block disagree.

The camera measurements are synthesized (pinhole projection of the block's true
position + pixel/depth noise) instead of rendered from images. The calibration
maths only needs "pixel + depth" per camera, which is exactly what a real
detector plus depth sensor would supply. The *solver* never sees
any ground truth; MuJoCo's true camera poses are used only to synthesize the
measurements and afterwards to grade the result.

Camera-frame convention throughout is MuJoCo's: x right, y up, camera looks
down its own -z axis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .gl import configure_gl

_DEFAULT_SCENE = (
    Path(__file__).resolve().parent.parent.parent / "stationlite" / "urdf" / "stationlite_pick_place.xml"
)

CAMERAS = ("headcam", "refcam", "left_handcam", "right_handcam")
HAND_CAMERAS = ("left_handcam", "right_handcam")
REFERENCE_CAMERA = "headcam"  # its frame becomes the shared frame
IMAGE_SIZE = (640, 480)  # (width, height); the scene XML leaves resolution unset
_RAY_GROUPS = np.array([1, 1, 0, 1, 1, 1], dtype=np.uint8)  # all geom groups except 2 (visual meshes)
# Line-of-sight ray targets, in units of the block box's half-size: its centre
# and a point just inside each corner.
_BOX_SAMPLES = np.array(
    [[0, 0, 0]] + [[sx, sy, sz] for sx in (-0.9, 0.9) for sy in (-0.9, 0.9) for sz in (-0.9, 0.9)]
)

# Where the block may be placed (world frame, metres): across the table, from
# table height up to a few cm above it.
_BLOCK_X = (0.35, 0.70)
_BLOCK_Y = (-0.42, 0.42)
_BLOCK_Z = (-0.1026, -0.05)


@dataclass(frozen=True)
class RigidTransform:
    """``p_out = R @ p_in + t``."""

    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    def apply(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points) @ self.R.T + self.t

    def inverse(self) -> "RigidTransform":
        return RigidTransform(self.R.T, -self.R.T @ self.t)

    def then(self, other: "RigidTransform") -> "RigidTransform":
        """Apply ``self`` first, then ``other``."""
        return RigidTransform(other.R @ self.R, other.R @ self.t + other.t)

    @staticmethod
    def identity() -> "RigidTransform":
        return RigidTransform(np.eye(3), np.zeros(3))


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera model derived from MuJoCo's ``fovy`` and the image size."""

    width: int
    height: int
    fovy_deg: float

    @property
    def focal_px(self) -> float:
        return (self.height / 2) / np.tan(np.radians(self.fovy_deg) / 2)

    def project(self, p_cam: np.ndarray) -> tuple[float, float, float]:
        """Camera-frame point -> (u, v, depth). Depth is distance along the view axis."""
        depth = -p_cam[2]
        u = self.width / 2 + self.focal_px * p_cam[0] / depth
        v = self.height / 2 - self.focal_px * p_cam[1] / depth
        return u, v, depth

    def back_project(self, u: float, v: float, depth: float) -> np.ndarray:
        """(u, v, depth) -> camera-frame point. Inverse of `project`."""
        x = (u - self.width / 2) * depth / self.focal_px
        y = -(v - self.height / 2) * depth / self.focal_px
        return np.array([x, y, -depth])


def fit_rigid_transform(src: np.ndarray, dst: np.ndarray) -> RigidTransform:
    """Least-squares rigid transform mapping ``src`` points onto ``dst`` (Kabsch).

    Both are (N, 3) with row i of ``src`` corresponding to row i of ``dst``.
    Needs N >= 3 points that are not all on one line.
    """
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    cross_cov = (src - src_mean).T @ (dst - dst_mean)
    U, _, Vt = np.linalg.svd(cross_cov)
    # The det term stops the SVD from returning a mirror image (reflection)
    # instead of a proper rotation.
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return RigidTransform(R, dst_mean - R @ src_mean)


def calibrate(pairs: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, RigidTransform]:
    """Per-camera transform from its own frame into the reference camera's frame.

    ``pairs[name] = (in_camera, in_reference)``: two (N, 3) arrays holding the
    block's position at N block positions, as measured by camera ``name`` and by
    the reference camera. Every camera may use its own set of positions.
    """
    transforms = {REFERENCE_CAMERA: RigidTransform.identity()}
    for name, (in_camera, in_reference) in pairs.items():
        transforms[name] = fit_rigid_transform(in_camera, in_reference)
    return transforms


@dataclass(frozen=True)
class CameraCalibration:
    """Calibrated camera -> reference transforms, including cameras that ride on an arm.

    ``transforms`` come from `calibrate`, measured with the arms at home.
    ``home_mount_poses[name]`` is each hand camera's pose at that same home
    pose, from forward kinematics (`SimulatedCameras.mount_pose`). At runtime
    a hand camera's point is first carried from where the camera is now back
    to where it was during calibration, then converted as usual.
    """

    transforms: dict[str, RigidTransform]
    home_mount_poses: dict[str, RigidTransform]

    def to_reference(self, name: str, p_cam: np.ndarray, cams: "SimulatedCameras") -> np.ndarray:
        """Camera ``name``'s measurement -> reference (headcam) frame, for the arm pose in ``cams.data``."""
        if name in self.home_mount_poses:
            # camera now -> robot base (FK now) -> camera at home (inverse of FK at home)
            now_to_home = cams.mount_pose(name).then(self.home_mount_poses[name].inverse())
            p_cam = now_to_home.apply(p_cam)
        return self.transforms[name].apply(p_cam)


def calibrate_cameras(
    model: mujoco.MjModel,
    n_calib: int,
    rng: np.random.Generator,
    pixel_sigma: float = 0.5,
    depth_sigma: float = 0.002,
) -> CameraCalibration:
    """Calibrate every camera on a scratch MjData at the home pose.

    It must be a scratch copy: calibration teleports the block (`place_block`).
    """
    scratch = mujoco.MjData(model)
    _setup_home_pose(model, scratch)
    cams = SimulatedCameras(model, scratch, pixel_sigma=pixel_sigma, depth_sigma=depth_sigma, rng=rng)
    transforms = calibrate(_collect_calibration_pairs(cams, n_calib, rng))
    return CameraCalibration(transforms, {name: cams.mount_pose(name) for name in HAND_CAMERAS})


class SimulatedCameras:
    """Stand-in for 'run a detector on each camera image and read the depth'.

    `observe` returns the block's 3D position in a camera's own frame - or None
    if the block is out of that camera's view or blocked by something.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        block_body: str = "zebra_legs",
        pixel_sigma: float = 0.5,
        depth_sigma: float = 0.002,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model, self.data = model, data
        self.pixel_sigma, self.depth_sigma = pixel_sigma, depth_sigma
        self.rng = rng or np.random.default_rng(0)
        self.intrinsics = {}
        self.cam_id = {}
        for name in CAMERAS:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            self.cam_id[name] = i
            self.intrinsics[name] = Intrinsics(*IMAGE_SIZE, float(model.cam_fovy[i]))
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, block_body)
        self._block_qpos = model.jnt_qposadr[model.body_jntadr[body]]
        # The block's solid collision box - its other geoms are group-2 visual
        # meshes, which the line-of-sight rays skip.
        geoms = range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body])
        boxes = [g for g in geoms if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and _RAY_GROUPS[model.geom_group[g]]]
        if len(boxes) != 1:
            raise ValueError(f"{block_body}: expected one ray-visible collision box, found {len(boxes)}")
        self._block_geom = boxes[0]

    def place_block(self, position: np.ndarray) -> None:
        self.data.qpos[self._block_qpos : self._block_qpos + 3] = position
        self.data.qpos[self._block_qpos + 3 : self._block_qpos + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(self.model, self.data)

    def cam_pose(self, name: str) -> RigidTransform:
        """True camera-frame -> world transform. For synthesis and grading only."""
        i = self.cam_id[name]
        return RigidTransform(self.data.cam_xmat[i].reshape(3, 3).copy(), self.data.cam_xpos[i].copy())

    def mount_pose(self, name: str) -> RigidTransform:
        """Camera-frame -> robot-base transform from forward kinematics.

        This is the pose of the link the camera is mounted on (from the joint
        angles), composed with the camera's fixed mount on that link (from the
        robot model). A real robot knows both from its joint encoders plus its
        URDF or hand-eye calibration, so unlike `cam_pose` the perception side
        may use it. In sim it equals `cam_pose`.
        """
        i = self.cam_id[name]
        link = self.model.cam_bodyid[i]
        link_R = self.data.xmat[link].reshape(3, 3)
        mount_R = np.zeros(9)
        mujoco.mju_quat2Mat(mount_R, self.model.cam_quat[i])
        return RigidTransform(link_R @ mount_R.reshape(3, 3), self.data.xpos[link] + link_R @ self.model.cam_pos[i])

    def _clear_line_of_sight(self, name: str) -> bool:
        """True if any part of the block's box is visible from the camera.

        Rays go to the box centre and to a point just inside each of its
        corners. If a ray's first hit is the box itself, that part of the
        block is visible. Don't aim at the block's origin: that point is the
        centre of the brick's top face, which is exactly the bottom face of
        any brick stacked on it. The ray then hits whichever box
        floating-point noise favours, which hid lower stacked bricks from
        every camera.
        """
        i = self.cam_id[name]
        origin = self.data.cam_xpos[i].copy()
        g = self._block_geom
        centre, box_R, half = self.data.geom_xpos[g], self.data.geom_xmat[g].reshape(3, 3), self.model.geom_size[g]
        hit_geom = np.zeros(1, dtype=np.int32)
        for local in _BOX_SAMPLES * half:
            direction = centre + box_R @ local - origin
            direction /= np.linalg.norm(direction)
            # bodyexclude: don't let the camera's own housing block its view.
            # geomgroup: skip group 2 (the bricks' visual meshes) so only
            # their solid collision boxes count - the two share faces exactly.
            mujoco.mj_ray(self.model, self.data, origin, direction, _RAY_GROUPS, 1, self.model.cam_bodyid[i], hit_geom)
            if hit_geom[0] == g:
                return True
        return False

    def observe(self, name: str, noisy: bool = True) -> np.ndarray | None:
        block_world = self.data.qpos[self._block_qpos : self._block_qpos + 3]
        pose = self.cam_pose(name)
        p_cam = pose.R.T @ (block_world - pose.t)
        if p_cam[2] >= 0:  # behind the camera
            return None
        intr = self.intrinsics[name]
        u, v, depth = intr.project(p_cam)
        margin = 20
        if not (margin < u < intr.width - margin and margin < v < intr.height - margin):
            return None
        if not self._clear_line_of_sight(name):
            return None
        if noisy:
            u += self.rng.normal(0, self.pixel_sigma)
            v += self.rng.normal(0, self.pixel_sigma)
            depth += self.rng.normal(0, self.depth_sigma)
        return intr.back_project(u, v, depth)


def _setup_home_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Arms at their original home pose; the other two zebra pieces parked out of the way."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    # zebra_body/zebra_head would otherwise sit in the scene and get in the
    # cameras' way while we calibrate/track zebra_legs. Nothing is simulated
    # here (no mj_step), so tucking them under the table is safe.
    for other in ("zebra_body", "zebra_head"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, other)
        data.qpos[model.jnt_qposadr[model.body_jntadr[body]] + 2] = -0.6
    mujoco.mj_forward(model, data)


def _random_block_position(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform((_BLOCK_X[0], _BLOCK_Y[0], _BLOCK_Z[0]), (_BLOCK_X[1], _BLOCK_Y[1], _BLOCK_Z[1]))


def _collect_calibration_pairs(cams: SimulatedCameras, n: int, rng: np.random.Generator):
    """For each non-reference camera, n block positions that it AND headcam can see.

    Returns {camera: (measured by camera (n, 3), measured by headcam (n, 3))}.
    """
    pairs = {name: ([], []) for name in CAMERAS if name != REFERENCE_CAMERA}
    attempts = 0
    while any(len(own) < n for own, _ in pairs.values()):
        attempts += 1
        if attempts > 2000 * n:
            raise RuntimeError("Could not find enough block positions visible to each camera")
        cams.place_block(_random_block_position(rng))
        ref = cams.observe(REFERENCE_CAMERA)
        if ref is None:
            continue
        for name, (own, in_ref) in pairs.items():
            if len(own) >= n:
                continue
            seen = cams.observe(name)
            if seen is not None:
                own.append(seen)
                in_ref.append(ref)
    return {name: (np.array(own), np.array(in_ref)) for name, (own, in_ref) in pairs.items()}


def _collect_test_positions(cams: SimulatedCameras, n: int, rng: np.random.Generator):
    """n new block positions seen by headcam and at least one hand camera.

    Returns (world positions (n, 3), [ {camera: measurement} for each position ]);
    each dict only has the cameras that could see the block.
    """
    positions, observations = [], []
    attempts = 0
    while len(positions) < n:
        attempts += 1
        if attempts > 2000 * n:
            raise RuntimeError("Could not find test positions visible to a hand camera")
        pos = _random_block_position(rng)
        cams.place_block(pos)
        seen = {name: cams.observe(name) for name in CAMERAS}
        seen = {name: p for name, p in seen.items() if p is not None}
        if REFERENCE_CAMERA in seen and any(h in seen for h in HAND_CAMERAS):
            positions.append(pos)
            observations.append(seen)
    return np.array(positions), observations


def _observe_live_position(cams: SimulatedCameras) -> dict:
    """Cameras' view of the block wherever it *currently* is - unlike
    `_collect_test_positions`, this never calls `place_block`, so it doesn't
    teleport the block. `_collect_test_positions` exists to synthesize fresh
    ground-truth positions for testing calibration accuracy; this exists for
    the opposite case, tracking a block that a live simulation (physics, or
    an arm carrying it) is actually moving on its own.

    Returns {camera: measurement}, only cameras that can currently see it.
    """
    seen = {name: cams.observe(name) for name in CAMERAS}
    return {name: p for name, p in seen.items() if p is not None}


def _show_in_viewer(model, data, cams: SimulatedCameras, test_world: np.ndarray, aligned_world: list[dict]) -> None:
    """Open the MuJoCo viewer and step the block through the held-out positions.

    Each camera's aligned estimate is drawn as a small coloured sphere; if the
    calibration is good, the spheres pile up on top of the (made see-through)
    block. (Don't enable the viewer's camera-marker overlay: looking through a
    camera from inside its own marker fills the whole view with green.)
    """
    import mujoco.viewer

    colours = {
        "headcam": (1.0, 0.2, 0.2, 1),
        "refcam": (0.2, 1.0, 0.2, 1),
        "left_handcam": (0.3, 0.5, 1.0, 1),
        "right_handcam": (1.0, 0.9, 0.1, 1),
    }
    model.geom_rgba[cams._block_geom, 3] = 0.35
    print("\nViewer open: block hops through the held-out positions; close the window to exit.")
    print("Spheres: headcam=red, refcam=green, left_handcam=blue, right_handcam=yellow")
    print("(only cameras that can see the block at that spot draw a sphere)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        k, last = 0, 0.0
        while viewer.is_running():
            if time.time() - last > 1.5:
                cams.place_block(test_world[k])
                with viewer.lock():
                    viewer.user_scn.ngeom = len(aligned_world[k])
                    for j, (name, point) in enumerate(aligned_world[k].items()):
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[j],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.008, 0, 0]),
                            point,
                            np.eye(3).flatten(),
                            np.array(colours[name], dtype=np.float32),
                        )
                viewer.sync()
                k, last = (k + 1) % len(test_world), time.time()
            time.sleep(0.02)


def _print_position(label: int | str, truth: np.ndarray, aligned: dict[str, np.ndarray]) -> None:
    print(f"Position {label} - truth {np.array2string(truth, precision=3, suppress_small=True)}")
    for name in CAMERAS:
        if name in aligned:
            al = np.array2string(aligned[name], precision=3, suppress_small=True)
        else:
            al = "cannot see block here"
        print(f"  {name:<15}{al}")
    print()


def run_calibration(
    scene_path: str | None = None,
    n_calib: int = 15,
    n_test: int = 6,
    pixel_sigma: float = 0.5,
    depth_sigma: float = 0.002,
    seed: int = 0,
    view: bool = False,
    prefer_gl: str = "egl",
    loop: bool = False,
    interval: float = 1.0,
) -> None:
    if view:
        configure_gl(prefer_gl)  # must happen before the viewer is created
    model = mujoco.MjModel.from_xml_path(str(scene_path or _DEFAULT_SCENE))
    data = mujoco.MjData(model)
    _setup_home_pose(model, data)
    rng = np.random.default_rng(seed)
    cams = SimulatedCameras(model, data, pixel_sigma=pixel_sigma, depth_sigma=depth_sigma, rng=rng)

    # --- 1-3: collect measurements and solve ---------------------------------
    pairs = _collect_calibration_pairs(cams, n_calib, rng)
    transforms = calibrate(pairs)
    ref_true = cams.cam_pose(REFERENCE_CAMERA)

    print(f"Block position, in metres, in the shared frame ({REFERENCE_CAMERA}):\n")

    if loop:
        # --- keep reporting fresh positions until stopped ---------------------
        # Stands in for "the block is moving in a live simulation": each tick
        # samples a new visible position instead of replaying a fixed batch.
        # A real publisher (step 4) ticks this same loop and sends each
        # position over the network instead of just printing it - so the
        # interval matters for real use: Victor's WorldModel marks a part
        # stale after 2s with no update, so this needs to run faster than that.
        print(f"Looping every {interval}s - press Ctrl+C to stop.\n")
        k = 0
        try:
            while True:
                k += 1
                world, obs = _collect_test_positions(cams, 1, rng)
                one_aligned = {name: transforms[name].apply(p) for name, p in obs[0].items()}
                one_truth = (world[0] - ref_true.t) @ ref_true.R
                _print_position(k, one_truth, one_aligned)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Stopped.")
        return

    # --- 4: check on block positions not used for fitting --------------------
    test_world, test_obs = _collect_test_positions(cams, n_test, rng)
    aligned = [{name: transforms[name].apply(p) for name, p in obs.items()} for obs in test_obs]
    truth = (test_world - ref_true.t) @ ref_true.R  # truth expressed in the shared frame

    for k in range(n_test):
        _print_position(k + 1, truth[k], aligned[k])

    if view:
        # Aligned estimates live in the headcam frame; the viewer draws in world coordinates.
        aligned_world = [{name: ref_true.apply(p) for name, p in a.items()} for a in aligned]
        _show_in_viewer(model, data, cams, test_world, aligned_world)
