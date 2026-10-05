#!/usr/bin/env python3
"""Two DUPLO bricks clicking together - a physics experiment on its own.

Nothing here uses the robot code. A tiny scene: a table, two DUPLO 2x4x2 bricks (the
zebra's), both upright and facing the same way. The LOWER brick stands on the table;
the UPPER one is held by a "ghost hand" (an invisible handle we move exactly, attached
to the brick through a slightly springy weld, like a gripper's give). The hand lowers
the upper brick onto the lower one and presses down; then it lifts, to see whether the
lower brick comes along (= they clicked).

The bricks have their real stud geometry here (measured from the LDraw mesh the scene
uses, stationlite/meshes/zebra_*_white.stl, in mm, origin = centre of the top face):
  - 2x4 studs, 16 mm apart (x -24/-8/8/24, y -8/8), 9.6 mm across, 4.4 mm tall
  - underneath: walls 1.6 mm thick, a deck 1.6 mm thick, and 3 tubes (x -16/0/16, y 0),
    12.8 mm across, from 1.6 mm above the bottom up to the deck
  - each stud sits between two tubes: by the drawing 0.11 mm clearance to each; the
    walls are 1.6 mm away. So the tubes do the gripping.
Real bricks have rounded stud tops and tube ends that guide a slightly-off brick in;
here they're a chamfer (CHAMFER) on the stud tops and the tube bottoms.

What we found (2026-10-05, pressed with ~28 N, then pulled apart):
  - The tubes must SQUEEZE the studs a little and GIVE a little, like plastic: overlap
    0.01 mm with 0.05 mm of give (the defaults, --fit/--flex). Then it clicks every time
    and holds ~1.6 N (about 5x the brick's weight) before coming apart.
  - Harder contacts or more overlap JAM (stuck 1.2 mm short even at 56 N - MuJoCo can't
    slide overlapping round shapes past each other); a gap or just touching holds
    nothing or only sometimes (just touching: 0 N in one of 12 cases).
  - How far off it may start and still click: up to 2 mm in ONE direction (along or
    across), 1 mm + 1 mm + 2 deg together, ~5 deg turned. 2 mm + 1 mm together lands
    tilted (7.5 deg); 3 mm or 8 deg lands on the studs.
    The robot places bricks 7-16 mm off today - for clicking it must get within ~1-2 mm.

usage: python3 experiments/brick_click.py [--dx MM] [--dy MM] [--yaw DEG] [--fit MM]
                                          [--png FILE]
  --dx/--dy/--yaw: how far the upper brick starts off from perfectly lined up
  --fit: how much the tubes squeeze the studs (mm of overlap; 0 = they just touch,
         -0.11 = the drawing's clearance; default 0.01)
  --flex: how much studs and tubes give, like flexing plastic (mm; default 0.05)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

MM = 0.001
H = 38.4 * MM  # brick body height (bottom to top face)
STUD_R, STUD_H = 4.8 * MM, 4.4 * MM
STUDS = [(x * MM, y * MM) for x in (-24, -8, 8, 24) for y in (-8, 8)]
WALL = 1.6 * MM
TUBE_R = 6.4 * MM
TUBES = [(x * MM, 0.0) for x in (-16, 0, 16)]
TUBE_BOTTOM = 1.6 * MM  # tubes end this far above the brick's bottom
CHAMFER = 0.8 * MM
MASS = 0.03  # kg per brick


def _frustum(name: str, r_bottom: float, r_top: float, z_bottom: float, z_top: float,
             r_mid: float | None = None, z_mid: float | None = None, n: int = 32) -> str:
    """A convex mesh: a cylinder with one chamfered end (rings at z_bottom, [z_mid], z_top)."""
    rings = [(r_bottom, z_bottom)] + ([(r_mid, z_mid)] if r_mid is not None else []) + [(r_top, z_top)]
    verts = []
    for r, z in rings:
        for k in range(n):
            a = 2 * np.pi * k / n
            verts += [r * np.cos(a), r * np.sin(a), z]
    return f'<mesh name="{name}" vertex="{" ".join(f"{v:.6f}" for v in verts)}"/>'


def _brick_geoms(prefix: str, margin: float = 0.0, soft: str = "") -> str:
    """One brick, origin at the centre of its bottom face: body walls/deck/tubes + studs."""
    m = MASS / 10  # mass spread over the parts (deck, 4 walls, 3 tubes, studs group)
    g = []
    # deck (top plate) and walls
    g.append(f'<geom type="box" size="{32*MM} {16*MM} {WALL/2}" pos="0 0 {H - WALL/2}" mass="{3*m}"/>')
    hw = (H - WALL) / 2
    g.append(f'<geom type="box" size="{32*MM} {WALL/2} {hw}" pos="0 {16*MM - WALL/2} {hw}" mass="{m}"/>')
    g.append(f'<geom type="box" size="{32*MM} {WALL/2} {hw}" pos="0 {-16*MM + WALL/2} {hw}" mass="{m}"/>')
    g.append(f'<geom type="box" size="{WALL/2} {16*MM - WALL} {hw}" pos="{32*MM - WALL/2} 0 {hw}" mass="{m}"/>')
    g.append(f'<geom type="box" size="{WALL/2} {16*MM - WALL} {hw}" pos="{-32*MM + WALL/2} 0 {hw}" mass="{m}"/>')
    # tubes (convex: solid; the hollow inside doesn't touch anything)
    for tx, ty in TUBES:
        g.append(f'<geom type="mesh" mesh="{prefix}_tube" pos="{tx} {ty} 0" mass="{m / 3}" margin="{margin}"{soft}/>')
    # studs on top
    for sx, sy in STUDS:
        g.append(f'<geom type="mesh" mesh="stud" pos="{sx} {sy} {H}" mass="{m / 8}" margin="{margin}"{soft}/>')
    return "\n        ".join(g)


def build(fit: float, start_xyz, start_yaw: float, margin: float = 0.0, flex: float = 0.0) -> mujoco.MjModel:
    """`margin`: studs and tubes start pushing on each other this far before they touch
    (MuJoCo's contact margin) - a squeeze without the shapes overlapping. Overlapping round
    shapes jam in MuJoCo (a 0.02 mm overlap stuck 1.2 mm short even at 56 N): its contact
    direction between overlapping convex meshes isn't the sideways one a sliding fit needs."""
    touch = np.hypot(8 * MM, 8 * MM) - STUD_R  # tube radius that just touches the studs (6.51 mm)
    r_tube = touch + fit
    meshes = [
        _frustum("stud", STUD_R, STUD_R - CHAMFER, 0.0, STUD_H, STUD_R, STUD_H - CHAMFER),
        _frustum("upper_tube", r_tube - CHAMFER, r_tube, TUBE_BOTTOM, H - WALL, r_tube, TUBE_BOTTOM + CHAMFER),
        _frustum("lower_tube", r_tube - CHAMFER, r_tube, TUBE_BOTTOM, H - WALL, r_tube, TUBE_BOTTOM + CHAMFER),
    ]
    # `flex` (m): studs and tubes give this much at full force, like plastic that flexes
    # (contact impedance ramps up over this depth) - 0 = as hard as everything else
    soft = f' solimp="0.5 0.99 {flex}" solref="0.005 1"' if flex > 0 else ""
    x, y, z = start_xyz
    q = [np.cos(start_yaw / 2), 0, 0, np.sin(start_yaw / 2)]
    xml = f"""
<mujoco model="brick_click">
  <option timestep="0.0002" noslip_iterations="10"/>
  <default>
    <geom condim="4" friction="0.4 0.01 0.001" solref="0.002 1" solimp="0.95 0.99 0.0002"
          rgba="0.95 0.95 0.95 1"/>
  </default>
  <asset>
    {chr(10).join(meshes)}
  </asset>
  <worldbody>
    <light pos="0.1 -0.2 0.4" dir="-0.3 0.6 -1"/>
    <geom name="table" type="plane" size="0.3 0.3 0.01" rgba="0.6 0.5 0.4 1"/>
    <body name="lower" pos="0 0 0">
      <freejoint/>
      {_brick_geoms("lower", margin, soft)}
    </body>
    <body name="upper" pos="{x} {y} {z}" quat="{q[0]} {q[1]} {q[2]} {q[3]}">
      <freejoint/>
      {_brick_geoms("upper", margin, soft)}
    </body>
    <body name="hand" mocap="true" pos="{x} {y} {z}" quat="{q[0]} {q[1]} {q[2]} {q[3]}">
      <geom type="sphere" size="0.003" contype="0" conaffinity="0" rgba="1 0 0 0.5"/>
    </body>
  </worldbody>
  <equality>
    <weld name="grip" body1="hand" body2="upper" solref="0.01 1"/>
  </equality>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def contact_force(model, data, a: int, b: int) -> float:
    """Total upward force from body a on body b (N) over their contacts."""
    f6 = np.zeros(6)
    total = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if {b1, b2} == {a, b}:
            mujoco.mj_contactForce(model, data, i, f6)
            normal = c.frame[:3]
            fz = f6[0] * normal[2]  # normal force along world z
            total += abs(fz)
    return total


def run(dx=0.0, dy=0.0, yaw_deg=0.0, fit=0.01, png=None, verbose=True, press_mm=5.0, margin=0.0,
        flex=0.05) -> dict:
    start = np.array([dx * MM, dy * MM, H + STUD_H + 3 * MM])  # upper's bottom 3 mm above the stud tops
    model = build(fit * MM, start, np.radians(yaw_deg), margin * MM, flex * MM)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lower, upper = model.body("lower").id, model.body("upper").id
    dt = model.opt.timestep
    seated_z = H  # upper's bottom on lower's top face

    def step_to(z_target, speed, settle=0.0):
        z0 = data.mocap_pos[0, 2]
        n = int(abs(z_target - z0) / speed / dt)
        for i in range(n):
            data.mocap_pos[0, 2] = z0 + (z_target - z0) * (i + 1) / n
            mujoco.mj_step(model, data)
        for _ in range(int(settle / dt)):
            mujoco.mj_step(model, data)

    for _ in range(int(0.2 / dt)):  # settle the lower brick on the table
        mujoco.mj_step(model, data)
    # 1. lower at 10 mm/s to press_mm below fully seated: the weld's give turns that into a push
    step_to(seated_z - press_mm * MM, 0.010, settle=0.3)
    press = contact_force(model, data, lower, upper)
    gap = (data.xpos[upper][2] - seated_z) / MM
    tilt = np.degrees(np.arccos(np.clip(data.xmat[upper].reshape(3, 3)[2, 2], -1, 1)))
    off = np.linalg.norm(data.xpos[upper][:2] - data.xpos[lower][:2]) / MM
    if png:
        _render(model, data, png)
    # 2. lift 2 cm at 10 mm/s: does the lower brick come along?
    lower_z0 = data.xpos[lower][2]
    step_to(data.mocap_pos[0, 2] + 20 * MM + press_mm * MM, 0.010, settle=0.3)
    lifted = (data.xpos[lower][2] - lower_z0) / MM
    # 3. pull test: hold the upper brick still and pull the lower one down with a force
    # growing 0 -> 30 N over 3 s; the force when they come apart (> 1 mm) is how firm the click is
    pull = 0.0
    if lifted > 15:
        n = int(3.0 / dt)
        for i in range(n):
            data.xfrc_applied[lower, 2] = -30.0 * i / n
            mujoco.mj_step(model, data)
            if (data.xpos[upper][2] - data.xpos[lower][2] - H) > 1 * MM:
                pull = 30.0 * i / n
                break
        else:
            pull = 30.0  # still together at 30 N
        data.xfrc_applied[lower, 2] = 0.0
    result = {"gap_mm": gap, "tilt_deg": tilt, "off_mm": off, "press_N": press, "lower_lifted_mm": lifted,
              "pull_N": pull,
              "seated": abs(gap) < 0.5 and tilt < 1.0, "clicked": lifted > 15}
    if verbose:
        print(f"start off by dx {dx} mm, dy {dy} mm, yaw {yaw_deg} deg, fit {fit} mm:")
        print(f"  pressed: upper's bottom {gap:+.2f} mm from fully seated, tilted {tilt:.1f} deg, "
              f"{off:.2f} mm off centre; bricks pushing on each other with {press:.1f} N")
        print(f"  lifted 2 cm: lower brick came up {lifted:.1f} mm -> "
              f"{'CLICKED (stays together)' if result['clicked'] else 'not attached'}")
        if result["clicked"]:
            print(f"  pull test: came apart at {pull:.1f} N" if pull < 30 else "  pull test: still together at 30 N")
    return result


def _render(model, data, path):
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, H]
    cam.distance, cam.azimuth, cam.elevation = 0.16, 120, -15
    renderer.update_scene(data, cam)
    from PIL import Image
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(renderer.render()).save(path)
    print(f"  picture: {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--fit", type=float, default=0.01)
    p.add_argument("--flex", type=float, default=0.05)
    p.add_argument("--margin", type=float, default=0.0,
                   help="studs and tubes start pushing this far (mm) before touching: a squeeze without overlap")
    p.add_argument("--press", type=float, default=5.0, help="how far below fully seated the hand goes (mm): deeper = harder push")
    p.add_argument("--png", default=None)
    a = p.parse_args()
    run(a.dx, a.dy, a.yaw, a.fit, a.png, press_mm=a.press, margin=a.margin, flex=a.flex)
