"""Integration tests for ``create_world(terrain="rough")`` on the MuJoCo backend.

A flat ground plane measures a locomotion policy's command tracking but never
its robustness to terrain. ``create_world(terrain="rough")`` lays down a
deterministic rough-ground heightfield (:mod:`strands_robots.simulation.terrain`)
instead of the flat plane so a floating-base robot is spawned and evaluated on
non-flat ground - the ground-generation primitive a terrain curriculum builds
on. These tests build a real world and step real physics (GL-free - no
rendering) to verify the heightfield ground is laid down, actually collides
(an object rests higher on a bump than on flat ground), the flat default is
unchanged, an unknown kind is rejected, ``ground_plane=False`` stays the master
floor switch, an attached robot's own ground plane is stripped over terrain,
and the agent-dispatch router accepts the ``terrain`` kwarg.
"""

from __future__ import annotations

import tempfile

import mujoco

from strands_robots.simulation import terrain
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_HFIELD = int(mujoco.mjtGeom.mjGEOM_HFIELD)


def _ground_geom_type(sim: MuJoCoSimEngine) -> int:
    assert sim._world is not None
    m = sim._world._model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    return -1 if gid < 0 else int(m.geom_type[gid])


def _interior_peak_xy() -> tuple[float, float]:
    """World (x, y) of the highest heightfield cell well inside the field rim."""
    n = terrain.TERRAIN_RESOLUTION
    h = terrain.generate_heightfield("rough", resolution=n, seed=terrain.TERRAIN_SEED)
    best_v, best_i, best_j = -1.0, 0, 0
    for i in range(6, n - 6):
        for j in range(6, n - 6):
            v = h[i * n + j]
            if v > best_v:
                best_v, best_i, best_j = v, i, j
    # MuJoCo hfield: row-major, row 0 -> min y, col 0 -> min x, spanning +/-radius.
    y = -terrain.TERRAIN_RADIUS + (best_i / (n - 1)) * 2 * terrain.TERRAIN_RADIUS
    x = -terrain.TERRAIN_RADIUS + (best_j / (n - 1)) * 2 * terrain.TERRAIN_RADIUS
    return x, y


def _box_rest_z(terrain_kind: str | None, x: float, y: float) -> float:
    sim = MuJoCoSimEngine()
    try:
        r = sim.create_world(terrain=terrain_kind)
        assert r["status"] == "success", r
        sim.add_object("blk", shape="box", size=[0.1, 0.1, 0.04], position=[x, y, 0.5], color=[1, 0, 0, 1])
        assert sim._world is not None
        m, d = sim._world._model, sim._world._data
        for _ in range(2500):
            mujoco.mj_step(m, d)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "blk")
        return float(d.xpos[bid][2])
    finally:
        sim.destroy()


def test_terrain_builds_a_heightfield_ground() -> None:
    sim = MuJoCoSimEngine()
    try:
        assert sim.create_world(terrain="rough")["status"] == "success"
        assert sim._world is not None
        m = sim._world._model
        assert _ground_geom_type(sim) == _HFIELD
        assert int(m.nhfield) == 1
        # the compiled heightfield is genuinely non-flat
        assert float(m.hfield_data.max()) - float(m.hfield_data.min()) > 0.5
    finally:
        sim.destroy()


def test_flat_ground_default_is_a_plane() -> None:
    sim = MuJoCoSimEngine()
    try:
        assert sim.create_world()["status"] == "success"
        assert _ground_geom_type(sim) == _PLANE
        assert sim._world is not None
        assert int(sim._world._model.nhfield) == 0
    finally:
        sim.destroy()


def test_terrain_collides_and_lifts_object_above_flat() -> None:
    x, y = _interior_peak_xy()
    z_terrain = _box_rest_z("rough", x, y)
    z_flat = _box_rest_z(None, x, y)
    # The object rests on the heightfield bump, meaningfully higher than on the
    # flat plane at the identical (x, y) -> the terrain both renders AND collides.
    assert z_terrain > z_flat + 0.03, (z_terrain, z_flat)


def test_unknown_terrain_is_rejected() -> None:
    sim = MuJoCoSimEngine()
    try:
        r = sim.create_world(terrain="bogus")
        assert r["status"] == "error"
        assert "rough" in r["content"][0]["text"]
        # no world was left half-built
        assert sim._world is None or sim._world._model is None
    finally:
        if sim._world is not None:
            sim.destroy()


def test_ground_plane_false_is_the_master_switch() -> None:
    sim = MuJoCoSimEngine()
    try:
        assert sim.create_world(terrain="rough", ground_plane=False)["status"] == "success"
        assert _ground_geom_type(sim) == -1  # no ground geom at all
        assert sim._world is not None
        assert int(sim._world._model.nhfield) == 0
    finally:
        sim.destroy()


_PLANEBOT_MJCF = """<mujoco model="planebot">
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="link" pos="0 0 0.4">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="link_g" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_attached_robot_ground_plane_is_stripped_over_terrain() -> None:
    sim = MuJoCoSimEngine()
    try:
        assert sim.create_world(terrain="rough")["status"] == "success"
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
            fh.write(_PLANEBOT_MJCF)
            path = fh.name
        assert sim.add_robot("planebot", urdf_path=path)["status"] == "success"
        assert sim._world is not None
        m = sim._world._model
        plane_geoms = [g for g in range(m.ngeom) if int(m.geom_type[g]) == _PLANE]
        hfield_geoms = [g for g in range(m.ngeom) if int(m.geom_type[g]) == _HFIELD]
        # The world owns the (heightfield) ground, so the robot's own z=0 plane
        # is stripped -> exactly one hfield ground, no leftover flat plane
        # coplanar with (and hiding) the terrain.
        assert len(hfield_geoms) == 1, hfield_geoms
        assert len(plane_geoms) == 0, plane_geoms
    finally:
        sim.destroy()


def test_router_dispatch_accepts_terrain_kwarg() -> None:
    sim = MuJoCoSimEngine()
    try:
        r = sim(action="create_world", terrain="rough")
        assert r["status"] == "success", r
        assert _ground_geom_type(sim) == _HFIELD
    finally:
        sim.destroy()


def test_router_dispatch_rejects_unknown_terrain() -> None:
    sim = MuJoCoSimEngine()
    try:
        r = sim(action="create_world", terrain="stairs")
        assert r["status"] == "error"
        assert "rough" in r["content"][0]["text"]
    finally:
        if sim._world is not None:
            sim.destroy()
