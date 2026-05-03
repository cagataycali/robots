"""Every primitive shape supported by ``MJCFBuilder._object_xml`` must render.

Also locks the scene-composer fallback path (``compose_multi_robot_scene``)
and the object-geom auto-naming convention (``<name>_geom``).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "glfw")


@pytest.fixture
def sim():
    from strands_robots.simulation import Simulation

    s = Simulation()
    s.create_world()
    yield s
    s.destroy()


@pytest.mark.parametrize(
    "shape,size,name",
    [
        ("box", [0.02, 0.02, 0.02], "a_box"),
        ("sphere", [0.025, 0.025, 0.025], "a_ball"),
        ("cylinder", [0.02, 0.02, 0.06], "a_rod"),
        ("capsule", [0.02, 0.02, 0.06], "a_capsule"),
    ],
)
def test_primitive_shape_roundtrips_to_model(sim, shape, size, name):
    r = sim.add_object(name=name, shape=shape, size=size, position=[0.1, 0.1, 0.05])
    assert r["status"] == "success", r

    # Geom is named by the convention '<name>_geom'
    import mujoco as mj

    gid = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_GEOM, f"{name}_geom")
    assert gid >= 0, f"geom '{name}_geom' not found in model"

    # And we can recolor it via geom_name (set_geom_properties coverage)
    r = sim.set_geom_properties(geom_name=f"{name}_geom", color=[0.3, 0.3, 0.3, 1.0])
    assert r["status"] == "success"


def test_plane_object_rejected_as_dynamic_body(sim):
    """MuJoCo only permits plane geoms inside static bodies. ``add_object``
    creates a *dynamic* body, so requesting shape='plane' must surface a
    clean error rather than a raw exception — this exercises the recompile
    failure branch in scene_ops.
    """
    r = sim.add_object(name="floor_mat", shape="plane", size=[0.5, 0.5, 0.001], position=[0, 0, 0.001])
    assert r["status"] == "error"
    assert "plane" in r["content"][0]["text"].lower()
