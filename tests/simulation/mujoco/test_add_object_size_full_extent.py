"""Behavioral tests for ``add_object`` size semantics on the MuJoCo backend.

These pin the documented contract: ``size`` is the **full extent in meters**
along each axis, which MuJoCo stores internally as half-extents. They also
cover the rejection of degenerate (non-positive) extents.
"""

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_size_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    yield s
    s.cleanup()


def _geom_size(world, geom_name):
    model = world._model
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
    assert gid >= 0, f"geom {geom_name!r} not found in compiled model"
    return list(model.geom_size[gid])


def test_box_full_extent_compiles_to_half(sim):
    """A 5 cm box (full extent) compiles to MuJoCo half-extents of 2.5 cm."""
    full = [0.06, 0.04, 0.02]
    result = sim.add_object("cube", shape="box", size=full)
    assert result["status"] == "success"

    half = _geom_size(sim._world, "cube_geom")
    assert half[0] == pytest.approx(full[0] / 2)
    assert half[1] == pytest.approx(full[1] / 2)
    assert half[2] == pytest.approx(full[2] / 2)


def test_sphere_size_is_diameter(sim):
    """``size[0]`` is the sphere diameter; the compiled radius is half of it."""
    result = sim.add_object("ball", shape="sphere", size=[0.1, 0.1, 0.1])
    assert result["status"] == "success"
    radius = _geom_size(sim._world, "ball_geom")[0]
    assert radius == pytest.approx(0.05)


def test_cylinder_diameter_and_full_height(sim):
    """``size[0]`` is the diameter and ``size[2]`` is the full height; both halve."""
    result = sim.add_object("can", shape="cylinder", size=[0.08, 0.0, 0.2])
    assert result["status"] == "success"
    out = _geom_size(sim._world, "can_geom")
    assert out[0] == pytest.approx(0.04)  # radius = diameter / 2
    assert out[1] == pytest.approx(0.1)  # half-height = full height / 2


def test_degenerate_zero_extent_rejected(sim):
    """A zero extent is rejected and the object is not added."""
    result = sim.add_object("flat", shape="box", size=[0.05, 0.05, 0.0])
    assert result["status"] == "error"
    assert "extent must be > 0" in result["content"][0]["text"]
    assert sim.list_objects()["content"][0]["text"] == "No objects."


def test_degenerate_negative_extent_rejected(sim):
    """A negative extent is rejected before any scene mutation."""
    result = sim.add_object("bad", shape="box", size=[-0.05, 0.05, 0.05])
    assert result["status"] == "error"
    assert "extent must be > 0" in result["content"][0]["text"]


def test_cylinder_ignores_unused_middle_component(sim):
    """``size[1]`` is unused for a cylinder, so a zero there is accepted."""
    result = sim.add_object("rod", shape="cylinder", size=[0.06, 0.0, 0.15])
    assert result["status"] == "success"
