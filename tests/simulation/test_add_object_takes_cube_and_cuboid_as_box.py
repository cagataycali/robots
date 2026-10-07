"""``add_object(shape="cube" | "cuboid")`` builds a box on every backend.

The docs, the README quickstart and Isaac's own ``DynamicCuboid`` class all
call a box a cube, so both words are shape names, resolved by one table
(:data:`strands_robots.simulation.models.SHAPE_ALIASES`).
"""

import pytest

from strands_robots.simulation.mjlab.simulation import MjlabEngine
from strands_robots.simulation.models import canonical_shape

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.mark.parametrize("alias", ["cube", "cuboid"])
def test_mujoco_builds_a_box_for_cube_and_cuboid(alias):
    sim = Simulation(tool_name="test_cube_alias_sim", mesh=False)
    try:
        sim.create_world()
        result = sim.add_object("red_cube", shape=alias, size=[0.05, 0.05, 0.05], position=[0.0, 0.0, 0.1])
        assert result["status"] == "success", result
        assert sim._world.objects["red_cube"].shape == "box"
        model = sim._world._model
        geoms = [g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name == "red_cube"]
        assert geoms and {int(model.geom_type[g]) for g in geoms} == {int(mj.mjtGeom.mjGEOM_BOX)}
    finally:
        sim.cleanup()


@pytest.mark.parametrize(
    ("shape", "expected"),
    [("cube", "box"), ("cuboid", "box"), ("box", "box"), ("sphere", "sphere"), ("torus", "torus"), (3, 3)],
)
def test_canonical_shape_maps_only_the_aliases(shape, expected):
    assert canonical_shape(shape) == expected


@pytest.mark.parametrize(("shape", "hint"), [("sfere", " Did you mean 'sphere'?"), ("torus", "")])
def test_mjlab_names_the_closest_shape(shape, hint):
    result = MjlabEngine.__new__(MjlabEngine).add_object(name="obj", shape=shape)
    assert result["content"][0]["text"] == (
        f"shape must be one of ['box', 'capsule', 'cylinder', 'sphere'], got {shape!r}.{hint}"
    )
