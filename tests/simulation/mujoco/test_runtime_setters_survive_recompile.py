"""A runtime property setter's value survives the next scene recompile.

``world._model`` is derived state: every scene mutation (``add_object``,
``add_camera``, ``add_robot``, ...) recompiles the scene spec over it. A setter
that writes only the model therefore reports a value that the next unrelated
mutation silently discards, and callers cannot predict when that happens because
none of those methods documents a recompile.

These tests pin the contract for every runtime property the MuJoCo backend lets a
caller set - a body's mass, a geom's colour / friction / size, and the world's
gravity / timestep - and check the durable value equals the one the setter
reported, so the fast path and the recompile cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.simulation.mujoco.scene_ops import (
    persist_body_mass,
    persist_geom_properties,
    persist_world_option,
)

mujoco = pytest.importorskip("mujoco")

from strands_robots import Simulation  # noqa: E402

# Two bodies covering both ways MuJoCo derives an inertial: "explicit" declares
# its own <inertial> (explicitinertial, as every menagerie robot link does),
# while "derived" has mass and inertia integrated from a geom density. The
# explicit body's geom is deliberately unnamed - most geoms in a robot scene are,
# so the id path has to work without a name.
SCENE = """<mujoco>
  <worldbody>
    <body name="explicit" pos="0 0 1">
      <freejoint/>
      <inertial pos="0 0 0" mass="2.0" diaginertia="0.01 0.02 0.03"/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="derived" pos="0.3 0 1">
      <freejoint/>
      <geom name="derived_geom" type="box" size="0.05 0.05 0.05" density="500"/>
    </body>
  </worldbody>
</mujoco>"""


@pytest.fixture
def sim():
    """A world holding one crate, torn down after the test."""
    engine = Simulation(tool_name="recompile_test", backend="mujoco", mesh=False)
    engine.create_world()
    assert (
        engine.add_object(name="crate", shape="box", size=[0.1, 0.1, 0.1], position=[0, 0, 0.5], mass=0.1)["status"]
        == "success"
    )
    yield engine
    engine.cleanup()


@pytest.fixture
def dual_sim():
    """A world holding an explicit-inertial body and a density-derived body."""
    engine = Simulation(tool_name="recompile_dual", backend="mujoco", mesh=False)
    engine.create_world()
    assert engine.replace_scene_mjcf(SCENE)["status"] == "success"
    yield engine
    engine.cleanup()


def _recompile(engine, name="trigger"):
    """Trigger a recompile the way a caller would: add an unrelated object."""
    result = engine.add_object(name=name, shape="sphere", size=[0.03], position=[1.0, 0, 0.5])
    assert result["status"] == "success", result
    return engine._world._model


def _body(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _geom(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


class TestGeomPropertiesSurviveRecompile:
    """colour / friction / size outlive an unrelated scene mutation."""

    def test_color_survives(self, sim):
        assert sim.set_geom_properties(geom_name="crate", color=[0.0, 1.0, 0.0, 1.0])["status"] == "success"
        model = _recompile(sim)
        assert np.allclose(model.geom_rgba[_geom(model, "crate_geom")], [0.0, 1.0, 0.0, 1.0])

    def test_friction_survives(self, sim):
        assert sim.set_geom_properties(geom_name="crate", friction=[0.2, 0.1, 0.0005])["status"] == "success"
        model = _recompile(sim)
        assert np.allclose(model.geom_friction[_geom(model, "crate_geom")], [0.2, 0.1, 0.0005])

    def test_size_survives(self, sim):
        assert sim.set_geom_properties(geom_name="crate", size=[0.02, 0.03, 0.04])["status"] == "success"
        model = _recompile(sim)
        assert np.allclose(model.geom_size[_geom(model, "crate_geom")], [0.02, 0.03, 0.04])

    def test_unnamed_geom_addressed_by_id_survives(self, dual_sim):
        """Most geoms in a robot scene carry no name, so the id path must persist too."""
        model = dual_sim._world._model
        geom_id = int(model.body_geomadr[_body(model, "explicit")])
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) is None
        assert dual_sim.set_geom_properties(geom_id=geom_id, color=[1.0, 0.0, 0.0, 1.0])["status"] == "success"
        model = _recompile(dual_sim)
        assert np.allclose(model.geom_rgba[geom_id], [1.0, 0.0, 0.0, 1.0])


class TestBodyMassSurvivesRecompile:
    """A body's mass outlives an unrelated scene mutation, however it is derived."""

    def test_geom_derived_mass_survives_and_matches_the_reported_inertial(self, sim):
        """The durable inertial equals the one the setter already reported."""
        assert sim.set_body_properties(body_name="crate", mass=5.0)["status"] == "success"
        model = sim._world._model
        body_id = _body(model, "crate")
        reported_mass = float(model.body_mass[body_id])
        reported_inertia = model.body_inertia[body_id].copy()

        model = _recompile(sim)
        body_id = _body(model, "crate")
        assert reported_mass == pytest.approx(5.0)
        assert float(model.body_mass[body_id]) == pytest.approx(reported_mass)
        assert np.allclose(model.body_inertia[body_id], reported_inertia)

    def test_explicit_inertial_mass_and_inertia_survive(self, dual_sim):
        """A body declaring its own <inertial> keeps both mass and inertia."""
        assert dual_sim.set_body_properties(body_name="explicit", mass=4.0)["status"] == "success"
        model = _recompile(dual_sim)
        body_id = _body(model, "explicit")
        assert float(model.body_mass[body_id]) == pytest.approx(4.0)
        # mass doubled at fixed geometry, so the inertia doubles with it
        assert np.allclose(model.body_inertia[body_id], [0.02, 0.04, 0.06])

    def test_density_derived_mass_survives(self, dual_sim):
        """A geom stating a density (not a mass) has that density scaled instead."""
        assert dual_sim.set_body_properties(body_name="derived", mass=0.25)["status"] == "success"
        model = _recompile(dual_sim)
        assert float(model.body_mass[_body(model, "derived")]) == pytest.approx(0.25)

    def test_a_body_with_no_mass_is_refused(self, sim):
        """The world body has no inertial and no geom, so a mass cannot be scaled onto it."""
        result = sim.set_body_properties(body_name="world", mass=5.0)
        assert result["status"] == "error"
        assert "no mass of its own" in result["content"][0]["text"]


class TestWorldOptionsSurviveRecompile:
    """gravity / timestep outlive an unrelated scene mutation."""

    def test_gravity_survives(self, sim):
        assert sim.set_gravity(gravity=[0.0, 0.0, -1.62])["status"] == "success"
        model = _recompile(sim)
        assert np.allclose(model.opt.gravity, [0.0, 0.0, -1.62])

    def test_timestep_survives(self, sim):
        assert sim.set_timestep(timestep=0.001)["status"] == "success"
        model = _recompile(sim)
        assert float(model.opt.timestep) == pytest.approx(0.001)


class TestRefusedWhenTheChangeCannotBeRecorded:
    """A change that cannot be made durable is refused, not reported as applied."""

    def test_setters_refuse_a_world_with_no_spec(self, sim):
        """Without a spec there is nowhere to record the change, so nothing is written."""
        model = sim._world._model
        geom_id = _geom(model, "crate_geom")
        before_rgba = model.geom_rgba[geom_id].copy()
        before_mass = float(model.body_mass[_body(model, "crate")])
        before_timestep = float(model.opt.timestep)
        sim._world._backend_state.pop("spec")

        for result in (
            sim.set_geom_properties(geom_name="crate", color=[0.0, 1.0, 0.0, 1.0]),
            sim.set_body_properties(body_name="crate", mass=5.0),
            sim.set_timestep(timestep=0.001),
            sim.set_gravity(gravity=[0.0, 0.0, -1.62]),
        ):
            assert result["status"] == "error", result
            assert "no live spec" in result["content"][0]["text"]

        # refused before either representation was touched
        assert np.allclose(model.geom_rgba[geom_id], before_rgba)
        assert float(model.body_mass[_body(model, "crate")]) == pytest.approx(before_mass)
        assert float(model.opt.timestep) == pytest.approx(before_timestep)

    def test_an_id_outside_the_spec_is_reported(self, sim):
        """An id the spec cannot resolve is named rather than written somewhere else."""
        reason = persist_geom_properties(sim._world, 10_000, color=[1.0, 0.0, 0.0, 1.0])
        assert reason is not None
        assert "outside the scene spec" in reason
        assert persist_body_mass(sim._world, -1, mass_ratio=2.0) is not None

    def test_persisting_nothing_is_a_no_op(self, sim):
        """Called with no property, the helpers report success and change nothing."""
        assert persist_geom_properties(sim._world, _geom(sim._world._model, "crate_geom")) is None
        assert persist_world_option(sim._world) is None
