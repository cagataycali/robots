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

Part of a ``geom_size`` row can be beyond a setter's reach: a geom declared with
``<fromto>`` has the compiler fix its extent along that axis, and a box's or
ellipsoid's cross-section as well, so a value written into the spec's ``size``
row for one of those components never survives a compile. Such a change is
refused rather than reported, and the tests below pin both halves - the refusal,
and that the components the ``fromto`` leaves alone still apply durably.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.simulation.mujoco.scene_ops import (
    fromto_fixed_size_components,
    persist_body_mass,
    persist_geom_properties,
    persist_world_option,
)

mujoco = pytest.importorskip("mujoco")

from strands_robots import Simulation  # noqa: E402
from strands_robots.simulation.mujoco.physics import _GEOM_SIZE_LAYOUTS  # noqa: E402

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


# Every geom type ``fromto`` can be used with, each declaring its axis extent by
# endpoints, plus a plain capsule declaring the same extent through ``size``.
# Densities make every body derive its inertial row from geometry, so a resize
# that slipped through would also leave mass and inertia describing another shape.
FROMTO_SCENE = """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    <body name="cap_body" pos="0 0 0.6">
      <freejoint/>
      <geom name="cap" type="capsule" fromto="0 0 -0.15  0 0 0.15" size="0.04" density="800"/>
    </body>
    <body name="cyl_body" pos="0.4 0 0.6">
      <freejoint/>
      <geom name="cyl" type="cylinder" fromto="0 0 -0.2  0 0 0.2" size="0.05" density="800"/>
    </body>
    <body name="box_body" pos="0.8 0 0.6">
      <freejoint/>
      <geom name="bx" type="box" fromto="0 0 -0.1  0 0 0.1" size="0.03 0.03" density="800"/>
    </body>
    <body name="ell_body" pos="1.2 0 0.6">
      <freejoint/>
      <geom name="ell" type="ellipsoid" fromto="0 0 -0.07  0 0 0.07" size="0.02 0.02" density="800"/>
    </body>
    <body name="plain_body" pos="1.6 0 0.6">
      <freejoint/>
      <geom name="plain" type="capsule" size="0.04 0.15" density="800"/>
    </body>
  </worldbody>
</mujoco>"""

# One case per type ``fromto`` governs: the geom, its type, the fixed component a
# resize would violate (index, name, the value the compiler keeps producing), a
# size that changes it, and a size that changes only what the fromto leaves alone.
# The axis extent is component 2 of a capsule / cylinder and component 3 of a box /
# ellipsoid, so the refused component differs by type.
FROMTO_CASES = [
    ("cap", "capsule", 1, "half-length", 0.15, [0.04, 0.30], [0.08, 0.15]),
    ("cyl", "cylinder", 1, "half-length", 0.20, [0.05, 0.44], [0.09, 0.20]),
    ("bx", "box", 2, "z half-extent", 0.10, [0.03, 0.03, 0.22], [0.06, 0.06, 0.10]),
    ("ell", "ellipsoid", 2, "z semi-axis", 0.07, [0.02, 0.02, 0.19], [0.05, 0.05, 0.07]),
]
FROMTO_IDS = [case[1] for case in FROMTO_CASES]

# A box's / ellipsoid's cross-section is made square from the first component, so
# its second one is fixed to whatever the caller passes first rather than to the
# value the geom compiles to today: (geom, type, name, size, expected).
FROMTO_SQUARE_CASES = [
    ("bx", "box", "y half-extent", [0.06, 0.03, 0.10], 0.06),
    ("ell", "ellipsoid", "y semi-axis", [0.05, 0.04, 0.07], 0.05),
]
FROMTO_SQUARE_IDS = [case[1] for case in FROMTO_SQUARE_CASES]


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


@pytest.fixture
def fromto_sim():
    """A world whose geoms declare their axis extent with ``fromto``."""
    engine = Simulation(tool_name="recompile_fromto", backend="mujoco", mesh=False)
    engine.create_world()
    assert engine.replace_scene_mjcf(FROMTO_SCENE)["status"] == "success"
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


class TestFromtoFixedSizeIsRefused:
    """A ``geom_size`` component a ``<fromto>`` fixes is refused, not reported."""

    PARAMS = ("geom", "geom_type", "index", "component", "expected", "changed", "others")

    @pytest.mark.parametrize(PARAMS, FROMTO_CASES, ids=FROMTO_IDS)
    def test_changing_a_fixed_component_writes_nothing(
        self, fromto_sim, geom, geom_type, index, component, expected, changed, others
    ):
        """Refused before either representation is touched, for every governed type."""
        model = fromto_sim._world._model
        geom_id = _geom(model, geom)
        body_id = int(model.geom_bodyid[geom_id])
        spec_geom = fromto_sim._world._backend_state["spec"].geoms[geom_id]
        before_model = model.geom_size[geom_id].copy()
        before_spec = np.array(spec_geom.size).copy()
        before_mass = float(model.body_mass[body_id])
        before_inertia = model.body_inertia[body_id].copy()

        result = fromto_sim.set_geom_properties(geom_name=geom, size=changed)

        assert result["status"] == "error", result
        assert np.allclose(model.geom_size[geom_id], before_model)
        assert np.allclose(np.array(spec_geom.size), before_spec)
        assert float(model.body_mass[body_id]) == pytest.approx(before_mass)
        assert np.allclose(model.body_inertia[body_id], before_inertia)

    @pytest.mark.parametrize(PARAMS, FROMTO_CASES, ids=FROMTO_IDS)
    def test_the_refusal_names_the_component_and_a_remedy(
        self, fromto_sim, geom, geom_type, index, component, expected, changed, others
    ):
        """The message names fromto, the component it fixes, and what to do instead."""
        text = fromto_sim.set_geom_properties(geom_name=geom, size=changed)["content"][0]["text"]
        assert text.startswith("set_geom_properties: ")
        assert "<fromto>" in text
        assert f"{component} (size component {index + 1} of a {geom_type})" in text
        assert f"restores {expected}" in text
        assert "edit the fromto to resize along its axis" in text

    @pytest.mark.parametrize(PARAMS, FROMTO_CASES, ids=FROMTO_IDS)
    def test_changing_only_what_the_fromto_leaves_alone_survives_a_recompile(
        self, fromto_sim, geom, geom_type, index, component, expected, changed, others
    ):
        """The fixed components are out of reach; the rest is still recorded durably.

        This is what refusing every resize of a ``fromto`` geom would have cost:
        thickening such a capsule is honored today, and the inertial row the setter
        reports is the one the next recompile reproduces.
        """
        model = fromto_sim._world._model
        geom_id = _geom(model, geom)
        body_id = int(model.geom_bodyid[geom_id])

        assert fromto_sim.set_geom_properties(geom_name=geom, size=others)["status"] == "success"
        reported_size = model.geom_size[geom_id].copy()
        reported_mass = float(model.body_mass[body_id])
        reported_inertia = model.body_inertia[body_id].copy()
        assert reported_size[index] == pytest.approx(expected)

        recompiled = _recompile(fromto_sim, name=f"trigger_{geom}")

        assert np.allclose(recompiled.geom_size[geom_id], reported_size)
        assert float(recompiled.body_mass[body_id]) == pytest.approx(reported_mass)
        assert np.allclose(recompiled.body_inertia[body_id], reported_inertia)

    @pytest.mark.parametrize(
        ("geom", "geom_type", "component", "size", "expected"),
        FROMTO_SQUARE_CASES,
        ids=FROMTO_SQUARE_IDS,
    )
    def test_a_square_cross_section_is_fixed_to_the_first_component(
        self, fromto_sim, geom, geom_type, component, size, expected
    ):
        """A fromto box's second component follows its first, so passing today's fails.

        The compiler copies component 1 over component 2, so what that component
        must equal is the caller's own first value - not the value the geom
        currently compiles to.
        """
        model = fromto_sim._world._model
        before = model.geom_size[_geom(model, geom)].copy()

        result = fromto_sim.set_geom_properties(geom_name=geom, size=size)

        assert result["status"] == "error", result
        assert component in result["content"][0]["text"]
        assert f"restores {expected}" in result["content"][0]["text"]
        assert np.allclose(model.geom_size[_geom(model, geom)], before)

    def test_a_geom_declaring_its_size_directly_is_unaffected(self, fromto_sim):
        """The same type without a fromto resizes every component as before."""
        model = fromto_sim._world._model
        geom_id = _geom(model, "plain")

        assert fromto_sim.set_geom_properties(geom_name="plain", size=[0.06, 0.33])["status"] == "success"
        recompiled = _recompile(fromto_sim, name="trigger_plain")

        assert np.allclose(recompiled.geom_size[geom_id][:2], [0.06, 0.33])

    def test_the_helper_reports_only_the_components_a_fromto_fixes(self, fromto_sim):
        """Empty for anything a fromto does not govern, so nothing is over-refused."""
        world = fromto_sim._world
        model = world._model

        assert fromto_fixed_size_components(world, _geom(model, "cap")) == {1: ("half-length", None)}
        assert fromto_fixed_size_components(world, _geom(model, "cyl")) == {1: ("half-length", None)}
        assert fromto_fixed_size_components(world, _geom(model, "bx")) == {
            1: ("y half-extent", 0),
            2: ("z half-extent", None),
        }
        assert fromto_fixed_size_components(world, _geom(model, "ell")) == {
            1: ("y semi-axis", 0),
            2: ("z semi-axis", None),
        }
        # declares its size directly, a type fromto cannot be used with, and an id
        # the spec cannot resolve
        assert fromto_fixed_size_components(world, _geom(model, "plain")) == {}
        assert fromto_fixed_size_components(world, _geom(model, "floor")) == {}
        assert fromto_fixed_size_components(world, 10_000) == {}

        world._backend_state.pop("spec")
        assert fromto_fixed_size_components(world, _geom(model, "cap")) == {}

    def test_every_fixed_component_is_within_the_accepted_size_length(self, fromto_sim):
        """The refusal indexes the caller's ``size`` on this, so nothing guards it.

        ``set_geom_properties`` requires a size of the geom type's exact component
        count and then indexes that vector at every component the helper reports -
        including the one a square cross-section copies from. Both indices have to
        fall inside the accepted count for each type a ``fromto`` governs, which is
        a property of the two tables together rather than of either alone, and the
        reason the refusal needs no bounds check to stand in for it.
        """
        world = fromto_sim._world
        model = world._model

        for geom_name, gtype, *_rest in FROMTO_CASES:
            fixed = fromto_fixed_size_components(world, _geom(model, geom_name))
            assert fixed, f"{gtype} declares a fromto, so the helper must report what it fixes"
            assert gtype in _GEOM_SIZE_LAYOUTS, f"{gtype} accepts no size, so it cannot be resized at all"
            accepted = _GEOM_SIZE_LAYOUTS[gtype][0]

            for index, (_component, follows) in fixed.items():
                assert 0 <= index < accepted, f"{gtype} fixes component {index} of an accepted {accepted}"
                if follows is None:
                    continue
                assert 0 <= follows < accepted, f"{gtype} copies component {follows} of an accepted {accepted}"
                # The remedy is "pass the value the compiler produces", so the
                # component a fixed one copies must itself be settable - if it
                # were fixed too, no size could satisfy both.
                assert follows not in fixed, f"{gtype} component {index} copies a component that is fixed too"


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
