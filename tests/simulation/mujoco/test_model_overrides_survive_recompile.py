# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Runtime model edits must survive the next scene recompile.

``set_body_properties`` writes ``model.body_mass``/``body_inertia``,
``set_geom_properties`` writes ``model.geom_rgba``/``geom_friction``/``geom_size``,
and ``randomize`` writes all of those plus ``mat_rgba``. Every scene mutation
(``add_object``, ``remove_object``, ``add_camera``, ``add_robot``, static
``move_object``, ``attach_bodies``, ``actuate_robot``) recompiles from
``_backend_state["spec"]``, so anything written only to ``model.*`` was silently
reverted.

Two distinct defects were found here, and only one was what the ledger described:

1. ``set_geom_properties`` DID mirror onto the spec, but under the name the CALLER
   passed. ``add_object`` names geoms ``{object}_geom`` and the id resolution
   accepts the bare object name as an alias, so the mirror looked up
   ``spec.geom("cube")`` - which RETURNS None rather than raising, so the
   ``except (KeyError, ValueError)`` never fired and the miss was invisible.
   Measured: friction 2.0 -> 1.0 and the colour back to grey on the next
   ``add_object``. Now mirrored under the geom's real compiled name, and a spec
   miss warns instead of being swallowed.

2. ``randomize`` never touched the spec at all, so one ``add_object`` after a
   randomize reverted the WHOLE sample::

       post-DR              mass=0.1307  friction=1.2535  rgba=0.561
       post-DR + add_object mass=0.1000  friction=1.0000  rgba=0.500

   A rollout that adds a distractor mid-episode was then training on the
   un-randomised domain while reporting the randomized one.

``set_body_properties``' mass already survived (an earlier fix routes it to the
geom that carries it, since a write to ``MjsBody.mass`` is ignored by the compiler
without an ``explicitinertial`` block); the tests below pin that too so it cannot
regress.

Note for anyone extending these: resolve geoms by their COMPILED name
(``cube_geom``), not the object name. ``mj_name2id`` returns -1 for the object
name and ``model.geom_friction[-1]`` silently reads the LAST geom - which is the
newly-added distractor, making a passing fix look broken.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    engine = Simulation(tool_name="devx_model_overrides", mesh=False)
    engine.create_world()
    assert (
        engine.add_object("cube", shape="box", position=[0.2, 0.0, 0.1], size=[0.03, 0.03, 0.03], mass=0.1)["status"]
        == "success"
    )
    try:
        yield engine
    finally:
        engine.cleanup(policy_stop_timeout=0.5)


def _geom(sim, field: str, geom_name: str = "cube_geom", component: int = 0) -> float:
    """Read one component of a geom field by COMPILED name (never by index)."""
    mj = sim._mj
    model = sim._world._model
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
    assert gid >= 0, f"geom {geom_name!r} not in the compiled model"
    return float(getattr(model, field)[gid][component])


def _mass(sim, body_name: str = "cube") -> float:
    mj = sim._mj
    model = sim._world._model
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    assert bid >= 0, f"body {body_name!r} not in the compiled model"
    return float(model.body_mass[bid])


def _mutate(sim, kind: str) -> None:
    """Trigger one scene mutation, i.e. one recompile from the spec."""
    if kind == "add_object":
        assert (
            sim.add_object("distractor", shape="sphere", position=[-0.2, 0.0, 0.1], size=[0.02], mass=0.05)["status"]
            == "success"
        )
    elif kind == "remove_object":
        assert (
            sim.add_object("tmp", shape="sphere", position=[-0.3, 0.0, 0.1], size=[0.02], mass=0.05)["status"]
            == "success"
        )
        assert sim.remove_object("tmp")["status"] == "success"
    elif kind == "add_camera":
        assert sim.add_camera("extra", position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])["status"] == "success"
    else:  # pragma: no cover - guard against a typo in a parametrize list
        raise AssertionError(f"unknown mutation {kind!r}")


class TestExplicitGeomEditsSurvive:
    @pytest.mark.parametrize("mutation", ["add_object", "remove_object", "add_camera"])
    def test_friction_survives_a_scene_mutation(self, sim, mutation):
        """Regression: friction 2.0 came back as 1.0 on the next add_object."""
        assert sim.set_geom_properties("cube", friction=[2.0, 0.1, 0.01])["status"] == "success"
        assert _geom(sim, "geom_friction") == pytest.approx(2.0)

        _mutate(sim, mutation)

        assert _geom(sim, "geom_friction") == pytest.approx(2.0), f"reverted after {mutation}"

    @pytest.mark.parametrize("mutation", ["add_object", "remove_object", "add_camera"])
    def test_color_survives_a_scene_mutation(self, sim, mutation):
        assert sim.set_geom_properties("cube", color=[1.0, 0.0, 0.0, 1.0])["status"] == "success"
        assert _geom(sim, "geom_rgba") == pytest.approx(1.0)

        _mutate(sim, mutation)

        assert _geom(sim, "geom_rgba") == pytest.approx(1.0), f"reverted after {mutation}"

    def test_size_survives_a_scene_mutation(self):
        engine = Simulation(tool_name="devx_model_overrides_size", mesh=False)
        try:
            engine.create_world()
            assert (
                engine.add_object("cube", shape="box", position=[0.2, 0.0, 0.1], size=[0.03, 0.03, 0.03])["status"]
                == "success"
            )
            assert engine.set_geom_properties("cube", size=[0.06, 0.06, 0.06])["status"] == "success"
            before = _geom(engine, "geom_size")

            _mutate(engine, "add_object")

            assert _geom(engine, "geom_size") == pytest.approx(before)
        finally:
            engine.cleanup(policy_stop_timeout=0.5)

    def test_the_bare_object_name_alias_still_works(self, sim):
        """The caller may pass "cube"; the mirror must resolve "cube_geom"."""
        assert sim.set_geom_properties("cube", friction=[2.0, 0.1, 0.01])["status"] == "success"
        assert sim.set_geom_properties("cube_geom", color=[0.0, 1.0, 0.0, 1.0])["status"] == "success"

        _mutate(sim, "add_object")

        assert _geom(sim, "geom_friction") == pytest.approx(2.0)
        assert _geom(sim, "geom_rgba", component=1) == pytest.approx(1.0)


class TestExplicitBodyEditsSurvive:
    @pytest.mark.parametrize("mutation", ["add_object", "remove_object", "add_camera"])
    def test_mass_survives_a_scene_mutation(self, sim, mutation):
        assert sim.set_body_properties("cube", mass=5.0)["status"] == "success"
        assert _mass(sim) == pytest.approx(5.0)

        _mutate(sim, mutation)

        assert _mass(sim) == pytest.approx(5.0, rel=1e-3), f"reverted after {mutation}"

    def test_mass_and_geom_edits_survive_together(self, sim):
        """The combination in the ledger's evidence."""
        assert sim.set_body_properties("cube", mass=5.0)["status"] == "success"
        assert sim.set_geom_properties("cube", friction=[2.0, 0.1, 0.01])["status"] == "success"

        _mutate(sim, "add_object")

        assert _mass(sim) == pytest.approx(5.0, rel=1e-3)
        assert _geom(sim, "geom_friction") == pytest.approx(2.0)


class TestRandomizationSurvives:
    @pytest.mark.parametrize("mutation", ["add_object", "remove_object", "add_camera"])
    def test_the_whole_sample_survives_a_scene_mutation(self, sim, mutation):
        """Regression: one add_object reverted the entire randomize sample."""
        assert sim.randomize(randomize_colors=True, randomize_physics=True, seed=1)["status"] == "success"
        mass = _mass(sim)
        friction = _geom(sim, "geom_friction")
        rgba = _geom(sim, "geom_rgba")
        # The fixture must actually have moved off the defaults, or this proves
        # nothing.
        assert friction != pytest.approx(1.0), "randomize did not change friction"
        assert rgba != pytest.approx(0.5), "randomize did not change the colour"

        _mutate(sim, mutation)

        assert _mass(sim) == pytest.approx(mass, rel=1e-3), f"mass reverted after {mutation}"
        assert _geom(sim, "geom_friction") == pytest.approx(friction, rel=1e-3)
        assert _geom(sim, "geom_rgba") == pytest.approx(rgba, rel=1e-3)

    def test_colors_only_randomization_survives(self, sim):
        assert sim.randomize(randomize_colors=True, randomize_physics=False, seed=2)["status"] == "success"
        rgba = _geom(sim, "geom_rgba")

        _mutate(sim, "add_object")

        assert _geom(sim, "geom_rgba") == pytest.approx(rgba, rel=1e-3)

    def test_physics_only_randomization_survives(self, sim):
        assert sim.randomize(randomize_colors=False, randomize_physics=True, seed=3)["status"] == "success"
        mass = _mass(sim)
        friction = _geom(sim, "geom_friction")

        _mutate(sim, "add_object")

        assert _mass(sim) == pytest.approx(mass, rel=1e-3)
        assert _geom(sim, "geom_friction") == pytest.approx(friction, rel=1e-3)

    def test_a_later_randomize_still_resamples(self, sim):
        """Persistence must not freeze the domain: each call is a fresh sample."""
        assert sim.randomize(randomize_colors=True, randomize_physics=True, seed=1)["status"] == "success"
        first = (_mass(sim), _geom(sim, "geom_friction"))

        assert sim.randomize(randomize_colors=True, randomize_physics=True, seed=99)["status"] == "success"
        second = (_mass(sim), _geom(sim, "geom_friction"))

        assert second != pytest.approx(first), "the second randomize returned the same sample"

    def test_the_ground_plane_is_still_excluded_from_recolor(self, sim):
        """The existing exclusion must not be re-introduced through the mirror."""
        mj = sim._mj
        model = sim._world._model
        ground = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "ground")
        if ground < 0:
            pytest.skip("no named ground geom in this world")
        before = [float(v) for v in model.geom_rgba[ground]]

        assert sim.randomize(randomize_colors=True, randomize_physics=False, seed=4)["status"] == "success"
        _mutate(sim, "add_object")

        model = sim._world._model
        ground = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "ground")
        assert [float(v) for v in model.geom_rgba[ground]] == pytest.approx(before)


class TestSpecMissIsLoud:
    def test_a_name_the_spec_does_not_know_warns(self, sim, caplog):
        """spec.geom() returns None for an unknown name instead of raising.

        That is why the original miss was invisible: the except clause could
        never fire. A silently reverted property is the defect, so say so.
        """
        with caplog.at_level("WARNING"):
            sim._sync_spec_geom("no_such_geom", friction=[2.0, 0.1, 0.01])

        warnings = [record.getMessage() for record in caplog.records if "not in the live spec" in record.getMessage()]
        assert warnings, [record.getMessage() for record in caplog.records]
        assert "no_such_geom" in warnings[0]
        assert "friction" in warnings[0]
        assert warnings[0].isascii()

    def test_a_known_name_does_not_warn(self, sim, caplog):
        with caplog.at_level("WARNING"):
            sim._sync_spec_geom("cube_geom", friction=[2.0, 0.1, 0.01])

        assert not [r for r in caplog.records if "not in the live spec" in r.getMessage()]


class TestImmediateEffectIsUnchanged:
    def test_reset_and_step_still_preserve_edits(self, sim):
        """The pre-existing behaviour the ledger verified must not regress."""
        assert sim.set_body_properties("cube", mass=5.0)["status"] == "success"

        sim.step(n_steps=10)
        assert _mass(sim) == pytest.approx(5.0, rel=1e-3)

        assert sim.reset()["status"] == "success"
        assert _mass(sim) == pytest.approx(5.0, rel=1e-3)
