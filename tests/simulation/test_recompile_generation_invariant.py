"""Regression tests: EVERY model swap bumps ``_recompile_generation``.

``_recompile_generation`` is the only discriminator consumers have for "the
arrays I cached no longer describe this model". Two read it:

* ``load_state`` - to reject a checkpoint whose nq/nv/na/nu happen to be
  unchanged by a same-shape recompile.
* ``randomization._dr_baseline`` - to re-snapshot the un-randomised ``model.*``
  arrays that ``randomize`` scales from.

Only ``_recompile_preserving_state`` bumped it. Four other paths swapped
``world._model`` silently - ``eject_robot_from_scene`` (the full rebuild that any
``remove_robot`` triggers), ``replace_scene_mjcf``, ``patch_scene_mjcf`` and
``remove_camera`` - so the baseline kept indexing arrays sized for the OLD model.

Measured, a 0.05 kg body after ``remove_robot`` + six ``randomize`` calls:

    light mass         = 0.6441 kg     legal window is [0.025, 0.100]
    baseline body_mass = len 24        live nbody = 13

Shrinking the model corrupted the sample silently; GROWING it raised
``IndexError: index 2 is out of bounds for axis 0 with size 2`` straight out of
``randomize``, escaping the tool's error contract entirely.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# randomize()'s default mass_range; a base mass m is legal in [0.5 m, 2.0 m]
# scaled against the ORIGINAL model, i.e. never further than a factor of two.
_MASS_RANGE = (0.5, 2.0)


def _randomize(sim) -> dict:
    return sim.randomize(randomize_physics=True, randomize_colors=False, randomize_lighting=False)


def _mass_of(sim, body: str) -> float:
    model = sim.mj_model
    return float(model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])


def _generation(sim) -> int:
    return int(sim._world._recompile_generation)


@pytest.fixture
def sim():
    s = Simulation(tool_name="recompile_generation_invariant", mesh=False)
    s.create_world()
    yield s
    s.destroy()


def _add(sim, name: str, x: float, mass: float = 0.05):
    return sim.add_object(name=name, shape="box", size=[0.04] * 3, position=[x, 0, 0.3], mass=mass)


def test_full_rebuild_bumps_the_generation(sim) -> None:
    """``remove_robot`` runs SpecBuilder.build, not spec.recompile."""
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    before = _generation(sim)
    assert sim.remove_robot(name="b")["status"] == "success"
    assert _generation(sim) > before, "eject_robot_from_scene swapped the model silently"


def test_replace_scene_mjcf_bumps_the_generation(sim) -> None:
    before = _generation(sim)
    xml = '<mujoco><worldbody><light pos="0 0 3"/><geom name="ground" type="plane" size="5 5 .1"/></worldbody></mujoco>'
    assert sim.replace_scene_mjcf(xml=xml)["status"] == "success"
    assert _generation(sim) > before


def test_patch_scene_mjcf_bumps_the_generation(sim) -> None:
    _add(sim, "cube", 0.3)
    before = _generation(sim)
    assert (
        sim.patch_scene_mjcf(ops=[{"op": "set_body_pos", "name": "cube", "pos": [0.5, 0, 0.4]}])["status"] == "success"
    )
    assert _generation(sim) > before


def test_remove_camera_bumps_the_generation(sim) -> None:
    assert sim.add_camera(name="cam", position=[1, 1, 1], target=[0, 0, 0])["status"] == "success"
    before = _generation(sim)
    assert sim.remove_camera(name="cam")["status"] == "success"
    assert _generation(sim) > before


def test_randomize_stays_in_window_across_a_full_rebuild(sim) -> None:
    """The user-visible consequence of the shrinking case."""
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    _add(sim, "light", 0.4)
    _randomize(sim)

    assert sim.remove_robot(name="b")["status"] == "success"
    for _ in range(6):
        assert _randomize(sim)["status"] == "success"

    mass = _mass_of(sim, "light")
    lo, hi = 0.05 * _MASS_RANGE[0], 0.05 * _MASS_RANGE[1]
    assert lo - 1e-9 <= mass <= hi + 1e-9, f"mass {mass} escaped [{lo}, {hi}] - stale baseline"


def test_baseline_arrays_match_the_live_model_after_a_rebuild(sim) -> None:
    """The direct invariant: a baseline indexed by body id must be nbody long."""
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    _randomize(sim)
    assert sim.remove_robot(name="b")["status"] == "success"
    _randomize(sim)

    baseline = sim._world._backend_state["dr_baseline"]
    model = sim.mj_model
    assert len(baseline["body_mass"]) == model.nbody
    assert len(baseline["geom_friction"]) == model.ngeom


def test_randomize_after_the_model_grows_does_not_raise(sim) -> None:
    """The growing case used to raise IndexError out of the tool."""
    _add(sim, "c", 0.3, mass=0.2)
    _randomize(sim)

    bodies = "".join(
        f'<body name="x{i}" pos="{i * 0.15} 1 0.3"><freejoint/>'
        f'<geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body>'
        for i in range(12)
    )
    xml = (
        f'<mujoco><worldbody><light pos="0 0 3"/>'
        f'<geom name="ground" type="plane" size="5 5 .1"/>{bodies}</worldbody></mujoco>'
    )
    assert sim.replace_scene_mjcf(xml=xml)["status"] == "success"
    assert sim.mj_model.nbody > 2

    assert _randomize(sim)["status"] == "success"
    for i in range(12):
        mass = _mass_of(sim, f"x{i}")
        assert 0.1 * _MASS_RANGE[0] - 1e-9 <= mass <= 0.1 * _MASS_RANGE[1] + 1e-9, (i, mass)


def test_checkpoint_saved_before_a_full_rebuild_is_refused(sim) -> None:
    """A same-shape rebuild is exactly what the generation stamp exists for."""
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    _add(sim, "cube", 0.4, mass=0.2)
    assert sim.save_state(name="cp")["status"] == "success"

    # Swap the object for an identically-shaped one: nq/nv/na/nu are unchanged,
    # so only the generation can tell the checkpoint is no longer applicable.
    assert sim.remove_object(name="cube")["status"] == "success"
    assert _add(sim, "other", 0.6, mass=0.2)["status"] == "success"

    result = sim.load_state(name="cp")
    assert result["status"] == "error"
    assert "stale" in result["content"][0]["text"]
