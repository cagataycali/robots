"""Regression tests: the domain-randomisation baseline tracks the CURRENT model.

``randomize(randomize_physics=True)`` scales mass/friction from a cached snapshot
of the un-randomised model so repeated calls cannot compound. That cache was keyed
on ``ngeom``, which misses any rebuild leaving the counts unchanged: a
``remove_object`` + ``add_object`` pair returns ``ngeom`` to its old value while
the arrays now describe DIFFERENT bodies, so the stale baseline was applied to
whichever body took the freed slot.

Measured - object with a 0.1 kg base, scaled against the removed body's 0.5 kg
baseline, after six calls:

    mass = 0.2907 kg     (legal window for a 0.1 kg base is [0.05, 0.2])

The cache is now keyed on ``_recompile_generation``, which increments on EVERY
recompile regardless of shape.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# randomize()'s default mass_range is (0.5, 2.0), so a base mass m is legal in
# [0.5 m, 1.0 m]... expressed against the *base*, the window is [m/2, m].
_MASS_RANGE = (0.5, 2.0)


def _mass_of(sim, body_name: str) -> float:
    model = sim.mj_model
    return float(model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)])


def _randomize(sim) -> None:
    assert (
        sim.randomize(randomize_physics=True, randomize_colors=False, randomize_lighting=False)["status"] == "success"
    )


@pytest.fixture
def sim():
    s = Simulation(tool_name="dr_baseline_invalidation", mesh=False)
    s.create_world()
    yield s
    s.destroy()


def test_baseline_is_rebuilt_when_a_slot_is_reused(sim) -> None:
    """The core defect: same ngeom, different bodies."""
    sim.add_object(name="a", shape="box", size=[0.04] * 3, position=[0.2, 0, 0.3], mass=0.5)
    sim.add_object(name="b", shape="box", size=[0.04] * 3, position=[0.6, 0, 0.3], mass=0.5)
    _randomize(sim)

    # Swap b (0.5 kg) for c (0.1 kg): ngeom is unchanged, so the ngeom guard
    # could not detect that the arrays now describe a different body.
    assert sim.remove_object(name="b")["status"] == "success"
    assert (
        sim.add_object(name="c", shape="box", size=[0.04] * 3, position=[0.9, 0, 0.3], mass=0.1)["status"] == "success"
    )

    for _ in range(6):
        _randomize(sim)

    mass = _mass_of(sim, "c")
    lo, hi = 0.1 * _MASS_RANGE[0], 0.1 * _MASS_RANGE[1]
    assert lo - 1e-9 <= mass <= hi + 1e-9, f"mass {mass} outside [{lo}, {hi}] - stale baseline"


def test_cache_is_keyed_on_the_recompile_generation(sim) -> None:
    sim.add_object(name="a", shape="box", size=[0.04] * 3, position=[0.2, 0, 0.3], mass=0.5)
    _randomize(sim)
    first = sim._world._backend_state["dr_baseline"]["generation"]

    sim.add_object(name="b", shape="box", size=[0.03] * 3, position=[1, 1, 0.3], mass=0.2)
    _randomize(sim)
    assert sim._world._backend_state["dr_baseline"]["generation"] > first


def test_growth_still_rebaselines(sim) -> None:
    """The case the ngeom guard already covered must keep working."""
    sim.add_object(name="a", shape="box", size=[0.04] * 3, position=[0.2, 0, 0.3], mass=0.5)
    _randomize(sim)
    sim.add_object(name="b", shape="box", size=[0.03] * 3, position=[1, 0, 0.3], mass=0.3)
    _randomize(sim)

    for name, base in (("a", 0.5), ("b", 0.3)):
        mass = _mass_of(sim, name)
        assert base * _MASS_RANGE[0] - 1e-9 <= mass <= base * _MASS_RANGE[1] + 1e-9, (name, mass)


def test_no_compounding_over_many_episodes_with_scene_churn(sim) -> None:
    """The original no-compounding guarantee must survive repeated rebuilds."""
    sim.add_object(name="keep", shape="box", size=[0.04] * 3, position=[0.2, 0, 0.3], mass=0.5)
    lo, hi = 0.5 * _MASS_RANGE[0], 0.5 * _MASS_RANGE[1]

    for episode in range(12):
        sim.reset()
        _randomize(sim)
        mass = _mass_of(sim, "keep")
        assert lo - 1e-9 <= mass <= hi + 1e-9, f"episode {episode}: mass {mass} escaped [{lo}, {hi}]"
        if episode % 3 == 0:
            sim.add_object(name=f"t{episode}", shape="box", size=[0.03] * 3, position=[1, 1, 0.3], mass=0.2)
            sim.remove_object(name=f"t{episode}")


def test_the_baseline_keeps_pristine_values_across_a_recompile(sim) -> None:
    """A generation bump must RE-MAP the baseline, not re-read the live model.

    Keying the cache on the generation made invalidation fire correctly, but the
    re-snapshot read the ALREADY-RANDOMISED arrays, so each churn made the current
    randomisation the new "original" and the window walked away:

        for ep: reset(); randomize(); (add_object + remove_object)
        baseline_mass["keep"]  0.5000 -> 0.5134 -> 0.4078 -> 0.6664 ...
        live mass at ep 7     1.1423        (legal window is [0.25, 1.0])

    The pristine value is now carried across by body/geom NAME.
    """
    sim.add_object(name="keep", shape="box", size=[0.04] * 3, position=[0.2, 0, 0.3], mass=0.5)
    _randomize(sim)
    model = sim.mj_model
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "keep")

    for episode in range(9):
        sim.reset()
        _randomize(sim)
        baseline = sim._world._backend_state["dr_baseline"]
        assert float(baseline["body_mass"][body]) == pytest.approx(0.5), (
            f"episode {episode}: the baseline drifted to {float(baseline['body_mass'][body])}"
        )
        if episode % 3 == 0:
            sim.add_object(name=f"t{episode}", shape="box", size=[0.03] * 3, position=[1, 1, 0.3], mass=0.2)
            sim.remove_object(name=f"t{episode}")


def test_no_compounding_over_forty_churned_episodes(sim) -> None:
    """The long-run form: the ceiling must hold, not merely the first few episodes."""
    sim.add_object(name="k", shape="box", size=[0.04] * 3, position=[0.2, 0, 0.3], mass=0.5)
    model = sim.mj_model
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "k")
    lo, hi = 0.5 * _MASS_RANGE[0], 0.5 * _MASS_RANGE[1]

    for episode in range(40):
        sim.reset()
        _randomize(sim)
        mass = float(sim.mj_model.body_mass[body])
        assert lo - 1e-9 <= mass <= hi + 1e-9, f"episode {episode}: mass {mass} escaped [{lo}, {hi}]"
        if episode % 2 == 0:
            sim.add_object(name=f"t{episode}", shape="box", size=[0.03] * 3, position=[1, 1, 0.3], mass=0.2)
            sim.remove_object(name=f"t{episode}")


def test_a_newly_added_body_baselines_from_the_live_model(sim) -> None:
    """An entity the old baseline never saw is un-randomised by definition.

    The remap is lazy - it runs on the next ``randomize``, not on the mutation -
    so the assertion below reads the baseline only after one more call.
    """
    sim.add_object(name="a", shape="box", size=[0.04] * 3, position=[0.2, 0, 0.3], mass=0.5)
    _randomize(sim)
    sim.add_object(name="fresh", shape="box", size=[0.04] * 3, position=[0.9, 0, 0.3], mass=0.1)
    _randomize(sim)

    baseline = sim._world._backend_state["dr_baseline"]
    model = sim.mj_model
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fresh")
    assert float(baseline["body_mass"][body]) == pytest.approx(0.1)

    for _ in range(8):
        _randomize(sim)
    mass = float(sim.mj_model.body_mass[mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "fresh")])
    assert 0.1 * _MASS_RANGE[0] - 1e-9 <= mass <= 0.1 * _MASS_RANGE[1] + 1e-9, mass


def test_friction_is_also_carried_by_name(sim) -> None:
    sim.add_object(name="a", shape="box", size=[0.04] * 3, position=[0.2, 0, 0.3], mass=0.5)
    base_friction = float(
        sim.mj_model.geom_friction[mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "a_geom")][0]
    )
    _randomize(sim)
    sim.add_object(name="b", shape="box", size=[0.03] * 3, position=[1, 1, 0.3], mass=0.2)

    baseline = sim._world._backend_state["dr_baseline"]
    geom = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "a_geom")
    assert float(baseline["geom_friction"][geom][0]) == pytest.approx(base_friction)
