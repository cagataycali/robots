"""Regression tests: ``add_object(mass=...)`` is validated like every sibling field.

``add_object`` validates ``position`` / ``orientation`` / ``size`` / ``color`` for
finiteness up front - its docstring says so - but ``mass`` was left out, while
``set_body_properties`` has enforced "finite and > 0" all along. The recompile
catches a negative or zero mass, but not a non-finite one:

    add_object(mass=nan) -> status="success", body_mass = 0.125
                            (MuJoCo silently substituted a density-derived
                             default; NOT the mass the caller asked for)
    add_object(mass=inf) -> status="success", body_mass = inf, inertia = inf,
                            and data.qpos non-finite after a single step

so the first case gave a *different object than requested* under a success
result, and the second poisoned the whole world.

Negative and zero masses were already refused, but only via "spec recompile
refused", which hides the actionable reason; they now get the same clear message.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="add_object_mass_validation", mesh=False)
    s.create_world()
    yield s
    s.destroy()


def _add(sim, mass, name="c"):
    return sim.add_object(name=name, shape="box", size=[0.05] * 3, position=[0.4, 0, 0.3], mass=mass)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_mass_is_rejected(sim, bad) -> None:
    result = _add(sim, bad)
    assert result["status"] == "error"
    assert "finite" in result["content"][0]["text"]
    assert mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "c") < 0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_mass_is_rejected_with_a_clear_reason(sim, bad) -> None:
    """Previously refused only as an opaque 'spec recompile refused'."""
    result = _add(sim, bad)
    assert result["status"] == "error"
    assert "mass" in result["content"][0]["text"]


def test_non_numeric_mass_is_rejected(sim) -> None:
    result = _add(sim, "heavy")
    assert result["status"] == "error"
    assert "mass" in result["content"][0]["text"]


def test_the_world_stays_integrable_after_a_rejected_mass(sim) -> None:
    """The inf case used to drive qpos non-finite on the first step."""
    for bad in (float("nan"), float("inf")):
        _add(sim, bad)
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert sim.step(n_steps=50)["status"] == "success"
    model, data = sim.mj_model, sim.mj_data
    assert bool(np.all(np.isfinite(model.body_mass)))
    assert bool(np.all(np.isfinite(model.body_inertia)))
    assert bool(np.all(np.isfinite(data.qpos)))
    assert bool(np.all(np.isfinite(data.qvel)))


def test_a_valid_mass_is_honoured_exactly(sim) -> None:
    """Guard against the fix degenerating, and against the nan-substitution bug:
    the compiled mass must be what was asked for, not a density-derived default."""
    for i, mass in enumerate((0.01, 0.2, 5.0)):
        name = f"o{i}"
        assert _add(sim, mass, name=name)["status"] == "success"
        body = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert float(sim.mj_model.body_mass[body]) == pytest.approx(mass)


def test_a_static_object_ignores_mass(sim) -> None:
    """Static bodies are welded to the worldbody and carry no mass at all."""
    result = sim.add_object(
        name="fixture", shape="box", size=[0.05] * 3, position=[0.4, 0, 0.3], mass=0.0, is_static=True
    )
    assert result["status"] == "success", "mass is irrelevant for a static object"
    body = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "fixture")
    assert int(sim.mj_model.body_dofnum[body]) == 0


def test_a_plane_still_works(sim) -> None:
    """A plane is forced static, so it must not trip the mass guard."""
    assert (
        sim.add_object(name="ground2", shape="plane", size=[5.0, 5.0, 0.1], position=[0, 0, 0])["status"] == "success"
    )


def test_matches_set_body_properties(sim) -> None:
    """The two entry points must agree on what a legal mass is."""
    assert _add(sim, 0.2)["status"] == "success"
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        assert _add(sim, bad, name="other")["status"] == "error", bad
        assert sim.set_body_properties(body_name="c", mass=bad)["status"] == "error", bad
