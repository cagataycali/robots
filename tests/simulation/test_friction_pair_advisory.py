"""Regression tests: lowering one geom's friction below its partner's is announced.

MuJoCo combines a contact pair's friction as the **max** of the two geoms (absent an
explicit ``<pair>``). The scene's ground plane declares no friction and so takes
MuJoCo's default ``1.0``, which means every ``friction < 1.0`` an agent sets on an
object is a silent no-op against the floor:

    set_geom_properties("b_geom", friction=[0.1, ...])   -> success
    model.geom_friction["b_geom"]                        -> [0.1, ...]   faithful
    travel under a fixed 3 N push                        -> 0.0028 m

    ... identical to mu=0.3 and mu=1.0. The analytic prediction for mu=0.1 is
    0.90 m, so the requested value was 326x off, with nothing to indicate it.

This is MuJoCo's rule, not a bug in the write - which is why the fix is an advisory
note on the success result rather than a changed value. Verified actionable: setting
the GROUND to 0.1 as well takes the same push from 0.0028 m to 0.9004 m, matching
the analytic 0.90 m.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="friction_pair_advisory", mesh=False)
    s.create_world()
    assert s.add_object(name="b", shape="box", size=[0.06] * 3, position=[0, 0, 0.031], mass=0.5)["status"] == "success"
    yield s
    s.destroy()


def _set(sim, geom, mu):
    return sim.set_geom_properties(geom_name=geom, friction=[mu, 0.005, 0.0001])


@pytest.mark.parametrize("mu", [0.0, 0.1, 0.5, 0.99])
def test_a_friction_below_the_ground_is_flagged(sim, mu) -> None:
    """The core defect: a silently inert value."""
    result = _set(sim, "b_geom", mu)
    assert result["status"] == "success"
    text = result["content"][0]["text"]
    assert "no effect against the floor" in text
    assert "MAX" in text


@pytest.mark.parametrize("mu", [1.0, 2.5])
def test_a_friction_at_or_above_the_ground_is_not_flagged(sim, mu) -> None:
    """Guard against advisory noise on values that DO take effect."""
    result = _set(sim, "b_geom", mu)
    assert result["status"] == "success"
    assert "no effect" not in result["content"][0]["text"]


def test_the_value_is_still_written(sim) -> None:
    """The note is advisory - it must not suppress the requested write."""
    assert _set(sim, "b_geom", 0.1)["status"] == "success"
    model = sim.mj_model
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "b_geom")
    assert float(model.geom_friction[geom][0]) == pytest.approx(0.1)


def test_lowering_the_ground_first_silences_the_note(sim) -> None:
    """Following the advice must make the note go away."""
    assert _set(sim, "ground", 0.05)["status"] == "success"
    result = _set(sim, "b_geom", 0.1)
    assert result["status"] == "success"
    assert "no effect" not in result["content"][0]["text"]


def test_setting_the_ground_itself_is_never_flagged(sim) -> None:
    """The ground cannot be below itself."""
    result = _set(sim, "ground", 0.1)
    assert result["status"] == "success"
    assert "no effect" not in result["content"][0]["text"]


def _travel(sim, force=3.0, steps=300):
    for _ in range(2000):
        sim.step(n_steps=1)
    model, data = sim.mj_model, sim.mj_data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")
    start = float(data.xpos[body][0])
    assert sim.apply_force(body_name="b", force=[force, 0, 0])["status"] == "success"
    for _ in range(steps):
        sim.step(n_steps=1)
    return float(sim.mj_data.xpos[body][0]) - start


def test_the_advisory_describes_a_real_no_op(sim) -> None:
    """Pin the behaviour the note documents, so it cannot outlive it."""
    assert _set(sim, "b_geom", 0.1)["status"] == "success"
    assert _travel(sim) < 0.05, "a friction below the ground's should NOT have freed the block"


def test_following_the_advice_changes_the_physics(sim) -> None:
    """And pin that the suggested remedy actually works."""
    assert _set(sim, "ground", 0.1)["status"] == "success"
    assert _set(sim, "b_geom", 0.1)["status"] == "success"
    travelled = _travel(sim)
    # Analytic: a = (F - mu*m*g)/m = 5.02 m/s^2 over 0.6 s -> ~0.90 m.
    assert travelled > 0.5, f"lowering both frictions should free the block, got {travelled}"
    assert bool(np.isfinite(travelled))
