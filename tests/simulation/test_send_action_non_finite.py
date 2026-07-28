"""Regression tests: a non-finite action is rejected, not silently discarded.

``_coerce_action`` validated that every action value *coerces* to a float but
never that it is finite, so a ``nan`` / ``inf`` was written straight into
``data.ctrl`` while the call reported success:

    send_action({"joint2": nan})  -> status="success", "Action applied to 'a' (1 keys)."
    data.ctrl[1]                  -> nan

MuJoCo then printed ``Nan, Inf or huge value in CTRL at ACTUATOR 1. The
simulation is unstable.`` on stderr and zeroed the control internally - so the
robot did not move, and nothing in the tool result said so. A policy emitting
``nan`` (un-normalised observations, a diverged checkpoint) is the usual way to
reach this, i.e. precisely when a silent no-op is most misleading.

The guard lives in ``_coerce_action`` - the same up-front, all-or-nothing
validation that already rejected non-numeric values - so both the mapping and the
ordered-vector forms are covered for every backend.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_NON_FINITE = [float("nan"), float("inf"), -float("inf")]


@pytest.fixture
def sim():
    s = Simulation(tool_name="send_action_non_finite", mesh=False)
    s.create_world()
    assert s.add_robot(name="a", data_config="panda")["status"] == "success"
    yield s
    s.destroy()


@pytest.mark.parametrize("value", _NON_FINITE)
def test_mapping_form_rejects_non_finite(sim, value) -> None:
    result = sim.send_action(action={"joint2": value}, robot_name="a")
    assert result["status"] == "error"
    assert "not finite" in result["content"][0]["text"]


@pytest.mark.parametrize("value", _NON_FINITE)
def test_vector_form_rejects_non_finite(sim, value) -> None:
    keys = sim.robot_action_keys("a")
    vector = [0.0] * len(keys)
    vector[-1] = value
    result = sim.send_action(action=vector, robot_name="a")
    assert result["status"] == "error"
    assert "not finite" in result["content"][0]["text"]
    # The error names the offending position so a policy author can find it.
    assert f"index {len(vector) - 1}" in result["content"][0]["text"]


def test_ctrl_is_left_untouched(sim) -> None:
    """All-or-nothing: a rejected action must not have partially applied."""
    before = np.array(sim.mj_data.ctrl).copy()
    result = sim.send_action(action={"joint1": 0.5, "joint2": float("nan")}, robot_name="a")
    assert result["status"] == "error"
    assert np.array_equal(np.array(sim.mj_data.ctrl), before), "a rejected action wrote to ctrl"


def test_ctrl_stays_finite(sim) -> None:
    """The user-visible symptom: MuJoCo reporting the simulation unstable."""
    for value in _NON_FINITE:
        sim.send_action(action={"joint2": value}, robot_name="a")
    assert bool(np.all(np.isfinite(sim.mj_data.ctrl)))
    assert sim.step(n_steps=10)["status"] == "success"
    assert bool(np.all(np.isfinite(sim.mj_data.qpos)))


def test_finite_actions_still_apply(sim) -> None:
    """Guard against the fix degenerating into 'reject everything'."""
    result = sim.send_action(action={"joint2": 0.4}, robot_name="a")
    assert result["status"] == "success"
    model, data = sim.mj_model, sim.mj_data
    act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a/actuator2")
    assert float(data.ctrl[act]) == pytest.approx(0.4)


def test_finite_vector_still_applies(sim) -> None:
    keys = sim.robot_action_keys("a")
    result = sim.send_action(action=[0.1] * len(keys), robot_name="a")
    assert result["status"] == "success"
    assert all(math.isfinite(float(v)) for v in sim.mj_data.ctrl)


def test_non_numeric_rejection_is_unchanged(sim) -> None:
    """The pre-existing type guard must keep its own message."""
    result = sim.send_action(action={"joint2": "x"}, robot_name="a")
    assert result["status"] == "error"
    assert "scalar number" in result["content"][0]["text"]
