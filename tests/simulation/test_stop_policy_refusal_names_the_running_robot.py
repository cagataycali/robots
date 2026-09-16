"""``stop_policy`` without a robot names the robots whose rollouts are in flight.

``robot_name`` stays required (it is never defaulted to the sole robot - see
``stop_policy``), but ``start_policy`` on the sole robot DOES default, so an agent
that launched without a name met "stop_policy requires 'robot_name'." and had to
go find one. The refusal now names the running robots with the exact call, or
says nothing is running, and stays bare on a backend with no rollout registry.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


@pytest.fixture
def sim():
    s = MuJoCoSimEngine(tool_name="stop_names", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    yield s
    s.cleanup()


def test_the_refusal_names_the_running_robot_and_the_exact_call(sim):
    assert (
        sim.start_policy("so101", policy_provider="mock", duration=5.0, control_frequency=30.0)["status"] == "success"
    )
    try:
        deadline = time.monotonic() + 5.0
        while "so101" not in sim._rollouts_in_flight() and time.monotonic() < deadline:
            time.sleep(0.05)
        r = sim.stop_policy("")
        assert r["status"] == "error"
        assert "stop_policy requires 'robot_name'." in _text(r)
        assert "Running now: so101 - stop_policy(robot_name='so101')." in _text(r)
    finally:
        assert sim.stop_policy("so101")["status"] == "success"


def test_the_refusal_says_when_nothing_is_running(sim):
    r = sim.stop_policy("")
    assert r["status"] == "error"
    assert "stop_policy requires 'robot_name'." in _text(r)
    assert "No policy is running now (list_policies_running)." in _text(r)
    assert "Running now" not in _text(r)


def test_the_refusal_still_never_defaults_to_the_sole_robot(sim):
    assert (
        sim.start_policy("so101", policy_provider="mock", duration=5.0, control_frequency=30.0)["status"] == "success"
    )
    try:
        time.sleep(0.3)
        assert sim.stop_policy("")["status"] == "error"
        assert "so101" in sim._rollouts_in_flight(), "an empty name must not have stopped the rollout"
    finally:
        sim.stop_policy("so101")


def test_a_backend_with_no_rollout_registry_keeps_the_bare_requirement():
    from tests.simulation.test_stop_policy_base_contract import _MinimalEngine

    engine = _MinimalEngine()
    assert engine._rollouts_in_flight() is None
    assert engine._stop_policy_requires_name_msg() == "stop_policy requires 'robot_name'."
    assert _text(engine.stop_policy("")) == "stop_policy requires 'robot_name'."
