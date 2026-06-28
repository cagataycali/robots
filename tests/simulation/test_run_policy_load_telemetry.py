"""run_policy / eval_policy surface the driving policy's load telemetry.

A multi-episode harness that pays a model reload per episode looks identical in
the result text to one that reuses a warm policy -- the only machine-checkable
difference is whether the load was a cache hit. These tests pin that
``run_policy`` and ``eval_policy`` carry ``policy_load_time_s`` and
``policy_load_cache_hit`` in their JSON block, that the values default honestly
for policies without load telemetry (MockPolicy), and that a policy_object
carrying telemetry has it echoed through verbatim.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim():
    s = Simulation(tool_name="load_telemetry", mesh=False)
    s.create_world()
    s.add_robot(name="so100", data_config="so100")
    yield s
    s.cleanup()


def _json_block(result: dict) -> dict:
    for blk in result.get("content", []):
        if isinstance(blk, dict) and isinstance(blk.get("json"), dict):
            return blk["json"]
    raise AssertionError(f"no json content block in {result}")


def test_run_policy_payload_has_telemetry_keys(sim):
    run = sim.run_policy(
        robot_name="so100",
        policy_provider="mock",
        n_steps=4,
        control_frequency=50,
        fast_mode=True,
    )
    assert run["status"] == "success", run
    payload = _json_block(run)
    # Keys always present so an agent never has to regex the text block.
    assert "policy_load_time_s" in payload
    assert "policy_load_cache_hit" in payload
    # MockPolicy exposes no load telemetry -> honest defaults.
    assert payload["policy_load_time_s"] == 0.0
    assert payload["policy_load_cache_hit"] is False


def test_run_policy_echoes_policy_object_telemetry(sim):
    policy = MockPolicy()
    # Simulate a warm, cache-hit load on the supplied policy object.
    policy.load_time_s = 0.0
    policy.load_cache_hit = True
    run = sim.run_policy(
        robot_name="so100",
        policy_object=policy,
        n_steps=4,
        control_frequency=50,
        fast_mode=True,
    )
    assert run["status"] == "success", run
    payload = _json_block(run)
    assert payload["policy_load_cache_hit"] is True


def test_eval_policy_payload_has_telemetry_keys(sim):
    ev = sim.eval_policy(
        robot_name="so100",
        policy_provider="mock",
        n_episodes=2,
        max_steps=4,
        control_frequency=50,
    )
    assert ev["status"] == "success", ev
    payload = _json_block(ev)
    assert "policy_load_time_s" in payload
    assert "policy_load_cache_hit" in payload
