# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""run_policy / eval_policy surface the driving policy's routing-degradation flags.

A run that silently degraded (camera routed positionally, or observation.state
composed from generic joint_0..N keys) reports ``status="success"`` exactly
like a healthy run -- the only machine-checkable difference is the
``positional_fallback_used`` / ``generic_state_keys_used`` flags the policy
raises. These tests pin that both keys are always present in the JSON block and
default honestly to False for a policy that exposes no such attribute
(MockPolicy), so an agent can self-correct without parsing the text or scraping
logs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim():
    s = Simulation(tool_name="routing_telemetry", mesh=False)
    s.create_world()
    s.add_robot(name="so100", data_config="so100")
    yield s
    s.cleanup()


def _json_block(result: dict) -> dict:
    for blk in result.get("content", []):
        if isinstance(blk, dict) and isinstance(blk.get("json"), dict):
            return blk["json"]
    raise AssertionError(f"no json content block in {result}")


def test_run_policy_payload_has_routing_flags(sim):
    run = sim.run_policy(
        robot_name="so100",
        policy_provider="mock",
        n_steps=4,
        control_frequency=50,
        fast_mode=True,
    )
    assert run["status"] == "success", run
    payload = _json_block(run)
    assert payload["positional_fallback_used"] is False
    assert payload["generic_state_keys_used"] is False


def test_eval_policy_payload_has_routing_flags(sim):
    ev = sim.eval_policy(
        robot_name="so100",
        policy_provider="mock",
        n_episodes=1,
        max_steps=4,
        control_frequency=50,
    )
    assert ev["status"] == "success", ev
    payload = _json_block(ev)
    assert payload["positional_fallback_used"] is False
    assert payload["generic_state_keys_used"] is False
