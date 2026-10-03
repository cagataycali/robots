# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A rollout that pins a servo at its force limit says so in the run_policy payload.

Every action-health field is keyed on key resolution, so an SO-101 pressing its
gripper into the floor reported ``action_errors=0`` and ``partial_action_failure_rate=0.0``
exactly like a rollout that reached its command. ``saturated_step_rate`` and
``saturation_rate`` read the backend's force clamp instead.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies import Policy  # noqa: E402
from strands_robots.simulation import Simulation  # noqa: E402
from strands_robots.simulation.policy_runner import PolicyRunner  # noqa: E402


class _HoldPolicy(Policy):
    """Emits the same action dict every tick."""

    def __init__(self, action: dict[str, float]) -> None:
        self._action = action

    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any) -> list[Any]:
        return [dict(self._action)]

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        pass

    @property
    def requires_images(self) -> bool:
        return False

    @property
    def provider_name(self) -> str:
        return "hold"


@pytest.fixture(scope="module")
def sim():
    s = Simulation(tool_name="saturation_test", mesh=False)
    s.create_world()
    s.add_robot(name="arm", data_config="so101")
    yield s
    s.cleanup()


def _run(sim: Simulation, overrides: dict[str, float]) -> tuple[str, dict[str, Any]]:
    sim.reset()
    keys = sim.robot_action_keys("arm")
    policy = _HoldPolicy({**dict.fromkeys(keys, 0.0), **overrides})
    result = PolicyRunner(sim).run("arm", policy, duration=1.0, control_frequency=50, fast_mode=True)
    text = "\n".join(b["text"] for b in result["content"] if "text" in b)
    payload = next(b["json"] for b in result["content"] if "json" in b)
    assert result["status"] == "success" and payload["action_errors"] == 0
    assert payload["partial_action_failure_rate"] == 0.0
    return text, payload


@pytest.mark.parametrize(
    ("overrides", "pinned"),
    [
        pytest.param({}, set(), id="clear: home pose is reachable"),
        pytest.param({"2": 1.7, "3": 1.6}, {"2", "3"}, id="stall: shoulder and elbow press the gripper into the floor"),
    ],
)
def test_saturation_is_reported_next_to_resolution(sim, overrides, pinned):
    text, payload = _run(sim, overrides)
    assert {n for n, rate in payload["saturation_rate"].items() if rate > 0.5} == pinned
    if pinned:
        assert payload["saturated_step_rate"] > 0.5
        assert "Force-limited" in text
    else:
        assert payload["saturated_step_rate"] == 0.0
        assert "Force-limited" not in text


def test_a_backend_that_cannot_tell_reports_none(sim, monkeypatch):
    monkeypatch.setattr(sim, "saturated_actuators", lambda robot_name: None)
    _, payload = _run(sim, {"2": 1.7})
    assert payload["saturated_step_rate"] is None and payload["saturation_rate"] is None
