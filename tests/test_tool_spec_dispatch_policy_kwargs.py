"""Regression tests: tool_spec dispatcher must forward policy-related kwargs
through **policy_kwargs to create_policy().

Context: PR #85 shipped a hardcoded whitelist in Simulation._dispatch_action
that silently dropped observation_mapping / action_mapping / data_config /
host / port and any other policy kwargs. This broke sim↔real transfer via
the AgentTool interface (tool_spec advertises `run_policy` / `eval_policy`
/ `start_policy` but agents couldn't actually wire mappings through).

These tests pin the forwarding behaviour without requiring MuJoCo — they
build a Simulation instance and call _dispatch_action directly, with
patched methods that capture the kwargs.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest

# Skip the whole module if mujoco isn't available (dev env without [sim-mujoco]).
# The dispatcher logic is still exercised in CI / any env with mujoco installed.
pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim() -> Generator[Simulation, None, None]:
    """Build a Simulation — dispatcher logic is tested in isolation via
    patched method replacements, so no world/state setup is required."""
    s = Simulation(tool_name="dispatch_test", mesh=False)
    yield s
    s.cleanup()


def _capture_kwargs(captured: dict[str, Any]):
    """Build a replacement method that stores all kwargs it receives."""

    def fake(**kwargs: Any) -> dict[str, Any]:
        captured.clear()
        captured.update(kwargs)
        return {"status": "success", "content": [{"text": "ok"}]}

    return fake


class TestDispatcherForwardsPolicyKwargs:
    """`_dispatch_action` must pass unknown keys through **policy_kwargs."""

    def test_run_policy_forwards_observation_and_action_mapping(self, sim):
        captured: dict[str, Any] = {}
        with patch.object(sim, "run_policy", _capture_kwargs(captured)):
            sim._dispatch_action(
                "run_policy",
                {
                    "robot_name": "so100",
                    "policy_provider": "mock",
                    "instruction": "pick up the red cube",
                    "duration": 3.0,
                    "observation_mapping": {
                        "front": "video.front",
                        "wrist": "video.wrist",
                        "joint_position": "state.single_arm",
                    },
                    "action_mapping": {
                        "action.single_arm": "joint_position",
                    },
                    "data_config": "so100",
                    "device": "mps",
                },
            )
        # Named params routed correctly
        assert captured["robot_name"] == "so100"
        assert captured["policy_provider"] == "mock"
        assert captured["instruction"] == "pick up the red cube"
        assert captured["duration"] == 3.0
        # Policy kwargs forwarded via **policy_kwargs
        assert captured["observation_mapping"] == {
            "front": "video.front",
            "wrist": "video.wrist",
            "joint_position": "state.single_arm",
        }
        assert captured["action_mapping"] == {"action.single_arm": "joint_position"}
        assert captured["data_config"] == "so100"
        assert captured["device"] == "mps"

    def test_eval_policy_forwards_pretrained_name_and_device(self, sim):
        captured: dict[str, Any] = {}
        with patch.object(sim, "eval_policy", _capture_kwargs(captured)):
            sim._dispatch_action(
                "eval_policy",
                {
                    "robot_name": "so100",
                    "policy_provider": "lerobot_local",
                    "pretrained_name_or_path": "lerobot/smolvla_base",
                    "device": "mps",
                    "trust_remote_code": True,
                    "actions_per_step": 4,
                    "n_episodes": 2,
                    "max_steps": 100,
                },
            )
        assert captured["robot_name"] == "so100"
        assert captured["policy_provider"] == "lerobot_local"
        assert captured["n_episodes"] == 2
        assert captured["max_steps"] == 100
        # Passthrough kwargs
        assert captured["pretrained_name_or_path"] == "lerobot/smolvla_base"
        assert captured["device"] == "mps"
        assert captured["trust_remote_code"] is True
        assert captured["actions_per_step"] == 4

    def test_start_policy_forwards_service_config(self, sim):
        captured: dict[str, Any] = {}
        with patch.object(sim, "start_policy", _capture_kwargs(captured)):
            sim._dispatch_action(
                "start_policy",
                {
                    "robot_name": "so100",
                    "policy_provider": "groot",
                    "host": "localhost",
                    "port": 5555,
                    "api_token": "dummy-token",
                    "data_config": "so100_dualcam",
                    "observation_mapping": {"front": "video.front"},
                    "action_mapping": {"action.single_arm": "joint_position"},
                    "instruction": "tidy the desk",
                },
            )
        assert captured["policy_provider"] == "groot"
        assert captured["host"] == "localhost"
        assert captured["port"] == 5555
        assert captured["api_token"] == "dummy-token"
        assert captured["data_config"] == "so100_dualcam"
        assert captured["observation_mapping"] == {"front": "video.front"}
        assert captured["action_mapping"] == {"action.single_arm": "joint_position"}

    def test_non_policy_action_does_not_pick_up_policy_kwargs(self, sim):
        """Actions without **kwargs must not accidentally accept unknown keys."""
        captured: dict[str, Any] = {}

        def fake_set_gravity(gravity: list[float] | None = None) -> dict[str, Any]:
            captured["gravity"] = gravity
            return {"status": "success", "content": [{"text": "ok"}]}

        with patch.object(sim, "set_gravity", fake_set_gravity):
            sim._dispatch_action(
                "set_gravity",
                {
                    "gravity": [0, 0, -9.81],
                    # These must be ignored (no **kwargs on set_gravity)
                    "observation_mapping": {"x": "y"},
                    "device": "mps",
                },
            )
        assert captured["gravity"] == [0, 0, -9.81]
        # No crash: unknown keys filtered when no **kwargs


class TestToolSpecAdvertisesPolicyKwargs:
    """tool_spec.json must expose the new kwargs so agents can discover them."""

    def test_tool_spec_has_mapping_properties(self):
        import json
        from pathlib import Path

        spec_path = Path(__file__).parent.parent / "strands_robots" / "simulation" / "mujoco" / "tool_spec.json"
        spec = json.loads(spec_path.read_text())
        props = spec["properties"]
        for key in (
            "observation_mapping",
            "action_mapping",
            "host",
            "port",
            "api_token",
            "trust_remote_code",
            "actions_per_step",
            "use_processor",
            "processor_overrides",
            "device",
        ):
            assert key in props, f"tool_spec.json missing '{key}'"
        # Mapping-typed keys must declare object type
        assert props["observation_mapping"]["type"] == "object"
        assert props["action_mapping"]["type"] == "object"
