"""Dispatcher tests for the nested ``policy_config`` shape.

After the backend-agnostic ``PolicyRunner`` refactor, the AgentTool
dispatcher is schema-driven: every method parameter is explicit, and
policy-provider-specific kwargs are nested under ``policy_config`` — they
are NEVER advertised as top-level properties in ``tool_spec.json`` and
NEVER forwarded via ``**kwargs``.

These tests pin:

1. ``policy_config`` nested forwarding works for ``run_policy`` /
   ``eval_policy`` / ``start_policy``.
2. ``tool_spec.json`` advertises ``policy_config`` and does NOT advertise
   any of the old leaked provider-specific fields.
3. Unknown top-level keys are dropped silently (no ``**kwargs`` passthrough).
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest

# Skip the whole module if mujoco isn't available (dev env without [sim-mujoco]).
pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim() -> Generator[Simulation, None, None]:
    s = Simulation(tool_name="dispatch_test", mesh=False)
    yield s
    s.cleanup()


def _capture_kwargs(captured: dict[str, Any], sim: Simulation, method_name: str):
    """Build a replacement that preserves the original signature so the
    schema-driven dispatcher binds the kwargs correctly."""
    import inspect
    from functools import wraps

    original = getattr(sim, method_name)

    @wraps(original)
    def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        # Bind positional args to parameter names for uniform capture
        sig = inspect.signature(original)
        bound = sig.bind_partial(*args, **kwargs)
        captured.clear()
        captured.update(bound.arguments)
        return {"status": "success", "content": [{"text": "ok"}]}

    return fake


class TestDispatcherForwardsPolicyConfig:
    """Nested ``policy_config`` routes verbatim to the method."""

    def test_run_policy_forwards_policy_config_as_single_dict(self, sim):
        captured: dict[str, Any] = {}
        cfg = {
            "observation_mapping": {
                "front": "video.front",
                "wrist": "video.wrist",
                "joint_position": "state.single_arm",
            },
            "action_mapping": {"action.single_arm": "joint_position"},
            "device": "mps",
        }
        with patch.object(sim, "run_policy", _capture_kwargs(captured, sim, "run_policy")):
            sim._dispatch_action(
                "run_policy",
                {
                    "robot_name": "so100",
                    "policy_provider": "mock",
                    "instruction": "pick up the red cube",
                    "duration": 3.0,
                    "policy_config": cfg,
                },
            )
        assert captured["robot_name"] == "so100"
        assert captured["policy_provider"] == "mock"
        assert captured["instruction"] == "pick up the red cube"
        assert captured["duration"] == 3.0
        # policy_config reaches the method as a single opaque dict
        assert captured["policy_config"] == cfg

    def test_eval_policy_forwards_policy_config(self, sim):
        captured: dict[str, Any] = {}
        cfg = {
            "pretrained_name_or_path": "lerobot/smolvla_base",
            "device": "mps",
            "trust_remote_code": True,
            "actions_per_step": 4,
        }
        with patch.object(sim, "eval_policy", _capture_kwargs(captured, sim, "eval_policy")):
            sim._dispatch_action(
                "eval_policy",
                {
                    "robot_name": "so100",
                    "policy_provider": "lerobot_local",
                    "n_episodes": 2,
                    "max_steps": 100,
                    "policy_config": cfg,
                },
            )
        assert captured["robot_name"] == "so100"
        assert captured["policy_provider"] == "lerobot_local"
        assert captured["n_episodes"] == 2
        assert captured["max_steps"] == 100
        assert captured["policy_config"] == cfg

    def test_start_policy_forwards_policy_config(self, sim):
        captured: dict[str, Any] = {}
        cfg = {
            "host": "localhost",
            "port": 5555,
            "api_token": "dummy-token",
            "observation_mapping": {"front": "video.front"},
            "action_mapping": {"action.single_arm": "joint_position"},
        }
        with patch.object(sim, "start_policy", _capture_kwargs(captured, sim, "start_policy")):
            sim._dispatch_action(
                "start_policy",
                {
                    "robot_name": "so100",
                    "policy_provider": "groot",
                    "instruction": "tidy the desk",
                    "policy_config": cfg,
                },
            )
        assert captured["policy_provider"] == "groot"
        assert captured["instruction"] == "tidy the desk"
        assert captured["policy_config"] == cfg


class TestDispatcherDropsUnknownTopLevelKeys:
    """Unknown top-level keys must be dropped silently — no ``**kwargs`` passthrough."""

    def test_run_policy_ignores_legacy_top_level_policy_kwargs(self, sim):
        """Old-shape top-level keys are simply not forwarded."""
        captured: dict[str, Any] = {}
        with patch.object(sim, "run_policy", _capture_kwargs(captured, sim, "run_policy")):
            sim._dispatch_action(
                "run_policy",
                {
                    "robot_name": "so100",
                    "policy_provider": "mock",
                    # These are no longer accepted at the top level:
                    "observation_mapping": {"x": "y"},
                    "device": "mps",
                    "pretrained_name_or_path": "lerobot/smolvla_base",
                },
            )
        assert captured["robot_name"] == "so100"
        assert captured["policy_provider"] == "mock"
        # Leaked legacy keys NOT forwarded
        assert "observation_mapping" not in captured
        assert "device" not in captured
        assert "pretrained_name_or_path" not in captured
        # policy_config defaults to None when not provided
        assert captured.get("policy_config") is None

    def test_non_policy_action_does_not_pick_up_unknown_kwargs(self, sim):
        captured: dict[str, Any] = {}

        def fake_set_gravity(gravity: list[float] | None = None) -> dict[str, Any]:
            captured["gravity"] = gravity
            return {"status": "success", "content": [{"text": "ok"}]}

        with patch.object(sim, "set_gravity", fake_set_gravity):
            sim._dispatch_action(
                "set_gravity",
                {"gravity": [0, 0, -9.81], "device": "mps", "policy_config": {}},
            )
        assert captured["gravity"] == [0, 0, -9.81]


class TestToolSpecIsClean:
    """tool_spec.json must advertise ``policy_config`` and NOT the old leaked keys."""

    def test_tool_spec_declares_policy_config(self):
        import json
        from pathlib import Path

        spec_path = Path(__file__).parent.parent / "strands_robots" / "simulation" / "mujoco" / "tool_spec.json"
        spec = json.loads(spec_path.read_text())
        props = spec["properties"]

        # policy_config must be present as an object
        assert "policy_config" in props, "tool_spec.json missing 'policy_config'"
        assert props["policy_config"]["type"] == "object"

        # Legacy top-level policy fields must NOT be advertised
        for leaked in (
            "observation_mapping",
            "action_mapping",
            "host",
            "port",
            "api_token",
            "policy_host",
            "policy_port",
            "pretrained_name_or_path",
            "trust_remote_code",
            "actions_per_step",
            "use_processor",
            "processor_overrides",
            "device",
            "model_path",
        ):
            assert leaked not in props, (
                f"tool_spec.json must not advertise top-level '{leaked}' — it belongs under policy_config"
            )
