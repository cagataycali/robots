"""Behavioural tests for PolicyRunner — run/replay/evaluate with a mock policy."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner, VideoConfig, _resolve_coroutine


@pytest.fixture
def sim_with_robot():
    s = Simulation(tool_name="pr_test", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    yield s
    s.cleanup()


# -- run() ----------------------------------------------------------------


class TestPolicyRunnerRun:
    def test_run_returns_success(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)
        result = runner.run(
            "alice",
            policy,
            duration=0.1,
            control_frequency=50,
            fast_mode=True,
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "alice" in text

    def test_run_invokes_on_frame_hook(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

        calls: list[int] = []

        def on_frame(step: int, obs: dict, action: dict) -> None:
            calls.append(step)

        runner = PolicyRunner(sim_with_robot)
        runner.run(
            "alice",
            policy,
            duration=0.04,
            control_frequency=50,
            fast_mode=True,
            on_frame=on_frame,
        )
        assert calls, "on_frame should fire at least once"
        # Step indices must be non-decreasing.
        assert calls == sorted(calls)


# -- evaluate() ----------------------------------------------------------------


class TestPolicyRunnerEvaluate:
    def test_evaluate_default_success_fn(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)

        result = runner.evaluate(
            "alice",
            policy,
            n_episodes=2,
            max_steps=5,
            success_fn=None,
        )
        assert result["status"] == "success"
        payload = result["content"][-1]["json"]
        assert payload["n_episodes"] == 2
        assert payload["max_steps"] == 5
        assert 0 <= payload["success_rate"] <= 1
        assert len(payload["episodes"]) == 2

    def test_evaluate_unknown_success_fn_errors(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)
        result = runner.evaluate(
            "alice",
            policy,
            n_episodes=1,
            max_steps=2,
            success_fn="__nope__",
        )
        assert result["status"] == "error"


# -- require_default_robot / _maybe_sim_time ----------------------------------


class TestHelpers:
    def test_maybe_sim_time_reads_state(self, sim_with_robot):
        runner = PolicyRunner(sim_with_robot)
        t = runner._maybe_sim_time()
        # Empty sim at t=0 should return 0.0.
        assert t == pytest.approx(0.0, abs=1e-9)

    def test_maybe_sim_time_on_broken_sim_returns_none(self):
        fake = MagicMock()
        fake.get_state.side_effect = RuntimeError("boom")
        runner = PolicyRunner(fake)
        assert runner._maybe_sim_time() is None

    def test_maybe_sim_time_no_get_state_returns_none(self):
        fake = object()
        runner = PolicyRunner(fake)  # type: ignore[arg-type]
        assert runner._maybe_sim_time() is None

    def test_require_default_robot_empty_raises(self):
        fake = MagicMock()
        fake.list_robots.return_value = []
        runner = PolicyRunner(fake)
        with pytest.raises(ValueError, match="No robots"):
            runner._require_default_robot()

    def test_require_default_robot_returns_first(self):
        fake = MagicMock()
        fake.list_robots.return_value = ["alpha", "beta"]
        runner = PolicyRunner(fake)
        assert runner._require_default_robot() == "alpha"


# -- replay() error paths (no lerobot → clean error) -------------------------


class TestReplayErrorPaths:
    def test_replay_missing_lerobot_clean_error(self, sim_with_robot, monkeypatch):
        """When lerobot isn't importable, replay returns a friendly error
        instead of propagating ImportError to the caller."""

        def _boom(*a, **kw):
            raise ImportError("no lerobot")

        # Patch the lazy import inside replay().
        import builtins

        real_import = builtins.__import__

        def _patched_import(name, *args, **kwargs):
            if name.startswith("strands_robots.dataset_recorder"):
                raise ImportError("no lerobot (test-forced)")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _patched_import)

        runner = PolicyRunner(sim_with_robot)
        result = runner.replay(
            repo_id="fake/ds",
            robot_name="alice",
            episode=0,
        )
        assert result["status"] == "error"
        assert "lerobot" in result["content"][0]["text"].lower()


# -- coroutine resolver ------------------------------------------------------


class TestResolveCoroutine:
    def test_passthrough_for_plain_list(self):
        assert _resolve_coroutine([{"j": 0.1}]) == [{"j": 0.1}]

    def test_awaits_coroutine(self):
        async def inner():
            return [{"j": 0.2}]

        assert _resolve_coroutine(inner()) == [{"j": 0.2}]


# -- VideoConfig -----------------------------------------------------------


class TestVideoConfig:
    def test_enabled_with_path(self):
        v = VideoConfig(path="/tmp/x.mp4", fps=30)
        assert v.enabled is True

    def test_disabled_without_path(self):
        v = VideoConfig()
        assert v.enabled is False
