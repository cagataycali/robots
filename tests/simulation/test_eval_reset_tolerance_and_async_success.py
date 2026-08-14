"""Per-episode reset tolerance and async-RTC success detection in ``evaluate``.

:meth:`PolicyRunner.evaluate` claims parity with the synchronous rollout twice,
in comments over two branches, and neither branch was driven:

* the per-episode ``policy.reset(seed=...)`` call is wrapped "best-effort, like
  every other ``reset`` call site", so a policy whose reset raises costs that
  episode its reseed and nothing else - but the warning that reports it had no
  test reference anywhere in the tree.
* the async-RTC arm checks the success criterion after each applied action and
  breaks, "mirrors the synchronous path". The module that owns
  ``async_rtc=True`` (:mod:`tests.simulation.test_eval_policy_async_rtc`) drives
  overlap, telemetry and step accounting but never supplies a success criterion,
  so the branch that detects success and stops the episode early has never run.

Both matter to a caller. A service-mode policy forwards ``reset`` over the wire
(see :meth:`strands_robots.policies.base.Policy.reset`), so a transport hiccup
reaches the tolerance on a real evaluation; and a chunk-emitting VLA evaluated
under the realistic inference latency ``async_rtc=True`` exists for is exactly
the policy whose success must still stop the episode when it arrives mid-chunk.

These tests pin the reported-and-continued contract, the early stop, and the
parity the comment asserts - the two paths agree on the same criterion.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from strands_robots.simulation.policy_runner import CooperativeStop, PolicyRunner

from .test_eval_policy_async_rtc import _CHUNK, _ChunkPolicy, _CountingSim

_SEED = 4242
_SUCCESS_AT = 6  # applied actions; mid-chunk for _CHUNK == 4
_MAX_STEPS = 40


class _ObservedCountSim(_CountingSim):
    """Reports the live applied-action count in the observation.

    The base double returns a constant observation, which cannot distinguish a
    live post-action read from a stale one held over from the chunk boundary.
    """

    def get_observation(self, robot_name: Any = None, *, skip_images: bool = False) -> dict[str, Any]:
        with self._lock:
            return {"applied": float(self.send_count)}


class _RaisingResetPolicy(_ChunkPolicy):
    """Chunk policy whose per-episode reset always fails.

    Models a service-mode policy whose reset is forwarded to a server: the call
    can fail for reasons unrelated to the rollout about to run.
    """

    def __init__(self, sim: _CountingSim) -> None:
        super().__init__(sim, chunk=_CHUNK, infer_sleep=0.0)
        self.reset_seeds: list[int | None] = []

    def reset(self, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)
        raise RuntimeError("policy reset endpoint unreachable")


class _RecordingResetPolicy(_ChunkPolicy):
    """Chunk policy whose per-episode reset succeeds, recording the seed."""

    def __init__(self, sim: _CountingSim) -> None:
        super().__init__(sim, chunk=_CHUNK, infer_sleep=0.0)
        self.reset_seeds: list[int | None] = []

    def reset(self, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)


class _StopOnResetPolicy(_ChunkPolicy):
    """Chunk policy whose per-episode reset raises a cooperative stop.

    Models an operator stopping the evaluation while the runner is between
    episodes: the request arrives as ``CooperativeStop``, which sits outside the
    ``Exception`` hierarchy precisely so a best-effort tolerance cannot eat it.
    """

    def __init__(self, sim: _CountingSim, chunk: int = 1) -> None:
        super().__init__(sim, chunk=chunk)
        self.resets = 0

    def reset(self, seed: int | None = None) -> None:
        self.resets += 1
        raise CooperativeStop("stop requested")


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    """Return the ``json`` block of an agent-tool envelope."""
    return next(block["json"] for block in result["content"] if "json" in block)


def _evaluate(policy: Any, sim: _CountingSim, **kwargs: Any) -> dict[str, Any]:
    """Drive ``evaluate`` on the double, defaulting the pacing knobs."""
    runner = PolicyRunner(sim)
    return runner.evaluate(
        "arm",
        policy,
        max_steps=kwargs.pop("max_steps", _MAX_STEPS),
        action_horizon=kwargs.pop("action_horizon", _CHUNK),
        control_frequency=kwargs.pop("control_frequency", 1000.0),
        **kwargs,
    )


def _reset_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Records reporting a failed per-episode policy reset."""
    return [r for r in caplog.records if "reset" in r.getMessage().lower() and r.levelno >= logging.WARNING]


class TestPerEpisodeResetIsBestEffort:
    """A policy whose reset raises loses its reseed, not its episode."""

    def test_a_raising_reset_is_reported_and_the_evaluation_continues(self, caplog: pytest.LogCaptureFixture) -> None:
        sim = _CountingSim()
        policy = _RaisingResetPolicy(sim)
        with caplog.at_level(logging.DEBUG, logger="strands_robots.simulation.policy_runner"):
            result = _evaluate(policy, sim, n_episodes=1, seed=_SEED)

        assert result["status"] == "success", result
        payload = _payload(result)
        assert payload["episodes_completed"] == 1, payload
        assert sim.send_count > 0, "the rollout did not run after the reset failed"

        reported = _reset_warnings(caplog)
        assert reported, "a failed per-episode reset left no record at all"
        message = reported[0].getMessage()
        assert "unreachable" in message, message
        # The reported seed is the per-episode seed derived from the master RNG,
        # not the caller's ``seed``, so the record names the reseed that was
        # actually attempted.
        assert policy.reset_seeds, "reset was never called"
        assert str(policy.reset_seeds[0]) in message, (message, policy.reset_seeds)

    def test_every_episode_is_attempted_when_reset_keeps_raising(self, caplog: pytest.LogCaptureFixture) -> None:
        sim = _CountingSim()
        policy = _RaisingResetPolicy(sim)
        with caplog.at_level(logging.DEBUG, logger="strands_robots.simulation.policy_runner"):
            result = _evaluate(policy, sim, n_episodes=3, seed=_SEED)

        payload = _payload(result)
        assert payload["episodes_completed"] == 3, payload
        assert len(policy.reset_seeds) == 3, policy.reset_seeds
        assert len(_reset_warnings(caplog)) == 3, "one failure per episode should be reported once each"
        # Each episode is reseeded from the master RNG rather than from the
        # caller's seed, so a failure in one episode cannot make two episodes
        # share a stream.
        assert len(set(policy.reset_seeds)) == 3, policy.reset_seeds
        assert _SEED not in policy.reset_seeds, policy.reset_seeds

    def test_a_clean_reset_is_called_and_reports_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        sim = _CountingSim()
        policy = _RecordingResetPolicy(sim)
        with caplog.at_level(logging.DEBUG, logger="strands_robots.simulation.policy_runner"):
            result = _evaluate(policy, sim, n_episodes=2, seed=_SEED)

        assert result["status"] == "success", result
        assert len(policy.reset_seeds) == 2, policy.reset_seeds
        assert all(isinstance(s, int) for s in policy.reset_seeds), policy.reset_seeds
        assert _reset_warnings(caplog) == [], "a clean reset must not report a failure"

    def test_a_cooperative_stop_from_reset_is_not_swallowed(self) -> None:
        """A stop request raised by ``reset`` must propagate, not be tolerated.

        The tolerance exists so a *seeding* failure costs only reproducibility.
        A stop request is not a seeding failure, and the narrow ``except
        Exception`` lets it through because the sentinel derives straight from
        ``BaseException`` -- so widening that clause would silently convert an
        operator's stop into a full-length evaluation.
        """
        assert not issubclass(CooperativeStop, Exception), (
            "the sentinel must stay outside the Exception hierarchy for the narrow reset tolerance to let it through"
        )
        sim = _CountingSim()
        policy = _StopOnResetPolicy(sim, chunk=1)
        result = PolicyRunner(sim).evaluate(
            robot_name="arm",
            policy=policy,
            n_episodes=3,
            max_steps=4,
            seed=7,
            success_fn=lambda _s: False,
        )
        payload = _payload(result)
        assert payload["stopped_early"] is True, (
            "a stop request from reset must end the evaluation, not be tolerated as a seeding failure"
        )
        assert sim.send_count == 0, "the stop landed before any action was applied"
        assert policy.resets == 1, "the runner must not go on to reset the next episode after a stop"


class TestAsyncRtcDetectsSuccessAndStopsEarly:
    """The async arm checks the criterion per applied action and breaks."""

    def test_async_rtc_records_the_success(self) -> None:
        sim = _ObservedCountSim()
        policy = _ChunkPolicy(sim, chunk=_CHUNK, infer_sleep=0.0)
        result = _evaluate(
            policy,
            sim,
            n_episodes=1,
            async_rtc=True,
            success_fn=lambda obs: obs["applied"] >= _SUCCESS_AT,
        )

        payload = _payload(result)
        assert payload["rtc_async_enabled"] is True, "the async arm silently ran synchronously"
        assert payload["success_rate"] == 1.0, payload
        assert payload["n_success"] == 1, payload
        assert payload["success_measured"] is True, payload

    def test_async_rtc_stops_the_episode_at_the_success(self) -> None:
        sim = _ObservedCountSim()
        policy = _ChunkPolicy(sim, chunk=_CHUNK, infer_sleep=0.0)
        result = _evaluate(
            policy,
            sim,
            n_episodes=1,
            async_rtc=True,
            success_fn=lambda obs: obs["applied"] >= _SUCCESS_AT,
        )

        payload = _payload(result)
        assert payload["rtc_async_enabled"] is True, "the async arm silently ran synchronously"
        assert payload["episodes"] == [{"episode": 0, "steps": _SUCCESS_AT, "success": True}], payload
        assert _SUCCESS_AT < _MAX_STEPS, "the fixture must succeed before the budget"
        assert sim.send_count == _SUCCESS_AT, sim.send_count

    def test_the_criterion_reads_the_live_post_action_observation(self) -> None:
        sim = _ObservedCountSim()
        policy = _ChunkPolicy(sim, chunk=_CHUNK, infer_sleep=0.0)
        seen: list[float] = []

        def check(obs: dict[str, Any]) -> bool:
            seen.append(obs["applied"])
            return obs["applied"] >= _SUCCESS_AT

        result = _evaluate(policy, sim, n_episodes=1, async_rtc=True, success_fn=check)

        payload = _payload(result)
        assert payload["rtc_async_enabled"] is True, "the async arm silently ran synchronously"
        assert payload["n_success"] == 1, payload
        assert seen == [float(i) for i in range(1, _SUCCESS_AT + 1)], seen


class TestTheTwoPathsAgreeOnOneCriterion:
    """The comment claims the async arm mirrors the synchronous path."""

    @pytest.mark.parametrize("async_rtc", [False, True], ids=["sync", "async"])
    def test_both_paths_stop_early_on_the_same_criterion(self, async_rtc: bool) -> None:
        sim = _ObservedCountSim()
        policy = _ChunkPolicy(sim, chunk=_CHUNK, infer_sleep=0.0)
        payload = _payload(
            _evaluate(
                policy,
                sim,
                n_episodes=1,
                async_rtc=async_rtc,
                success_fn=lambda obs: obs["applied"] >= _SUCCESS_AT,
            )
        )
        assert payload["rtc_async_enabled"] is async_rtc, payload
        assert payload["n_success"] == 1, payload
        assert payload["episodes"] == [{"episode": 0, "steps": _SUCCESS_AT, "success": True}], payload

    def test_the_two_paths_report_the_same_success_accounting(self) -> None:
        outcomes = {}
        for async_rtc in (False, True):
            sim = _ObservedCountSim()
            policy = _ChunkPolicy(sim, chunk=_CHUNK, infer_sleep=0.0)
            payload = _payload(
                _evaluate(
                    policy,
                    sim,
                    n_episodes=2,
                    async_rtc=async_rtc,
                    success_fn=lambda obs: obs["applied"] >= _SUCCESS_AT,
                )
            )
            assert payload["rtc_async_enabled"] is async_rtc, payload
            outcomes[async_rtc] = (
                payload["success_rate"],
                payload["n_success"],
                payload["avg_steps"],
                tuple(sorted(e["steps"] for e in payload["episodes"])),
            )
        assert outcomes[False] == outcomes[True], outcomes
