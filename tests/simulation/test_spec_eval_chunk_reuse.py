# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Spec-based eval must reuse the action chunk, not re-infer for every step.

``_evaluate_with_spec``'s outer loop was ``for _ in range(max_steps):``. The inner
loop DOES drain ``actions[:_chunk]`` and increments ``steps`` per applied action -
but the outer loop iterated ``max_steps`` times regardless, re-running
``get_observation`` + ``get_actions`` every time. So N steps cost N inferences and
every action but the first of each chunk was discarded.

Measured pre-fix with ``chunk=4``, ``max_steps=20``: **20** inference calls where 5
suffice. On a SmolVLA-realistic ``chunk=50`` at 80 ms it was 294 of 300 inferences
wasted - 23.5 s of a 28.5 s episode.

It also silently reinstated the closed-loop ``action_horizon=1`` behaviour that this
method's own docstring records as "a contributing factor to ``success_rate=0``",
because only the FIRST action of each chunk was ever applied to a fresh
observation - the opposite of the open-loop chunk replay chunk-emitting models are
trained for.

The sibling loops already bound on applied steps (legacy ``evaluate()`` uses
``while steps < max_steps``).
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402
from strands_robots.simulation.policy_runner import PolicyRunner  # noqa: E402

_JOINTS = [str(i) for i in range(1, 7)]


class _CountingPolicy(Policy):
    """Emits a fixed-length chunk and counts how often it is queried."""

    def __init__(self, chunk: int = 4, empty: bool = False) -> None:
        super().__init__()
        self.calls = 0
        self._chunk = chunk
        self._empty = empty

    @property
    def provider_name(self) -> str:
        return "counting"

    def set_robot_state_keys(self, keys) -> None:
        pass

    @property
    def actions_per_step(self) -> int:
        return self._chunk

    def is_chunk_emitting(self) -> bool:
        return True

    @property
    def execution_horizon(self) -> int:
        return self._chunk

    async def get_actions(self, observation, instruction, **kwargs):
        self.calls += 1
        if self._empty:
            return []
        return [dict.fromkeys(_JOINTS, 0.0) for _ in range(self._chunk)]


class _Spec:
    """Minimal BenchmarkProtocol implementation."""

    name = "test_spec"

    def __init__(self, max_steps: int = 20, succeed_after: int | None = None) -> None:
        self.max_steps = max_steps
        self.on_step_calls = 0
        self._succeed_after = succeed_after

    def augment_observation(self, sim, obs):
        return obs

    def on_episode_start(self, sim, episode) -> None:
        pass

    def on_step(self, sim, obs, action):
        from strands_robots.benchmarks.libero.adapter import StepInfo

        self.on_step_calls += 1
        return StepInfo(reward=0.0, done=False)

    def reset(self, sim, episode) -> None:
        pass

    def is_success(self, sim) -> bool:
        return self._succeed_after is not None and self.on_step_calls >= self._succeed_after

    def is_failure(self, sim) -> bool:
        return False

    def on_episode_end(self, sim, episode) -> None:
        pass


def _sim() -> MuJoCoSimEngine:
    sim = MuJoCoSimEngine()
    sim.create_world()
    assert sim.add_robot("so101")["status"] == "success"
    return sim


def _evaluate(sim, policy, spec, action_horizon: int = 8) -> dict:
    return PolicyRunner(sim)._evaluate_with_spec(
        "so101", policy, spec, instruction="", n_episodes=1, seed=0, action_horizon=action_horizon
    )


class TestChunkIsReused:
    def test_one_inference_feeds_a_whole_chunk(self):
        """Regression: 20 steps with chunk=4 cost 20 inferences, not 5."""
        sim = _sim()
        policy = _CountingPolicy(chunk=4)
        spec = _Spec(max_steps=20)
        try:
            result = _evaluate(sim, policy, spec)

            assert result["status"] == "success", result
            assert policy.calls == 5, f"expected 5 inferences for 20 steps at chunk=4, got {policy.calls}"
            # Every step still runs the spec hook: the fix removes wasted
            # INFERENCE, not applied actions.
            assert spec.on_step_calls == 20
        finally:
            sim.destroy()

    @pytest.mark.parametrize("chunk,expected_calls", [(1, 20), (2, 10), (4, 5), (5, 4), (20, 1)])
    def test_inference_count_scales_with_the_chunk(self, chunk, expected_calls):
        sim = _sim()
        policy = _CountingPolicy(chunk=chunk)
        spec = _Spec(max_steps=20)
        try:
            _evaluate(sim, policy, spec)

            assert policy.calls == expected_calls
            assert spec.on_step_calls == 20
        finally:
            sim.destroy()

    def test_a_chunk_longer_than_max_steps_needs_one_inference(self):
        """The inner loop's `steps >= max_steps` break must end the episode."""
        sim = _sim()
        policy = _CountingPolicy(chunk=50)
        spec = _Spec(max_steps=10)
        try:
            _evaluate(sim, policy, spec)

            assert policy.calls == 1
            assert spec.on_step_calls == 10
        finally:
            sim.destroy()

    def test_action_horizon_one_is_still_closed_loop(self):
        """action_horizon=1 must remain per-step inference when asked for.

        The defect made every chunk behave this way; the CONFIGURED behaviour must
        still be available. resolve_chunk_length gives a chunk-emitting policy its
        full trained chunk, so this asserts the non-chunk-emitting case.
        """

        class _SingleStep(_CountingPolicy):
            def is_chunk_emitting(self) -> bool:
                return False

            @property
            def execution_horizon(self) -> int:
                return 1

            @property
            def actions_per_step(self) -> int:
                return 1

        sim = _sim()
        policy = _SingleStep(chunk=1)
        spec = _Spec(max_steps=12)
        try:
            _evaluate(sim, policy, spec, action_horizon=1)

            assert policy.calls == 12
        finally:
            sim.destroy()


class TestEarlyTerminationStillWorks:
    def test_success_ends_the_episode_early(self):
        sim = _sim()
        policy = _CountingPolicy(chunk=4)
        spec = _Spec(max_steps=100, succeed_after=8)
        try:
            _evaluate(sim, policy, spec)

            assert spec.on_step_calls < 100, "early termination did not fire"
            # And the inference count is bounded by the steps actually taken.
            assert policy.calls <= -(-spec.on_step_calls // 4) + 1
        finally:
            sim.destroy()


class TestDegeneratePolicyStillTerminates:
    def test_an_empty_chunk_policy_does_not_spin_forever(self):
        """The outer loop now bounds on `steps`, so the empty branch must count one.

        With the old `for _ in range(max_steps)` form, stepping physics was enough
        to terminate. Under `while steps < max_steps` it is not - an empty-chunk
        policy would re-query forever (the D14 hang, in this method).
        """
        sim = _sim()
        policy = _CountingPolicy(chunk=4, empty=True)
        spec = _Spec(max_steps=20)
        box: dict = {}

        def target() -> None:
            try:
                box["result"] = _evaluate(sim, policy, spec)
            except Exception as exc:  # noqa: BLE001
                box["result"] = {"status": "raised", "exc": f"{type(exc).__name__}: {exc}"}

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=30.0)
        try:
            assert not thread.is_alive(), "spec eval spun forever on an empty-chunk policy"
            # Bounded by max_steps: one query + one counted step per iteration.
            assert policy.calls <= spec.max_steps
        finally:
            sim.destroy()
