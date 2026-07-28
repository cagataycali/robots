# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Spec-based eval must declare the RTC observed delay, not let it be estimated.

``_evaluate_with_spec`` called ``policy.get_actions(...)`` without ever calling
``policy.set_rtc_observed_delay(...)``, so ``Policy.rtc_observed_delay_steps``
stayed ``None`` for the whole eval. ``LerobotLocalPolicy._run_rtc_inference`` then
selects its ``_estimate_inference_delay(fps=...)`` WALL-CLOCK branch, which that
code's own comment calls "non-reproducible - it warms up within an episode and
varies run-to-run, so two otherwise-identical seeded episodes drift apart at the
seam".

Measured across the three eval paths, one policy recording the value per inference::

    _evaluate_with_spec : [None, None, None, None]
    evaluate (legacy)   : [0, 0, 0, 0]
    run (synchronous)   : [0, 0, 0, 0]

This matters most on THIS path of all of them: it is the one the codebase
advertises as reproducible - ``evaluate`` rejects ``async_rtc=True`` when a spec is
set for "bit-stable reproducibility", and each episode does its own
``set_eval_seed``. The world is PAUSED during inference here, so the exact answer
is the trivially known ``0``; an estimated 4 steps at 50Hz would feed lerobot's
``get_prefix_weights`` the wrong freeze length.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.benchmarks.libero.adapter import StepInfo  # noqa: E402
from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402
from strands_robots.simulation.policy_runner import PolicyRunner  # noqa: E402

_JOINTS = [str(i) for i in range(1, 7)]


class _DelayProbe(Policy):
    """Records ``rtc_observed_delay_steps`` as seen at each inference."""

    def __init__(self, chunk: int = 4) -> None:
        super().__init__()
        self._chunk = chunk
        self.seen: list[int | None] = []
        self.chunk_lengths: list[int] = []

    @property
    def provider_name(self) -> str:
        return "probe"

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
        self.seen.append(self.rtc_observed_delay_steps)
        return [dict.fromkeys(_JOINTS, 0.0) for _ in range(self._chunk)]


class _Spec:
    name = "delay_spec"

    def __init__(self, max_steps: int = 16) -> None:
        self.max_steps = max_steps
        self.on_step_calls = 0
        self.instruction = ""

    def augment_observation(self, sim, obs):
        return obs

    def on_episode_start(self, sim, episode) -> None:
        pass

    def on_step(self, sim, obs, action):
        self.on_step_calls += 1
        return StepInfo(reward=0.0, done=False)

    def reset(self, sim, episode) -> None:
        pass

    def is_success(self, sim) -> bool:
        return False

    def is_failure(self, sim) -> bool:
        return False

    def on_episode_end(self, sim, episode) -> None:
        pass


def _sim() -> MuJoCoSimEngine:
    sim = MuJoCoSimEngine()
    sim.create_world()
    assert sim.add_robot("so101")["status"] == "success"
    return sim


class TestObservedDelayIsDeclared:
    def test_every_inference_sees_zero_not_none(self):
        """Regression: the value stayed None for the whole eval."""
        sim = _sim()
        policy = _DelayProbe()
        try:
            PolicyRunner(sim)._evaluate_with_spec(
                "so101", policy, _Spec(), instruction="", n_episodes=1, seed=0, action_horizon=8
            )

            assert policy.seen, "the policy was never queried"
            # A None anywhere means that inference fell back to the
            # non-reproducible wall-clock estimate.
            assert None not in policy.seen
            assert all(value == 0 for value in policy.seen), policy.seen
        finally:
            sim.destroy()

    def test_it_matches_the_legacy_evaluate_path(self):
        """The three eval paths must not disagree about a known-exact quantity."""
        sim = _sim()
        spec_probe = _DelayProbe()
        legacy_probe = _DelayProbe()
        try:
            runner = PolicyRunner(sim)
            runner._evaluate_with_spec(
                "so101", spec_probe, _Spec(), instruction="", n_episodes=1, seed=0, action_horizon=8
            )
            runner.evaluate("so101", legacy_probe, n_episodes=1, max_steps=16, action_horizon=8)

            assert set(spec_probe.seen) == set(legacy_probe.seen) == {0}
        finally:
            sim.destroy()

    def test_it_holds_across_multiple_episodes(self):
        """Per-episode reseeding must not reintroduce the estimate branch."""
        sim = _sim()
        policy = _DelayProbe()
        try:
            PolicyRunner(sim)._evaluate_with_spec(
                "so101", policy, _Spec(max_steps=8), instruction="", n_episodes=3, seed=0, action_horizon=8
            )

            assert len(policy.seen) >= 3
            assert all(value == 0 for value in policy.seen), policy.seen
        finally:
            sim.destroy()


class TestChunkTruncationHasOneSourceOfTruth:
    def test_the_chunk_is_truncated_once_by_the_helper(self):
        """The helper applies resolve_chunk_length; the loop must not re-slice.

        Two independent truncations of the same bound is how they drift apart.
        """
        sim = _sim()
        # A policy whose chunk EXCEEDS the horizon: resolve_chunk_length gives a
        # chunk-emitting policy its full trained chunk, so all 6 must be applied.
        policy = _DelayProbe(chunk=6)
        spec = _Spec(max_steps=12)
        try:
            PolicyRunner(sim)._evaluate_with_spec(
                "so101", policy, spec, instruction="", n_episodes=1, seed=0, action_horizon=2
            )

            assert spec.on_step_calls == 12
            # 12 steps at a 6-action chunk is exactly 2 inferences.
            assert len(policy.seen) == 2, policy.seen
        finally:
            sim.destroy()

    def test_max_steps_still_bounds_a_long_chunk(self):
        sim = _sim()
        policy = _DelayProbe(chunk=50)
        spec = _Spec(max_steps=7)
        try:
            PolicyRunner(sim)._evaluate_with_spec(
                "so101", policy, spec, instruction="", n_episodes=1, seed=0, action_horizon=8
            )

            assert spec.on_step_calls == 7
            assert len(policy.seen) == 1
        finally:
            sim.destroy()
