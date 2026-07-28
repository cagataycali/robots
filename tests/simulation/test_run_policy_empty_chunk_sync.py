# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The SYNCHRONOUS rollout loop must not hang on an empty action chunk.

``run()``'s synchronous branch is ``while step_count < total_steps:`` ->
``chunk = _query_chunk(observation)`` -> ``for action_dict in chunk:``. When the
policy returns ``[]`` the ``for`` body never executes, ``step_count`` never
increments, and the ``while`` condition never changes - so the loop re-queries the
policy at full speed forever. Measured pre-fix: ``alive_after_5s=True`` with
**59095** ``get_actions`` calls, and via the public surface ``run_policy`` reached
**69519** calls without returning.

This is the branch a SINGLE-STEP policy lands on: ``async_rtc=None`` (the default)
resolves from ``is_chunk_emitting()``, so MockPolicy, a classical planner, or any
``actions_per_step=1`` provider takes it. Empty chunks are a real transient, not a
hypothetical - ``return []`` appears in ``curobo/_next_chunk`` (cached trajectory
exhausted), ``moveit2/_unpack_trajectory`` (empty trajectory) and groot at two
sites.

Every sibling path already guarded this: the ASYNC branch of the same method,
``_ChunkPipeline._iter_sync``, legacy ``evaluate()``, ``_evaluate_with_spec`` and
``run_multi_policy``. Only the synchronous branch did not - and the existing
empty-chunk tests
(``test_policy_runner_async_rtc_empty_chunk.py``) all drive ``async_rtc=True``, so
the guard on this branch was untested.

Every test here is wall-clock bounded, so a regression FAILS rather than wedging
the suite.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

_JOINTS = [str(i) for i in range(1, 7)]
_DEADLINE_S = 15.0


class _EmptyChunkPolicy(Policy):
    """Always returns an empty chunk - the transient that caused the hang."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "empty"

    def set_robot_state_keys(self, keys) -> None:
        pass

    async def get_actions(self, observation, instruction, **kwargs):
        self.calls += 1
        return []


class _EventuallyEmptyPolicy(_EmptyChunkPolicy):
    """Returns one good chunk, then goes empty (a mid-rollout transient)."""

    async def get_actions(self, observation, instruction, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return [dict.fromkeys(_JOINTS, 0.0)]
        return []


def _sim() -> MuJoCoSimEngine:
    sim = MuJoCoSimEngine()
    sim.create_world()
    assert sim.add_robot("so101")["status"] == "success"
    return sim


def _run_bounded(call) -> dict:
    """Run ``call`` on a thread and fail if it does not return in time.

    A bare call would hang the whole suite on a regression; this converts the hang
    into an assertion failure that names it.
    """
    box: dict = {}

    def target() -> None:
        try:
            box["result"] = call()
        except Exception as exc:  # noqa: BLE001 - surfaced via the box
            box["result"] = {"status": "raised", "exc": f"{type(exc).__name__}: {exc}"}

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=_DEADLINE_S)
    assert not thread.is_alive(), (
        f"the synchronous rollout did not return within {_DEADLINE_S}s - it is re-querying the policy "
        f"forever on an empty chunk (the D14 hang)"
    )
    return box["result"]


def _text(result: dict) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []))


class TestSynchronousEmptyChunk:
    def test_run_policy_returns_an_error_instead_of_hanging(self):
        """Regression: this reached 69519 get_actions calls without returning."""
        sim = _sim()
        policy = _EmptyChunkPolicy()
        try:
            result = _run_bounded(
                lambda: sim.run_policy(policy_object=policy, robot_name="so101", n_steps=8, control_frequency=50.0)
            )

            assert result["status"] == "error", result
            assert "empty action chunk" in _text(result)
        finally:
            sim.destroy()

    def test_the_policy_is_not_queried_in_a_hot_loop(self):
        """The guard must fire on the FIRST empty chunk, not after N retries."""
        sim = _sim()
        policy = _EmptyChunkPolicy()
        try:
            _run_bounded(
                lambda: sim.run_policy(policy_object=policy, robot_name="so101", n_steps=8, control_frequency=50.0)
            )

            assert policy.calls <= 2, f"policy was re-queried {policy.calls} times"
        finally:
            sim.destroy()

    def test_a_mid_rollout_empty_chunk_also_stops(self):
        """The transient shape: a policy that works, then exhausts its trajectory."""
        sim = _sim()
        policy = _EventuallyEmptyPolicy()
        try:
            result = _run_bounded(
                lambda: sim.run_policy(policy_object=policy, robot_name="so101", n_steps=64, control_frequency=50.0)
            )

            assert result["status"] == "error", result
            assert "empty action chunk" in _text(result)
        finally:
            sim.destroy()

    def test_the_runner_path_is_covered_too(self):
        """Assert on the branch directly, with async_rtc pinned False.

        ``run_policy`` resolves ``async_rtc`` from ``is_chunk_emitting()``; pinning
        it False guarantees the SYNCHRONOUS branch is the one under test rather
        than relying on that resolution.
        """
        from strands_robots.simulation.policy_runner import PolicyRunner

        sim = _sim()
        policy = _EmptyChunkPolicy()
        try:
            result = _run_bounded(
                lambda: PolicyRunner(sim).run(
                    "so101",
                    policy,
                    instruction="",
                    n_steps=8,
                    control_frequency=50.0,
                    async_rtc=False,
                )
            )

            assert result["status"] == "error", result
            assert "empty action chunk" in _text(result)
        finally:
            sim.destroy()

    def test_error_text_is_plain_ascii(self):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        sim = _sim()
        try:
            result = _run_bounded(
                lambda: sim.run_policy(policy_object=_EmptyChunkPolicy(), robot_name="so101", n_steps=8)
            )

            assert _text(result).isascii()
        finally:
            sim.destroy()


class TestNonEmptyChunksAreUnaffected:
    def test_a_normal_policy_still_completes(self):
        class _Hold(Policy):
            @property
            def provider_name(self) -> str:
                return "hold"

            def set_robot_state_keys(self, keys) -> None:
                pass

            async def get_actions(self, observation, instruction, **kwargs):
                return [dict.fromkeys(_JOINTS, 0.0)]

        sim = _sim()
        try:
            result = _run_bounded(
                lambda: sim.run_policy(policy_object=_Hold(), robot_name="so101", n_steps=8, control_frequency=50.0)
            )

            assert result["status"] == "success", result
        finally:
            sim.destroy()
