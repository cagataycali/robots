"""Each hardware task starts the policy from a clean per-episode state.

``Policy.reset`` exists to clear the state that must not cross an episode
boundary - action chunk caches, sampler RNG, KV-caches (see its docstring in
MODULE ``strands_robots.policies.base``). The sim runner calls it once per
episode in MODULE ``strands_robots.simulation.policy_runner``; these tests pin
that the hardware control loop does the same, because there the leftover
actions of the previous task are commanded to a physical arm.

Both hardware entry points - ``run_policy(policy_object=...)`` and the
provider-backed ``start_task`` - funnel through ``_execute_task_async``, so the
reset is asserted through each of them.
"""

from __future__ import annotations

import logging
from typing import Any

from tests.test_hardware_robot_lifecycle import _FakeLeRobot, _make_robot


class _ChunkCachingPolicy:
    """A VLA-shaped policy: one inference per episode, chunk cached after that.

    ``reset()`` drops the cache, which is the whole contract under test. The
    cached chunk is derived from the instruction, so an action carrying the
    wrong value is proof that a previous task's chunk was replayed.
    """

    def __init__(self) -> None:
        self._chunk: list[dict[str, Any]] | None = None
        self.reset_calls = 0
        self.actions_at_reset: list[int] = []
        self.driven: list[tuple[str, float]] = []
        self.robot: _FakeLeRobot | None = None

    def set_robot_state_keys(self, keys: list[str]) -> None:
        pass

    def set_control_frequency(self, hz: float) -> None:
        pass

    def set_rtc_observed_delay(self, steps: int | None) -> None:
        pass

    def reset(self, seed: int | None = None) -> None:
        self.reset_calls += 1
        self._chunk = None
        if self.robot is not None:
            # How many actions had already reached the arm when reset ran.
            self.actions_at_reset.append(len(self.robot.sent_actions))

    async def get_actions(self, observation: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
        if self._chunk is None:
            # Inference happens once; every later step consumes the cache.
            self._chunk = [{"j0.pos": 1.0 if instruction == "stack the cups" else 9.0}]
        self.driven.append((instruction, self._chunk[0]["j0.pos"]))
        return [dict(self._chunk[0])]


class _ResetRaisesPolicy(_ChunkCachingPolicy):
    """A policy whose per-episode reset fails."""

    def reset(self, seed: int | None = None) -> None:
        self.reset_calls += 1
        raise RuntimeError("remote reset endpoint unreachable")


def _run_two_tasks(hw: Any, policy: Any) -> None:
    """Drive one policy object through two tasks with different instructions."""
    hw.run_policy(policy_object=policy, instruction="stack the cups", duration=5.0, n_steps=2)
    hw.run_policy(policy_object=policy, instruction="open the drawer", duration=5.0, n_steps=2)


class TestPerTaskPolicyReset:
    def test_a_second_task_does_not_drive_the_first_tasks_cached_chunk(self):
        """The defect: a reused policy object replayed the previous instruction.

        Driving one object through two tasks is the documented
        ``run_policy(policy_object=...)`` usage. Without a per-task reset the
        cached chunk inferred for task one is commanded to the arm during task
        two, so the arm executes actions for an instruction nobody asked for.
        """
        fake = _FakeLeRobot()
        hw = _make_robot(fake)
        hw.action_horizon = 1
        policy = _ChunkCachingPolicy()

        _run_two_tasks(hw, policy)

        second_task = {value for instruction, value in policy.driven if instruction == "open the drawer"}
        assert second_task == {9.0}, f"second task drove the first task's chunk: {policy.driven}"
        hw.cleanup()

    def test_reset_runs_once_per_task_before_any_action_reaches_the_arm(self):
        """One clear per task, and it precedes the first commanded action.

        A reset issued after the loop has started would leave the first
        actions of the task drawn from stale state, which is the failure this
        guards against - so the ordering is part of the contract, not just the
        call count.
        """
        fake = _FakeLeRobot()
        hw = _make_robot(fake)
        hw.action_horizon = 1
        policy = _ChunkCachingPolicy()
        policy.robot = fake

        _run_two_tasks(hw, policy)

        assert policy.reset_calls == 2
        # Task one resets with nothing sent; task two resets with only task
        # one's actions on the bus - never mid-task.
        assert policy.actions_at_reset == [0, 2]
        hw.cleanup()

    def test_a_policy_whose_reset_raises_is_still_driven(self, caplog):
        """Fail-soft, matching the sim runner: a failed reset never aborts a task.

        A policy that forwards ``reset`` to a remote inference server can fail
        for reasons unrelated to the task. Refusing to run would strand the
        operator, so the task proceeds and the warning names the stale state.
        """
        fake = _FakeLeRobot()
        hw = _make_robot(fake)
        hw.action_horizon = 1
        policy = _ResetRaisesPolicy()

        with caplog.at_level(logging.WARNING):
            hw.run_policy(policy_object=policy, instruction="stack the cups", duration=5.0, n_steps=2)

        assert policy.reset_calls == 1
        assert fake.sent_actions, "a failed reset must not stop the task from running"
        assert "possibly stale per-episode state" in caplog.text
        hw.cleanup()

    def test_the_provider_backed_start_task_path_also_resets(self, monkeypatch):
        """``start_task`` builds its policy from a provider, and resets it too.

        Both entry points share ``_execute_task_async``; this pins that the
        reset is on the shared path rather than only on the object path, so a
        server-backed provider cannot carry state between tasks either.
        """
        fake = _FakeLeRobot()
        hw = _make_robot(fake)
        hw.action_horizon = 1
        policy = _ChunkCachingPolicy()

        async def _fake_get_policy(port, host, provider):
            return policy

        monkeypatch.setattr(hw, "_get_policy", _fake_get_policy)

        hw._execute_task_sync("stack the cups", 5555, "localhost", "cosmos3", 5.0)

        assert policy.reset_calls == 1
        assert fake.sent_actions
        hw.cleanup()
