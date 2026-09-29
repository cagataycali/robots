"""Guard: ``run_policy(n_episodes=N)`` resets the POLICY at every episode boundary.

``SimEngine._run_policy_episodes`` resets the simulator between episodes, and
``PolicyRunner.run`` resets the policy only when a seed was given. An unseeded
multi-episode recording with a provider that keeps an observation history or an
action queue (flux3_action, groot, any RTC policy) therefore started episode N+1
conditioned on episode N's frames while the scene had jumped back to rest. On
the SO-101 a 10-episode flux3_action recording drifted its ``shoulder_lift``
command by a further ~5 rad per episode until 4400 of 4500 ticks were clamped at
the joint limit; a second recording made after ``PolicyRunner.evaluate`` alone
was fixed showed the identical monotonic drift, because the recorder goes
through ``run_policy``, not ``evaluate``.

Graded against a counting policy on a real engine: N episodes must see the
policy reset at least N - 1 times (once per boundary), seeded or not.
"""

from __future__ import annotations

from typing import Any

import pytest

import strands_robots
from strands_robots.policies.mock import MockPolicy

_EPISODES = 3


class _CountingPolicy(MockPolicy):
    """Mock policy that records every ``reset`` it receives."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.resets: list[int | None] = []

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.resets.append(seed)


@pytest.mark.parametrize("seed", [None, 7], ids=["unseeded", "seeded"])
def test_every_episode_boundary_resets_the_policy(seed: int | None) -> None:
    sim = strands_robots.Robot("so101", mode="sim", mesh=False)
    policy = _CountingPolicy()
    try:
        result = sim.run_policy(
            robot_name="so101",
            policy_object=policy,
            instruction="collect",
            n_steps=4,
            control_frequency=30.0,
            fast_mode=True,
            n_episodes=_EPISODES,
            reset_between=True,
            seed=seed,
        )
    finally:
        sim.cleanup()
    assert result["status"] == "success", result
    boundaries = _EPISODES - 1
    assert len(policy.resets) >= boundaries, (
        f"run_policy(n_episodes={_EPISODES}, seed={seed}) reset the policy {len(policy.resets)} time(s); "
        f"expected at least one reset per episode boundary ({boundaries}). A history-keeping "
        "provider otherwise conditions episode N+1 on episode N."
    )
    if seed is not None:
        # Seeded: the boundary resets carry the next episode's seed (seed + ep).
        assert {seed + ep for ep in range(1, _EPISODES)} <= set(policy.resets)


def test_reset_between_false_does_not_reset_the_policy_between_episodes() -> None:
    """``reset_between=False`` is the caller asking for continuity; honour it on the policy too."""
    sim = strands_robots.Robot("so101", mode="sim", mesh=False)
    policy = _CountingPolicy()
    try:
        result = sim.run_policy(
            robot_name="so101",
            policy_object=policy,
            instruction="collect",
            n_steps=4,
            control_frequency=30.0,
            fast_mode=True,
            n_episodes=_EPISODES,
            reset_between=False,
        )
    finally:
        sim.cleanup()
    assert result["status"] == "success", result
    assert policy.resets == []
