"""Single-rollout reproducibility via ``run_policy(seed=...)``.

A single ``PolicyRunner.run`` / ``SimEngine.run_policy`` rollout drives a
stochastic policy (VLA action-chunk sampling, diffusion noise) from the
process-global RNG. Without an explicit seed, the same scene + policy produces
a different trajectory on every run - so a manipulation policy can grasp on one
run and miss on the next with no scene change. Multi-episode ``evaluate`` already
seeds per episode; this pins the same reproducibility contract for the
single-rollout path.

Regression for the no-grasp-on-re-run report: identical seed -> identical
trajectory; the seed is forwarded to ``policy.reset(seed=...)``.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy
from strands_robots.simulation.policy_runner import PolicyRunner

from .test_policy_runner import FakeSim


class StochasticPolicy(Policy):
    """Policy whose actions are drawn from the global Python/NumPy RNGs.

    Stands in for a real VLA whose action-chunk sampling is stochastic. The
    per-step draw is recorded so two rollouts can be compared bit-for-bit.
    Also records the seed passed to :meth:`reset` so the forwarding contract
    can be asserted.
    """

    def __init__(self) -> None:
        self.robot_state_keys: list[str] = []
        self.draws: list[float] = []
        self.reset_seeds: list[int | None] = []

    @property
    def provider_name(self) -> str:
        return "stochastic"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = robot_state_keys

    def reset(self, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        # Draw from BOTH global RNGs so the test exercises the python+numpy
        # seeding that set_eval_seed performs.
        draw = random.random() + float(np.random.random())
        self.draws.append(draw)
        return [{k: draw for k in self.robot_state_keys}]


def _run_once(seed: int | None) -> StochasticPolicy:
    sim = FakeSim()
    policy = StochasticPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
    result = PolicyRunner(sim).run(
        "fake_robot",
        policy,
        duration=0.5,
        control_frequency=10.0,  # -> 5 steps
        action_horizon=1,
        fast_mode=True,
        seed=seed,
    )
    assert result["status"] == "success"
    return policy


def test_same_seed_same_trajectory():
    """Two single rollouts at the same seed draw identical action sequences."""
    a = _run_once(seed=1234)
    b = _run_once(seed=1234)

    assert a.draws, "policy produced no action draws"
    assert a.draws == b.draws, (
        "run_policy(seed=...) is not reproducible: identical seed produced "
        f"different trajectories\n  run A: {a.draws}\n  run B: {b.draws}"
    )


def test_different_seed_different_trajectory():
    """Different seeds diverge - the seed actually drives sampling."""
    a = _run_once(seed=1234)
    b = _run_once(seed=5678)
    assert a.draws != b.draws


def test_seed_forwarded_to_policy_reset():
    """The master seed is forwarded to ``policy.reset(seed=...)`` so service-mode
    policies (e.g. a remote VLA server) can re-init their own RNG."""
    policy = _run_once(seed=42)
    assert 42 in policy.reset_seeds


def test_no_seed_leaves_reset_untouched():
    """Default (seed=None) preserves historical behaviour: reset is not forced."""
    policy = _run_once(seed=None)
    assert policy.reset_seeds == [], (
        f"run_policy with no seed should not call policy.reset(seed=...); got {policy.reset_seeds}"
    )
