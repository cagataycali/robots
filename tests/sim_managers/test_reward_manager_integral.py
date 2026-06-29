"""Integral test: a full manager-driven rollout loop with a closed-loop reward.

Exercises the four managers together over a synthetic episode and asserts the
emergent behaviour reviewers care about: a perfectly tracking agent accrues
more reward than a stationary one, the per-term breakdown sums to the total,
and dt-scaling makes total reward invariant to control frequency.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.sim_managers import EnvState, build_managers

_CONFIG = {
    "command_manager": {
        "terms": [
            {
                "name": "base_velocity",
                "func": "uniform_velocity",
                "params": {"lin_vel_x": [0.5, 0.5], "lin_vel_y": [0.0, 0.0], "ang_vel_z": [0.0, 0.0]},
            }
        ]
    },
    "observation_manager": {
        "terms": [
            {"func": "base_lin_vel"},
            {"func": "base_ang_vel"},
            {"func": "projected_gravity"},
            {"func": "joint_pos"},
            {"func": "joint_vel"},
            {"func": "last_action"},
            {"func": "velocity_commands"},
        ]
    },
    "reward_manager": {
        "terms": [
            {"name": "track_lin", "func": "track_lin_vel_xy_exp", "weight": 1.0, "params": {"std": 0.25}},
            {"name": "track_ang", "func": "track_ang_vel_z_exp", "weight": 0.5, "params": {"std": 0.25}},
            {"name": "lin_vel_z", "func": "lin_vel_z_l2", "weight": -2.0},
            {"name": "action_rate", "func": "action_rate_l2", "weight": -0.01},
            {"name": "alive", "func": "alive", "weight": 0.25},
        ]
    },
    "termination_manager": {
        "terms": [
            {"func": "time_out"},
            {"func": "bad_orientation", "params": {"limit_angle": 1.0}},
        ]
    },
}


def _rollout(track_command: bool, n_steps: int = 50, dt: float = 0.02) -> float:
    ms = build_managers(_CONFIG)
    assert ms.command and ms.observation and ms.reward and ms.termination
    ms.reward.reset()
    ms.command.reset(rng=np.random.default_rng(0))
    total = 0.0
    last_action = np.zeros(6)
    for step in range(n_steps):
        state = EnvState(
            joint_pos=np.zeros(6),
            joint_vel=np.zeros(6),
            action=last_action,
            last_action=last_action,
            base_lin_vel=np.array([0.5, 0.0, 0.0]) if track_command else np.zeros(3),
            base_ang_vel=np.zeros(3),
            dt=dt,
            step_count=step,
            max_episode_length=n_steps,
        )
        ms.command.compute(state)
        obs = ms.observation.compute(state)
        assert obs.shape == (3 + 3 + 3 + 6 + 6 + 6 + 3,)
        reward = ms.reward.compute(state)
        # breakdown sums to total (within float tolerance)
        assert sum(ms.reward.term_values.values()) == reward
        term = ms.termination.compute(state)
        total += reward
        if term.done:
            break
    return total


def test_tracking_agent_outscores_stationary_agent():
    assert _rollout(track_command=True) > _rollout(track_command=False)


def test_breakdown_keys_match_configured_terms():
    ms = build_managers(_CONFIG)
    assert ms.command and ms.reward
    state = EnvState(joint_pos=np.zeros(6), joint_vel=np.zeros(6), action=np.zeros(6), last_action=np.zeros(6))
    ms.command.compute(state)
    ms.reward.compute(state)
    assert set(ms.reward.term_values) == {"track_lin", "track_ang", "lin_vel_z", "action_rate", "alive"}


def test_reward_is_dt_invariant_over_fixed_horizon():
    # Same physical episode duration, half the dt -> twice the steps -> equal total.
    coarse = _rollout(track_command=True, n_steps=50, dt=0.02)
    fine = _rollout(track_command=True, n_steps=100, dt=0.01)
    assert coarse == pytest.approx(fine)
