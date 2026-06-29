"""WBT integral rollout: a tracking agent outscores a stationary one.

Drives a 6-joint clip through the full manager stack (command + observation +
reward + termination) for a synthetic agent that perfectly follows the
reference, and a stationary agent that never moves. The tracking agent must
accumulate a strictly higher pose-tracking reward and never trigger the
divergence termination, while the stationary agent diverges - the end-to-end
contract a WBT trainer relies on.
"""

from __future__ import annotations

import numpy as np

from strands_robots.sim_managers import (
    CommandManager,
    EnvState,
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from strands_robots.sim_managers.motion import MOTION_TARGET_POS, MOTION_TARGET_VEL

# A smooth 16-frame sinusoidal clip on 6 joints.
_N_FRAMES = 16
_N_JOINTS = 6
_t = np.linspace(0.0, 2 * np.pi, _N_FRAMES, endpoint=False)
FRAMES_POS = np.stack([0.5 * np.sin(_t + j) for j in range(_N_JOINTS)], axis=1)
FRAMES_VEL = np.stack([0.5 * np.cos(_t + j) for j in range(_N_JOINTS)], axis=1)
FPS = 20.0
DT = 0.05


def _managers():
    command = CommandManager.from_config(
        {
            "terms": [
                {
                    "name": "motion",
                    "func": "motion_clip",
                    "params": {"frames_pos": FRAMES_POS.tolist(), "frames_vel": FRAMES_VEL.tolist(), "fps": FPS},
                }
            ]
        }
    )
    observation = ObservationManager.from_config(
        {
            "terms": [
                {"func": "motion_phase"},
                {"func": "joint_pos"},
                {"func": "joint_pos_error"},
            ]
        }
    )
    reward = RewardManager.from_config(
        {
            "terms": [
                {"name": "track_pos", "func": "track_joint_pos_exp", "weight": 1.0, "params": {"std": 0.4}},
                {"name": "track_vel", "func": "track_joint_vel_exp", "weight": 0.5, "params": {"std": 2.0}},
            ]
        }
    )
    termination = TerminationManager.from_config(
        {"terms": [{"func": "time_out"}, {"func": "motion_divergence", "params": {"threshold": 0.7}}]}
    )
    return command, observation, reward, termination


def _rollout(*, track: bool, n_steps: int = 40):
    command, observation, reward, termination = _managers()
    command.reset()
    reward.reset()
    total = 0.0
    diverged = False
    obs_dim = 0
    for step in range(n_steps):
        state = EnvState(
            joint_pos=np.zeros(_N_JOINTS),
            joint_vel=np.zeros(_N_JOINTS),
            action=np.zeros(_N_JOINTS),
            last_action=np.zeros(_N_JOINTS),
            dt=DT,
            step_count=step,
            max_episode_length=n_steps,
        )
        command.compute(state)
        if track:
            # perfect follower: actual joints equal the published targets
            state.joint_pos = np.asarray(state.extras[MOTION_TARGET_POS], dtype=float)
            state.joint_vel = np.asarray(state.extras[MOTION_TARGET_VEL], dtype=float)
        obs = observation.compute(state)
        obs_dim = obs.shape[0]
        total += reward.compute(state)
        result = termination.compute(state)
        if result.terminated:
            diverged = True
            break
    return total, diverged, obs_dim


def test_tracking_agent_outscores_stationary():
    track_reward, track_diverged, obs_dim = _rollout(track=True)
    still_reward, _, _ = _rollout(track=False)
    assert track_reward > still_reward
    assert not track_diverged  # perfect follower never diverges
    # obs = phase(2) + joint_pos(6) + joint_pos_error(6) = 14
    assert obs_dim == 14


def test_stationary_agent_diverges_from_motion():
    # With a clip amplitude of 0.5 across 6 joints, the stationary agent's
    # error norm (~0.85) crosses the 0.7 threshold within the rollout.
    _, diverged, _ = _rollout(track=False)
    assert diverged
