"""Shared fixtures for sim_managers tests."""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.sim_managers import EnvState


@pytest.fixture
def state() -> EnvState:
    """A representative 6-joint locomotion state with an active command."""
    return EnvState(
        joint_pos=np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.5]),
        joint_vel=np.array([0.0, 1.0, -1.0, 0.0, 2.0, 0.0]),
        action=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]),
        last_action=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        base_lin_vel=np.array([0.5, 0.0, 0.1]),
        base_ang_vel=np.array([0.0, 0.0, 0.3]),
        projected_gravity=np.array([0.0, 0.0, -1.0]),
        base_height=0.7,
        joint_torque=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        joint_acc=np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
        commands={"base_velocity": np.array([0.5, 0.0, 0.3])},
        dt=0.02,
        step_count=5,
        max_episode_length=100,
    )
