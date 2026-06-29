"""EnvState contract: coercion, defaults, and command lookup behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.sim_managers import EnvState


def test_array_fields_coerced_to_1d_float64():
    state = EnvState(joint_pos=[1, 2, 3], joint_vel=[0, 0, 0], action=[0.0], last_action=[0.0])
    assert state.joint_pos.dtype == np.float64
    assert state.joint_pos.shape == (3,)
    assert state.num_joints == 3


def test_optional_arrays_default_to_zeros_sized_to_joints():
    state = EnvState(joint_pos=np.zeros(4), joint_vel=np.zeros(4), action=np.zeros(2), last_action=np.zeros(2))
    assert state.joint_torque is not None and state.joint_torque.shape == (4,)
    assert state.joint_acc is not None and state.joint_acc.shape == (4,)
    assert state.default_joint_pos is not None and state.default_joint_pos.shape == (4,)


def test_joint_vel_length_mismatch_raises():
    with pytest.raises(ValueError, match="joint_vel length"):
        EnvState(joint_pos=np.zeros(3), joint_vel=np.zeros(2), action=np.zeros(1), last_action=np.zeros(1))


def test_2d_field_rejected():
    with pytest.raises(ValueError, match="must be 1-D"):
        EnvState(joint_pos=np.zeros((2, 2)), joint_vel=np.zeros(4), action=np.zeros(1), last_action=np.zeros(1))


def test_command_lookup_missing_raises_with_available_names():
    state = EnvState(joint_pos=np.zeros(2), joint_vel=np.zeros(2), action=np.zeros(2), last_action=np.zeros(2))
    state.commands["base_velocity"] = np.zeros(3)
    with pytest.raises(KeyError, match="base_velocity"):
        state.command("missing")
