"""Numeric behaviour of individual locomotion reward terms."""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.sim_managers import EnvState, get_term_class


def _make(func, **params):
    return get_term_class("reward", func)(**params)


def test_track_lin_vel_exact_match_is_one(state):
    # base_lin_vel xy == command xy -> perfect tracking -> exp(0) == 1.
    term = _make("track_lin_vel_xy_exp", std=0.25)
    assert term(state) == pytest.approx(1.0)


def test_track_lin_vel_decreases_with_error(state):
    term = _make("track_lin_vel_xy_exp", std=0.25)
    state.base_lin_vel = np.array([0.0, 0.0, 0.0])  # command is [0.5, 0, 0.3]
    assert 0.0 < term(state) < 1.0


def test_track_ang_vel_exact_match_is_one(state):
    term = _make("track_ang_vel_z_exp", std=0.25)
    assert term(state) == pytest.approx(1.0)


def test_lin_vel_z_l2_is_squared_vertical_velocity(state):
    term = _make("lin_vel_z_l2")
    assert term(state) == pytest.approx(0.1**2)


def test_ang_vel_xy_l2_ignores_yaw(state):
    term = _make("ang_vel_xy_l2")
    assert term(state) == pytest.approx(0.0)  # only yaw is non-zero


def test_flat_orientation_zero_when_upright(state):
    assert _make("flat_orientation_l2")(state) == pytest.approx(0.0)


def test_orientation_l2_zero_when_upright(state):
    assert _make("orientation_l2")(state) == pytest.approx(0.0)


def test_orientation_l2_penalises_tilt(state):
    state.projected_gravity = np.array([0.0, 0.0, -0.5])
    assert _make("orientation_l2")(state) == pytest.approx(0.25)


def test_action_rate_is_squared_delta(state):
    # action - last_action = [0.1, 0.2, 0.3, 0, 0, 0] -> 0.14
    assert _make("action_rate_l2")(state) == pytest.approx(0.01 + 0.04 + 0.09)


def test_dof_torques_and_acc_and_vel(state):
    assert _make("dof_torques_l2")(state) == pytest.approx(1.0)
    assert _make("dof_acc_l2")(state) == pytest.approx(4.0)
    assert _make("dof_vel_l2")(state) == pytest.approx(1.0 + 1.0 + 4.0)


def test_alive_is_constant(state):
    assert _make("alive")(state) == 1.0


def test_termination_penalty_tracks_flag(state):
    assert _make("termination_penalty")(state) == 0.0
    state.terminated = True
    assert _make("termination_penalty")(state) == 1.0


def test_joint_pos_limits_zero_when_within(state):
    state.joint_pos_limits = np.array([[-2.0, 2.0]] * 6)
    assert _make("joint_pos_limits")(state) == pytest.approx(0.0)


def test_joint_pos_limits_penalises_violation():
    state = EnvState(
        joint_pos=np.array([1.0]),
        joint_vel=np.array([0.0]),
        action=np.array([0.0]),
        last_action=np.array([0.0]),
        joint_pos_limits=np.array([[-1.0, 1.0]]),
        soft_joint_pos_limit_factor=0.5,
    )
    # soft upper = 0.5; pos 1.0 -> violation 0.5.
    assert _make("joint_pos_limits")(state) == pytest.approx(0.5)


def test_joint_pos_limits_zero_without_limits(state):
    assert state.joint_pos_limits is None
    assert _make("joint_pos_limits")(state) == 0.0


def test_feet_air_time_rewards_first_contact():
    state = EnvState(
        joint_pos=np.zeros(2),
        joint_vel=np.zeros(2),
        action=np.zeros(2),
        last_action=np.zeros(2),
        feet_air_time=np.array([0.6, 0.0]),
        feet_contact=np.array([True, False]),
        commands={"base_velocity": np.array([0.5, 0.0, 0.0])},
    )
    term = _make("feet_air_time", threshold=0.4)
    term.reset(state)
    # foot 0 just made contact with 0.6s air time -> (0.6 - 0.4) == 0.2.
    assert term(state) == pytest.approx(0.2)
    # second step, still in contact -> no first-contact event.
    assert term(state) == pytest.approx(0.0)


def test_feet_slide_penalises_sliding_contact():
    state = EnvState(
        joint_pos=np.zeros(2),
        joint_vel=np.zeros(2),
        action=np.zeros(2),
        last_action=np.zeros(2),
        feet_contact=np.array([True, False]),
        extras={"feet_lin_vel": np.array([[1.0, 0.0], [5.0, 0.0]])},
    )
    # only the in-contact foot (1 m/s) is penalised.
    assert _make("feet_slide")(state) == pytest.approx(1.0)
