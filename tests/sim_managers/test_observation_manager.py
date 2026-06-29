"""Observation terms + ObservationManager concatenation / scale / clip / slices."""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.sim_managers import EnvState, ObservationManager, get_term_class


def test_joint_pos_is_relative_to_default():
    state = EnvState(
        joint_pos=np.array([1.0, 2.0]),
        joint_vel=np.zeros(2),
        action=np.zeros(2),
        last_action=np.zeros(2),
        default_joint_pos=np.array([0.5, 0.5]),
    )
    term = get_term_class("observation", "joint_pos")()
    np.testing.assert_allclose(term(state), [0.5, 1.5])


def test_manager_concatenates_in_order_and_records_slices(state):
    mgr = ObservationManager.from_config(
        {"terms": [{"func": "base_lin_vel"}, {"func": "joint_vel"}, {"func": "velocity_commands"}]}
    )
    obs = mgr.compute(state)
    assert obs.shape == (3 + 6 + 3,)
    assert mgr.term_slices["base_lin_vel"] == slice(0, 3)
    assert mgr.term_slices["joint_vel"] == slice(3, 9)
    assert mgr.term_slices["velocity_commands"] == slice(9, 12)
    np.testing.assert_allclose(obs[mgr.term_slices["base_lin_vel"]], state.base_lin_vel)


def test_scale_and_clip_applied(state):
    mgr = ObservationManager.from_config({"terms": [{"func": "base_lin_vel", "scale": 10.0, "clip": [-2.0, 2.0]}]})
    obs = mgr.compute(state)
    # base_lin_vel = [0.5, 0, 0.1] * 10 = [5, 0, 1] clipped to [-2, 2] -> [2, 0, 1].
    np.testing.assert_allclose(obs, [2.0, 0.0, 1.0])


def test_empty_manager_returns_empty_vector(state):
    mgr = ObservationManager.from_config({"terms": []})
    assert mgr.compute(state).shape == (0,)


def test_duplicate_labels_rejected():
    with pytest.raises(ValueError, match="duplicate term labels"):
        ObservationManager.from_config(
            {"terms": [{"name": "x", "func": "base_lin_vel"}, {"name": "x", "func": "joint_vel"}]}
        )
