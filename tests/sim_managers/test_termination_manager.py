"""Termination terms + manager failure/timeout classification."""

from __future__ import annotations

import numpy as np

from strands_robots.sim_managers import TerminationManager, get_term_class


def test_time_out_fires_at_horizon(state):
    state.step_count = 100
    assert get_term_class("termination", "time_out")()(state) is True
    state.step_count = 99
    assert get_term_class("termination", "time_out")()(state) is False


def test_bad_orientation_triggers_when_tilted(state):
    term = get_term_class("termination", "bad_orientation")(limit_angle=0.5)
    assert term(state) is False  # upright
    state.projected_gravity = np.array([1.0, 0.0, 0.0])  # 90 deg tilt
    assert term(state) is True


def test_base_height_below_threshold(state):
    term = get_term_class("termination", "base_height_below_threshold")(min_height=0.5)
    assert term(state) is False  # height 0.7
    state.base_height = 0.2
    assert term(state) is True


def test_manager_separates_timeout_from_failure(state):
    mgr = TerminationManager.from_config(
        {"terms": [{"func": "time_out"}, {"func": "base_height_below_threshold", "params": {"min_height": 0.5}}]}
    )
    # nothing fires
    res = mgr.compute(state)
    assert not res.done and not res.time_out and not res.terminated

    # failure only
    state.base_height = 0.1
    res = mgr.compute(state)
    assert res.done and res.terminated and not res.time_out

    # timeout
    state.base_height = 0.7
    state.step_count = 100
    res = mgr.compute(state)
    assert res.done and res.time_out and not res.terminated


def test_joint_pos_limit_safe_without_limits(state):
    assert get_term_class("termination", "joint_pos_limit")()(state) is False
