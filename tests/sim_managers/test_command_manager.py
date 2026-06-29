"""Command terms + manager: sampling, resampling, seeding, publication."""

from __future__ import annotations

import numpy as np

from strands_robots.sim_managers import CommandManager, EnvState, get_term_class


def _state(dt=0.02):
    return EnvState(joint_pos=np.zeros(2), joint_vel=np.zeros(2), action=np.zeros(2), last_action=np.zeros(2), dt=dt)


def test_command_within_ranges():
    term = get_term_class("command", "uniform_velocity")(
        lin_vel_x=(0.5, 0.5), lin_vel_y=(-0.1, 0.1), ang_vel_z=(1.0, 1.0)
    )
    cmd = term(_state())
    assert cmd[0] == 0.5
    assert -0.1 <= cmd[1] <= 0.1
    assert cmd[2] == 1.0


def test_seeded_reset_is_reproducible():
    term = get_term_class("command", "uniform_velocity")(lin_vel_x=(-1.0, 1.0))
    term.reset(rng=np.random.default_rng(0))
    a = term(_state()).copy()
    term.reset(rng=np.random.default_rng(0))
    b = term(_state()).copy()
    np.testing.assert_allclose(a, b)


def test_resamples_after_interval():
    term = get_term_class("command", "uniform_velocity")(lin_vel_x=(-1.0, 1.0), resampling_time=0.05)
    term.reset(rng=np.random.default_rng(1))
    first = term(_state()).copy()
    # advance 2 steps of dt=0.02 (0.04s) -> no resample yet
    term.update(0.02)
    term.update(0.02)
    np.testing.assert_allclose(term(_state()), first)
    term.update(0.02)  # now 0.06s >= 0.05 -> resample
    assert not np.allclose(term(_state()), first)


def test_manager_publishes_command_onto_state():
    mgr = CommandManager.from_config(
        {"terms": [{"name": "base_velocity", "func": "uniform_velocity", "params": {"lin_vel_x": [0.3, 0.3]}}]}
    )
    state = _state()
    out = mgr.compute(state)
    assert "base_velocity" in state.commands
    np.testing.assert_allclose(state.command("base_velocity"), out["base_velocity"])
    assert state.command("base_velocity")[0] == 0.3
