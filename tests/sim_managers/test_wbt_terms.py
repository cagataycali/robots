"""WBT terms: command publication, observations, rewards, termination, errors."""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.sim_managers import (
    CommandManager,
    EnvState,
    ObservationManager,
    get_term_class,
)
from strands_robots.sim_managers.motion import (
    MOTION_PHASE,
    MOTION_TARGET_POS,
    MOTION_TARGET_VEL,
)

# A 2-frame clip on 2 joints: [0,0] -> [1,-1], fps=10 (duration 0.2s).
FRAMES_POS = [[0.0, 0.0], [1.0, -1.0]]
FRAMES_VEL = [[0.0, 0.0], [2.0, -2.0]]


def _state(joint_pos, joint_vel=None, dt=0.02):
    jp = np.asarray(joint_pos, dtype=float)
    jv = np.zeros_like(jp) if joint_vel is None else np.asarray(joint_vel, dtype=float)
    return EnvState(joint_pos=jp, joint_vel=jv, action=np.zeros_like(jp), last_action=np.zeros_like(jp), dt=dt)


def _command_manager():
    return CommandManager.from_config(
        {
            "terms": [
                {
                    "name": "motion",
                    "func": "motion_clip",
                    "params": {"frames_pos": FRAMES_POS, "frames_vel": FRAMES_VEL, "fps": 10.0},
                }
            ]
        }
    )


def test_command_publishes_targets_to_extras():
    mgr = _command_manager()
    state = _state([0.0, 0.0])
    out = mgr.compute(state)  # t=0 after one update of dt -> still near frame 0
    assert MOTION_TARGET_POS in state.extras
    assert MOTION_TARGET_VEL in state.extras
    assert MOTION_PHASE in state.extras
    # returned command vector equals the published target pose
    np.testing.assert_allclose(out["motion"], state.extras[MOTION_TARGET_POS])


def test_command_phase_advances_over_steps():
    mgr = _command_manager()
    state = _state([0.0, 0.0], dt=0.05)  # quarter of the 0.2s clip per step
    phases = []
    for _ in range(4):
        mgr.compute(state)
        phases.append(state.extras[MOTION_PHASE])
    # phase strictly increases through the first cycle then wraps
    assert phases[0] < phases[1] < phases[2]


def test_observation_terms_dims_and_values():
    mgr = _command_manager()
    obs_mgr = ObservationManager.from_config(
        {
            "terms": [
                {"func": "motion_phase"},
                {"func": "motion_target_joint_pos"},
                {"func": "motion_target_joint_vel"},
                {"func": "joint_pos_error"},
            ]
        }
    )
    state = _state([0.25, 0.25])
    mgr.compute(state)
    obs = obs_mgr.compute(state)
    # phase(2) + target_pos(2) + target_vel(2) + error(2) = 8
    assert obs.shape == (8,)
    # joint_pos_error slice == joint_pos - target
    sl = obs_mgr.term_slices["joint_pos_error"]
    np.testing.assert_allclose(obs[sl], state.joint_pos - state.extras[MOTION_TARGET_POS])


def test_track_joint_pos_exp_peaks_when_matched():
    term = get_term_class("reward", "track_joint_pos_exp")(std=0.5)
    state = _state([0.0, 0.0])
    state.extras[MOTION_TARGET_POS] = np.array([0.0, 0.0])
    matched = term(state)
    state2 = _state([0.0, 0.0])
    state2.extras[MOTION_TARGET_POS] = np.array([1.0, -1.0])
    mismatched = term(state2)
    assert matched == pytest.approx(1.0)
    assert mismatched < matched


def test_track_joint_vel_exp_peaks_when_matched():
    term = get_term_class("reward", "track_joint_vel_exp")(std=1.0)
    state = _state([0.0, 0.0], joint_vel=[0.5, -0.5])
    state.extras[MOTION_TARGET_VEL] = np.array([0.5, -0.5])
    assert term(state) == pytest.approx(1.0)


def test_motion_divergence_terminates_when_far():
    term = get_term_class("termination", "motion_divergence")(threshold=0.5)
    state = _state([1.0, -1.0])
    state.extras[MOTION_TARGET_POS] = np.array([0.0, 0.0])  # error norm ~1.41 > 0.5
    assert term(state) is True
    assert getattr(term, "is_time_out", False) is False
    near = _state([0.1, 0.0])
    near.extras[MOTION_TARGET_POS] = np.array([0.0, 0.0])
    assert term(near) is False


def test_missing_motion_target_raises_actionable_error():
    term = get_term_class("reward", "track_joint_pos_exp")()
    state = _state([0.0, 0.0])  # no command term ran -> extras empty
    with pytest.raises(KeyError, match="motion_clip"):
        term(state)


def test_target_dimension_mismatch_raises():
    term = get_term_class("observation", "joint_pos_error")()
    state = _state([0.0, 0.0, 0.0])  # 3 joints
    state.extras[MOTION_TARGET_POS] = np.array([0.0, 0.0])  # 2-joint clip
    with pytest.raises(ValueError, match="one column per joint"):
        term(state)
