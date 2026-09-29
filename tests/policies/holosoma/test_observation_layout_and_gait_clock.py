"""Holosoma observation layout and gait clock, checked against the upstream loop.

The reference numbers reproduce ``holosoma_inference`` (commit ``bccd4d7``):
``policies/base.py`` sorts a group's terms alphabetically before flattening
(line 135), ``policies/locomotion.py`` advances the two-foot phase and pins it
to ``pi`` when standing (lines 56-67), and the ``loco-g1-29dof`` preset scales
``base_ang_vel`` by 0.25 and ``dof_vel`` by 0.05.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.policies.holosoma import (
    ACTOR_OBS_DIMS,
    ACTOR_OBS_TERMS,
    HOLOSOMA_OBS_DIM,
    GaitPhase,
    HolosomaConfig,
    actor_obs_slices,
    build_actor_obs,
)


def _upstream_flatten(terms: dict[str, np.ndarray]) -> np.ndarray:
    """What ``BasePolicy._update_obs_history`` does with history length 1."""
    return np.concatenate([terms[name].ravel() for name in sorted(terms)]).astype(np.float32)


def test_terms_are_alphabetical_and_sum_to_the_onnx_width() -> None:
    assert list(ACTOR_OBS_TERMS) == sorted(ACTOR_OBS_TERMS)
    assert sum(ACTOR_OBS_DIMS[t] for t in ACTOR_OBS_TERMS) == HOLOSOMA_OBS_DIM == 100
    slices = actor_obs_slices()
    assert slices["actions"] == slice(0, 29)
    assert slices["base_ang_vel"] == slice(29, 32)
    assert slices["command_ang_vel"] == slice(32, 33)
    assert slices["command_lin_vel"] == slice(33, 35)
    assert slices["cos_phase"] == slice(35, 37)
    assert slices["dof_pos"] == slice(37, 66)
    assert slices["dof_vel"] == slice(66, 95)
    assert slices["projected_gravity"] == slice(95, 98)
    assert slices["sin_phase"] == slice(98, 100)


def test_build_actor_obs_matches_the_sorted_upstream_flatten() -> None:
    cfg = HolosomaConfig()
    rng = np.random.default_rng(7)
    last_action = rng.normal(size=29)
    ang = rng.normal(size=3)
    lin = np.array([0.5, -0.1])
    yaw = 0.3
    phase = np.array([0.4, 0.4 + np.pi])
    qj = rng.normal(size=29)
    dqj = rng.normal(size=29)
    grav = np.array([0.1, -0.2, -0.97])
    default = np.asarray(cfg.default_angles)

    upstream = _upstream_flatten(
        {
            "base_ang_vel": ang * 0.25,
            "projected_gravity": grav,
            "command_lin_vel": lin,
            "command_ang_vel": np.array([yaw]),
            "dof_pos": qj - default,
            "dof_vel": dqj * 0.05,
            "actions": last_action,
            "sin_phase": np.sin(phase),
            "cos_phase": np.cos(phase),
        }
    )
    ours = build_actor_obs(
        cfg,
        last_action=last_action,
        base_ang_vel=ang,
        command_lin_vel=lin,
        command_ang_vel=yaw,
        phase=phase,
        qj=qj,
        dqj=dqj,
        proj_gravity=grav,
    )
    assert ours.dtype == np.float32
    np.testing.assert_allclose(ours, upstream, rtol=0, atol=1e-6)


def test_build_actor_obs_refuses_a_wrong_width() -> None:
    cfg = HolosomaConfig()
    with pytest.raises(ValueError, match="qj must have 29"):
        build_actor_obs(
            cfg,
            last_action=np.zeros(29),
            base_ang_vel=np.zeros(3),
            command_lin_vel=np.zeros(2),
            command_ang_vel=0.0,
            phase=np.array([0.0, np.pi]),
            qj=np.zeros(15),
            dqj=np.zeros(29),
            proj_gravity=np.array([0.0, 0.0, -1.0]),
        )


def _upstream_phase(n_ticks: int, lin: np.ndarray, ang: float, dt: float) -> list[np.ndarray]:
    """``LocomotionPolicy.update_phase_time`` replayed for ``n_ticks``."""
    phase = np.array([[0.0, np.pi]])
    standing = False
    out = []
    for _ in range(n_ticks):
        phase = np.fmod(phase + dt + np.pi, 2 * np.pi) - np.pi
        if np.linalg.norm(lin) < 0.01 and np.linalg.norm([ang]) < 0.01:
            phase[0, :] = np.pi * np.ones(2)
            standing = True
        elif standing:
            phase = np.array([[0.0, np.pi]])
            standing = False
        out.append(phase[0].copy())
    return out


def test_gait_phase_replays_upstream_for_a_walk() -> None:
    cfg = HolosomaConfig()
    assert cfg.phase_dt == pytest.approx(2 * np.pi / 50.0)
    clock = GaitPhase(cfg)
    lin = np.array([0.5, 0.0])
    expected = _upstream_phase(120, lin, 0.0, cfg.phase_dt)
    for want in expected:
        got = clock.step(lin, 0.0)
        np.testing.assert_allclose(got, want, atol=1e-12)
    assert clock.is_standing is False


def test_gait_phase_pins_both_feet_when_standing_and_restarts_on_the_first_moving_tick() -> None:
    cfg = HolosomaConfig()
    clock = GaitPhase(cfg)
    still = np.zeros(2)
    for _ in range(10):
        np.testing.assert_allclose(clock.step(still, 0.0), [np.pi, np.pi])
    assert clock.is_standing is True
    first_moving = clock.step(np.array([0.3, 0.0]), 0.0)
    np.testing.assert_allclose(first_moving, [0.0, np.pi])
    second_moving = clock.step(np.array([0.3, 0.0]), 0.0)
    np.testing.assert_allclose(second_moving, [cfg.phase_dt, -np.pi + cfg.phase_dt], atol=1e-12)


def test_gait_phase_yaw_alone_counts_as_moving() -> None:
    clock = GaitPhase(HolosomaConfig())
    got = clock.step(np.zeros(2), 0.5)
    assert not clock.is_standing
    assert not np.allclose(got, [np.pi, np.pi])


def test_reset_restarts_the_clock() -> None:
    clock = GaitPhase(HolosomaConfig())
    for _ in range(7):
        clock.step(np.array([0.5, 0.0]), 0.0)
    clock.reset()
    np.testing.assert_allclose(clock.phase, [0.0, np.pi])
    assert clock.is_standing is False
