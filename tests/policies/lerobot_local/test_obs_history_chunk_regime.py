"""Observation-history policies must run per-step via ``select_action()``.

Diffusion (``n_obs_steps=2``) and VQBeT (``n_obs_steps=5``) consume a short
observation history. LeRobot builds that history inside ``select_action()``'s
queue; ``predict_action_chunk()``'s offline branch stacks a SINGLE live frame as
``n_obs_steps=1`` and ``generate_actions()`` trips
``assert n_obs_steps == config.n_obs_steps`` -> crash.

``LerobotLocalPolicy._auto_detect_actions_per_step`` therefore must keep
``actions_per_step=1`` (the ``select_action()`` regime) for any checkpoint whose
config declares ``n_obs_steps > 1``, instead of adopting ``n_action_steps`` and
routing to the crashing ``predict_action_chunk()`` path.
"""

from __future__ import annotations

from unittest.mock import patch

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


class _Cfg:
    """Minimal loaded-policy config stub read by _auto_detect_actions_per_step."""

    def __init__(self, *, n_obs_steps=1, n_action_steps=1, temporal_ensemble_coeff=None):
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.temporal_ensemble_coeff = temporal_ensemble_coeff


class _Pol:
    def __init__(self, cfg):
        self.config = cfg


def _unloaded(**kw) -> LerobotLocalPolicy:
    with patch.object(LerobotLocalPolicy, "_load_model"):
        return LerobotLocalPolicy(pretrained_name_or_path="test/model", **kw)


def test_obs_history_policy_keeps_actions_per_step_1():
    """n_obs_steps>1 (Diffusion=2) -> stay at 1 so select_action() is used."""
    pol = _unloaded()
    pol._policy = _Pol(_Cfg(n_obs_steps=2, n_action_steps=32))
    assert pol.actions_per_step == 1  # default
    pol._auto_detect_actions_per_step()
    # Must NOT adopt n_action_steps (that routes to predict_action_chunk and
    # crashes on a single-frame obs). Stay at 1 -> per-step select_action().
    assert pol.actions_per_step == 1


def test_vqbet_style_obs_history_keeps_actions_per_step_1():
    """VQBeT's larger history (n_obs_steps=5) is handled the same way."""
    pol = _unloaded()
    pol._policy = _Pol(_Cfg(n_obs_steps=5, n_action_steps=8))
    pol._auto_detect_actions_per_step()
    assert pol.actions_per_step == 1


def test_single_obs_chunk_policy_still_auto_adopts_horizon():
    """Control: ACT-style (n_obs_steps=1, chunk=100) is unchanged -> 100."""
    pol = _unloaded()
    pol._policy = _Pol(_Cfg(n_obs_steps=1, n_action_steps=100))
    pol._auto_detect_actions_per_step()
    assert pol.actions_per_step == 100


def test_missing_n_obs_steps_defaults_to_chunk_replay():
    """A config with no n_obs_steps attr behaves as before (adopts chunk)."""
    pol = _unloaded()

    class _NoObs:
        def __init__(self):
            self.n_action_steps = 30

    pol._policy = _Pol(_NoObs())
    pol._auto_detect_actions_per_step()
    assert pol.actions_per_step == 30


def test_explicit_horizon_respected_even_with_obs_history():
    """An explicit caller horizon is never overridden by the obs-history guard."""
    pol = _unloaded(actions_per_step=4)
    pol._policy = _Pol(_Cfg(n_obs_steps=2, n_action_steps=32))
    pol._auto_detect_actions_per_step()
    assert pol.actions_per_step == 4
