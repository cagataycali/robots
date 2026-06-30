"""Tests for ``GymSimEnv`` - the gymnasium adapter over ``SimEnv``.

CPU-only with a fake engine: pins the gymnasium 5-tuple contract, the
terminated-vs-truncated split (the SB3 TimeLimit footgun done right), the
critic-obs passthrough, and ``gymnasium.utils.env_checker.check_env`` compliance.
An optional SB3 smoke-train runs only when stable-baselines3 is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from strands_robots.training.rl import GymSimEnv, SimEnv  # noqa: E402


class _FakeEngine:
    """1-DOF fake engine; joint ``J`` integrates the action."""

    def __init__(self) -> None:
        self._j = 0.0
        self._v = 0.0

    def list_robots(self) -> list[str]:
        return ["fake"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["J"]

    def reset(self) -> dict:
        self._j = 0.0
        self._v = 0.0
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images: bool = False) -> dict:
        return {"J": self._j, "J.vel": self._v}

    def send_action(self, action, robot_name=None, n_substeps: int = 1) -> dict:
        a = float(action[0]) if len(action) else 0.0
        self._v = 0.1 * a
        self._j += self._v
        return {"status": "success"}


def _make_sim_env(*, success_fn=None, max_steps=5):  # type: ignore[no-untyped-def]
    return SimEnv(
        _FakeEngine(),
        actor_obs_keys=["J", "J.vel"],
        critic_obs_keys=["J", "J.vel"],
        reward_terms=[lambda e: 1.0],
        action_dim=1,
        max_episode_steps=max_steps,
        success_fn=success_fn,
    )


def test_spaces_shapes() -> None:
    env = GymSimEnv(_make_sim_env())
    assert env.observation_space.shape == (2,)
    assert env.action_space.shape == (1,)
    assert env.action_space.low.min() == -1.0
    assert env.action_space.high.max() == 1.0


def test_reset_returns_obs_and_info() -> None:
    env = GymSimEnv(_make_sim_env())
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)
    assert obs.dtype == np.float32
    assert "critic_obs" in info
    assert info["critic_obs"].shape == (2,)


def test_step_returns_five_tuple() -> None:
    env = GymSimEnv(_make_sim_env())
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
    assert isinstance(obs, np.ndarray) and obs.shape == (2,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "critic_obs" in info


def test_truncation_on_timeout_not_terminated() -> None:
    """No success_fn -> episode ends via TIME-OUT -> truncated=True, terminated=False."""
    env = GymSimEnv(_make_sim_env(success_fn=None, max_steps=3))
    env.reset()
    term_seen, trunc_seen = False, False
    for _ in range(3):
        _obs, _r, terminated, truncated, _info = env.step(np.array([0.0], dtype=np.float32))
        term_seen = term_seen or terminated
        trunc_seen = trunc_seen or truncated
    assert trunc_seen is True, "time-out must surface as truncated"
    assert term_seen is False, "time-out must NOT surface as terminated"


def test_termination_on_success_not_truncated() -> None:
    """success_fn always True -> genuine TERMINAL -> terminated=True, truncated=False."""
    env = GymSimEnv(_make_sim_env(success_fn=lambda e: True, max_steps=10))
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    assert terminated is True, "success must surface as terminated"
    assert truncated is False, "a real terminal is not a truncation"


def test_check_env_compliance() -> None:
    """gymnasium's own env checker must pass (full API conformance)."""
    from gymnasium.utils.env_checker import check_env

    env = GymSimEnv(_make_sim_env(success_fn=lambda e: False, max_steps=5))
    # skip_render_check: we advertise no render modes by design.
    check_env(env, skip_render_check=True)


def test_sb3_baseline_smoke_train() -> None:
    """If stable-baselines3 is installed, PPO must train on GymSimEnv without error.

    This is the external-baseline lever: our from-scratch PPO can now be
    sanity-checked against SB3 PPO on the SAME env. Gated so CI without SB3
    skips cleanly.
    """
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    env = GymSimEnv(_make_sim_env(success_fn=lambda e: False, max_steps=10))
    model = PPO("MlpPolicy", env, n_steps=32, batch_size=16, n_epochs=2, verbose=0, seed=0)
    model.learn(total_timesteps=64)
    # Smoke: the learned policy produces a valid action for an observation.
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1,)
