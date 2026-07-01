"""Unit tests for ``SimEnv`` - the single-environment RL wrapper over a ``SimEngine``.

CPU-only, fake engine. Pins the construction-time validation contract (obs keys,
reward terms, substeps, action-dim inference / requirement) and the reset
lifecycle hooks (custom ``reset_fn``, stateful reward-term ``reset``), plus the
``close`` no-op contract that keeps ``SimEnv`` interface-compatible with
``VecSimEnv`` / ``GymSimEnv``. These are behaviors the RL trainers rely on: a
typo in obs keys or a missing action dim must fail loudly at construction, not
mid-rollout.
"""

from __future__ import annotations

from typing import cast

import pytest

torch = pytest.importorskip("torch")

from strands_robots.simulation.base import SimEngine  # noqa: E402 - after torch importorskip
from strands_robots.training.rl import SimEnv  # noqa: E402 - after torch importorskip


class _OneJointEngine:
    """Minimal fake engine with a single robot ``fake`` and one joint ``J``."""

    def list_robots(self) -> list[str]:
        return ["fake"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["J"]

    def reset(self) -> dict:
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images: bool = False) -> dict:
        return {"J": 0.0, "J.vel": 1.0}

    def send_action(self, action, robot_name=None, n_substeps: int = 1) -> dict:
        return {"status": "success"}


class _NoRobotEngine(_OneJointEngine):
    """Fake engine with no registered robots (``list_robots`` is empty)."""

    def list_robots(self) -> list[str]:
        return []


def _engine(obj: object) -> SimEngine:
    """Present a duck-typed fake as a ``SimEngine`` for the ``SimEnv`` constructor.

    ``SimEngine`` is a nominal ABC, but ``SimEnv`` only touches the handful of
    methods the fakes implement (list_robots / robot_joint_names / reset /
    get_observation / send_action), so the cast is safe for these unit tests.
    """
    return cast(SimEngine, obj)


def test_rejects_empty_actor_obs_keys() -> None:
    with pytest.raises(ValueError, match="actor_obs_keys"):
        SimEnv(_engine(_OneJointEngine()), actor_obs_keys=[], reward_terms=[lambda e: 1.0], action_dim=1)


def test_rejects_empty_reward_terms() -> None:
    with pytest.raises(ValueError, match="reward_terms"):
        SimEnv(_engine(_OneJointEngine()), actor_obs_keys=["J"], reward_terms=[], action_dim=1)


def test_rejects_nonpositive_n_substeps() -> None:
    with pytest.raises(ValueError, match="n_substeps"):
        SimEnv(
            _engine(_OneJointEngine()),
            actor_obs_keys=["J"],
            reward_terms=[lambda e: 1.0],
            action_dim=1,
            n_substeps=0,
        )


def test_infers_action_dim_from_robot_joints() -> None:
    # No action_dim given -> derived from the robot's joint count (one joint -> 1).
    env = SimEnv(_engine(_OneJointEngine()), actor_obs_keys=["J"], reward_terms=[lambda e: 1.0])
    assert env.num_actions == 1


def test_requires_action_dim_when_no_robot() -> None:
    with pytest.raises(ValueError, match="action_dim must be given"):
        SimEnv(_engine(_NoRobotEngine()), actor_obs_keys=["J"], reward_terms=[lambda e: 1.0])


def test_reset_fn_invoked_instead_of_engine_reset() -> None:
    calls: dict[str, int] = {"reset_fn": 0, "engine_reset": 0}

    class _TrackingEngine(_OneJointEngine):
        def reset(self) -> dict:
            calls["engine_reset"] += 1
            return {"status": "success"}

    def reset_fn(engine: SimEngine) -> None:
        calls["reset_fn"] += 1

    env = SimEnv(
        _engine(_TrackingEngine()),
        actor_obs_keys=["J"],
        reward_terms=[lambda e: 1.0],
        action_dim=1,
        reset_fn=reset_fn,
    )
    env.reset()
    assert calls["reset_fn"] == 1
    # Custom reset_fn takes over: the engine's own reset() must not be called.
    assert calls["engine_reset"] == 0


def test_stateful_reward_term_reset_called_on_reset() -> None:
    class _StatefulTerm:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

        def __call__(self, engine: SimEngine) -> float:
            return 1.0

    term = _StatefulTerm()
    env = SimEnv(_engine(_OneJointEngine()), actor_obs_keys=["J"], reward_terms=[term], action_dim=1)
    # Construction does not reset the term; the first reset() does.
    assert term.reset_calls == 0
    env.reset()
    assert term.reset_calls == 1


def test_close_is_noop() -> None:
    env = SimEnv(_engine(_OneJointEngine()), actor_obs_keys=["J"], reward_terms=[lambda e: 1.0], action_dim=1)
    # close() owns no resources (engine lifecycle is the caller's); it must not raise.
    env.close()
