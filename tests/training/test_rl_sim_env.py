"""Unit tests for ``SimEnv`` - the single-environment RL wrapper over a ``SimEngine``.

CPU-only, fake engine. Pins the construction-time validation contract (obs keys,
reward terms, substeps, action-dim inference / requirement) and the reset
lifecycle hooks (custom ``reset_fn``, stateful reward-term ``reset``), plus the
``close`` no-op contract that keeps ``SimEnv`` interface-compatible with
``VecSimEnv`` / ``GymSimEnv``. These are behaviors the RL trainers rely on: a
typo in obs keys or a missing action dim must fail loudly at construction, not
mid-rollout. Also pins the asymmetric actor-critic observation contract: the
critic sees the actor's keys plus any privileged ones, so a privileged key never
costs the critic the state it is valuing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from strands_robots.simulation.base import SimEngine  # noqa: E402 - after torch importorskip
from strands_robots.training.rl import SimEnv  # noqa: E402 - after torch importorskip
from tests.training._engine_stand_in import EngineStandIn  # noqa: E402


def _engine(**kwargs: object) -> EngineStandIn:
    """The engine these tests wrap: one joint ``J`` free-running at unit velocity."""
    return EngineStandIn(gain=0.0, drift=1.0, **kwargs)  # type: ignore[arg-type]


def test_rejects_empty_actor_obs_keys() -> None:
    with pytest.raises(ValueError, match="actor_obs_keys"):
        SimEnv(_engine(), actor_obs_keys=[], reward_terms=[lambda e: 1.0], action_dim=1)


def test_rejects_empty_reward_terms() -> None:
    with pytest.raises(ValueError, match="reward_terms"):
        SimEnv(_engine(), actor_obs_keys=["J"], reward_terms=[], action_dim=1)


def test_rejects_nonpositive_n_substeps() -> None:
    with pytest.raises(ValueError, match="n_substeps"):
        SimEnv(
            _engine(),
            actor_obs_keys=["J"],
            reward_terms=[lambda e: 1.0],
            action_dim=1,
            n_substeps=0,
        )


def test_infers_action_dim_from_robot_action_keys() -> None:
    # No action_dim given -> derived from the robot's action-key count, which
    # send_action binds a vector against (one actuator -> 1).
    env = SimEnv(_engine(), actor_obs_keys=["J"], reward_terms=[lambda e: 1.0])
    assert env.num_actions == 1


def test_requires_action_dim_when_no_robot() -> None:
    with pytest.raises(ValueError, match="action_dim must be given"):
        SimEnv(_engine(robots=()), actor_obs_keys=["J"], reward_terms=[lambda e: 1.0])


def test_reset_fn_invoked_instead_of_engine_reset() -> None:
    calls: dict[str, int] = {"reset_fn": 0}

    def reset_fn(engine: SimEngine) -> None:
        calls["reset_fn"] += 1

    engine = _engine()
    env = SimEnv(
        engine,
        actor_obs_keys=["J"],
        reward_terms=[lambda e: 1.0],
        action_dim=1,
        reset_fn=reset_fn,
    )
    env.reset()
    assert calls["reset_fn"] == 1
    # Custom reset_fn takes over: the engine's own reset() must not be called.
    assert engine.resets == 0


def test_stateful_reward_term_reset_called_on_reset() -> None:
    class _StatefulTerm:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

        def __call__(self, engine: SimEngine) -> float:
            return 1.0

    term = _StatefulTerm()
    env = SimEnv(_engine(), actor_obs_keys=["J"], reward_terms=[term], action_dim=1)
    # Construction does not reset the term; the first reset() does.
    assert term.reset_calls == 0
    env.reset()
    assert term.reset_calls == 1


def test_close_is_noop() -> None:
    env = SimEnv(_engine(), actor_obs_keys=["J"], reward_terms=[lambda e: 1.0], action_dim=1)
    # close() owns no resources (engine lifecycle is the caller's); it must not raise.
    env.close()


# --- termination vs truncation classification (the SAC/PPO bootstrap contract) ---
#
# ``step`` MUST report a time-out (episode-length limit) and a genuine success
# terminal as DISTINCT events: a time-out is a truncation whose successor value
# is still bootstrapped, while a success is a terminal that stops the value
# backup. Collapsing the two -- surfacing a time-out as ``terminated`` -- silently
# breaks the off-policy target (the FastTD3 truncation-bootstrap bug, fixed
# upstream Jun 2025). These pin that the flags never collapse.


def test_step_timeout_is_truncation_not_terminal() -> None:
    """A time-out sets done=1 but reports time_out (bootstrappable), NOT terminated."""
    # max_episode_steps=1 -> the first step is a time-out; no success_fn -> never a terminal.
    env = SimEnv(
        _engine(),
        actor_obs_keys=["J"],
        reward_terms=[lambda e: 1.0],
        action_dim=1,
        max_episode_steps=1,
    )
    env.reset()
    _, _, done, info = env.step(torch.zeros(1, 1))
    assert float(done.reshape(-1)[0]) == 1.0  # episode ends (the caller resets)
    assert info["time_out"] is True  # ...but as a truncation -> bootstrap the value
    assert info["terminated"] is False  # a time-out is NOT a terminal state


def test_step_success_is_terminal_not_truncation() -> None:
    """A success_fn hit sets terminated (no bootstrap), NOT time_out; disproves always-False."""
    # success on the first step, with head-room before the time-out limit so the
    # two conditions are unambiguously separable.
    env = SimEnv(
        _engine(),
        actor_obs_keys=["J"],
        reward_terms=[lambda e: 1.0],
        action_dim=1,
        max_episode_steps=99,
        success_fn=lambda e: True,
    )
    env.reset()
    _, _, done, info = env.step(torch.zeros(1, 1))
    assert float(done.reshape(-1)[0]) == 1.0
    assert info["terminated"] is True  # genuine terminal -> stop the value backup
    assert info["time_out"] is False  # not a truncation


def test_step_truncation_boundary_is_exact() -> None:
    """time_out fires exactly at step == max_episode_steps, not the step before."""
    env = SimEnv(
        _engine(),
        actor_obs_keys=["J"],
        reward_terms=[lambda e: 1.0],
        action_dim=1,
        max_episode_steps=2,
    )
    env.reset()
    _, _, done1, info1 = env.step(torch.zeros(1, 1))
    assert float(done1.reshape(-1)[0]) == 0.0  # step 1 of 2 -> not yet a time-out
    assert info1["time_out"] is False
    _, _, done2, info2 = env.step(torch.zeros(1, 1))
    assert float(done2.reshape(-1)[0]) == 1.0  # step 2 == limit -> time-out
    assert info2["time_out"] is True


class TestCriticObservationComposition:
    """The critic observes the actor's keys plus the privileged ones, not instead.

    ``critic_obs_keys`` names the *extra*, simulation-only keys of an asymmetric
    actor-critic. Reading it as the critic's whole observation silently costs the
    critic every actor key the moment one privileged key is named: the value/Q
    head is then sized for, and fed, a vector that no longer contains the state
    whose value it is estimating, and training runs to completion reporting a
    loss either way.
    """

    @staticmethod
    def _env(critic_obs_keys: object = "unset") -> SimEnv:
        kwargs = {} if critic_obs_keys == "unset" else {"critic_obs_keys": critic_obs_keys}
        return SimEnv(
            _engine(extra_obs={"cube_dist": 2.0}),
            actor_obs_keys=["J", "J.vel"],
            reward_terms=[lambda e: 1.0],
            action_dim=1,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_a_privileged_key_is_added_to_the_actor_keys(self) -> None:
        env = self._env(["cube_dist"])
        assert env.critic_obs_keys == ["J", "J.vel", "cube_dist"]
        assert env.num_critic_obs == 3
        assert env.num_actor_obs == 2

    def test_the_critic_vector_carries_the_actor_values_and_then_the_privileged_one(self) -> None:
        # The values, not just the key list: the vector is what reaches the head.
        obs = self._env(["cube_dist"]).reset()
        assert obs["actor_obs"].reshape(-1).tolist() == [0.0, 1.0]
        assert obs["critic_obs"].reshape(-1).tolist() == [0.0, 1.0, 2.0]

    def test_repeating_an_actor_key_does_not_widen_the_critic_observation(self) -> None:
        # A second copy of a scalar the critic already holds is not information,
        # and the width it would add is stamped into a checkpoint.
        env = self._env(["J", "cube_dist"])
        assert env.critic_obs_keys == ["J", "J.vel", "cube_dist"]
        assert env.num_critic_obs == 3

    def test_an_empty_privileged_list_leaves_the_critic_symmetric(self) -> None:
        # "nothing to add", not "the critic observes nothing" - a zero-width
        # critic observation is not a configuration any caller can want.
        env = self._env([])
        assert env.critic_obs_keys == ["J", "J.vel"]
        assert env.num_critic_obs == 2

    def test_omitting_the_privileged_keys_leaves_the_critic_symmetric(self) -> None:
        # Control: the documented default was already correct and is unchanged.
        env = self._env()
        assert env.critic_obs_keys == ["J", "J.vel"]
        assert env.num_critic_obs == 2
