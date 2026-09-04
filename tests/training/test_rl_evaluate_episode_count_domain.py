"""``BaseRLAlgo.evaluate`` holds its episode count to the shared count domain.

``num_episodes`` is read three times inside ``evaluate``: as the ``range()``
bound of the episode loop, as the denominator of the reported ``success_rate``,
and verbatim as the ``num_episodes`` field of the returned metrics dict. A bare
``num_episodes <= 0`` test screened none of those, and the shared domain's own
docstring names the two ways it fails: an integral or non-finite float "raises
``TypeError`` ... rather than being coerced", and ``bool`` "is an ``int``
subclass, so a bare ``value < 1`` test lets ``True`` through as a silent count
of 1 while rejecting ``False``".

Both consequences were reachable through the public method:

* ``evaluate(num_episodes=True)`` ran one episode and reported
  ``{"num_episodes": True, "success_rate": <successes>/True}`` - a ``bool`` in a
  field documented ``int``, with a success rate divided by a flag.
* ``2.5`` / ``nan`` / ``inf`` passed the comparison and raised ``TypeError`` out
  of ``range()`` - after the env, networks and optimizers had been built and a
  checkpoint loaded - from a method documenting ``Raises: ValueError``. That
  raise escaped the eval-mode window before it was closed, and
  :class:`~strands_robots.training.rl.normalization.EmpiricalNormalization`
  freezes its running statistics in eval mode, so observation normalization
  silently stopped learning for the rest of the run.

The three sibling ``range()``-bound counts of an RL run (``total_timesteps`` /
``rollout_steps`` / ``num_envs``) already consult
:func:`~strands_robots.utils.positive_count_error`; this pins that the episode
count does too, and that a count it cannot honor leaves nothing frozen.

Same CPU-only shape as ``test_rl_evaluate.py``: a tiny stand-in engine drives a
deterministic 1-DOF env, so the normalizer, the actor-critic and the eval loop
are the real ones while the physics is not needed.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from strands_robots.simulation.base import SimEngine  # noqa: E402
from strands_robots.training.rl import PpoTrainer, RLTrainSpec, SimEnv  # noqa: E402
from strands_robots.training.rl.normalization import EmpiricalNormalization  # noqa: E402
from strands_robots.utils import positive_count_error  # noqa: E402

# Every value the method cannot honor as an episode count. ``0`` and ``-1`` are
# the two the replaced ``<= 0`` test already refused, kept as the rows that hold
# both before and after the fix; the rest are the ones it let through.
REFUSED: list[Any] = [0, -1, False, True, 2.5, 3.0, float("nan"), float("inf"), "2", None]
ACCEPTED: list[int] = [1, 3]


class _StandInEngine:
    """One joint ``J`` integrated by the action - no physics backend needed."""

    def __init__(self) -> None:
        self._j = 0.0
        self._vel = 0.0

    def list_robots(self) -> list[str]:
        return ["arm"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["J"]

    def robot_action_keys(self, robot_name: str) -> list[str]:
        return ["J"]

    def reset(self) -> dict[str, Any]:
        self._j = 0.0
        self._vel = 0.0
        return {"status": "success"}

    def get_observation(self, robot_name: Any = None, *, skip_images: bool = False) -> dict[str, float]:
        return {"J": self._j, "J.vel": self._vel}

    def send_action(self, action: Any, robot_name: Any = None, n_substeps: int = 1) -> dict[str, Any]:
        self._vel = 0.1 * (float(action[0]) if len(action) else 0.0)
        self._j += self._vel
        return {"status": "success"}


def _env_factory(reward: Any = None) -> Any:
    """Build a zero-arg factory for a deterministic 1-DOF env."""

    def factory() -> SimEnv:
        # ``SimEngine`` is a nominal ABC and ``SimEnv`` touches only the handful of
        # methods above, so the cast is safe here - the same idiom
        # ``test_rl_sim_env.py`` uses for its own stand-ins.
        return SimEnv(
            cast(SimEngine, _StandInEngine()),
            actor_obs_keys=["J", "J.vel"],
            reward_terms=[reward or (lambda e: 1.0)],
            action_dim=1,
            max_episode_steps=3,
        )

    return factory


def _live_trainer(tmp_path: Any, reward: Any = None) -> PpoTrainer:
    """A trainer that has been ``setup`` and has an observation normalizer."""
    trainer = PpoTrainer()
    trainer.setup(
        RLTrainSpec(
            env_factory=_env_factory(reward),
            output_dir=str(tmp_path),
            rollout_steps=4,
            num_mini_batches=2,
            hidden_dims=(8,),
            normalize_obs=True,
            seed=0,
        )
    )
    _normalizer(trainer)  # fail here, not mid-cell, if the spec built none
    return trainer


def _normalizer(trainer: PpoTrainer) -> EmpiricalNormalization:
    """The trainer's live observation normalizer.

    Asserted rather than skipped: every cell below is about what happens to the
    running statistics, so a trainer built without one would pass vacuously.
    """
    norm = trainer.actor_norm
    assert norm is not None, "this pin needs a live observation normalizer (normalize_obs=True)"
    return norm


def _normalizer_samples(trainer: PpoTrainer) -> int:
    """Samples the observation normalizer has folded into its running stats."""
    return int(_normalizer(trainer).count.item())


class TestTheEpisodeCountIsHeldToTheSharedDomain:
    """Only a positive integer can be honored, so only one is accepted."""

    @pytest.mark.parametrize("value", REFUSED, ids=repr)
    def test_a_value_that_is_not_a_positive_integer_is_refused(self, tmp_path: Any, value: Any) -> None:
        trainer = _live_trainer(tmp_path)
        with pytest.raises(ValueError) as excinfo:
            trainer.evaluate(num_episodes=value)
        assert "num_episodes" in str(excinfo.value), (
            f"evaluate({value!r}) was refused without naming the parameter: {excinfo.value}"
        )

    def test_the_refusal_is_the_shared_domains_own_wording(self, tmp_path: Any) -> None:
        """Graded through the answer, not the source: a hand-rolled copy would drift.

        The method must not re-implement the rule - it must ask the domain the
        three sibling counts ask, so the same count cannot be refused by one
        surface of an RL run and accepted by another.
        """
        trainer = _live_trainer(tmp_path)
        for value in REFUSED:
            expected = positive_count_error(value, "num_episodes", "evaluate")
            assert expected is not None, "the shared domain does not refuse this value - fix the table"
            with pytest.raises(ValueError) as excinfo:
                trainer.evaluate(num_episodes=value)
            assert str(excinfo.value) == expected, (
                f"evaluate({value!r}) answered {str(excinfo.value)!r}, not the shared domain's {expected!r}"
            )

    @pytest.mark.parametrize("value", ACCEPTED)
    def test_a_positive_integer_is_still_honored(self, tmp_path: Any, value: int) -> None:
        """Over-reach control: the accepted domain is unchanged."""
        out = _live_trainer(tmp_path).evaluate(num_episodes=value)
        assert out["num_episodes"] == value
        assert len(out["returns"]) == value
        assert 0.0 <= out["success_rate"] <= 1.0

    def test_a_bool_is_not_reported_as_the_episode_count(self, tmp_path: Any) -> None:
        """``True`` used to run one episode and be echoed into the metrics dict.

        ``success_rate`` is ``successes / num_episodes``, so a flag reaching the
        field made it the denominator of a reported rate as well as the loop
        bound - and ``False`` was refused while ``True`` was not.
        """
        trainer = _live_trainer(tmp_path)
        with pytest.raises(ValueError, match="num_episodes"):
            trainer.evaluate(num_episodes=True)


class TestACountItCannotHonorLeavesNothingFrozen:
    """The severity of the old guard: a refused count froze normalization.

    ``evaluate`` flips the actor-critic and the observation normalizer into
    ``eval()`` mode. ``collect_rollout`` re-enters ``actor_critic.train()``
    itself, but ``evaluate`` is the only place in the package that puts the
    normalizer back - so a raise that skipped the restore stopped the running
    statistics from ever updating again, with nothing reporting that.
    """

    @pytest.mark.parametrize("value", [2.5, float("nan"), float("inf")], ids=repr)
    def test_the_normalizer_still_learns_after_a_refused_count(self, tmp_path: Any, value: Any) -> None:
        trainer = _live_trainer(tmp_path)
        trainer.collect_rollout()
        before = _normalizer_samples(trainer)
        assert before > 0, "the normalizer folded nothing, so this cell would pass vacuously"

        with pytest.raises(ValueError, match="num_episodes"):
            trainer.evaluate(num_episodes=value)

        assert _normalizer(trainer).training is True, f"evaluate({value!r}) left the normalizer in eval() mode"
        trainer.collect_rollout()
        assert _normalizer_samples(trainer) > before, (
            f"observation normalization stopped learning after evaluate({value!r}): "
            f"the running stats are still at {before} samples"
        )


class TestTheEvalModeWindowIsClosedOnEveryExit:
    """A reward term raising on a live engine must not freeze the trainer either.

    The count guard removes the trigger this module is named for; the window is
    closed in ``finally`` so the stated "side-effect-free" contract does not
    depend on which exception was possible.
    """

    @staticmethod
    def _reward_that_fails_after(n: int) -> Any:
        calls = {"n": 0}

        def reward(engine: Any) -> float:
            calls["n"] += 1
            if calls["n"] > n:
                raise RuntimeError("reward term failed on a live engine")
            return 1.0

        reward.calls = calls  # type: ignore[attr-defined]
        return reward

    def test_an_exception_from_the_rollout_restores_the_modes(self, tmp_path: Any) -> None:
        reward = self._reward_that_fails_after(6)
        trainer = _live_trainer(tmp_path, reward=reward)
        trainer.collect_rollout()
        before = _normalizer_samples(trainer)

        with pytest.raises(RuntimeError, match="reward term failed"):
            trainer.evaluate(num_episodes=3)

        assert trainer.actor_critic.training is True, "actor_critic left in eval() after a raising rollout"
        assert _normalizer(trainer).training is True, "actor_norm left in eval() after a raising rollout"

        reward.calls["n"] = -(10**6)  # let the reward work again
        trainer.collect_rollout()
        assert _normalizer_samples(trainer) > before, "observation normalization stopped learning after a raise"

    def test_the_exception_is_not_swallowed(self, tmp_path: Any) -> None:
        """Control: the restore is a ``finally``, never a ``return``/``except``.

        Holds both before and after the fix - it is what stops the remedy from
        turning a failed evaluation into a reported one.
        """
        trainer = _live_trainer(tmp_path, reward=self._reward_that_fails_after(6))
        trainer.collect_rollout()
        with pytest.raises(RuntimeError, match="reward term failed"):
            trainer.evaluate(num_episodes=3)
