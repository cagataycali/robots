"""Unit tests for ``BaseRLAlgo.evaluate()`` - the deterministic eval peer of train().

CPU-only: a tiny fake ``SimEngine`` drives a deterministic 1-DOF env so the test
needs neither mujoco nor model downloads. Pins the eval contract: deterministic
rollout, frozen normalizer, schema, success_rate, whether that rate measured the
policy at all, and the not-setup guard. The ``num_episodes`` domain is pinned as
a table in ``test_rl_evaluate_episode_count_domain.py``.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from strands_robots.training.rl import PpoTrainer, RLTrainSpec, SimEnv  # noqa: E402


class _FakeEngine:
    """Minimal SimEngine stand-in: one joint ``J`` integrated by the action.

    ``J`` starts at 0.0; each ``send_action([a])`` moves it by ``0.1 * a``.
    ``J.vel`` reports the last delta. Enough to exercise reset/step/observe and
    a success predicate without any physics backend.
    """

    def __init__(self) -> None:
        self._j = 0.0
        self._vel = 0.0

    def list_robots(self) -> list[str]:
        return ["fake"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["J"]

    def robot_action_keys(self, robot_name: str) -> list[str]:
        # These fakes are duck-typed rather than ``SimEngine`` subclasses, so
        # they do not inherit the default that mirrors the joint names. This
        # robot's one joint is its one actuator, so the two vocabularies agree -
        # which is the shape ``SimEnv`` sizes its action head from.
        return ["J"]

    def reset(self) -> dict:
        self._j = 0.0
        self._vel = 0.0
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images: bool = False) -> dict:
        return {"J": self._j, "J.vel": self._vel}

    def send_action(self, action, robot_name=None, n_substeps: int = 1) -> dict:
        a = float(action[0]) if len(action) else 0.0
        self._vel = 0.1 * a
        self._j += self._vel
        return {"status": "success"}


def _make_env():  # type: ignore[no-untyped-def]
    eng = _FakeEngine()
    return SimEnv(
        eng,
        actor_obs_keys=["J", "J.vel"],
        reward_terms=[lambda e: -abs(float(e.get_observation(skip_images=True)["J"]) - 0.2)],
        action_dim=1,
        max_episode_steps=5,
    )


#: A success threshold below anything the joint can reach: satisfied at reset and
#: after every step, so the first step terminates the episode whatever the policy
#: commands. The pure form of a criterion on the wrong side of the initial state.
_UNREACHABLY_LOW = -1e9

#: A threshold the policy has to drive ``J`` up to, so success is something it earns.
_EARNED = 0.2


def _threshold_env_factory(threshold: float, **kwargs):  # type: ignore[no-untyped-def]
    """Factory for an env whose success predicate is ``J >= threshold``.

    ``J`` starts every episode at 0.0 and each step moves it by ``0.1 * a``, so
    the threshold decides whether the predicate is something the policy has to
    achieve or something the reset already handed it - which is the whole axis
    the tests below vary. Note the two are not symmetric about the resting value:
    a threshold AT 0.0 holds at reset but an action driving ``J`` negative
    un-satisfies it, so the misconfiguration that pins a hard 1.0 is a threshold
    the joint can never fall below - :data:`_UNREACHABLY_LOW`.
    """

    def factory():  # type: ignore[no-untyped-def]
        eng = _FakeEngine()
        return SimEnv(
            eng,
            actor_obs_keys=["J", "J.vel"],
            reward_terms=[lambda e: 1.0],
            action_dim=1,
            max_episode_steps=5,
            success_fn=lambda e: float(e.get_observation(skip_images=True)["J"]) >= threshold,
            **kwargs,
        )

    return factory


def _make_env_with_success():  # type: ignore[no-untyped-def]
    # Already satisfied at reset, so the episode terminates on its first step
    # whatever the policy commands. Kept as the fixture for the terminal-counting
    # tests because it is the shortest way to produce a terminal, and those tests
    # now also assert that evaluate() reports the rate as unearned - see
    # TestSuccessRateSaysWhetherThePolicyEarnedIt.
    return _threshold_env_factory(_UNREACHABLY_LOW)()


def test_evaluate_schema_and_determinism(tmp_path) -> None:  # type: ignore[no-untyped-def]
    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir=str(tmp_path),
        rollout_steps=4,
        num_mini_batches=2,
        hidden_dims=(8,),
        seed=0,
    )
    trainer.setup(spec)

    out = trainer.evaluate(num_episodes=3)
    # Schema.
    for k in (
        "num_episodes",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
        "mean_length",
        "success_rate",
        "success_measured",
        "episodes_successful_at_reset",
        "returns",
    ):
        assert k in out, f"missing key {k}"
    assert out["num_episodes"] == 3
    assert len(out["returns"]) == 3
    assert out["mean_length"] == 5.0  # no success_fn -> always times out at max_episode_steps
    assert out["success_rate"] == 0.0
    assert out["success_measured"] is False  # ... and that 0.0 measured nothing

    # Determinism: same weights -> identical eval returns.
    out2 = trainer.evaluate(num_episodes=3)
    assert out["returns"] == out2["returns"]


def test_evaluate_success_rate_counts_terminals(tmp_path) -> None:  # type: ignore[no-untyped-def]
    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env_with_success,
        output_dir=str(tmp_path),
        rollout_steps=4,
        num_mini_batches=2,
        hidden_dims=(8,),
        seed=0,
    )
    trainer.setup(spec)
    out = trainer.evaluate(num_episodes=4)
    # Predicate already true at reset -> every episode terminates on step 1, and
    # the rate is reported as one the scene handed the policy.
    assert out["success_rate"] == 1.0
    assert out["mean_length"] == 1.0
    assert out["episodes_successful_at_reset"] == 4


def test_evaluate_requires_spec_when_not_setup() -> None:
    trainer = PpoTrainer()
    with pytest.raises(ValueError, match="setup"):
        trainer.evaluate(num_episodes=2)


def test_evaluate_loads_checkpoint_fresh_instance(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Train briefly, checkpoint, then evaluate from a FRESH trainer + checkpoint.
    t1 = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir=str(tmp_path),
        total_timesteps=4 * 3,
        rollout_steps=4,
        num_mini_batches=2,
        num_learning_epochs=1,
        hidden_dims=(8,),
        seed=0,
    )
    result = t1.train(spec)
    assert result.status == "success"

    t2 = PpoTrainer()
    out = t2.evaluate(spec=spec, num_episodes=2)  # discovers latest checkpoint under output_dir
    assert out["num_episodes"] == 2
    assert len(out["returns"]) == 2


def test_evaluate_works_for_sac_too(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """evaluate() lives on BaseRLAlgo, so FastSAC inherits it unchanged.

    SAC's deterministic action is ``tanh(mean)`` via the same ``act_inference``
    contract PPO uses, so no per-subclass override is needed.
    """
    from strands_robots.training.rl import FastSacTrainer

    trainer = FastSacTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env_with_success,
        output_dir=str(tmp_path),
        rollout_steps=4,
        batch_size=16,
        learning_starts=16,
        hidden_dims=(8,),
        seed=0,
    )
    trainer.setup(spec)
    out = trainer.evaluate(num_episodes=3)
    assert out["num_episodes"] == 3
    assert out["success_rate"] == 1.0  # predicate already true at reset
    assert out["mean_length"] == 1.0
    assert out["episodes_successful_at_reset"] == 3  # inherited from BaseRLAlgo too


def test_evaluate_restores_train_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """evaluate() must be side-effect-free w.r.t. train/eval mode.

    Regression guard: a train -> evaluate -> train continuation needs the
    actor-critic (and obs normalizer) returned to training mode, else
    BatchNorm/Dropout running stats silently freeze on resumed training.
    """
    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir=str(tmp_path),
        rollout_steps=4,
        num_mini_batches=2,
        hidden_dims=(8,),
        seed=0,
    )
    trainer.setup(spec)

    # Force a known training mode, then evaluate, then assert it is restored.
    trainer.actor_critic.train()
    assert trainer.actor_critic.training is True
    actor_norm = getattr(trainer, "actor_norm", None)
    if actor_norm is not None:
        actor_norm.train()

    trainer.evaluate(num_episodes=2)

    assert trainer.actor_critic.training is True, "actor_critic left in eval() after evaluate()"
    if actor_norm is not None:
        assert actor_norm.training is True, "actor_norm left in eval() after evaluate()"

    # And the inverse: an eval-mode caller stays in eval mode.
    trainer.actor_critic.eval()
    trainer.evaluate(num_episodes=1)
    assert trainer.actor_critic.training is False, "evaluate() wrongly flipped a model that was in eval()"


class TestSuccessRateSaysWhetherThePolicyEarnedIt:
    """``evaluate()`` reports whether its ``success_rate`` measured the policy.

    The rate degenerates to a constant from two opposite directions, and neither
    is visible in the number itself:

    - No ``success_fn`` on the env: nothing can terminate, so every episode times
      out and the rate is a hard ``0.0`` - reading exactly like a policy that was
      scored and failed everything.
    - A predicate that already holds at reset: ``SimEnv.step`` samples it only
      after an applied action, so the episode terminates on its first step
      whatever the policy commands, contributing a hard ``1.0``.

    ``PolicyRunner.evaluate`` and ``PolicyRunner.evaluate_benchmark`` already
    report both for the same ``success_rate``; this is the third route publishing
    it. Each cell below pins the reported field and that the rate itself is left
    exactly as measured, since silently correcting it would hide the
    misconfigured predicate that produced it.
    """

    @staticmethod
    def _evaluate(tmp_path, factory, num_episodes=4):  # type: ignore[no-untyped-def]
        trainer = PpoTrainer()
        trainer.setup(
            RLTrainSpec(
                env_factory=factory,
                output_dir=str(tmp_path),
                rollout_steps=4,
                num_mini_batches=2,
                hidden_dims=(8,),
                seed=0,
            )
        )
        return trainer.evaluate(num_episodes=num_episodes)

    def test_the_premise_a_threshold_at_the_resting_value_already_holds(self) -> None:
        """Anti-vacuity: the misconfigured predicate is true before any step runs."""
        env = _threshold_env_factory(_UNREACHABLY_LOW)()
        env.reset()
        assert env.success_fn is not None
        assert env.success_fn(env.engine) is True, "the reset state already satisfies the threshold"

        reachable = _threshold_env_factory(_EARNED)()
        reachable.reset()
        assert reachable.success_fn is not None
        assert reachable.success_fn(reachable.engine) is False, "the earned threshold needs the policy to act"

    def test_a_predicate_already_true_at_reset_is_counted(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        out = self._evaluate(tmp_path, _threshold_env_factory(_UNREACHABLY_LOW))
        assert out["episodes_successful_at_reset"] == 4
        assert out["success_measured"] is True
        # Reported, not corrected.
        assert out["success_rate"] == 1.0
        assert out["mean_length"] == 1.0

    def test_a_predicate_the_policy_must_satisfy_is_not_counted(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The control: a threshold above the resting value leaves the count at zero."""
        out = self._evaluate(tmp_path, _threshold_env_factory(_EARNED))
        assert out["episodes_successful_at_reset"] == 0
        assert out["success_measured"] is True

    def test_a_missing_predicate_is_flagged_rather_than_read_as_failure(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        out = self._evaluate(tmp_path, _make_env)
        assert out["success_measured"] is False
        assert out["episodes_successful_at_reset"] == 0  # nothing to sample
        assert out["success_rate"] == 0.0  # left as measured

    @pytest.mark.parametrize(
        ("factory", "expected"),
        [
            (_threshold_env_factory(_UNREACHABLY_LOW), "already satisfied the success criterion at reset"),
            (_make_env, "does NOT measure task success"),
        ],
        ids=["true-at-reset", "no-predicate"],
    )
    def test_each_degenerate_rate_is_warned_about(self, tmp_path, caplog, factory, expected) -> None:  # type: ignore[no-untyped-def]
        with caplog.at_level(logging.WARNING, logger="strands_robots.training.rl.base_algo"):
            self._evaluate(tmp_path, factory)
        assert expected in caplog.text, caplog.text

    def test_an_honest_evaluation_is_not_warned_about(self, tmp_path, caplog) -> None:  # type: ignore[no-untyped-def]
        """The control for the warnings: a scored, unearned-free rate stays quiet."""
        with caplog.at_level(logging.WARNING, logger="strands_robots.training.rl.base_algo"):
            self._evaluate(tmp_path, _threshold_env_factory(_EARNED))
        assert "at reset" not in caplog.text, caplog.text
        assert "does NOT measure task success" not in caplog.text, caplog.text

    def test_the_warning_names_only_the_figures_this_route_reports(self, tmp_path, caplog) -> None:  # type: ignore[no-untyped-def]
        """The shared helper's default names ``pass_hat_k``, which this route has not got.

        A remedy naming a field absent from the result sends the reader looking
        for nothing, so ``evaluate`` narrows the helper to the figure it does
        publish.
        """
        with caplog.at_level(logging.WARNING, logger="strands_robots.training.rl.base_algo"):
            out = self._evaluate(tmp_path, _threshold_env_factory(_UNREACHABLY_LOW))
        assert "success_rate" in caplog.text
        assert "pass_hat_k" not in caplog.text, "named a figure evaluate() does not return"
        assert "pass_hat_k" not in out
        # The field the message tells the reader to look up is really there.
        assert "episodes_successful_at_reset" in caplog.text
        assert "episodes_successful_at_reset" in out

    def test_a_predicate_that_raises_at_reset_is_not_fatal(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The reset sample is diagnostic, so it must not abort an otherwise-fine eval.

        A predicate can legitimately read state that only a first step
        establishes. It raises on the reset sample here and answers normally
        afterwards, which is exactly that shape.
        """
        calls = {"n": 0}

        def flaky(engine) -> bool:  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] == 1:
                raise KeyError("a key the first step would have written")
            return False

        def factory():  # type: ignore[no-untyped-def]
            return SimEnv(
                _FakeEngine(),
                actor_obs_keys=["J", "J.vel"],
                reward_terms=[lambda e: 1.0],
                action_dim=1,
                max_episode_steps=3,
                success_fn=flaky,
            )

        out = self._evaluate(tmp_path, factory, num_episodes=1)
        assert out["num_episodes"] == 1
        assert out["episodes_successful_at_reset"] == 0  # not sampled, not counted
        assert calls["n"] > 1, "the predicate kept being consulted after the reset raise"
