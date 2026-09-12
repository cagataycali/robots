"""``learning_starts`` must name a replay fill the run it configures can reach.

``learning_starts`` is the buffer fill an off-policy loop waits for before its
first gradient step, and two other caller-supplied counts bound the fill a run
ever reaches: the step budget it collects,
``max(1, total_timesteps // steps) * steps`` for ``steps = rollout_steps *
num_envs``, and ``buffer_size``, the ring buffer's own capacity. Neither
relation was checked, and both were reachable with plain positive ``int``
values that pass every per-field domain. Measured on the MuJoCo reach env
(FastSAC and FastTD3, one otherwise-valid spec per case):
``total_timesteps=20`` against ``learning_starts=32``, and ``buffer_size=8``
against ``learning_starts=16``, each returned ``validate() == []`` and then
``status="success"`` with a written checkpoint and an exported ``policy.pt`` -
having taken **zero** gradient steps, so the artifact was the randomly
initialized network ``setup`` built.

That is the outcome the two gates on either side of this one already cite as
the one they exist to refuse. ``rl_replay_problems``' docstring names a run
that "takes zero gradient updates for the whole run and still reports success
with a written checkpoint", and the ``learning_starts >= batch_size`` relation
is gated on the strict-``int`` count domain because a non-finite threshold
"skips the warmup and takes zero gradient steps ... a run that reports success
having learned nothing". ``buffer_size=True`` is refused for exactly this
reason - a one-slot buffer that never reaches the threshold - while
``int(True) == 1`` and ``buffer_size=1`` was accepted: the same buffer, the same
run, two verdicts. :class:`TestTheSameOneSlotBufferGetsOneVerdict` pins that.

The relation is asked only of counts. Every operand already has a gate that
reports a non-count as one, so a non-count is left to that gate rather than
described as an unreachable threshold; :class:`TestANonCountIsLeftToItsOwnGate`
pins both halves. Only the two off-policy backends wait for a warmup, so only
they ask - :class:`TestABackendWithNoWarmupStaysQuiet` pins the scope line.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import warmup_reachability_problems

torch = pytest.importorskip("torch")

from strands_robots.training.rl import RLTrainSpec, SimEnv  # noqa: E402

#: The two backends whose loop waits for a warmup fill.
OFF_POLICY = ["fast_sac", "fast_td3"]

#: Backends that read no ``learning_starts``, so they must stay silent about it.
NO_WARMUP = ["ppo", "mock"]

#: Every operand of the relation. A non-count in any one leaves it undecidable.
OPERANDS = ["total_timesteps", "rollout_steps", "num_envs", "learning_starts", "buffer_size"]

#: Spellings that are not counts, each already refused by its own field's gate.
NOT_A_COUNT: list[Any] = [True, 0.5, float("nan"), float("inf"), "8", None]

#: The two ways a positive-``int`` spec can put the threshold out of reach, as
#: ``(overrides, the field the refusal must name)``.
UNREACHABLE: dict[str, tuple[dict[str, Any], str]] = {
    "a step budget below the threshold": ({"total_timesteps": 20, "learning_starts": 32}, "total_timesteps"),
    "a capacity below the threshold": ({"buffer_size": 8, "learning_starts": 16, "batch_size": 8}, "buffer_size"),
}


class _FakeEngine:
    """One-joint fake engine: enough surface for ``SimEnv`` to wrap."""

    def __init__(self) -> None:
        self._j = 0.0

    def list_robots(self) -> list[str]:
        return ["fake"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["J"]

    def robot_action_keys(self, robot_name: str) -> list[str]:
        return ["J"]

    def reset(self) -> dict:
        self._j = 0.0
        return {"status": "success"}

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict:
        return {"J": self._j, "J.vel": 0.0}

    def send_action(self, action: Any, robot_name: str | None = None, n_substeps: int = 1) -> dict:
        self._j += 0.1 * (float(action[0]) if len(action) else 0.0)
        return {"status": "success"}


def _make_env() -> SimEnv:
    return SimEnv(_FakeEngine(), actor_obs_keys=["J", "J.vel"], reward_terms=[lambda e: 0.0], action_dim=1)  # type: ignore[arg-type]


def _spec(**overrides: Any) -> RLTrainSpec:
    """A small, otherwise-valid off-policy spec: only the relation is exercised."""
    base: dict[str, Any] = {
        "env_factory": _make_env,
        "output_dir": "/tmp/warmup_reachability",
        "device": "cpu",
        "total_timesteps": 40,
        "rollout_steps": 10,
        "learning_starts": 16,
        "batch_size": 16,
        "gradient_steps": 1,
        "hidden_dims": (16,),
        "seed": 0,
    }
    base.update(overrides)
    return RLTrainSpec(**base)


def _about_the_threshold(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about reachability."""
    return [p for p in create_trainer(provider).validate(spec) if "is never reached" in p]


class TestAnUnreachableThresholdIsRefused:
    """A spec whose warmup fill is out of reach is reported, not run."""

    @pytest.mark.parametrize("provider", OFF_POLICY)
    @pytest.mark.parametrize("case", list(UNREACHABLE), ids=lambda c: c)
    def test_the_problem_names_the_threshold_and_the_short_count(self, provider: str, case: str) -> None:
        overrides, short_field = UNREACHABLE[case]
        (problem,) = _about_the_threshold(provider, _spec(**overrides))
        assert "learning_starts" in problem
        assert short_field in problem
        assert "zero gradient steps" in problem

    @pytest.mark.parametrize("provider", OFF_POLICY)
    @pytest.mark.parametrize("case", list(UNREACHABLE), ids=lambda c: c)
    def test_nothing_is_built_and_no_checkpoint_is_exported(self, provider: str, case: str, tmp_path: Any) -> None:
        """The refusal precedes ``setup``, so the run that learns nothing never starts.

        Before the relation was checked this returned ``status="success"`` with a
        written ``policy.pt`` - the randomly initialized network, no gradient
        step having run.
        """
        overrides, _ = UNREACHABLE[case]
        result = create_trainer(provider).train(_spec(output_dir=str(tmp_path), **overrides))
        assert result.status == "error"
        assert result.checkpoint_dir in (None, "")
        assert result.exported_model in (None, "")
        assert "is never reached" in result.message
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("provider", OFF_POLICY)
    def test_both_short_counts_are_reported_together(self, provider: str) -> None:
        """One problem per operand, so a caller sees every count it has to raise."""
        problems = _about_the_threshold(provider, _spec(total_timesteps=20, buffer_size=8, learning_starts=32))
        assert len(problems) == 2
        assert any("total_timesteps" in p for p in problems)
        assert any("buffer_size" in p for p in problems)


class TestTheSameOneSlotBufferGetsOneVerdict:
    """``buffer_size=True`` and ``buffer_size=1`` are one buffer, so one verdict.

    The count domain refuses ``True`` because it builds a one-slot buffer that
    never reaches ``learning_starts`` and takes zero gradient steps under
    ``status="success"``. ``int(True) == 1``, so that is a property of the
    capacity rather than of the spelling, and ``1`` was accepted.
    """

    def test_the_two_spellings_are_the_same_capacity(self) -> None:
        assert int(True) == 1

    @pytest.mark.parametrize("provider", OFF_POLICY)
    @pytest.mark.parametrize("capacity", [True, 1])
    def test_neither_spelling_reaches_a_run(self, provider: str, capacity: Any) -> None:
        spec = _spec(buffer_size=capacity, learning_starts=16, batch_size=16)
        assert create_trainer(provider).validate(spec), f"{provider} accepted buffer_size={capacity!r}"

    @pytest.mark.parametrize("provider", OFF_POLICY)
    def test_each_spelling_is_refused_by_the_gate_that_owns_its_reason(self, provider: str) -> None:
        """The type rule still owns ``True``; the relation owns the capacity."""
        as_bool = create_trainer(provider).validate(_spec(buffer_size=True))
        as_one = create_trainer(provider).validate(_spec(buffer_size=1))
        assert any("buffer_size must be a positive integer" in p for p in as_bool)
        assert [p for p in as_one if "is never reached" in p]


class TestTheUsableDomainIsUntouched:
    """A reachable threshold is accepted, and the run still learns."""

    @pytest.mark.parametrize("provider", OFF_POLICY)
    def test_the_shipped_defaults_are_accepted(self, provider: str) -> None:
        assert _about_the_threshold(provider, _spec(env_factory=_make_env, **{})) == []
        assert warmup_reachability_problems(RLTrainSpec(output_dir="/tmp/x"), context=provider) == []

    @pytest.mark.parametrize("provider", OFF_POLICY)
    def test_a_reachable_spec_still_takes_gradient_steps(self, provider: str, tmp_path: Any) -> None:
        """Non-vacuity: the gate refuses the unreachable spec, not the small one."""
        trainer = create_trainer(provider)
        spec = _spec(output_dir=str(tmp_path))
        assert trainer.validate(spec) == []
        result = trainer.train(spec)
        assert result.status == "success"
        assert "latest_loss" in result.metrics  # an update ran

    def test_a_threshold_met_exactly_by_the_budget_is_reachable(self) -> None:
        """The boundary is inclusive: the fill only has to *reach* the threshold."""
        assert warmup_reachability_problems(_spec(total_timesteps=40, learning_starts=40), context="fast_sac") == []
        assert warmup_reachability_problems(_spec(buffer_size=16, learning_starts=16), context="fast_sac") == []

    def test_a_budget_below_one_iteration_still_collects_a_whole_iteration(self) -> None:
        """``max(1, ...)`` means a short budget collects ``steps``, not ``total_timesteps``."""
        spec = _spec(total_timesteps=3, rollout_steps=10, learning_starts=10)
        assert warmup_reachability_problems(spec, context="fast_sac") == []


class TestANonCountIsLeftToItsOwnGate:
    """The relation is only asked of counts; a non-count keeps its own message."""

    @pytest.mark.parametrize("field", OPERANDS)
    @pytest.mark.parametrize("value", NOT_A_COUNT, ids=repr)
    def test_the_relation_reports_nothing_about_it(self, field: str, value: Any) -> None:
        spec = _spec(**{field: value})
        assert warmup_reachability_problems(spec, context="fast_sac") == []

    @pytest.mark.parametrize("field", OPERANDS)
    @pytest.mark.parametrize("value", NOT_A_COUNT, ids=repr)
    def test_the_field_is_still_refused_by_its_own_domain(self, field: str, value: Any) -> None:
        """Non-vacuity: the silence above is scoping, not an accepted non-count."""
        problems = create_trainer("fast_sac").validate(_spec(**{field: value}))
        assert [p for p in problems if field in p], f"fast_sac accepted {field}={value!r}"


class TestABackendWithNoWarmupStaysQuiet:
    """A backend that never waits for a replay fill reports nothing about one."""

    @pytest.mark.parametrize("provider", NO_WARMUP)
    @pytest.mark.parametrize("case", list(UNREACHABLE), ids=lambda c: c)
    def test_it_reports_nothing_about_the_threshold(self, provider: str, case: str) -> None:
        overrides, _ = UNREACHABLE[case]
        assert _about_the_threshold(provider, _spec(**overrides)) == []

    def test_ppo_still_refuses_the_run_size_it_does_read(self) -> None:
        """Non-vacuity: PPO's silence is scoping, not a backend that checks nothing."""
        assert [p for p in create_trainer("ppo").validate(_spec(total_timesteps=0)) if "total_timesteps" in p]
