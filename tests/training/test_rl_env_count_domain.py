"""``RLTrainSpec.num_envs`` is a count on every backend, whatever count each wants.

``num_envs`` is the third caller-supplied factor of the one loop bound every RL
backend derives::

    steps_per_iter = rollout_steps * num_envs
    num_iters = max(1, total_timesteps // steps_per_iter)
    for it in range(num_iters):  # collect, update

Its two siblings are held to :func:`~strands_robots.utils.positive_count_error`
by :func:`~strands_robots.training._validate.rl_run_size_problems`, whose own
docstring works through why a bare ``value <= 0`` test on a *derived* bound is
weaker than the same test on a field read straight off the spec. ``num_envs`` was
left out of that gate for a real reason - which counts are usable genuinely
differs per backend, PPO parallelizing over any positive count where the
MuJoCo-backed FastSAC requires exactly ``1`` - and each backend stated its own
rule with a bare comparison.

That reason covers which *counts* are usable. It does not cover whether the value
is a count, which is the same on both backends and was stated by neither.
Measured before this gate existed, over ``total_timesteps=1000`` and
``rollout_steps=64`` (intended: 15 iterations of 64 steps):

* ``nan`` passed PPO's ``< 1`` test. ``64 * nan`` is ``nan``, ``max(1, nan)``
  keeps ``1``, and ``1000 // 1`` is ``1000``: the run reported ``success`` and
  announced ``"1000 iterations x 1 steps complete"`` while each of those 1000
  iterations collected ``rollout_steps`` steps - 66x the requested budget, under a
  message that misdescribes both factors.
* ``inf`` passed the same test and ran **one** iteration, announcing
  ``"1 iterations x inf steps complete"``.
* ``2.5`` passed and made ``num_iters`` the float ``6.0``, which raises
  ``TypeError: 'float' object cannot be interpreted as an integer`` out of
  ``range()`` - after ``setup`` had built the environment, the networks and the
  optimizers, which is the cost a read-only preflight exists to precede. ``1.0``
  passed FastSAC's ``!= 1`` as well and raised the same way from ``15.0``. A
  *large* float does not: ``100.5`` makes ``1000 // 6432.0`` the float ``0.0``,
  which the clamp replaces with the int ``1``, so it runs one iteration instead.
* ``True`` passed both backends - it satisfies ``< 1`` being false and ``!= 1``
  being false - and is numerically ``1``, so the run length is right and what is
  wrong is that a value reading as a flag was accepted as a count. That is the
  reason the shared domain refuses a ``bool`` rather than testing a bound.
* ``"4"`` and ``None`` raised ``TypeError: '<' not supported between instances of
  'str' and 'int'`` out of PPO's comparison itself - from a
  :meth:`~strands_robots.training.base.Trainer.validate` documented to *return*
  problems.

So each backend now consults the shared domain first and asks its own count rule
only of a count. The scope line the exclusion draws is unchanged and is pinned
here too: ``4`` is usable for the parallel backend and refused by the single-env
one, on both trees.
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from typing import Any

import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import rl_run_size_problems
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.training.rl.fast_sac import FastSacTrainer
from strands_robots.training.rl.ppo import PpoTrainer
from strands_robots.utils import positive_count_error

# The backends that derive a loop bound from this factor.
RL_BACKENDS = ("ppo", "fast_sac", "fast_td3")

# Values that survive a bare per-backend comparison and cannot bound the loop.
# ``True`` / ``1.0`` survive *both* comparisons; the rest survive PPO's ``< 1``.
NOT_A_COUNT: list[Any] = [True, False, 1.0, 2.5, 100.5, float("nan"), float("inf"), "4", None, [1], {"n": 1}]

# The subset a bare comparison could not even report on: it raises instead.
RAISED_OUT_OF_THE_COMPARISON: list[Any] = ["4", None, [1], {"n": 1}]

# The values that SURVIVED a bare comparison, classified by what each did to the
# derived bound. Every other value in ``NOT_A_COUNT`` was already refused by one
# comparison or the other, so it is graded above but carries no harm of its own.
#
# ``max(1, ...)`` reads ``nan`` as ``1`` (nan compares false against everything)
# and ``inf`` as ``inf``, so the two land on opposite sides of the intended count.
WRONG_RUN_LENGTH: dict[Any, int] = {float("nan"): 1000, float("inf"): 1}

# An integral or fractional float keeps the bound a float, but only while
# ``total_timesteps // steps_per_iter`` is at least one - a large enough float is
# clamped back to the int ``1`` and lands in ``WRONG_RUN_LENGTH``'s territory
# instead. These two are the reachable float-bound cases at this run size.
FLOAT_BOUND: list[float] = [1.0, 2.5]

# ``True`` is numerically ``1``, so it does not change the run length at all: the
# harm is that a value which reads as a flag is accepted as a count of one, the
# same reason the shared domain refuses a ``bool`` everywhere else.
READS_AS_ONE: list[Any] = [True]

# A positive ``int`` is the whole shared half.
A_COUNT: list[Any] = [1, 2, 4, 1024]


@pytest.fixture
def spec() -> RLTrainSpec:
    """A spec whose every other field is usable, so only ``num_envs`` is in doubt."""
    return RLTrainSpec(
        output_dir="/tmp/strands-rl-env-count",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
        total_timesteps=1000,
        rollout_steps=64,
    )


def _num_envs_problems(provider: str, spec: RLTrainSpec, value: Any) -> list[str]:
    """The problems ``provider``'s real ``validate`` reports about ``num_envs``."""
    spec.num_envs = value
    return [p for p in create_trainer(provider).validate(spec) if "num_envs" in p]


class TestEveryBackendRefusesANonCount:
    """The shared half, through the real ``validate`` entry point."""

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("value", NOT_A_COUNT)
    def test_a_non_count_is_refused(self, provider: str, spec: RLTrainSpec, value: Any) -> None:
        assert _num_envs_problems(provider, spec, value)

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("value", NOT_A_COUNT)
    def test_the_refusal_names_the_field_and_the_domain(self, provider: str, spec: RLTrainSpec, value: Any) -> None:
        """A caller fixes a field it can name, against a rule it can read."""
        problems = _num_envs_problems(provider, spec, value)
        assert any("num_envs must be a positive integer" in p for p in problems), problems

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("value", NOT_A_COUNT)
    def test_the_refusal_names_the_backend_that_refused(self, provider: str, spec: RLTrainSpec, value: Any) -> None:
        """The shared domain carries the caller's own identity into the message."""
        trainer = create_trainer(provider)
        problems = _num_envs_problems(provider, spec, value)
        assert any(p.startswith(f"{trainer.provider_name}: num_envs") for p in problems), problems

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("value", RAISED_OUT_OF_THE_COMPARISON)
    def test_a_non_numeric_value_is_returned_rather_than_raised(
        self, provider: str, spec: RLTrainSpec, value: Any
    ) -> None:
        """``validate`` is documented to return problems, so it must not raise here."""
        spec.num_envs = value
        problems = create_trainer(provider).validate(spec)  # must not raise
        assert any("num_envs" in p for p in problems), problems


class TestTheHarmTheDomainReplaces:
    """The arithmetic that made a surviving non-count a wrong-length run.

    These read the derived bound rather than the trainers, because the values are
    refused now - the point is what they *would* do to the loop, which is why
    refusing them at a read-only preflight is the fix.
    """

    @pytest.mark.parametrize(("value", "expected"), list(WRONG_RUN_LENGTH.items()))
    def test_a_surviving_non_count_changes_the_run_length(self, value: Any, expected: int) -> None:
        """And by how much, in each direction, so a wrong number here fails loudly."""
        intended = max(1, 1000 // max(1, 64 * 1))
        assert intended == 15
        got = max(1, 1000 // max(1, 64 * value))
        assert got == expected != intended, (value, got, expected, intended)

    def test_nan_lengthens_the_run_where_inf_shortens_it(self) -> None:
        """The clamp, not the value, decides the direction - which is why it is silent.

        ``nan`` compares false against everything, so ``max(1, nan)`` keeps the
        ``1`` and the budget divides by a single step; ``inf`` survives the clamp
        and swallows the whole budget in one iteration.
        """
        assert math.isnan(64 * float("nan"))
        assert max(1, 64 * float("nan")) == 1
        assert max(1, 64 * float("inf")) == float("inf")

    @pytest.mark.parametrize("value", FLOAT_BOUND)
    def test_a_float_makes_the_loop_bound_unusable_as_a_range(self, value: float) -> None:
        num_iters = max(1, 1000 // max(1, 64 * value))
        assert isinstance(num_iters, float), (value, num_iters)
        with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
            range(num_iters)  # type: ignore[call-overload]

    def test_a_large_float_is_clamped_back_to_a_usable_bound_and_is_still_wrong(self) -> None:
        """The float-bound harm has a boundary; past it the harm is the length again.

        ``100.5`` makes ``1000 // 6432.0`` the float ``0.0``, which the clamp
        replaces with the int ``1`` - so ``range()`` is happy and the run is one
        iteration instead of fifteen. Recorded so the two harms are not conflated:
        a float is not always a float bound.
        """
        num_iters = max(1, 1000 // max(1, 64 * 100.5))
        assert num_iters == 1
        assert isinstance(num_iters, int)
        assert range(num_iters) is not None

    @pytest.mark.parametrize("value", READS_AS_ONE)
    def test_a_flag_is_numerically_one_so_the_length_is_not_the_harm(self, value: Any) -> None:
        """``True`` runs the intended length; what it changes is what the field means.

        This is the whole reason the shared domain refuses a ``bool`` rather than
        testing a bound: the arithmetic cannot tell a flag from the count ``1``.
        """
        assert max(1, 1000 // max(1, 64 * value)) == max(1, 1000 // max(1, 64 * 1))
        assert positive_count_error(value, "num_envs", "x") is not None


class TestTheDomainIsConsultedBeforeTheCountRule:
    """Ordering, structurally: a non-count is reported as one, not as a wrong count."""

    @pytest.mark.parametrize(("provider", "trainer_class"), [("ppo", PpoTrainer), ("fast_sac", FastSacTrainer)])
    def test_the_domain_call_precedes_any_local_comparison(self, provider: str, trainer_class: type[Trainer]) -> None:
        # ``inspect.getsource`` of a method is indented; ``ast.parse`` is not.
        tree = ast.parse(textwrap.dedent(inspect.getsource(trainer_class.validate)))
        domain_lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "positive_count_error"
            and any(_names_num_envs(arg) for arg in node.args)
        ]
        comparison_lines = [
            node.lineno for node in ast.walk(tree) if isinstance(node, ast.Compare) and _names_num_envs(node.left)
        ]
        assert len(domain_lines) == 1, (provider, domain_lines)
        assert all(domain_lines[0] < line for line in comparison_lines), (
            provider,
            domain_lines,
            comparison_lines,
        )

    def test_the_single_env_message_is_reserved_for_a_real_count(self, spec: RLTrainSpec) -> None:
        """A non-count is not reported as the wrong number of environments."""
        problems = _num_envs_problems("fast_sac", spec, float("nan"))
        assert not any("single-env" in p for p in problems), problems


class TestThePerBackendCountRuleIsUnchanged:
    """The scope line the exclusion draws, pinned on this tree too."""

    @pytest.mark.parametrize("value", A_COUNT)
    def test_the_parallel_backend_accepts_any_positive_count(self, spec: RLTrainSpec, value: int) -> None:
        assert _num_envs_problems("ppo", spec, value) == []

    def test_the_single_env_backend_accepts_exactly_one(self, spec: RLTrainSpec) -> None:
        assert _num_envs_problems("fast_sac", spec, 1) == []

    @pytest.mark.parametrize("value", [2, 4, 1024])
    def test_the_single_env_backend_still_refuses_a_larger_count(self, spec: RLTrainSpec, value: int) -> None:
        problems = _num_envs_problems("fast_sac", spec, value)
        assert any("single-env" in p for p in problems), problems

    def test_the_backends_still_disagree_about_the_same_count(self, spec: RLTrainSpec) -> None:
        """Two envs: usable for the parallel backend, refused by the single-env one."""
        assert _num_envs_problems("ppo", spec, 2) == []
        assert _num_envs_problems("fast_sac", spec, 2)


class TestThePremises:
    """What the fix rests on, so a change to any of it fails here rather than silently."""

    def test_the_shared_domain_refuses_every_probe_value(self) -> None:
        """Non-vacuity: the domain is what does the refusing, for all of them."""
        for value in NOT_A_COUNT:
            assert positive_count_error(value, "num_envs", "x") is not None, value

    def test_the_shared_domain_accepts_every_count(self) -> None:
        for value in A_COUNT:
            assert positive_count_error(value, "num_envs", "x") is None, value

    def test_the_domain_is_exactly_the_parallel_backends_own_rule(self) -> None:
        """Why PPO keeps no second comparison: over the accepted set, ``>= 1`` holds.

        A ``< 1`` branch after this domain would be unreachable, and dead code
        that reads as a rule is worse than none.
        """
        for value in A_COUNT:
            assert value >= 1, value

    def test_the_shared_gate_says_which_half_of_the_field_it_excludes(self) -> None:
        """The exclusion is documented as covering the count rule, not the field.

        The gate's docstring is where a contributor reads why ``num_envs`` is not
        in it. Left saying the whole field is out, it invites the next reader to
        leave the shared half open again - which is exactly how it was open.
        """
        doc = " ".join((rl_run_size_problems.__doc__ or "").split())
        assert "num_envs" in doc
        assert "which *counts* are usable differs between the backends" in doc
        assert "count *at all* is not per-backend" in doc

    def test_the_two_sibling_factors_are_held_to_the_same_domain(self, spec: RLTrainSpec) -> None:
        """The domain is shared with the two factors this one is multiplied by."""
        for field in ("total_timesteps", "rollout_steps"):
            setattr(spec, field, 0)
            problems = [p for p in create_trainer("ppo").validate(spec) if field in p]
            assert any("must be a positive integer" in p for p in problems), (field, problems)
            setattr(spec, field, 64)


def _names_num_envs(node: ast.AST) -> bool:
    """Whether ``node`` is the ``spec.num_envs`` attribute read."""
    return isinstance(node, ast.Attribute) and node.attr == "num_envs"
