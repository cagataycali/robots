# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``learning_starts`` is a count, so FastSAC asks it of the count domain.

``learning_starts`` is one side of the relation ``learning_starts >= batch_size``,
and that relation was the only thing standing between the field and its two
consumers. The relation's *other* operand was already asked of
:func:`~strands_robots.utils.positive_count_error` before being compared; this
one was not, on the stated grounds that it is "one side of a relation rather than
a bare count". That reason is about the shape of the *rule* and says nothing
about whether the value is a count at all, and a comparison is not a domain: a
non-finite value compares False against every ``int`` (nothing is greater than
``nan``, and ``inf`` is below no integer), so the relation passed and reported
nothing.

Both consumers then read a value that is not a count.
:meth:`~strands_robots.training.rl.fast_sac.FastSacTrainer.collect_rollout` tests
``buffer.size < learning_starts`` to decide whether to draw a random warmup
action, and ``train`` tests ``buffer.size >= learning_starts`` to decide whether
``update()`` runs at all. Measured on the MuJoCo reach env used by
``tests/training/test_rl_fast_sac.py`` (40 timesteps, ``rollout_steps=10``,
``batch_size=16``, ``gradient_steps=2``, one field mutated):

=========================  ==========  ========  =====================
``learning_starts``        ``validate``  ``train``  ``update()`` calls
=========================  ==========  ========  =====================
``16`` (a real count)      ``[]``      success   3
``float("nan")``           ``[]``      success   **0**
``float("inf")``           ``[]``      success   **0**
=========================  ==========  ========  =====================

So the run built the environment, the networks, the optimizers and the replay
buffer, collected every rollout, wrote a checkpoint and reported
``status="success"`` having taken no gradient step at all. That is the outcome
``_rl_replay_problems`` exists to refuse for ``buffer_size`` - its docstring
records the same "zero gradient updates, yet the run reported success" - reached
through the field its scope line excluded. ``nan`` additionally skips the random
warmup the field exists to provide, so the untrained actor drives from step one.

The domain applied here is the strict-``int``
:func:`~strands_robots.utils.positive_count_error` the relation's other operand
already uses, so the relation now compares two values drawn from one domain. Two
consequences of *strict*, both deliberate and pinned below: an integral float
such as ``1000.0`` and a ``numpy`` integer are now refused although the bare
comparison accepted them, because the field is annotated ``int`` and a relation
between a strict count and a loose one is the drift this closes; and a very large
``int`` is still accepted, because magnitude is not the axis - ``10**400`` is a
count, and a run that has not reached it has genuinely not finished warming up.

The scope of the *gate* is unchanged: ``_rl_replay_problems`` still reports
nothing about this field, because PPO reads neither it nor the three replay
counts. ``tests/training/test_rl_replay_domain.py`` pins that.

The field and its relation are owned by
:func:`~strands_robots.training._validate.warmup_batch_relation_problems`,
reached through ``Trainer._rl_warmup_batch_problems``. Both off-policy backends
state the rule identically, so the structural pins below read it off that one
owner and require that no backend restate it - a second copy is a bound that can
drift from the shared one.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
import textwrap
from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import rl_replay_problems, warmup_batch_relation_problems
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import positive_count_error
from tests.training._spec_field_reads import reads_spec_field

#: The batch size every case below pairs with, so the relation has a real operand.
BATCH_SIZE = 256

#: A count comfortably above :data:`BATCH_SIZE`, so the relation is satisfied and
#: only the domain can refuse.
VALID = 1_000

#: Values a bare ``learning_starts < batch_size`` accepted while being unusable as
#: a count. These are the silent half: ``validate`` reported nothing at all.
SILENTLY_ACCEPTED: list[Any] = [float("nan"), float("inf"), 1_000.0]

#: Values the bare comparison already rejected, but as a *relation* failure -
#: "must be >= batch_size" invites raising the value, when the value is not a
#: count in the first place.
MISDESCRIBED: list[Any] = [True, False, 0.5, -1.0, 0, -5, -1_000]

#: Values that reached ``<`` and raised out of a ``validate`` documented to
#: *return* its problems.
RAISED_OUT_OF_VALIDATE: list[Any] = ["1000", None, [1000]]

#: Every spelling that is not a usable count, however it failed before.
NOT_A_COUNT: list[Any] = [*SILENTLY_ACCEPTED, *MISDESCRIBED, *RAISED_OUT_OF_VALIDATE]


def _spec(**overrides: Any) -> RLTrainSpec:
    return RLTrainSpec(batch_size=BATCH_SIZE, **overrides)


def _problems(value: Any) -> list[str]:
    return create_trainer("fast_sac").validate(_spec(learning_starts=value))


def _about_learning_starts(value: Any) -> list[str]:
    return [p for p in _problems(value) if "learning_starts" in p]


class TestFastSacRefusesALearningStartsThatIsNotACount:
    """The regression: every unusable spelling is reported, none reaches a run."""

    @pytest.mark.parametrize("value", NOT_A_COUNT, ids=repr)
    def test_it_is_reported_as_a_problem(self, value: Any) -> None:
        assert _about_learning_starts(value)

    @pytest.mark.parametrize("value", NOT_A_COUNT, ids=repr)
    def test_validate_returns_rather_than_raising(self, value: Any) -> None:
        """A ``validate`` documented to return problems must not raise one."""
        assert isinstance(_problems(value), list)

    @pytest.mark.parametrize("value", NOT_A_COUNT, ids=repr)
    def test_the_message_names_the_field_and_the_domain(self, value: Any) -> None:
        problem = _about_learning_starts(value)[0]
        assert "learning_starts" in problem
        assert "positive integer" in problem

    @pytest.mark.parametrize("value", NOT_A_COUNT, ids=repr)
    def test_the_message_is_the_shared_domain_one_verbatim(self, value: Any) -> None:
        """The wording is the shared rule's, so the two cannot drift apart."""
        expected = positive_count_error(value, "learning_starts", "fast_sac")
        assert expected is not None
        assert expected in _about_learning_starts(value)

    @pytest.mark.parametrize("value", NOT_A_COUNT, ids=repr)
    def test_the_refusal_names_the_backend(self, value: Any) -> None:
        assert _about_learning_starts(value)[0].startswith("fast_sac:")


class TestTheHarmTheRelationCouldNotSee:
    """Why a comparison is not a domain, as arithmetic rather than as prose.

    These hold on any tree - they are properties of IEEE comparison, not of the
    fix - and they are what makes the refusal necessary: both consumers read one
    of these two comparisons to decide what the run does.
    """

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_the_relation_could_not_reject_it(self, value: float) -> None:
        """``learning_starts < batch_size`` is False, so the relation passed."""
        assert not value < BATCH_SIZE

    def test_a_nan_threshold_skips_the_warmup_the_field_exists_for(self) -> None:
        """``collect_rollout`` draws a random action while ``size < threshold``."""
        assert not 0 < float("nan")

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_no_buffer_ever_passes_the_threshold(self, value: float) -> None:
        """``train`` runs ``update()`` only while ``size >= threshold``."""
        assert not 10**9 >= value

    def test_the_gate_that_names_this_outcome_is_the_replay_one(self) -> None:
        """The sibling gate's docstring records the same zero-update success."""
        doc = " ".join((rl_replay_problems.__doc__ or "").split())
        assert "zero" in doc and "gradient" in doc


class TestTheUsableDomainIsUntouched:
    """A real count is accepted and the relation still does its own job."""

    def test_a_positive_integer_above_the_batch_is_accepted(self) -> None:
        assert not _about_learning_starts(VALID)

    def test_the_shipped_default_is_accepted(self) -> None:
        default = RLTrainSpec().learning_starts
        assert not _about_learning_starts(default)

    def test_a_count_below_the_batch_still_fails_the_relation(self) -> None:
        """The relation is preserved, not replaced: 1 is a count and still wrong."""
        problem = _about_learning_starts(1)[0]
        assert "must be >= batch_size" in problem

    def test_the_relation_message_is_not_the_domain_message(self) -> None:
        """A count that is merely too small must not be called a non-count."""
        assert "positive integer" not in _about_learning_starts(1)[0]

    def test_a_very_large_count_is_still_a_count(self) -> None:
        """Magnitude is not *this* axis: an enormous warmup is still a count.

        It is refused - no run's step budget or replay capacity reaches it, by
        the reachability relation graded in
        ``tests/training/test_warmup_threshold_is_reachable.py`` - but as a
        threshold out of reach rather than as a non-count, which is the only
        question this gate decides.
        """
        problems = _about_learning_starts(10**400)
        assert "positive integer" not in " ".join(problems)
        assert [p for p in problems if "is never reached" in p]  # not silently accepted either


class TestWhatStrictNewlyRefuses:
    """Two consequences of using the sibling operand's strict domain.

    Both were accepted by the bare comparison and both are refused now. They are
    recorded here rather than left to be discovered, because they are the price
    of the two operands sharing one domain.
    """

    @pytest.mark.parametrize("value", [1_000.0, np.int64(1_000)], ids=["integral-float", "numpy-int"])
    def test_it_is_refused_although_the_comparison_accepted_it(self, value: Any) -> None:
        assert _about_learning_starts(value)

    @pytest.mark.parametrize("value", [1_000.0, np.int64(1_000)], ids=["integral-float", "numpy-int"])
    def test_the_comparison_alone_would_have_admitted_it(self, value: Any) -> None:
        """Non-vacuity: these are refused by the domain, not by the relation."""
        assert not value < BATCH_SIZE

    def test_the_field_is_annotated_as_an_integer(self) -> None:
        """Which is why strict is the honest reading of the contract."""
        assert RLTrainSpec.__annotations__["learning_starts"] == "int"


class TestTheGateStillReportsNothingAboutIt:
    """The field-scoped gate's own scope is unchanged by this fix."""

    @pytest.mark.parametrize("value", NOT_A_COUNT, ids=repr)
    def test_the_replay_gate_stays_silent(self, value: Any) -> None:
        spec = _spec(learning_starts=value)
        assert not [p for p in rl_replay_problems(spec, context="fast_sac") if "learning_starts" in p]

    @pytest.mark.parametrize("provider", ["ppo", "mock"])
    def test_a_backend_that_does_not_read_it_stays_quiet(self, provider: str) -> None:
        """PPO has no replay warmup, so an SAC-only value is not its business."""
        spec = _spec(learning_starts=float("nan"))
        assert not [p for p in create_trainer(provider).validate(spec) if "learning_starts" in p]

    def test_tau_is_still_refused_on_its_own_axis(self) -> None:
        """The neighbouring exclusion is untouched: a coefficient is not a count."""
        problems = create_trainer("fast_sac").validate(_spec(learning_starts=VALID, tau=2.0))
        assert [p for p in problems if "tau" in p]
        assert not [p for p in problems if "positive integer" in p]


class TestBothOperandsOfTheRelationShareOneDomain:
    """The structural claim, read off the owner's source rather than prose."""

    @staticmethod
    def _owner_source() -> str:
        return textwrap.dedent(inspect.getsource(warmup_batch_relation_problems))

    def test_the_relation_reads_both_fields(self) -> None:
        """Premise: this is the statement the two operands meet in."""
        assert "learning_starts < batch_size" in self._owner_source()

    def test_each_operand_is_asked_of_the_shared_domain(self) -> None:
        asked = {
            call.args[1].value
            for call in ast.walk(ast.parse(self._owner_source()))
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "positive_count_error"
            and len(call.args) > 1
            and isinstance(call.args[1], ast.Constant)
        }
        assert {"learning_starts", "batch_size"} <= asked

    def test_the_domain_is_asked_before_the_relation(self) -> None:
        """A guard after the comparison would not stop the comparison."""
        source = self._owner_source()
        assert source.index('"learning_starts", context') < source.index("learning_starts < batch_size")

    def test_the_domain_is_not_restated_locally(self) -> None:
        """One owner: no hand-rolled isfinite / isinstance beside the shared call."""
        source = self._owner_source()
        assert "math.isfinite" not in source
        assert "isinstance(spec.learning_starts" not in source
        assert math.isfinite(1.0)  # the module imports math only for this premise


class TestOneOwnerForTheWarmupBatchRelation:
    """No off-policy backend may re-implement the relation or skip its gate.

    The set of backends in scope is derived from the tree rather than listed, so a
    third off-policy trainer that starts warming up a replay buffer fails these
    tests until it routes through the gate.
    """

    @staticmethod
    def _training_modules() -> list[pathlib.Path]:
        """Every training module except the one that owns the rule."""
        root = pathlib.Path(inspect.getfile(Trainer)).parent
        owner = pathlib.Path(inspect.getfile(warmup_batch_relation_problems)).resolve()
        return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)

    @staticmethod
    def _defines_validate(source: str) -> bool:
        return any(isinstance(n, ast.FunctionDef) and n.name == "validate" for n in ast.walk(ast.parse(source)))

    @staticmethod
    def _calls_the_gate(source: str) -> bool:
        return any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_rl_warmup_batch_problems"
            for n in ast.walk(ast.parse(source))
        )

    def test_no_module_re_implements_the_relation(self) -> None:
        """A local comparison is a second copy of the bound, and can drift."""
        offenders = [
            p.name for p in self._training_modules() if "spec.learning_starts < spec.batch_size" in p.read_text()
        ]
        assert offenders == [], f"modules compare the two operands locally: {offenders}"

    def test_every_preflight_that_reads_it_routes_through_the_gate(self) -> None:
        adrift = [
            p.name
            for p in self._training_modules()
            if self._defines_validate(src := p.read_text())
            and reads_spec_field(src, ("learning_starts",))
            and not self._calls_the_gate(src)
        ]
        assert adrift == [], f"preflights read learning_starts without the shared gate: {adrift}"

    def test_the_reader_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot sweep an empty tree clean."""
        readers = {
            p.name
            for p in self._training_modules()
            if self._defines_validate(src := p.read_text()) and reads_spec_field(src, ("learning_starts",))
        }
        assert readers == {"fast_sac.py", "fast_td3.py"}, readers

    def test_the_scanners_detect_a_planted_defect(self) -> None:
        """Both rules really report the shapes they are written to catch."""
        planted = "def validate(self, spec):\n    return [] if spec.learning_starts < spec.batch_size else []\n"
        assert self._defines_validate(planted)
        assert reads_spec_field(planted, ("learning_starts",))
        assert not self._calls_the_gate(planted)
        assert self._calls_the_gate("def validate(self, spec):\n    return self._rl_warmup_batch_problems(spec)\n")


class TestBothOffPolicyBackendsStateItIdentically:
    """One owner means one message, which is what a shared rule buys."""

    @pytest.mark.parametrize("value", NOT_A_COUNT, ids=repr)
    def test_a_non_count_reads_the_same_from_either_backend(self, value: Any) -> None:
        spec = _spec(learning_starts=value)
        sac = [p for p in create_trainer("fast_sac").validate(spec) if "learning_starts" in p]
        td3 = [p for p in create_trainer("fast_td3").validate(spec) if "learning_starts" in p]
        assert sac and [p.replace("fast_sac", "") for p in sac] == [p.replace("fast_td3", "") for p in td3]

    def test_a_short_warmup_reads_the_same_from_either_backend(self) -> None:
        """The relation names both values, and names them once per backend."""
        spec = _spec(learning_starts=10)
        expected = [
            f"learning_starts (10) must be >= batch_size ({BATCH_SIZE}) "
            "so the first gradient step can sample a full batch"
        ]
        for provider in ("fast_sac", "fast_td3"):
            assert [p for p in create_trainer(provider).validate(spec) if "sample a full batch" in p] == expected

    def test_a_satisfied_relation_is_silent_from_either_backend(self) -> None:
        spec = _spec(learning_starts=BATCH_SIZE)
        for provider in ("fast_sac", "fast_td3"):
            assert not [p for p in create_trainer(provider).validate(spec) if "sample a full batch" in p]
