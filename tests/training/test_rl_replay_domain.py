"""The three counts a FastSAC replay loop is built from are checked against a domain.

``RLTrainSpec.buffer_size``, ``RLTrainSpec.batch_size`` and
``RLTrainSpec.gradient_steps`` are the three caller-supplied counts of an
off-policy SAC run's replay loop - the buffer's capacity, the transitions
sampled per gradient step, and the SAC updates run per iteration. Each is
consumed directly as a count: ``buffer_size`` is a tensor dimension built in
``FastSacTrainer.setup``, ``batch_size`` is passed to ``ReplayBuffer.sample``,
and ``gradient_steps`` is a ``range()`` bound in the update loop.

Because they are consumed as counts, a bare ``value <= 0`` test is weaker than
the strict-``int`` :func:`~strands_robots.utils.positive_count_error` domain, and
both failure modes were measured on the MuJoCo reach env before this gate
existed (an otherwise-valid run, one field mutated): ``buffer_size=True`` built a
one-slot buffer that never reached ``learning_starts``, so **zero** gradient
updates ran, yet the run reported ``status="success"`` and "10 iterations x 4
steps complete"; ``buffer_size=0.5`` raised ``IndexError`` from
``ReplayBuffer.add`` (``int(0.5) == 0``, a zero-capacity buffer);
``batch_size=0.5`` / ``batch_size=True`` raised ``TypeError`` from
``torch.randint`` in ``ReplayBuffer.sample``; ``gradient_steps=0.5`` raised
``TypeError`` from ``range()`` in the update loop - each after the environment,
the networks, the optimizers and the replay buffer had been built - and ``"256"``
/ ``None`` raised ``TypeError`` out of the ``<= 0`` comparison itself, from a
``validate`` documented to *return* problems.

Only FastSAC reads these three fields. PPO sizes its minibatches from
``num_mini_batches`` and never reads them, so it must stay quiet about them, as
must every supervised backend; :class:`TestABackendThatDoesNotReadThemStaysQuiet`
pins that. ``learning_starts`` and ``tau`` are outside this gate's field set, and
:class:`TestTauAndLearningStartsAreNotInThisDomain` pins that scope line rather
than leaving it to prose. ``tau`` is a coefficient in ``(0, 1]`` and so shares no
part of this domain; it has its own gate on its own interval, graded in
``tests/training/test_polyak_coefficient_domain.py``. ``learning_starts`` is one side of a relation
(``>= batch_size``) and *does* share the domain: FastSAC asks it of the same
strict-``int`` rule as the relation's other operand, in its own ``validate``
rather than through this gate, because PPO reads neither field. That is graded in
``tests/training/test_learning_starts_count_domain.py``, which measured the cost
of leaving it to the relation alone - a non-finite value compares False against
every ``int``, so the relation passed and the run took zero gradient steps while
reporting success, the same outcome ``buffer_size=True`` produces above.

Every domain test here reaches the real ``validate`` entry point, so it covers
the wiring as well as the domain.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import rl_replay_problems
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import positive_count_error
from tests.training._spec_field_reads import reads_spec_field

# The backends that build a replay loop from these three fields.
SAC_BACKENDS = ("fast_sac", "fast_td3")

# Backends that never read them: PPO sizes minibatches from num_mini_batches, and
# the supervised backends have no replay loop at all. Each must stay quiet, or an
# off-policy-only value would be a false rejection of a field it does not use.
QUIET_BACKENDS = ("ppo", "mock")

# The three replay counts.
REPLAY_FIELDS = ("buffer_size", "batch_size", "gradient_steps")

# Non-positive counts: the half a local ``<= 0`` test already caught (though with
# a local message, not the shared-domain one this gate now emits).
NO_TRAINING: list[Any] = [0, -1, -1000, False]

# Values that survive ``value <= 0`` and cannot be a count. ``True`` is the silent
# half - a degenerate size of one - and the fractions / non-finite values reach a
# tensor allocation or a ``range()`` and raise there.
SILENTLY_DEGENERATE_OR_LATE: list[Any] = [True, 0.5, float("nan"), float("inf")]

# Values that reach a run and raise there, or raise out of the comparison itself.
NOT_A_COUNT: list[Any] = [*SILENTLY_DEGENERATE_OR_LATE, 100.5, 4.0, "256", None, [256], {"buffer_size": 256}]

UNUSABLE: list[Any] = [*NO_TRAINING, *NOT_A_COUNT]

# A positive ``int`` is the whole domain. All are <= the default learning_starts
# (1000), so setting batch_size to one cannot also trip the >= batch_size relation.
USABLE: list[Any] = [1, 8, 256]

# ``range()`` and torch accept an object with ``__index__``, so a numpy integer
# would bound a loop and size a tensor - and is still refused, because the shared
# count rule is strict about ``int``. Pinned so the documented cost is not implied.
REFUSED_BY_THE_SHARED_COUNT_RULE: list[Any] = [np.int64(256), np.int32(8), np.uint8(1)]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(output_dir="/tmp/rl_replay_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


def _field_problems(provider: str, spec: RLTrainSpec, field: str) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``field``.

    Filtered on the shared domain's ``"{context}: {param} "`` message shape, so a
    problem about a different field (the ``learning_starts >= batch_size``
    relation names ``learning_starts``, not ``batch_size``) can neither mask a
    missing refusal nor be mistaken for one.
    """
    prefix = f"{provider}: {field} "
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(prefix)]


class TestFastSacRefusesAnUnusableReplayCount:
    """FastSAC refuses every value its replay loop cannot be built from."""

    @pytest.mark.parametrize("field", REPLAY_FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: RLTrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        assert _field_problems("fast_sac", spec, field), f"fast_sac accepted {field}={value!r}"

    @pytest.mark.parametrize("field", REPLAY_FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_problem_names_the_field_and_the_domain(self, spec: RLTrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        (problem,) = _field_problems("fast_sac", spec, field)
        assert problem == positive_count_error(value, field, "fast_sac")

    @pytest.mark.parametrize("field", REPLAY_FIELDS)
    @pytest.mark.parametrize("value", NOT_A_COUNT)
    def test_a_non_count_never_reaches_a_run(self, spec: RLTrainSpec, field: str, value: Any) -> None:
        """The refusal precedes ``setup``, so nothing is built before it."""
        setattr(spec, field, value)
        result = create_trainer("fast_sac").train(spec)
        assert result.status == "error"
        assert result.checkpoint_dir in (None, "")
        assert f"{field} must be a positive integer" in result.message


class TestTheUsableDomainIsUntouched:
    """The gate refuses only what the replay loop cannot use."""

    @pytest.mark.parametrize("field", REPLAY_FIELDS)
    @pytest.mark.parametrize("value", USABLE)
    def test_a_positive_integer_is_accepted(self, spec: RLTrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        assert _field_problems("fast_sac", spec, field) == []

    @pytest.mark.parametrize("field", REPLAY_FIELDS)
    def test_the_shipped_default_is_accepted(self, spec: RLTrainSpec, field: str) -> None:
        assert _field_problems("fast_sac", spec, field) == []

    @pytest.mark.parametrize("field", REPLAY_FIELDS)
    @pytest.mark.parametrize("value", REFUSED_BY_THE_SHARED_COUNT_RULE)
    def test_a_numpy_integer_is_refused_although_range_would_accept_it(
        self, spec: RLTrainSpec, field: str, value: Any
    ) -> None:
        """The documented cost of the strict-``int`` rule, pinned not implied."""
        assert len(range(value)) == int(value)
        setattr(spec, field, value)
        assert _field_problems("fast_sac", spec, field)


class TestABackendThatDoesNotReadThemStaysQuiet:
    """A backend with no SAC replay loop reports nothing about these fields."""

    @pytest.mark.parametrize("provider", QUIET_BACKENDS)
    @pytest.mark.parametrize("field", REPLAY_FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_reports_nothing_about_the_field(self, provider: str, spec: RLTrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        assert _field_problems(provider, spec, field) == []

    def test_ppo_still_refuses_the_run_size_it_does_read(self, spec: RLTrainSpec) -> None:
        """Non-vacuity: PPO's silence here is scoping, not a backend that checks nothing."""
        spec.total_timesteps = 0
        assert [p for p in create_trainer("ppo").validate(spec) if "total_timesteps" in p]

    def test_mock_still_refuses_the_run_size_it_does_read(self, spec: RLTrainSpec) -> None:
        """Non-vacuity for the supervised backend."""
        spec.steps = 0
        assert _field_problems("mock", spec, "steps")


class TestTheDomainIsTheSharedCountRule:
    """The gate contributes no rule of its own beyond reading the three fields."""

    @pytest.mark.parametrize("field", REPLAY_FIELDS)
    @pytest.mark.parametrize("value", [*UNUSABLE, *USABLE, *REFUSED_BY_THE_SHARED_COUNT_RULE])
    def test_it_agrees_with_the_shared_count_domain(self, spec: RLTrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        shared = positive_count_error(value, field, "fast_sac")
        assert (rl_replay_problems(spec, context="fast_sac") == []) is (shared is None)

    @pytest.mark.parametrize("field", REPLAY_FIELDS)
    def test_the_message_is_the_shared_one_verbatim(self, spec: RLTrainSpec, field: str) -> None:
        setattr(spec, field, 0)
        assert rl_replay_problems(spec, context="fast_sac") == [positive_count_error(0, field, "fast_sac")]

    def test_all_three_counts_are_reported_together(self, spec: RLTrainSpec) -> None:
        """One call per field, so a caller sees every count it got wrong."""
        spec.buffer_size = 0
        spec.batch_size = -1
        spec.gradient_steps = 0.5  # type: ignore[assignment]
        assert rl_replay_problems(spec, context="fast_sac") == [
            positive_count_error(0, "buffer_size", "fast_sac"),
            positive_count_error(-1, "batch_size", "fast_sac"),
            positive_count_error(0.5, "gradient_steps", "fast_sac"),
        ]

    def test_the_context_names_the_backend_that_refused(self, spec: RLTrainSpec) -> None:
        spec.buffer_size = 0
        (problem,) = _field_problems("fast_sac", spec, "buffer_size")
        assert problem.startswith("fast_sac: ")

    def test_a_spec_without_the_rl_fields_reports_nothing(self) -> None:
        """The gate reads through ``getattr`` defaults, so a plain spec is quiet."""
        from strands_robots.training.base import TrainSpec

        assert rl_replay_problems(TrainSpec(output_dir="/tmp/x"), context="mock") == []


class TestTheCountsAreConsumedDirectlyAsCounts:
    """The premise the domain rests on, asserted rather than described.

    If a later change stops consuming these three as counts (a capacity, a sample
    size, a ``range()`` bound), these fail and the gate's stated reason gets
    re-examined instead of quietly becoming wrong.
    """

    def test_the_three_fields_are_consumed_as_counts(self) -> None:
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        source = inspect.getsource(FastSacTrainer)
        assert "spec.buffer_size" in source  # replay buffer capacity
        assert "self.buffer.sample(spec.batch_size)" in source  # sample size
        assert "for _ in range(spec.gradient_steps):" in source  # range() bound

    @pytest.mark.parametrize("value", SILENTLY_DEGENERATE_OR_LATE)
    def test_a_non_count_survives_the_local_comparison(self, value: Any) -> None:
        """Why a local ``<= 0`` is weaker: each of these passes it."""
        assert not (value <= 0)

    def test_a_bool_is_a_degenerate_size_of_one(self) -> None:
        """The silent half: ``True`` is a one-slot buffer / a batch of one."""
        assert int(True) == 1
        assert len(range(True)) == 1

    def test_a_fractional_capacity_truncates_to_zero(self) -> None:
        """``int(0.5) == 0`` - the zero-capacity buffer the IndexError came from."""
        assert int(0.5) == 0

    @pytest.mark.parametrize("value", ["256", None])
    def test_a_non_numeric_count_raises_out_of_the_comparison_itself(self, value: Any) -> None:
        """Why a local ``<= 0`` cannot even report: it raises before appending."""
        with pytest.raises(TypeError, match="not supported between instances"):
            _ = value <= 0


class TestTauAndLearningStartsAreNotInThisDomain:
    """The scope line, pinned: two SAC fields this gate does not carry.

    Neither is one of the three replay counts, so the gate must stay silent about
    both and FastSAC must still refuse them itself. What that refusal rests on
    differs: ``tau`` is a coefficient in ``(0, 1]`` and shares no part of the
    count domain, while ``learning_starts`` is one side of a relation
    (``>= batch_size``) whose *other* operand is already asked of the shared
    count rule - so FastSAC asks it of that rule too, in its own ``validate``.
    ``tests/training/test_learning_starts_count_domain.py`` grades that; here the
    claim is only that this gate reports nothing about either field.
    """

    @pytest.mark.parametrize("field", ["tau", "learning_starts"])
    def test_the_gate_reports_nothing_about_it(self, spec: RLTrainSpec, field: str) -> None:
        setattr(spec, field, 0)
        assert not [p for p in rl_replay_problems(spec, context="fast_sac") if field in p]

    def test_fast_sac_still_refuses_a_tau_outside_the_unit_interval(self, spec: RLTrainSpec) -> None:
        spec.tau = 2.0
        assert [p for p in create_trainer("fast_sac").validate(spec) if "tau" in p]

    def test_fast_sac_still_refuses_learning_starts_below_batch_size(self, spec: RLTrainSpec) -> None:
        spec.learning_starts = 1
        spec.batch_size = 256
        assert [p for p in create_trainer("fast_sac").validate(spec) if "learning_starts" in p]


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(rl_replay_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_replay(source: str) -> bool:
    """Does *source* read any of the three counts, by name or through a table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none in an attribute
    access.
    """
    return reads_spec_field(source, REPLAY_FIELDS)


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_rl_replay_problems"
        for node in ast.walk(ast.parse(source))
    )


def _defines_validate(source: str) -> bool:
    """Does *source* implement the preflight the biconditional is about?"""
    return any(isinstance(node, ast.FunctionDef) and node.name == "validate" for node in ast.walk(ast.parse(source)))


class TestOneOwnerForTheReplayCountDomain:
    """No backend that reads the three counts may skip the domain or re-implement it.

    The set of backends in scope is derived from the tree rather than listed, so a
    second trainer that starts building a replay loop from these fields fails this
    test until it routes through the gate.
    """

    def test_every_preflight_that_reads_them_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name
            for p in _training_modules()
            if _defines_validate(p.read_text())
            and _reads_the_replay(p.read_text())
            and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"preflights read the replay counts without the shared gate: {adrift}"

    def test_the_preflight_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        gated = {
            p.name for p in _training_modules() if _defines_validate(p.read_text()) and _reads_the_replay(p.read_text())
        }
        assert gated == {"fast_sac.py", "fast_td3.py"}, gated

    def test_ppo_reads_none_of_the_replay_fields(self) -> None:
        """The scope is SAC-only because PPO does not read these counts at all."""
        ppo = next(p for p in _training_modules() if p.name == "ppo.py")
        assert not _reads_the_replay(ppo.read_text())

    def test_the_scanner_detects_a_planted_preflight(self) -> None:
        """A preflight reading a count without the gate is really reported."""
        planted = "def validate(self, spec):\n    return [] if spec.buffer_size else []\n"
        assert _defines_validate(planted)
        assert _reads_the_replay(planted)
        assert not _calls_the_gate(planted)

    @pytest.mark.parametrize("field", REPLAY_FIELDS)
    def test_no_backend_re_implements_the_count_check(self, field: str) -> None:
        """A local ``<= 0`` copy would drift from the shared rule."""
        offenders = [
            p.name
            for p in _training_modules()
            if f"spec.{field} <" in p.read_text() or f"spec.{field} >" in p.read_text()
        ]
        assert offenders == [], f"modules compare spec.{field} locally: {offenders}"
