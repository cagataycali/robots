"""The on-policy optimization-epoch count is checked against its domain.

``RLTrainSpec.num_learning_epochs`` is the number of passes PPO's update makes
over each rollout batch, and it is consumed as a bare loop bound around the
whole optimizer step: ``for _ in range(spec.num_learning_epochs)`` encloses
every ``self.optimizer.step()`` in ``PpoTrainer.update``. So the field does not
merely scale how much optimization happens - a non-positive value removes all of
it, and nothing downstream notices.

Measured on this backend over a 60-step run before the gate existed: ``0`` and
``-3`` both reported ``status="success"``, took **zero** gradient steps, and
wrote a checkpoint whose parameters were bit-identical to each other, i.e. the
untrained initialisation. The reported losses read ``0.0`` rather than blank
because the update averages its accumulators through ``max(1, n_updates)``, so
an epoch count that ran no minibatch reports plausible metrics for a run that
learned nothing. ``True`` was a silent single epoch (12 optimizer steps instead
of 24), and ``2.7``/``nan``/``inf``/``"5"``/``None`` raised a bare ``TypeError``
out of ``range()`` after the environment, the networks and a full rollout had
been built.

Scoped to the on-policy backend: FastSAC optimizes per gradient step from a
replay buffer and has no epoch loop over a rollout batch, so it must not report
on a field it never reads. Every domain test here reaches the real ``validate``
entry point, so it covers the wiring as well as the domain.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap
from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import optimization_epochs_problems
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import positive_count_error
from tests.training._spec_field_reads import reads_spec_field

# The one backend whose update loops over a rollout batch.
ON_POLICY = "ppo"

# Backends with no epoch loop - they must stay quiet about the field.
NO_EPOCH_LOOP_BACKENDS = ("fast_sac", "mock")

# Non-positive counts. These are the silent half: the run succeeds having taken
# no gradient step, and reports losses of 0.0 through ``max(1, n_updates)``.
NO_OPTIMIZATION: list[Any] = [0, -1, -3, -100]

# Values ``range()`` cannot use as a bound, or reads as a different count than
# the caller wrote. ``True`` is ``range(1)``; an integral float still raises.
NOT_A_COUNT: list[Any] = [
    True,
    False,
    2.7,
    5.0,
    float("nan"),
    float("inf"),
    "5",
    None,
    [5],
    {"num_learning_epochs": 5},
]

UNUSABLE: list[Any] = [*NO_OPTIMIZATION, *NOT_A_COUNT]

# A positive ``int`` is the whole domain.
USABLE: list[Any] = [1, 2, 5, 5000]

# ``range()`` itself accepts any object with ``__index__``, so these three DO
# bound a loop (``len(range(np.int64(4))) == 4``) and are nonetheless refused:
# the shared count domain is strict about ``int``, and it is the domain every
# other count in this preflight already uses (``val_episodes``). The wider
# alternative, ``positive_whole_number_error``, would accept the integral float
# ``5.0`` that ``range()`` raises on - a guard accepting what its own consumer
# refuses - so the narrower rule is the correct one and this is its cost.
REFUSED_BY_THE_SHARED_COUNT_RULE: list[Any] = [np.int64(4), np.int32(3), np.uint8(2)]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(output_dir="/tmp/optimization_epochs_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


def _epoch_problems(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about the field.

    Filtered on the shared domains\' ``"{context}: {param} "`` message shape
    rather than on a bare substring, so an unrelated problem can neither mask a
    missing refusal nor be mistaken for one.
    """
    prefix = f"{provider}: num_learning_epochs "
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(prefix)]


class TestTheOnPolicyBackendRefusesAnUnusableEpochCount:
    """PPO refuses every value its epoch loop cannot honor."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: RLTrainSpec, value: Any) -> None:
        spec.num_learning_epochs = value
        assert _epoch_problems(ON_POLICY, spec), f"ppo accepted num_learning_epochs={value!r}"

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_problem_names_the_field_and_the_domain(self, spec: RLTrainSpec, value: Any) -> None:
        spec.num_learning_epochs = value
        (problem,) = _epoch_problems(ON_POLICY, spec)
        assert "num_learning_epochs" in problem
        assert "positive integer" in problem
        assert repr(value) in problem, problem

    @pytest.mark.parametrize("value", NO_OPTIMIZATION)
    def test_a_non_positive_count_never_reaches_a_run(self, spec: RLTrainSpec, value: Any) -> None:
        """``train`` is fail-closed on ``validate``, so no rollout is collected."""
        spec.num_learning_epochs = value
        result = create_trainer(ON_POLICY).train(spec)
        assert result.status == "error"
        assert "num_learning_epochs" in result.message


class TestTheUsableDomainIsUntouched:
    """Every count the epoch loop can honor still passes."""

    @pytest.mark.parametrize("value", USABLE)
    def test_a_positive_integer_is_accepted(self, spec: RLTrainSpec, value: Any) -> None:
        spec.num_learning_epochs = value
        assert _epoch_problems(ON_POLICY, spec) == []

    def test_the_shipped_default_is_accepted(self, spec: RLTrainSpec) -> None:
        """Non-vacuity: the fixture is not refused before the field is set."""
        assert spec.num_learning_epochs == 5
        assert _epoch_problems(ON_POLICY, spec) == []

    @pytest.mark.parametrize("value", REFUSED_BY_THE_SHARED_COUNT_RULE)
    def test_a_numpy_integer_is_refused_although_range_would_accept_it(self, spec: RLTrainSpec, value: Any) -> None:
        """The measured cost of using the strict shared count rule.

        These spellings do bound a loop, so refusing them is a narrowing rather
        than a fix. It is the right narrowing: the wider domain would accept the
        integral float ``5.0``, which ``range()`` raises on, and every other
        count in this preflight is already held to this same rule. No caller in
        the repository passes one - the examples, docs and tests all write a
        plain ``int`` literal.
        """
        assert len(range(value)) == int(value)
        spec.num_learning_epochs = value
        assert _epoch_problems(ON_POLICY, spec)


class TestABackendWithNoEpochLoopStaysQuiet:
    """A backend that never reads the field must not report on it."""

    @pytest.mark.parametrize("provider", NO_EPOCH_LOOP_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_reports_nothing_about_the_field(self, provider: str, spec: RLTrainSpec, value: Any) -> None:
        spec.num_learning_epochs = value
        assert _epoch_problems(provider, spec) == []

    @pytest.mark.parametrize("provider", NO_EPOCH_LOOP_BACKENDS)
    def test_but_it_still_refuses_a_field_it_does_read(self, provider: str, spec: RLTrainSpec) -> None:
        """Non-vacuity: silence is scoping, not a preflight that reports nothing.

        ``learning_rate`` is the field chosen here because it is the universal
        one - every backend reads it, including the non-RL ``mock``, which has no
        discount factor to report on either.
        """
        spec.learning_rate = -1.0
        prefix = f"{provider}: learning_rate "
        assert [p for p in create_trainer(provider).validate(spec) if p.startswith(prefix)]


class TestTheDomainIsTheSharedCountRule:
    """The gate contributes no rule of its own beyond reading the field."""

    @pytest.mark.parametrize("value", [*UNUSABLE, *USABLE, *REFUSED_BY_THE_SHARED_COUNT_RULE])
    def test_it_agrees_with_the_shared_count_domain(self, spec: RLTrainSpec, value: Any) -> None:
        spec.num_learning_epochs = value
        shared = positive_count_error(value, "num_learning_epochs", "ppo")
        assert (optimization_epochs_problems(spec, context="ppo") == []) is (shared is None)

    def test_the_message_is_the_shared_one_verbatim(self, spec: RLTrainSpec) -> None:
        spec.num_learning_epochs = 0
        assert optimization_epochs_problems(spec, context="ppo") == [
            positive_count_error(0, "num_learning_epochs", "ppo")
        ]

    def test_the_context_names_the_backend_that_refused(self, spec: RLTrainSpec) -> None:
        spec.num_learning_epochs = 0
        (problem,) = _epoch_problems(ON_POLICY, spec)
        assert problem.startswith("ppo: ")


# --- why the domain exists: the field bounds the whole optimizer step --------


def _ppo_update_source() -> str:
    """Source of ``PpoTrainer.update`` - read via the class, not a path."""
    from strands_robots.training.rl.ppo import PpoTrainer

    return inspect.getsource(PpoTrainer.update)


class TestTheFieldBoundsTheEntireOptimizerStep:
    """The premise the domain rests on, asserted rather than described.

    If a later change moves ``optimizer.step()`` out of the epoch loop, or stops
    averaging through ``max(1, n_updates)``, these fail and the gate's stated
    reason gets re-examined instead of quietly becoming wrong.
    """

    def test_the_epoch_range_encloses_the_optimizer_step(self) -> None:
        tree = ast.parse(textwrap.dedent(_ppo_update_source()))
        loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For) and ast.unparse(node.iter) == "range(spec.num_learning_epochs)"
        ]
        assert len(loops) == 1, "the epoch loop is no longer bounded by the field"
        stepped = [
            ast.unparse(call)
            for call in ast.walk(loops[0])
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "step"
        ]
        assert "self.optimizer.step()" in stepped, stepped

    def test_a_non_positive_bound_iterates_zero_times(self) -> None:
        """So a non-positive count takes no gradient step at all."""
        assert [len(range(n)) for n in (5, 1, 0)] == [5, 1, 0]
        assert len(range(-3)) == 0

    def test_true_is_a_silent_single_epoch(self) -> None:
        """Which is why ``bool`` must be refused rather than read as one."""
        assert len(range(True)) == 1

    def test_a_non_integer_bound_raises_out_of_range(self) -> None:
        bound: Any = 2.7
        with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
            range(bound)

    def test_the_reported_losses_are_averaged_through_a_clamp(self) -> None:
        """Which is why a zero-epoch run reports 0.0 rather than nothing."""
        assert "max(1, n_updates)" in _ppo_update_source()
        assert 0.0 / max(1, 0) == 0.0


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(optimization_epochs_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_epoch_count(source: str) -> bool:
    """Does *source* read ``spec.num_learning_epochs``, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, ("num_learning_epochs",))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_optimization_epochs_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheEpochCountDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* ``spec.num_learning_epochs`` must route it through the
    shared gate, so a second on-policy backend that starts looping over a
    rollout batch with the field fails this test until it does.
    """

    def test_every_module_that_reads_it_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name
            for p in _training_modules()
            if _reads_the_epoch_count(p.read_text()) and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"modules read spec.num_learning_epochs without the shared gate: {adrift}"

    def test_the_reader_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        readers = {p.name for p in _training_modules() if _reads_the_epoch_count(p.read_text())}
        assert readers == {"ppo.py"}, readers

    def test_the_scanner_detects_a_planted_reader(self) -> None:
        """A module reading the field without the gate is really reported."""
        planted = "def validate(self, spec):\n    return [] if spec.num_learning_epochs else []\n"
        assert _reads_the_epoch_count(planted)
        assert not _calls_the_gate(planted)

    def test_no_backend_re_implements_the_bound(self) -> None:
        """A local copy of the bound would drift from the shared rule."""
        offenders = [
            p.name
            for p in _training_modules()
            if "spec.num_learning_epochs <" in p.read_text() or "spec.num_learning_epochs >" in p.read_text()
        ]
        assert offenders == [], f"modules compare spec.num_learning_epochs locally: {offenders}"
