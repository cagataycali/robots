"""The two factors an RL run's length is derived from are checked against a domain.

``RLTrainSpec.total_timesteps`` and ``RLTrainSpec.rollout_steps`` are the two
caller-supplied factors of the one loop bound every RL backend derives::

    steps_per_iter = rollout_steps * num_envs
    num_iters = max(1, total_timesteps // steps_per_iter)
    for it in range(num_iters):  # collect, update

Because the bound is *derived* rather than read straight off the spec, a bare
``value <= 0`` test on either factor is weaker than the same test on a field
consumed directly: the ``max(1, ...)`` clamp turns every value that survives the
comparison but cannot divide into a silent single iteration. Measured on both
trainers over a 16-step run with ``rollout_steps=4`` before this gate existed,
``True``, ``0.5``, ``nan`` and ``inf`` each reported ``status="success"``, ran
**one** iteration, wrote a checkpoint and announced ``"1 iterations x 4 steps
complete"``; ``100.5`` raised ``TypeError: 'float' object cannot be interpreted
as an integer`` out of ``range()`` after the environment, the networks, the
optimizers and (FastSAC) the replay buffer had been built; and ``"16"``/``None``
raised ``TypeError`` out of the comparison itself, from a ``validate`` documented
to *return* problems. ``rollout_steps=True`` was worse than a short run: FastSAC
ran 16 single-step iterations instead of 4 of 4, and PPO normalized advantages
over a length-one batch and died inside torch's ``Normal`` constraint.

``num_envs``, the third factor of ``steps_per_iter``, is deliberately outside the
shared domain because its accepted set differs per backend - PPO parallelizes and
accepts any count ``>= 1``, the MuJoCo-backed FastSAC requires exactly ``1`` - so
it is not one shared rule. :class:`TestNumEnvsIsNotInTheSharedDomain` pins that
scope line rather than leaving it to prose.

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
from strands_robots.training._validate import rl_run_size_problems
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import positive_count_error
from tests.training._spec_field_reads import reads_spec_field

# The backends that size a training loop from these two fields.
RL_BACKENDS = ("ppo", "fast_sac")

# A backend that sizes its run from ``steps`` instead - it must stay quiet about
# fields it never reads, or an RL-only value would be a false rejection.
SUPERVISED_BACKENDS = ("mock",)

# The two factors, and the field the domain deliberately excludes.
RUN_SIZE_FIELDS = ("total_timesteps", "rollout_steps")

# Non-positive counts: the half a local ``<= 0`` test already caught.
NO_TRAINING: list[Any] = [0, -1, -1000]

# Values that survive ``value <= 0`` and cannot bound the loop. The first four
# are the silent half - ``max(1, ...)`` reads each as a single iteration, because
# ``True // n`` is ``0``, ``0.5 // n`` is ``0.0``, and both ``nan // n`` and
# ``inf // n`` are ``nan``, which compares false against everything.
SILENTLY_ONE_ITERATION: list[Any] = [True, 0.5, float("nan"), float("inf")]

# Values that reach the loop and raise there, or raise out of the comparison.
NOT_A_COUNT: list[Any] = [*SILENTLY_ONE_ITERATION, False, 100.5, 4.0, "16", None, [16], {"total_timesteps": 16}]

UNUSABLE: list[Any] = [*NO_TRAINING, *NOT_A_COUNT]

# A positive ``int`` is the whole domain.
USABLE: list[Any] = [1, 4, 24, 100_000]

# ``range()`` accepts any object with ``__index__``, so these DO bound a loop and
# are still refused: the shared count domain is strict about ``int``, which is
# what lets it refuse the integral float ``4.0`` that ``range()`` raises on. A
# guard that accepted what its own consumer refuses would be the wrong trade.
REFUSED_BY_THE_SHARED_COUNT_RULE: list[Any] = [np.int64(24), np.int32(4), np.uint8(8)]

# The two values PPO's divisibility relation used to mishandle. Annotated ``Any``
# because they are deliberately outside the field's declared type - which is the
# property under test, so the annotation must not stand in the way of asserting it.
A_NON_FINITE_ROLLOUT: Any = float("nan")
A_STRING_ROLLOUT: Any = "24"


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(output_dir="/tmp/rl_run_size_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


def _field_problems(provider: str, spec: RLTrainSpec, field: str) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``field``.

    Filtered on the shared domains' ``"{context}: {param} "`` message shape
    rather than on a bare substring, so an unrelated problem can neither mask a
    missing refusal nor be mistaken for one.
    """
    prefix = f"{provider}: {field} "
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(prefix)]


class TestEveryRlBackendRefusesAnUnusableRunSize:
    """Both trainers refuse every value their loop bound cannot be built from."""

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("field", RUN_SIZE_FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, provider: str, spec: RLTrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        assert _field_problems(provider, spec, field), f"{provider} accepted {field}={value!r}"

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("field", RUN_SIZE_FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_problem_names_the_field_and_the_domain(
        self, provider: str, spec: RLTrainSpec, field: str, value: Any
    ) -> None:
        setattr(spec, field, value)
        (problem,) = _field_problems(provider, spec, field)
        assert problem == positive_count_error(value, field, provider)

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("field", RUN_SIZE_FIELDS)
    @pytest.mark.parametrize("value", NOT_A_COUNT)
    def test_a_non_count_never_reaches_a_run(self, provider: str, spec: RLTrainSpec, field: str, value: Any) -> None:
        """The refusal precedes ``setup``, so nothing is built before it."""
        setattr(spec, field, value)
        result = create_trainer(provider).train(spec)
        assert result.status == "error"
        assert result.checkpoint_dir in (None, "")
        assert f"{field} must be a positive integer" in result.message


class TestTheUsableDomainIsUntouched:
    """The gate refuses only what the loop bound cannot use."""

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("field", RUN_SIZE_FIELDS)
    @pytest.mark.parametrize("value", USABLE)
    def test_a_positive_integer_is_accepted(self, provider: str, spec: RLTrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        assert _field_problems(provider, spec, field) == []

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("field", RUN_SIZE_FIELDS)
    def test_the_shipped_default_is_accepted(self, provider: str, spec: RLTrainSpec, field: str) -> None:
        assert _field_problems(provider, spec, field) == []

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("field", RUN_SIZE_FIELDS)
    @pytest.mark.parametrize("value", REFUSED_BY_THE_SHARED_COUNT_RULE)
    def test_a_numpy_integer_is_refused_although_range_would_accept_it(
        self, provider: str, spec: RLTrainSpec, field: str, value: Any
    ) -> None:
        """The documented cost of the strict-``int`` rule, pinned not implied."""
        assert len(range(value)) == int(value)
        setattr(spec, field, value)
        assert _field_problems(provider, spec, field)


class TestASupervisedBackendStaysQuiet:
    """A backend that sizes its run from ``steps`` reports nothing about these."""

    @pytest.mark.parametrize("provider", SUPERVISED_BACKENDS)
    @pytest.mark.parametrize("field", RUN_SIZE_FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_reports_nothing_about_the_field(self, provider: str, spec: RLTrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        assert _field_problems(provider, spec, field) == []

    @pytest.mark.parametrize("provider", SUPERVISED_BACKENDS)
    def test_but_it_still_refuses_the_run_size_it_does_read(self, provider: str, spec: RLTrainSpec) -> None:
        """Non-vacuity: silence is scoping, not a backend that checks nothing."""
        spec.steps = 0
        assert _field_problems(provider, spec, "steps")


class TestTheDomainIsTheSharedCountRule:
    """The gate contributes no rule of its own beyond reading the two fields."""

    @pytest.mark.parametrize("field", RUN_SIZE_FIELDS)
    @pytest.mark.parametrize("value", [*UNUSABLE, *USABLE, *REFUSED_BY_THE_SHARED_COUNT_RULE])
    def test_it_agrees_with_the_shared_count_domain(self, spec: RLTrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        shared = positive_count_error(value, field, "ppo")
        assert (rl_run_size_problems(spec, context="ppo") == []) is (shared is None)

    @pytest.mark.parametrize("field", RUN_SIZE_FIELDS)
    def test_the_message_is_the_shared_one_verbatim(self, spec: RLTrainSpec, field: str) -> None:
        setattr(spec, field, 0)
        assert rl_run_size_problems(spec, context="ppo") == [positive_count_error(0, field, "ppo")]

    def test_both_factors_are_reported_together(self, spec: RLTrainSpec) -> None:
        """One call per field, so a caller sees every factor it got wrong."""
        spec.total_timesteps = 0
        spec.rollout_steps = -1
        assert rl_run_size_problems(spec, context="ppo") == [
            positive_count_error(0, "total_timesteps", "ppo"),
            positive_count_error(-1, "rollout_steps", "ppo"),
        ]

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    def test_the_context_names_the_backend_that_refused(self, provider: str, spec: RLTrainSpec) -> None:
        spec.total_timesteps = 0
        (problem,) = _field_problems(provider, spec, "total_timesteps")
        assert problem.startswith(f"{provider}: ")

    def test_a_spec_without_the_rl_fields_reports_nothing(self) -> None:
        """The gate reads through ``getattr`` defaults, so a plain spec is quiet."""
        from strands_robots.training.base import TrainSpec

        assert rl_run_size_problems(TrainSpec(output_dir="/tmp/x"), context="mock") == []


# --- why the domain exists: the bound is derived, and the clamp hides that ----


def _train_loop_sources() -> dict[str, str]:
    """Source of every RL training loop, read via the class not a path."""
    from strands_robots.training.rl.base_algo import BaseRLAlgo
    from strands_robots.training.rl.fast_sac import FastSacTrainer

    return {
        "BaseRLAlgo.train": inspect.getsource(BaseRLAlgo.train),
        "FastSacTrainer.train": inspect.getsource(FastSacTrainer.train),
    }


class TestTheLoopBoundIsDerivedThroughAClamp:
    """The premise the domain rests on, asserted rather than described.

    If a later change drops the ``max(1, ...)`` clamp or stops deriving the bound
    from these two fields, these fail and the gate's stated reason gets
    re-examined instead of quietly becoming wrong.
    """

    @pytest.mark.parametrize("name", ["BaseRLAlgo.train", "FastSacTrainer.train"])
    def test_the_bound_is_the_clamped_quotient_of_the_two_factors(self, name: str) -> None:
        source = _train_loop_sources()[name]
        assert "num_iters = max(1, spec.total_timesteps // steps_per_iter)" in source
        assert "for it in range(num_iters):" in source

    @pytest.mark.parametrize("value", SILENTLY_ONE_ITERATION)
    def test_the_clamp_reads_a_non_count_as_a_single_iteration(self, value: Any) -> None:
        """The silent half: no exception, one iteration, a successful report."""
        assert len(range(max(1, value // 4))) == 1

    def test_a_fraction_above_one_iteration_raises_out_of_range_instead(self) -> None:
        """The late half: a float bound, raised only once the loop is reached."""
        assert 100.5 // 4 == 25.0
        with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
            range(max(1, 100.5 // 4))  # type: ignore[call-overload]

    def test_an_infinite_budget_is_silent_rather_than_unbounded(self) -> None:
        """``inf // n`` is ``nan``, and ``nan`` loses every comparison to ``1``."""
        assert np.isnan(float("inf") // 4)
        assert max(1, float("inf") // 4) == 1

    @pytest.mark.parametrize("value", ["16", None])
    def test_a_non_numeric_budget_raises_out_of_the_comparison_itself(self, value: Any) -> None:
        """Why a local ``<= 0`` cannot even report: it raises before appending."""
        with pytest.raises(TypeError, match="not supported between instances"):
            _ = value <= 0


class TestNumEnvsIsNotInTheSharedDomain:
    """The scope line, pinned: the third factor is per-backend by necessity.

    ``num_envs`` is the other factor of ``steps_per_iter``, so it looks like it
    belongs beside the two above. It cannot: the backends disagree about which
    counts are usable, so there is no one domain to share.
    """

    def test_the_gate_reports_nothing_about_it(self, spec: RLTrainSpec) -> None:
        spec.num_envs = 0
        assert rl_run_size_problems(spec, context="ppo") == []

    def test_the_backends_disagree_about_its_accepted_set(self, spec: RLTrainSpec) -> None:
        """Two envs: usable for the parallel backend, refused by the single-env one."""
        spec.num_envs = 2
        assert not [p for p in create_trainer("ppo").validate(spec) if "num_envs" in p]
        assert [p for p in create_trainer("fast_sac").validate(spec) if "num_envs" in p]

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    def test_each_backend_still_refuses_a_non_positive_count(self, provider: str, spec: RLTrainSpec) -> None:
        """Non-vacuity: excluding it from the domain did not leave it unchecked."""
        spec.num_envs = 0
        assert [p for p in create_trainer(provider).validate(spec) if "num_envs" in p]


class TestTheDivisibilityRelationIsAskedOnlyOfACount:
    """PPO's ``rollout_steps % num_mini_batches`` rule is between two counts.

    A relation can only be asked once both sides are the kind of thing it relates,
    so it sits after the domain that grades ``rollout_steps``. Before it did, a
    string reached ``%`` as string formatting (``TypeError`` out of a method
    documented to return problems) and ``nan`` was reported as *indivisible*,
    which named the wrong problem.
    """

    def test_a_non_count_is_reported_once_as_a_non_count(self, spec: RLTrainSpec) -> None:
        spec.rollout_steps = A_NON_FINITE_ROLLOUT
        problems = [p for p in create_trainer("ppo").validate(spec) if "rollout_steps" in p]
        assert problems == [positive_count_error(A_NON_FINITE_ROLLOUT, "rollout_steps", "ppo")]

    def test_a_string_no_longer_raises_out_of_the_relation(self, spec: RLTrainSpec) -> None:
        spec.rollout_steps = A_STRING_ROLLOUT
        problems = [p for p in create_trainer("ppo").validate(spec) if "rollout_steps" in p]
        assert problems == [positive_count_error(A_STRING_ROLLOUT, "rollout_steps", "ppo")]

    def test_the_relation_is_still_asked_of_a_count(self, spec: RLTrainSpec) -> None:
        """Over-reach guard: gating the relation did not disable it."""
        spec.rollout_steps = 25
        spec.num_mini_batches = 4
        assert [p for p in create_trainer("ppo").validate(spec) if "divisible by num_mini_batches" in p]

    def test_a_divisible_count_is_accepted(self, spec: RLTrainSpec) -> None:
        spec.rollout_steps = 24
        spec.num_mini_batches = 4
        assert not [p for p in create_trainer("ppo").validate(spec) if "divisible by num_mini_batches" in p]


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(rl_run_size_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_run_size(source: str) -> bool:
    """Does *source* read either factor, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none in an attribute
    access.
    """
    return reads_spec_field(source, RUN_SIZE_FIELDS)


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_rl_run_size_problems"
        for node in ast.walk(ast.parse(source))
    )


def _defines_validate(source: str) -> bool:
    """Does *source* implement the preflight the biconditional is about?

    The rule a shared domain states is about ``validate``: a backend whose
    preflight reads the field must route it through the gate. The abstract base
    that *consumes* the field in the shared training loop has no preflight of its
    own to route anything through, so scoping the scan to modules that implement
    ``validate`` is what keeps the rule a statement about preflights rather than
    about every mention of the field.
    """
    return any(isinstance(node, ast.FunctionDef) and node.name == "validate" for node in ast.walk(ast.parse(source)))


class TestOneOwnerForTheRlRunSizeDomain:
    """No RL backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed, so a
    third RL trainer that starts sizing its loop from these fields fails this
    test until it routes through the gate.
    """

    def test_every_preflight_that_reads_them_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name
            for p in _training_modules()
            if _defines_validate(p.read_text())
            and _reads_the_run_size(p.read_text())
            and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"preflights read the run-size factors without the shared gate: {adrift}"

    def test_the_preflight_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        gated = {
            p.name
            for p in _training_modules()
            if _defines_validate(p.read_text()) and _reads_the_run_size(p.read_text())
        }
        assert gated == {"fast_sac.py", "ppo.py"}, gated

    def test_the_shared_loop_reads_the_fields_and_has_no_preflight_to_gate(self) -> None:
        """Why the scan is scoped to preflights, pinned rather than asserted in prose.

        ``base_algo.py`` is where the two factors are *consumed* - it derives the
        loop bound from them - and it implements no ``validate``, so it is the one
        reader with nothing to route. A future ``validate`` there must gate, which
        the scan above then requires.
        """
        readers = {p.name for p in _training_modules() if _reads_the_run_size(p.read_text())}
        assert "base_algo.py" in readers
        base_algo = next(p for p in _training_modules() if p.name == "base_algo.py")
        assert not _defines_validate(base_algo.read_text())

    def test_the_scanner_detects_a_planted_preflight(self) -> None:
        """A preflight reading a factor without the gate is really reported."""
        planted = "def validate(self, spec):\n    return [] if spec.total_timesteps else []\n"
        assert _defines_validate(planted)
        assert _reads_the_run_size(planted)
        assert not _calls_the_gate(planted)

    @pytest.mark.parametrize("field", RUN_SIZE_FIELDS)
    def test_no_backend_re_implements_the_bound(self, field: str) -> None:
        """A local copy of the bound would drift from the shared rule."""
        offenders = [
            p.name
            for p in _training_modules()
            if f"spec.{field} <" in p.read_text() or f"spec.{field} >" in p.read_text()
        ]
        assert offenders == [], f"modules compare spec.{field} locally: {offenders}"
