"""The checkpoint cadence a TrainSpec asks for is one shared whole-number domain.

:attr:`~strands_robots.training.base.TrainSpec.save_freq` is the interval, in
optimizer steps, at which a run writes a checkpoint. Four providers read it - the
three supervised backends and the SageMaker transport - and each consumes it in
the same three shapes a process count is consumed in:

* a ``spec.save_freq > 0`` selector, which LeRobot uses twice to derive the
  validation cadence (``--eval_steps=`` on the argv path, ``cfg.eval_steps`` on
  the in-process one) because a non-positive cadence disables periodic saving;
* an argv or Hydra token - ``--save_freq=`` (LeRobot), ``--save_steps=``
  (GR00T), ``checkpoint.save_iter=`` (Cosmos) - decoded into an ``int`` field;
* a direct assignment into a typed config (``cfg.save_freq``, GR00T's
  ``save_steps`` kwarg, a forwarded SageMaker hyperparameter), none of which
  coerces.

Before this contract the field had no domain while its run-size neighbours
``steps`` and ``global_batch_size`` - the step count in the very same argv -
shared one, and every way a bad cadence failed was silent or late:

* ``nan`` and ``False`` compare as *not* greater than zero (``nan`` compares
  false against everything), so the selector read them as "periodic saving
  disabled" and a spec asking to checkpoint every ``nan`` steps evaluated once
  at the very end instead, under a successful result.
* ``2.7``, ``5000.0``, ``True`` and ``inf`` *are* greater than zero, so they
  passed through as the cadence itself and reached an ``int`` field as a float
  or a ``bool``. Their argv token then failed inside the launched run, after the
  dataset and model were already loaded.
* A string, ``None`` or a list raised ``TypeError`` out of the comparison itself
  - from inside a :meth:`Trainer.validate` documented to *return* problems.

What this domain deliberately does *not* decide is the floor. LeRobot documents
a non-positive ``save_freq`` as disabling periodic saving and its
``should_save_checkpoint`` implements exactly that, so ``0`` and a negative are
a capability rather than an unusable value - the ``eval_steps`` fallback above
exists for them. Only the type is graded, and the boundary is pinned below so it
cannot move silently.

The tests pin the contract in both directions: every provider that reads the
field refuses the same unusable values through the one shared gate with a
message naming itself, the disable capability and a usable cadence are
untouched, a backend that ignores the field reports nothing about it, and the
refused values are grounded in what the selector and LeRobot's own decoder do
with them.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
from typing import Any

import pytest

from strands_robots.training._validate import checkpoint_cadence_problems
from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.cosmos3 import Cosmos3Trainer
from strands_robots.training.groot import Gr00tTrainer
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.training.mock import MockTrainer
from strands_robots.training.sagemaker import SagemakerTrainer
from strands_robots.utils import positive_count_error
from tests.training._spec_field_reads import reads_spec_field

FIELD = "save_freq"

# The providers that read the cadence. SageMaker forwards it as a
# hyperparameter; the other three interpolate it into a native token.
CHECKPOINTING_BACKENDS = (LerobotTrainer, Gr00tTrainer, Cosmos3Trainer, SagemakerTrainer)

# Values no backend can honor, split by how each one failed before the gate.

# Read as "not more than zero" by the selector -> the cadence silently disabled.
SILENTLY_DISABLED = (float("nan"), False)

# Read as "more than zero" -> passed through as the cadence into an int field.
REACHED_THE_INT_FIELD = (2.7, 5000.0, True, float("inf"))

# Raised TypeError out of the comparison, from inside validate().
RAISED_OUT_OF_VALIDATE: tuple[Any, ...] = ("5000", None, [5000])

UNUSABLE = SILENTLY_DISABLED + REACHED_THE_INT_FIELD + RAISED_OUT_OF_VALIDATE

# The floor is NOT part of the domain: a non-positive cadence disables periodic
# saving, and a cadence above ``steps`` writes only the final checkpoint.
USABLE = (0, -1, 1, 500, 10**9)


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A spec whose cadence is the only thing under test.

    ``validate`` may well report unrelated problems (GR00T wants a checkout,
    Cosmos a recipe TOML, SageMaker a role); every assertion below filters for
    the field name, so an unrelated problem cannot mask - or fake - a cadence
    verdict.
    """
    return TrainSpec(
        dataset_root=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        base_model="lerobot/act",
        embodiment="new_embodiment",
    )


def _problems_about(trainer: Trainer, spec: TrainSpec) -> list[str]:
    """``validate`` problems that name the cadence field."""
    return [p for p in trainer.validate(spec) if FIELD in p]


def _token_decodes(value: Any) -> bool:
    """Does ``f"{value}"`` decode into an ``int`` the way LeRobot decodes it?

    LeRobot declares ``save_freq`` as a plain ``int`` and parses the argv token
    with draccus, so this is the real destination of the interpolated value.
    """
    draccus = pytest.importorskip("draccus")
    try:
        draccus.decode(int, f"{value}")
    except Exception:
        return False
    return True


def _comparison_survives(value: Any) -> bool:
    """Does the ``> 0`` selector run at all for *value*?"""
    try:
        _ = value > 0
    except TypeError:
        return False
    return True


class TestEveryReadingBackendRefusesAnUnusableCadence:
    """Each provider that reads the field refuses every unusable value."""

    @pytest.mark.parametrize("trainer_cls", CHECKPOINTING_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any) -> None:
        spec.save_freq = value
        problems = _problems_about(trainer_cls(), spec)
        assert problems, f"{trainer_cls.__name__} accepted {FIELD}={value!r}"
        assert any("must be a whole number of steps" in p for p in problems), problems

    @pytest.mark.parametrize("trainer_cls", CHECKPOINTING_BACKENDS)
    def test_the_problem_names_the_backend_that_refused_it(self, spec: TrainSpec, trainer_cls: type[Trainer]) -> None:
        trainer = trainer_cls()
        spec.save_freq = 2.7  # type: ignore[assignment]
        assert any(p.startswith(f"{trainer.provider_name}: {FIELD} ") for p in _problems_about(trainer, spec))

    @pytest.mark.parametrize("trainer_cls", CHECKPOINTING_BACKENDS)
    @pytest.mark.parametrize("value", RAISED_OUT_OF_VALIDATE)
    def test_a_non_numeric_cadence_is_a_problem_not_an_exception(
        self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any
    ) -> None:
        """``validate`` returns problems; it must not raise out of a comparison."""
        spec.save_freq = value
        problems = trainer_cls().validate(spec)  # must not raise
        assert any(FIELD in p for p in problems), problems

    @pytest.mark.parametrize("trainer_cls", CHECKPOINTING_BACKENDS)
    def test_the_refusal_names_the_disable_capability_it_kept(
        self, spec: TrainSpec, trainer_cls: type[Trainer]
    ) -> None:
        """A caller who meant "never checkpoint" must be told how to say it.

        The floor is outside the domain, so the message has to point at the
        spelling that *is* accepted rather than leave "whole number" to imply a
        positive one.
        """
        spec.save_freq = float("nan")  # type: ignore[assignment]
        assert any("disable periodic saving" in p for p in _problems_about(trainer_cls(), spec))


class TestTheDisableCapabilityAndAUsableCadenceAreUntouched:
    """The floor is not part of the domain, and a usable cadence is not a problem."""

    @pytest.mark.parametrize("trainer_cls", CHECKPOINTING_BACKENDS)
    @pytest.mark.parametrize("value", USABLE)
    def test_a_whole_number_cadence_is_not_a_problem(
        self, spec: TrainSpec, trainer_cls: type[Trainer], value: int
    ) -> None:
        spec.save_freq = value
        assert _problems_about(trainer_cls(), spec) == []

    @pytest.mark.parametrize("trainer_cls", CHECKPOINTING_BACKENDS)
    @pytest.mark.parametrize("value", (0, -1))
    def test_a_non_positive_cadence_is_the_documented_way_to_disable_saving(
        self, spec: TrainSpec, trainer_cls: type[Trainer], value: int
    ) -> None:
        """LeRobot's ``should_save_checkpoint`` reads it as "no periodic saves".

        Pinned separately from the usable-cadence case above because it is the
        boundary this domain deliberately declines to decide: were the floor
        folded in, these two values would be refused and the capability lost.
        """
        spec.save_freq = value
        assert _problems_about(trainer_cls(), spec) == []

    def test_the_default_cadence_is_inside_the_domain(self, spec: TrainSpec) -> None:
        """Non-vacuity for every assertion above: the default is not refused."""
        assert checkpoint_cadence_problems(spec, context="acme") == []
        assert isinstance(spec.save_freq, int) and not isinstance(spec.save_freq, bool)


class TestABackendThatIgnoresTheFieldReportsNothing:
    """A backend must not report on a field it never reads.

    :class:`TrainSpec` documents that a backend "reads the fields it supports
    and ignores the rest", so this gate is scoped to the checkpointing providers
    rather than made universal like the learning-rate one.
    """

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_mock_backend_checkpoints_from_no_cadence(self, spec: TrainSpec, value: Any) -> None:
        spec.save_freq = value
        assert _problems_about(MockTrainer(), spec) == []

    def test_an_rl_backend_checkpoints_from_no_cadence(self, tmp_path: pathlib.Path) -> None:
        pytest.importorskip("torch")
        from strands_robots.training.rl.base_algo import RLTrainSpec
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        rl_spec = RLTrainSpec(
            output_dir=str(tmp_path),
            env_factory=lambda: None,  # type: ignore[arg-type,return-value]
        )
        rl_spec.save_freq = 2.7  # type: ignore[assignment]
        assert [p for p in FastSacTrainer().validate(rl_spec) if FIELD in p] == []


class TestTheRefusedValuesAreOnesNoBackendCanHonor:
    """Ground the domain in what the selector and LeRobot's own decoder do."""

    @pytest.mark.parametrize("value", SILENTLY_DISABLED)
    def test_a_silent_value_reads_as_not_more_than_zero(self, value: Any) -> None:
        """The ``> 0`` selector routes each of these to the *disabled* branch."""
        assert not value > 0

    def test_nan_compares_false_against_every_bound(self) -> None:
        """Why ``nan`` slipped through a comparison-based reading in both directions."""
        nan = float("nan")
        assert not nan > 0
        assert not nan <= 0
        assert math.isnan(nan)

    @pytest.mark.parametrize("value", REACHED_THE_INT_FIELD)
    def test_a_passed_through_value_reads_as_more_than_zero(self, value: Any) -> None:
        assert value > 0

    @pytest.mark.parametrize("value", RAISED_OUT_OF_VALIDATE)
    def test_a_non_numeric_cadence_cannot_be_compared_at_all(self, value: Any) -> None:
        """Which is why the ungated comparison raised rather than reporting."""
        assert not _comparison_survives(value)

    @pytest.mark.parametrize("value", SILENTLY_DISABLED + REACHED_THE_INT_FIELD)
    def test_lerobots_own_decoder_refuses_the_token_it_renders(self, value: Any) -> None:
        """The argv token's real destination refuses every one of these.

        So nothing downstream would have told the caller either - the refusal
        lands inside the launched run once the dataset and model are loaded.
        """
        assert not _token_decodes(value)

    def test_every_unusable_value_fails_at_least_one_consumer(self) -> None:
        """The two failure modes together cover the refused set.

        Non-vacuity for the two tests above: a value that both decoded and
        compared cleanly would have no consumer to justify refusing it.
        """
        survivors = [v for v in UNUSABLE if _token_decodes(v) and _comparison_survives(v)]
        assert survivors == [], f"refused values no consumer objects to: {survivors}"

    def test_both_failure_modes_are_represented(self) -> None:
        """And that neither test above is measuring the whole set on its own."""
        assert [v for v in UNUSABLE if not _comparison_survives(v)]
        assert [v for v in UNUSABLE if _comparison_survives(v) and not _token_decodes(v)]

    def test_a_string_cadence_fails_only_the_comparison(self) -> None:
        """The one refused value whose *token* is perfectly decodable.

        ``"5000"`` renders to the same token as ``5000``, so a domain argued
        from the decoder alone would have admitted it - and then the ``> 0``
        selector raises. It takes both consumers to justify the type test.
        """
        assert _token_decodes("5000")
        assert not _comparison_survives("5000")

    @pytest.mark.parametrize("value", (5000.0, True, 2.7))
    def test_the_two_step_counts_in_one_argv_agree(self, value: Any) -> None:
        """``steps`` is refused for the same spellings, by its own shared domain.

        ``save_freq`` and ``steps`` are interpolated into the same argv and read
        by the same ``int`` decoder, so the same number must not be refused for
        one and accepted for the other. That is why the type test is strict
        rather than accepting an integral float.
        """
        assert positive_count_error(value, "steps", "acme") is not None
        assert checkpoint_cadence_problems(TrainSpec(save_freq=value), context="acme")


def _trainer_modules() -> list[pathlib.Path]:
    """Every backend module, INCLUDING the ``rl`` subpackage.

    Rooted at the module that defines :class:`Trainer` so the scan cannot
    silently point at the wrong tree. The module that *defines* the shared gate
    is excluded - derived from the gate itself rather than named, so the
    exclusion cannot drift - because it reads the field as its owner rather than
    as a consumer of it.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(checkpoint_cadence_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_cadence(source: str) -> bool:
    """Does *source* read the field, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, (FIELD,))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_checkpoint_cadence_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheCheckpointCadenceDomain:
    """No backend may re-implement the domain, and none may skip it.

    The set of providers in scope is derived from the tree rather than listed:
    a module that *reads* the field must route it through the shared gate, so a
    fifth backend that starts checkpointing from ``save_freq`` fails this test
    until it does.
    """

    def test_the_scan_finds_the_checkpointing_backends(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep of nothing."""
        readers = {p.name for p in _trainer_modules() if _reads_the_cadence(p.read_text())}
        assert readers == {"cosmos3.py", "groot.py", "lerobot.py", "sagemaker.py"}

    def test_every_backend_that_checkpoints_routes_through_the_shared_gate(self) -> None:
        adrift = sorted(
            p.name
            for p in _trainer_modules()
            if _reads_the_cadence(source := p.read_text()) and not _calls_the_gate(source)
        )
        assert adrift == [], f"modules reading the cadence without the shared gate: {adrift}"

    def test_no_backend_re_implements_the_domain(self) -> None:
        """A local type or floor test on the field is the hole this closed.

        A ``> 0`` comparison is a different question - "did the caller disable
        periodic saving" - and stays where it is.
        """
        offenders: list[str] = []
        for path in _trainer_modules():
            for line in path.read_text().splitlines():
                if f"spec.{FIELD}" in line and ("<= 0" in line or "< 1" in line or "!= int" in line):
                    offenders.append(f"{path.name}: {line.strip()}")
        assert offenders == [], f"local domain checks on the cadence: {offenders}"

    def test_the_selector_the_backends_do_keep_is_still_there(self) -> None:
        """The gate must not have been mistaken for the disable branch.

        ``save_freq if save_freq > 0 else steps`` is the reading that makes a
        non-positive cadence mean "evaluate once at the end"; gating the type
        does not remove it, and removing it would silently change what a
        disabled cadence does to the validation schedule.
        """
        source = pathlib.Path(inspect.getfile(LerobotTrainer)).read_text()
        assert source.count(f"spec.{FIELD} if spec.{FIELD} > 0 else spec.steps") == 2

    def test_the_scanners_detect_a_planted_defect(self) -> None:
        """A scanner that silently matched nothing would look like a clean tree."""
        planted = "def validate(self, spec):\n    return [] if spec.save_freq > 0 else []\n"
        assert _reads_the_cadence(planted)
        assert not _calls_the_gate(planted)

    def test_the_scanners_detect_a_table_driven_defect(self) -> None:
        """A backend that forwards the field by name is a reader too.

        The form the SageMaker transport takes: no attribute access mentions the
        field, so a scan keyed on ``spec.save_freq`` alone reports a clean sweep
        while that provider skips the gate.
        """
        planted = 'F = ("save_freq",)\ndef validate(self, spec):\n    return [getattr(spec, f) for f in F]\n'
        assert _reads_the_cadence(planted)
        assert not _calls_the_gate(planted)


class TestTheGateIsUsableOnItsOwn:
    """The shared gate's own contract, independent of any backend."""

    def test_it_reports_the_context_it_was_given(self, spec: TrainSpec) -> None:
        spec.save_freq = 2.7  # type: ignore[assignment]
        (problem,) = checkpoint_cadence_problems(spec, context="acme")
        assert problem.startswith("acme: save_freq must be a whole number of steps, got 2.7.")

    def test_a_usable_cadence_reports_nothing(self, spec: TrainSpec) -> None:
        spec.save_freq = 500
        assert checkpoint_cadence_problems(spec, context="acme") == []

    def test_it_reports_at_most_one_problem(self, spec: TrainSpec) -> None:
        """One field, one problem - the message has to name the value once."""
        spec.save_freq = None  # type: ignore[assignment]
        assert len(checkpoint_cadence_problems(spec, context="acme")) == 1
