"""The checkpoint cadence a TrainSpec asks for is one shared step-cadence domain.

:attr:`~strands_robots.training.base.TrainSpec.save_freq` is how often a run
writes a checkpoint, and four backends read it - LeRobot (``cfg.save_freq``
in-process and a ``--save_freq=`` argv token), GR00T (``--save_steps=``), Cosmos
(a ``checkpoint.save_iter=`` Hydra override) and SageMaker (a JSON-encoded
hyperparameter the job's container reads). Before this contract none of them
checked it, and the destinations do not agree about a single value:

* The in-process LeRobot path puts the value straight into lerobot's
  ``should_save_checkpoint(step, save_freq, total_steps)`` -
  ``(save_freq > 0 and step % save_freq == 0) or step == total_steps``. No parser
  stands between the spec and that expression, so ``True`` is a modulus of one
  and writes a full checkpoint **every single step**, a fractional or non-finite
  cadence never satisfies ``step % cadence == 0`` for an integral step and so
  silently becomes the *disabled* mode, and a ``str`` raises ``TypeError`` out of
  the comparison - inside the training loop, after the dataset and model load.
* The argv path renders the value into a token lerobot decodes into the same
  ``int`` field, where draccus refuses ``True`` / ``2.7`` / ``5000.0`` / ``nan``
  / ``inf`` - but accepts the token a ``"5000"`` renders, the one value the
  in-process path cannot use.

So the two routes of one backend disagree about the same spec: whichever value a
caller supplies outside the domain, one route refuses it and the other runs with
it. :attr:`TrainSpec.extra` already states the rule this breaks - one spec must
mean one run whichever path the backend takes.

The tests below pin the contract in both directions: every backend that reads the
field refuses the same unusable values through the one shared domain with a
message naming itself, a usable cadence is untouched (``0`` and a negative
included - they select the documented "disable periodic saving" mode), a backend
that ignores the field reports nothing about it, and the refused values are
grounded in what the real consumers do with them.
"""

from __future__ import annotations

import json
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

# The backends that checkpoint from the field.
CHECKPOINTING_BACKENDS = (LerobotTrainer, Gr00tTrainer, Cosmos3Trainer, SagemakerTrainer)

# Values no consumer can honor, split by how each one failed before the gate.

# Passes the ``> 0`` test and then means something the caller did not write:
# ``True`` a modulus of one, the rest a cadence no integral step satisfies.
SILENTLY_WRONG = (True, 2.7, 5000.0, float("nan"), float("inf"))

# Raised out of the consumer: the comparison itself refuses a non-number.
RAISED_IN_THE_CONSUMER = ("5000", [7])

UNUSABLE = SILENTLY_WRONG + RAISED_IN_THE_CONSUMER

# Cadences every consumer honors as themselves. ``0`` and a negative are the
# documented "disable periodic saving" mode, which is why this domain has no
# floor.
USABLE = (1, 100, 1_000, 20_000, 0, -1)
DISABLING = (0, -1)

# A run long enough for a cadence to differ visibly from the disabled mode.
STEPS = 10_000

# Two representative unusable cadences, named so the assertions outside a
# ``parametrize`` can use them. Typed ``Any`` for the same reason the
# parametrized values are: the annotation forbids them, and the callers this
# domain exists for supply values the annotation never saw - a JSON config, an
# agent tool call, a spec assembled from a form.
A_FRACTIONAL_CADENCE: Any = 2.7
A_STRING_CADENCE: Any = "5000"


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A spec whose ``save_freq`` is the only thing under test.

    ``validate`` may well report unrelated problems (Cosmos wants a recipe TOML,
    GR00T a checkout); every assertion below filters for the field name, so an
    unrelated problem can neither mask nor fake a cadence verdict.
    """
    return TrainSpec(
        dataset_root=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        base_model="lerobot/act",
        embodiment="new_embodiment",
        steps=STEPS,
    )


# The shape the shared domain emits: ``"{context}: save_freq must be ..."``.
# Matched rather than the bare word because pytest derives ``tmp_path`` from the
# test name, so a path in an unrelated problem could contain it too - a filter
# that picked that up could both mask and fake a verdict.
_NAMES_THE_CADENCE = ": save_freq "


def _cadence_problems_of(trainer: Trainer, spec: TrainSpec) -> list[str]:
    """``validate`` problems about ``save_freq``."""
    return [p for p in trainer.validate(spec) if _NAMES_THE_CADENCE in p]


def _should_save_checkpoint() -> Any:
    """lerobot's own in-process checkpoint gate, or skip when lerobot is absent."""
    return pytest.importorskip("lerobot.common.train_utils").should_save_checkpoint


def _periodic_saves(cadence: Any) -> int:
    """How many periodic checkpoints lerobot writes for *cadence* in one run.

    Counts ``1 .. STEPS - 1`` so the final step - which lerobot always saves,
    whatever the cadence - cannot mask the periodic behaviour under test.
    """
    gate = _should_save_checkpoint()
    return sum(1 for step in range(1, STEPS) if gate(step, cadence, STEPS))


def _decodes_as_the_argv_token(value: Any) -> bool:
    """Would lerobot decode the ``--save_freq=`` token this value renders?"""
    draccus = pytest.importorskip("draccus")
    try:
        draccus.decode(int, f"{value}")
    except Exception:
        return False
    return True


def _runs_in_process(value: Any) -> bool:
    """Does lerobot's in-process checkpoint gate accept this value at all?"""
    try:
        _periodic_saves(value)
    except Exception:
        return False
    return True


class TestEveryCheckpointingBackendRefusesAnUnusableCadence:
    """The first half of the biconditional: a reader must route through the gate."""

    @pytest.mark.parametrize("trainer_cls", CHECKPOINTING_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any) -> None:
        spec.save_freq = value
        assert _cadence_problems_of(trainer_cls(), spec) != []

    @pytest.mark.parametrize("trainer_cls", CHECKPOINTING_BACKENDS)
    def test_the_problem_names_the_backend_that_refused_it(self, spec: TrainSpec, trainer_cls: type[Trainer]) -> None:
        trainer = trainer_cls()
        spec.save_freq = A_FRACTIONAL_CADENCE
        problems = _cadence_problems_of(trainer, spec)
        assert problems and problems[0].startswith(f"{trainer.provider_name}: ")

    @pytest.mark.parametrize("trainer_cls", CHECKPOINTING_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_unusable_cadence_is_a_problem_not_an_exception(
        self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any
    ) -> None:
        """``validate`` is documented to *return* problems, so it must not raise.

        The in-process consumer raises ``TypeError`` from a bare comparison on a
        non-number, and the gate runs before any such comparison in ``validate``.
        """
        spec.save_freq = value
        assert isinstance(trainer_cls().validate(spec), list)

    @pytest.mark.parametrize("trainer_cls", CHECKPOINTING_BACKENDS)
    def test_the_message_says_how_to_disable_periodic_saving(self, spec: TrainSpec, trainer_cls: type[Trainer]) -> None:
        """A caller who wrote ``2.7`` may have meant "rarely"; say what to write.

        The disabled mode is reachable only through a non-positive value, so a
        refusal that did not mention it would leave the caller to guess.
        """
        spec.save_freq = A_FRACTIONAL_CADENCE
        assert "non-positive" in _cadence_problems_of(trainer_cls(), spec)[0]


class TestAUsableCadenceIsUntouched:
    """The controls: nothing the consumers honor may be refused."""

    @pytest.mark.parametrize("trainer_cls", CHECKPOINTING_BACKENDS)
    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_cadence_is_not_a_problem(self, spec: TrainSpec, trainer_cls: type[Trainer], value: int) -> None:
        spec.save_freq = value
        assert _cadence_problems_of(trainer_cls(), spec) == []

    @pytest.mark.parametrize("trainer_cls", CHECKPOINTING_BACKENDS)
    def test_the_default_cadence_is_not_a_problem(self, spec: TrainSpec, trainer_cls: type[Trainer]) -> None:
        """The field has a concrete default, so every spec carries a value."""
        assert _cadence_problems_of(trainer_cls(), spec) == []


class TestABackendThatIgnoresTheFieldReportsNothing:
    """A backend must not report on a field it never reads.

    :class:`TrainSpec` documents that a backend "reads the fields it supports and
    ignores the rest", so this gate is scoped to the checkpointing backends
    rather than made universal like the learning-rate one.
    """

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_checkpoints_from_nothing(self, spec: TrainSpec, value: Any) -> None:
        spec.save_freq = value
        assert _cadence_problems_of(MockTrainer(), spec) == []


class TestTheRefusedValuesAreOnesNoConsumerCanHonor:
    """Ground the domain in what the real consumers do with each value."""

    def test_a_boolean_cadence_checkpoints_every_single_step(self) -> None:
        """The headline: ``True`` is not rejected downstream, it is *obeyed*.

        As an ``int`` subclass it is a modulus of one, so a run that asked for a
        checkpoint cadence writes a full checkpoint at every optimizer step.
        """
        assert _periodic_saves(True) == STEPS - 1
        assert _periodic_saves(1_000) == 9, "the cadence a caller would have written"

    @pytest.mark.parametrize("value", (2.7, float("nan"), float("inf")))
    def test_a_non_integral_cadence_silently_becomes_the_disabled_mode(self, value: Any) -> None:
        """Indistinguishable from the documented ``0``, with nothing reported."""
        assert _periodic_saves(value) == 0 == _periodic_saves(0)

    @pytest.mark.parametrize("value", RAISED_IN_THE_CONSUMER)
    def test_a_non_numeric_cadence_raises_inside_the_training_loop(self, value: Any) -> None:
        """Not at submission: from the ``save_freq > 0`` comparison at step 1."""
        with pytest.raises(TypeError):
            _periodic_saves(value)

    def test_no_value_outside_the_domain_is_honored_by_both_lerobot_routes(self) -> None:
        """The parity failure the shared gate exists to close.

        LeRobot reaches the same field two ways - an argv token it decodes and an
        in-process assignment nothing decodes - and for every value outside the
        domain exactly one of them takes it. One spec would otherwise mean two
        different runs depending on which route the backend chose.
        """
        honored_by_both = [value for value in UNUSABLE if _decodes_as_the_argv_token(value) and _runs_in_process(value)]
        assert honored_by_both == [], f"unusable values both routes accept: {honored_by_both}"

    @pytest.mark.parametrize("value", USABLE)
    def test_every_value_inside_the_domain_is_honored_by_both_routes(self, value: int) -> None:
        """The other direction: the domain admits nothing either route refuses."""
        assert _decodes_as_the_argv_token(value)
        assert _runs_in_process(value)

    def test_the_two_routes_disagree_in_both_directions(self) -> None:
        """Non-vacuity for the parity test: neither route is simply the stricter.

        If every unusable value failed on the same route, one route's check would
        suffice and the shared domain would be redundant. They do not: the argv
        decoder refuses what the in-process path obeys, and vice versa.
        """
        argv_only = [v for v in UNUSABLE if _runs_in_process(v) and not _decodes_as_the_argv_token(v)]
        inproc_only = [v for v in UNUSABLE if _decodes_as_the_argv_token(v) and not _runs_in_process(v)]
        assert argv_only, "expected values the in-process path obeys and the decoder refuses"
        assert inproc_only, "expected values the decoder accepts and the in-process path refuses"

    @pytest.mark.parametrize("value", DISABLING)
    def test_a_non_positive_cadence_really_disables_periodic_saving(self, value: int) -> None:
        """Why the domain has no floor: the mode is real and is documented.

        Only the periodic saves stop; lerobot still writes the final checkpoint,
        which is what makes this a mode rather than a loss of the run's output.
        """
        assert _periodic_saves(value) == 0
        assert _should_save_checkpoint()(STEPS, value, STEPS) is True

    @pytest.mark.parametrize("value", (float("nan"), float("inf")))
    def test_sagemaker_would_forward_it_as_invalid_json(self, value: float) -> None:
        """The fourth consumer, whose route fails differently again.

        ``json.dumps`` renders these as ``NaN`` / ``Infinity``, which are not
        JSON - only a permissive decoder reads them back, so the container may
        reject the hyperparameter rather than the submission.
        """
        rendered = json.dumps(value)
        with pytest.raises(ValueError):
            json.loads(rendered, parse_constant=_reject)
        assert math.isnan(value) or math.isinf(value)


def _reject(constant: str) -> Any:
    """A strict-JSON hook: refuse the constants ``json.dumps`` emits for non-finites."""
    raise ValueError(constant)


class TestTheGateIsUsableOnItsOwn:
    """The shared gate's own contract, independent of any backend."""

    def test_it_reports_the_context_it_was_given(self, spec: TrainSpec) -> None:
        spec.save_freq = A_FRACTIONAL_CADENCE
        problems = checkpoint_cadence_problems(spec, context="acme")
        assert len(problems) == 1
        assert problems[0].startswith("acme: save_freq must be an integer number of steps, got 2.7.")

    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_cadence_reports_nothing(self, spec: TrainSpec, value: int) -> None:
        spec.save_freq = value
        assert checkpoint_cadence_problems(spec, context="acme") == []

    def test_it_reports_one_problem_at_a_time(self, spec: TrainSpec) -> None:
        """One field, one problem - so a caller fixes one thing per message."""
        spec.save_freq = A_STRING_CADENCE
        assert len(checkpoint_cadence_problems(spec, context="acme")) == 1
