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

Which backends refuse which values, in which words, is the table in
``tests/training/test_train_spec_field_domains.py``. The tests below
pin that every usable cadence is one both lerobot routes honor.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.cosmos3 import Cosmos3Trainer
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.training.sagemaker import SagemakerTrainer

# The backends that checkpoint from the field.
CHECKPOINTING_BACKENDS = (LerobotTrainer, Cosmos3Trainer, SagemakerTrainer)

# Values no consumer can honor, split by how each one failed before the gate.

# Passes the ``> 0`` test and then means something the caller did not write:
# ``True`` a modulus of one, the rest a cadence no integral step satisfies.
SILENTLY_WRONG = (True, 2.7, 5000.0, float("nan"), float("inf"))

# Raised out of the consumer: the comparison itself refuses a non-number.
RAISED_IN_THE_CONSUMER: tuple[Any, ...] = ("5000", [7])

UNUSABLE: tuple[Any, ...] = SILENTLY_WRONG + RAISED_IN_THE_CONSUMER

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
    def test_the_message_says_how_to_disable_periodic_saving(self, spec: TrainSpec, trainer_cls: type[Trainer]) -> None:
        """A caller who wrote ``2.7`` may have meant "rarely"; say what to write.

        The disabled mode is reachable only through a non-positive value, so a
        refusal that did not mention it would leave the caller to guess.
        """
        spec.save_freq = A_FRACTIONAL_CADENCE
        assert "non-positive" in _cadence_problems_of(trainer_cls(), spec)[0]


class TestTheRefusedValuesAreOnesNoConsumerCanHonor:
    """Ground the domain in what the real consumers do with each value."""

    def test_every_value_inside_the_domain_is_honored_by_both_routes(self) -> None:
        """The other direction: the domain admits nothing either route refuses."""
        for value in USABLE:
            assert _decodes_as_the_argv_token(value), f"{value!r}"
            assert _runs_in_process(value), f"{value!r}"

    def test_a_non_positive_cadence_really_disables_periodic_saving(self) -> None:
        """Why the domain has no floor: the mode is real and is documented.

        Only the periodic saves stop; lerobot still writes the final checkpoint,
        which is what makes this a mode rather than a loss of the run's output.
        """
        for value in DISABLING:
            assert _periodic_saves(value) == 0, f"{value!r}"
            assert _should_save_checkpoint()(STEPS, value, STEPS) is True, f"{value!r}"
