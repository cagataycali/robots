"""The interval an RL run checkpoints at is the same shared step-cadence domain.

:attr:`~strands_robots.training.rl.base_algo.RLTrainSpec.log_interval` is what
the RL loop paces :meth:`BaseRLAlgo.save_checkpoint` by. Its name and its spec
entry both said "iterations between progress logs", but no RL module emits a log
line at all: the field is read in exactly one expression, and that expression
decides whether an intermediate checkpoint is written::

    if spec.log_interval and (it % spec.log_interval == 0 or it == num_iters - 1):
        ckpt_dir = self.save_checkpoint(spec.output_dir, iteration=it + 1)

So it is the same *kind* of value as
:attr:`~strands_robots.training.base.TrainSpec.save_freq` - a periodic
checkpoint cadence consumed as the modulus of a step test - and it reached that
modulus with no domain at all, while the supervised field beside it on the same
spec has one (``tests/training/test_checkpoint_cadence_domain.py``). One spec
therefore meant two readings: a cadence refused for a supervised run was
accepted for an RL one.

Nothing downstream judges it. Measured on the inherited loop over a
20-iteration run, against the ``[1, 6, 11, 16, 20]`` an ``int`` cadence of 5
writes:

* ``True`` wrote 20 checkpoints - one every iteration, a modulus of one.
* ``2.5`` wrote the schedule of ``5``, so a caller who halved the cadence to
  double the checkpoints got the same five.
* ``nan`` wrote 1, at the final iteration: the modulus is never satisfied, so
  the field silently became the *disabled* mode under ``status="success"``.
  ``0.3`` and ``inf`` lose the periodic checkpoints the same way.
* ``"5"`` raised ``TypeError`` out of ``train()`` - past a ``validate``
  documented to return every problem a run has, and after ``setup`` had built
  the env, the networks and the optimizers.

For RL those intermediate checkpoints are not a convenience: return is
non-monotonic in training, so the deployable policy is often an earlier
iteration, and a run that silently kept only its last cannot be recovered
without training again.

The tests below pin the contract in both directions, against the same value sets
the supervised half uses - imported from that guard rather than restated, so the
two fields cannot drift on to two domains.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap
from typing import Any

import pytest

from strands_robots.training._validate import rl_checkpoint_interval_problems
from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.mock import MockTrainer
from strands_robots.training.rl.base_algo import BaseRLAlgo, RLTrainSpec
from strands_robots.training.rl.fast_sac import FastSacTrainer
from strands_robots.training.rl.fast_td3 import FastTd3Trainer
from strands_robots.training.rl.ppo import PpoTrainer

# The smallest legal BaseRLAlgo, reused rather than copied: its stubbed hooks let
# the *inherited* train loop run, which is the consumer this cadence reaches.
from tests.training._spec_field_reads import reads_spec_field

# One domain, so one value set: imported from the supervised guard rather than
# restated, which is what stops the two fields drifting on to two rules.
from tests.training.test_checkpoint_cadence_domain import (
    A_FRACTIONAL_CADENCE,
    A_STRING_CADENCE,
    CHECKPOINTING_BACKENDS,
    RAISED_IN_THE_CONSUMER,
    UNUSABLE,
    USABLE,
)
from tests.training.test_rl_base_lifecycle_contract import _BareRLAlgo

# Every RL backend checkpoints from the field - all three inherit the one loop.
RL_BACKENDS = (PpoTrainer, FastSacTrainer, FastTd3Trainer)

# Iterations in the measured run, and the cadence a caller writes. Sized so a
# wrong reading shows up as a different number of checkpoints.
RL_ITERS = 20
A_REQUESTED_CADENCE = 5
# What that cadence really writes, measured on the inherited loop.
THE_REQUESTED_SCHEDULE = [1, 6, 11, 16, 20]

# The shape the shared domain emits: ``"{context}: log_interval must be ..."``.
_NAMES_THE_INTERVAL = ": log_interval "


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> RLTrainSpec:
    """A launchable RL spec whose ``log_interval`` is the only thing under test.

    ``validate`` may report unrelated problems; every assertion below filters for
    the field name, so an unrelated problem can neither mask nor fake a verdict.
    """
    return RLTrainSpec(
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
        output_dir=str(tmp_path / "out"),
        total_timesteps=RL_ITERS,
    )


def _interval_problems_of(trainer: Trainer, spec: TrainSpec) -> list[str]:
    """``validate`` problems about ``log_interval``."""
    return [p for p in trainer.validate(spec) if _NAMES_THE_INTERVAL in p]


def _checkpoint_iterations(cadence: Any) -> list[int | None]:
    """Iterations the inherited RL loop writes a checkpoint at, for *cadence*.

    Drives the real :meth:`BaseRLAlgo.train` over stubbed hooks, so the schedule
    is the loop's own reading of the field rather than a restatement of it. The
    stub's ``validate`` reports nothing, which is what lets a value the gate now
    refuses still be measured against the consumer it used to reach.
    """
    algo = _BareRLAlgo()
    algo.train(RLTrainSpec(total_timesteps=RL_ITERS, log_interval=cadence, output_dir="/tmp/rl-out"))
    return list(algo.saved_iterations)


class TestEveryRLBackendRefusesAnUnusableInterval:
    """First half of the biconditional: a backend that checkpoints must refuse."""

    @pytest.mark.parametrize("trainer_cls", RL_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: RLTrainSpec, trainer_cls: type[Trainer], value: Any) -> None:
        spec.log_interval = value
        assert _interval_problems_of(trainer_cls(), spec) != []

    @pytest.mark.parametrize("trainer_cls", RL_BACKENDS)
    def test_the_problem_names_the_backend_that_refused_it(self, spec: RLTrainSpec, trainer_cls: type[Trainer]) -> None:
        trainer = trainer_cls()
        spec.log_interval = A_FRACTIONAL_CADENCE
        problems = _interval_problems_of(trainer, spec)
        assert problems and problems[0].startswith(f"{trainer.provider_name}: ")

    @pytest.mark.parametrize("trainer_cls", RL_BACKENDS)
    @pytest.mark.parametrize("value", RAISED_IN_THE_CONSUMER)
    def test_an_unusable_interval_is_a_problem_not_an_exception(
        self, spec: RLTrainSpec, trainer_cls: type[Trainer], value: Any
    ) -> None:
        """``validate`` is documented to *return* problems, for every spelling.

        A str or a list reaches the loop's ``%`` as an operand it cannot take, so
        a preflight that compared the value itself would raise here instead.
        """
        spec.log_interval = value
        assert _interval_problems_of(trainer_cls(), spec) != []

    @pytest.mark.parametrize("trainer_cls", RL_BACKENDS)
    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_interval_is_not_a_problem(
        self, spec: RLTrainSpec, trainer_cls: type[Trainer], value: int
    ) -> None:
        """Including ``0`` and a negative: this domain has no floor."""
        spec.log_interval = value
        assert _interval_problems_of(trainer_cls(), spec) == []

    @pytest.mark.parametrize("trainer_cls", RL_BACKENDS)
    def test_the_default_interval_is_not_a_problem(self, spec: RLTrainSpec, trainer_cls: type[Trainer]) -> None:
        assert _interval_problems_of(trainer_cls(), spec) == []


class TestASupervisedBackendReportsNothingAboutIt:
    """Second half of the biconditional: a non-reader must not report.

    Per :class:`~strands_robots.training.base.TrainSpec` a backend ignores the
    fields it does not support, so a supervised backend reporting on the RL
    loop's interval would be a false rejection - it has ``save_freq`` for the
    same question.
    """

    @pytest.mark.parametrize("trainer_cls", (*CHECKPOINTING_BACKENDS, MockTrainer))
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_says_nothing(self, spec: RLTrainSpec, trainer_cls: type[Trainer], value: Any) -> None:
        spec.log_interval = value
        assert _interval_problems_of(trainer_cls(), spec) == []


class TestWhatTheLoopDoesWithAnUnusableInterval:
    """Ground every refusal in the schedule the loop really writes."""

    def test_the_requested_cadence_is_the_reference(self) -> None:
        """Non-vacuity: an int cadence writes the schedule the others are read against."""
        assert _checkpoint_iterations(A_REQUESTED_CADENCE) == THE_REQUESTED_SCHEDULE

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_does_not_write_the_requested_schedule(self, value: Any) -> None:
        try:
            written = _checkpoint_iterations(value)
        except TypeError:
            return  # raised out of the loop - pinned by its own test below
        assert written != THE_REQUESTED_SCHEDULE

    def test_a_boolean_interval_checkpoints_every_single_iteration(self) -> None:
        """``True`` is a modulus of one: 20 checkpoints in a 20-iteration run."""
        assert _checkpoint_iterations(True) == list(range(1, RL_ITERS + 1))

    @pytest.mark.parametrize("value", (float("nan"), 0.3, float("inf")))
    def test_a_non_integral_interval_silently_loses_the_periodic_checkpoints(self, value: float) -> None:
        """The worst reading: only the final checkpoint survives, under success.

        ``nan`` never satisfies the modulus, so only the ``it == num_iters - 1``
        arm fires. For RL that is the whole point of the field - return is
        non-monotonic, so the deployable policy is often an earlier iteration.
        """
        written = _checkpoint_iterations(value)
        assert written[-1] == RL_ITERS
        assert len(written) < len(THE_REQUESTED_SCHEDULE)

    def test_a_fractional_interval_is_the_schedule_of_a_different_integer(self) -> None:
        """``2.5`` is indistinguishable from the ``5`` the caller did not write.

        The conflation a type check catches and a range check cannot.
        """
        assert _checkpoint_iterations(2.5) == _checkpoint_iterations(5) == THE_REQUESTED_SCHEDULE

    @pytest.mark.parametrize("value", RAISED_IN_THE_CONSUMER)
    def test_a_non_numeric_interval_raises_inside_the_training_loop(self, value: Any) -> None:
        """Not a ``TrainResult``: a traceback out of ``train()``, after ``setup``.

        The lifecycle's whole contract is to report a failed run as a terminal
        ``TrainResult``; this escaped it with the env, the networks and the
        optimizers already built.
        """
        with pytest.raises(TypeError):
            _checkpoint_iterations(value)

    @pytest.mark.parametrize("value", USABLE)
    def test_every_value_inside_the_domain_is_honored(self, value: int) -> None:
        written = _checkpoint_iterations(value)
        assert written and written[-1] == RL_ITERS

    def test_zero_disables_the_periodic_checkpoints_and_a_negative_does_not(self) -> None:
        """Which spelling disables is the loop's business, not the domain's.

        Unlike lerobot's ``save_freq > 0`` test, this loop guards on bare
        truthiness: ``0`` leaves only the single end-of-run checkpoint, while a
        negative is the cadence of its magnitude. Both are inside the domain -
        it has no floor - so this pins the asymmetry rather than hiding it.
        """
        assert _checkpoint_iterations(0) == [RL_ITERS]
        assert _checkpoint_iterations(-A_REQUESTED_CADENCE) == THE_REQUESTED_SCHEDULE


def _training_modules() -> list[pathlib.Path]:
    """Every training module, minus the one that defines the shared gate.

    Rooted at the module that defines :class:`Trainer` rather than at the RL
    package, so a supervised module that starts reading the field is in scope
    too. The module that *defines* the gate is excluded - derived from the gate
    itself rather than named, so the exclusion cannot drift - because it reads
    the field as its owner, not as a consumer.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(rl_checkpoint_interval_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_interval(source: str) -> bool:
    """Does *source* read ``spec.log_interval``, by name or through a table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read.
    """
    return reads_spec_field(source, ("log_interval",))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_rl_checkpoint_interval_problems"
        for node in ast.walk(ast.parse(source))
    )


def _concrete_rl_backends() -> set[type[BaseRLAlgo]]:
    """Every shipped :class:`BaseRLAlgo` subclass, discovered rather than listed.

    The subclasses are only ever read here (their ``validate`` source is
    inspected, never instantiated), which is why holding the abstract base's
    ``type`` is sound.
    """
    # type-abstract: the base is an ABC, and these are held rather than called.
    shipped = [c for c in BaseRLAlgo.__subclasses__() if c.__module__.startswith("strands_robots.")]
    return set(shipped)  # type: ignore[type-abstract]


class TestOneOwnerForTheRLCheckpointInterval:
    """No RL backend may skip the domain, and none may re-implement it.

    Scope is derived from the class hierarchy, not from the text, because a
    *textual* read is not what puts a backend in scope here: FastSAC and FastTD3
    override ``train`` and so name the field, while PPO inherits
    :meth:`BaseRLAlgo.train` and never mentions it - yet PPO runs the identical
    loop over the identical field. A scan keyed on the attribute access would
    report a clean sweep with PPO's interval ungated. What every backend in scope
    shares is that it *is* a ``BaseRLAlgo``, so that is the derivation, and a
    fourth RL backend fails these tests until its ``validate`` routes through the
    gate.
    """

    def test_the_discovery_finds_every_shipped_rl_backend(self) -> None:
        """Non-vacuity, and the agreement cell: a new backend cannot arrive unpinned."""
        assert sorted(c.__name__ for c in _concrete_rl_backends()) == sorted(c.__name__ for c in RL_BACKENDS)

    def test_every_rl_backend_routes_through_the_shared_gate(self) -> None:
        adrift = sorted(
            cls.__name__
            for cls in _concrete_rl_backends()
            if not _calls_the_gate(textwrap.dedent(inspect.getsource(cls.validate)))
        )
        assert adrift == [], f"RL backends whose validate skips the shared gate: {adrift}"

    def test_the_loop_that_consumes_the_interval_is_the_inherited_one(self) -> None:
        """Why the hierarchy is the right scope: PPO reads the field by inheriting.

        Pins the asymmetry the derivation is built on, so a refactor that moves
        the loop cannot quietly invalidate it.
        """
        readers = {p.name for p in _training_modules() if _reads_the_interval(p.read_text())}
        assert readers == {"base_algo.py", "fast_sac.py", "fast_td3.py"}
        assert PpoTrainer.train is BaseRLAlgo.train

    def test_no_backend_re_implements_the_domain(self) -> None:
        offenders: list[str] = []
        for path in _training_modules():
            for line in path.read_text().splitlines():
                if "spec.log_interval" in line and ("int(" in line or "isinstance" in line):
                    offenders.append(f"{path.name}: {line.strip()}")
        assert offenders == [], f"local domain checks on spec.log_interval: {offenders}"

    def test_the_scanners_detect_a_planted_defect(self) -> None:
        """A scanner that silently matched nothing would look like a clean tree."""
        planted = "def validate(self, spec):\n    return [f'--log_every={spec.log_interval}']\n"
        assert _reads_the_interval(planted)
        assert not _calls_the_gate(planted)

    def test_the_scanners_detect_a_table_driven_defect(self) -> None:
        """A backend that forwards the field by name is a reader too."""
        planted = (
            'FIELDS = ("log_interval",)\ndef validate(self, spec):\n    return [getattr(spec, f) for f in FIELDS]\n'
        )
        assert _reads_the_interval(planted)
        assert not _calls_the_gate(planted)


class TestTheSharedDomainSurface:
    """The gate itself, called directly."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_refuses_what_the_supervised_gate_refuses(self, value: Any) -> None:
        assert rl_checkpoint_interval_problems(RLTrainSpec(log_interval=value), context="acme") != []

    @pytest.mark.parametrize("value", USABLE)
    def test_it_accepts_what_the_supervised_gate_accepts(self, value: int) -> None:
        assert rl_checkpoint_interval_problems(RLTrainSpec(log_interval=value), context="acme") == []

    def test_it_reports_one_problem_at_a_time_naming_its_own_field(self) -> None:
        problems = rl_checkpoint_interval_problems(RLTrainSpec(log_interval=A_FRACTIONAL_CADENCE), context="acme")
        assert len(problems) == 1
        assert problems[0].startswith("acme: log_interval must be an integer number of steps, got 2.7.")

    def test_it_reports_the_context_it_was_given(self) -> None:
        problems = rl_checkpoint_interval_problems(RLTrainSpec(log_interval=A_STRING_CADENCE), context="acme")
        assert problems and problems[0].startswith("acme: ")

    def test_a_spec_without_the_field_reports_nothing(self) -> None:
        """``log_interval`` lives on ``RLTrainSpec``, so a plain spec carries no interval.

        Read defensively for the same reason the other RL-only gates are: the
        signature is the shared ``TrainSpec`` one.
        """
        assert rl_checkpoint_interval_problems(TrainSpec(), context="acme") == []
