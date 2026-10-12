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
two fields cannot drift on to two domains. That every backend inheriting the
loop routes the field through the one gate is a row of
``test_every_shared_domain_has_one_owner.py``.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from strands_robots.training._validate import rl_checkpoint_interval_problems
from strands_robots.training.base import TrainSpec
from strands_robots.training.rl.base_algo import RLTrainSpec
from strands_robots.training.rl.fast_sac import FastSacTrainer
from strands_robots.training.rl.fast_td3 import FastTd3Trainer
from strands_robots.training.rl.ppo import PpoTrainer

# The smallest legal BaseRLAlgo, reused rather than copied: its stubbed hooks let
# the *inherited* train loop run, which is the consumer this cadence reaches.
# One domain, so one value set: imported from the supervised guard rather than
# restated, which is what stops the two fields drifting on to two rules.
from tests.training.test_checkpoint_cadence_domain import (
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

    def test_a_non_integral_interval_silently_loses_the_periodic_checkpoints(self) -> None:
        """The worst reading: only the final checkpoint survives, under success.

        ``nan`` never satisfies the modulus, so only the ``it == num_iters - 1``
        arm fires. For RL that is the whole point of the field - return is
        non-monotonic, so the deployable policy is often an earlier iteration.
        """
        for value in (float("nan"), 0.3, float("inf")):
            written = _checkpoint_iterations(value)
            assert written[-1] == RL_ITERS, f"{value!r}"
            assert len(written) < len(THE_REQUESTED_SCHEDULE), f"{value!r}"

    def test_a_fractional_interval_is_the_schedule_of_a_different_integer(self) -> None:
        """``2.5`` is indistinguishable from the ``5`` the caller did not write.

        The conflation a type check catches and a range check cannot.
        """
        assert _checkpoint_iterations(2.5) == _checkpoint_iterations(5) == THE_REQUESTED_SCHEDULE

    def test_a_non_numeric_interval_raises_inside_the_training_loop(self) -> None:
        """Not a ``TrainResult``: a traceback out of ``train()``, after ``setup``.

        The lifecycle's whole contract is to report a failed run as a terminal
        ``TrainResult``; this escaped it with the env, the networks and the
        optimizers already built.
        """
        for value in RAISED_IN_THE_CONSUMER:
            with pytest.raises(TypeError):
                _checkpoint_iterations(value)

    def test_every_value_inside_the_domain_is_honored(self) -> None:
        for value in USABLE:
            written = _checkpoint_iterations(value)
            assert written and written[-1] == RL_ITERS, f"{value!r}"

    def test_zero_disables_the_periodic_checkpoints_and_a_negative_does_not(self) -> None:
        """Which spelling disables is the loop's business, not the domain's.

        Unlike lerobot's ``save_freq > 0`` test, this loop guards on bare
        truthiness: ``0`` leaves only the single end-of-run checkpoint, while a
        negative is the cadence of its magnitude. Both are inside the domain -
        it has no floor - so this pins the asymmetry rather than hiding it.
        """
        assert _checkpoint_iterations(0) == [RL_ITERS]
        assert _checkpoint_iterations(-A_REQUESTED_CADENCE) == THE_REQUESTED_SCHEDULE


class TestTheSharedDomainSurface:
    """The gate itself, called directly."""

    def test_a_spec_without_the_field_reports_nothing(self) -> None:
        """``log_interval`` lives on ``RLTrainSpec``, so a plain spec carries no interval.

        Read defensively for the same reason the other RL-only gates are: the
        signature is the shared ``TrainSpec`` one.
        """
        assert rl_checkpoint_interval_problems(TrainSpec(), context="acme") == []
