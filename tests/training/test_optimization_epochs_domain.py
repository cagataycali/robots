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
on a field it never reads. The refusal through the real ``validate`` entry point is one row of
``tests/training/test_rl_spec_field_domains.py``; this module keeps what the
field does to the run that consumes it.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import pytest

from strands_robots.training.rl import RLTrainSpec


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(output_dir="/tmp/optimization_epochs_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


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
