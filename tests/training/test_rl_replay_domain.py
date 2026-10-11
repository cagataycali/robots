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

The refusal through the real ``validate`` entry point is one row of
``tests/training/test_rl_spec_field_domains.py``; this module keeps what the
field does to the run that consumes it.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import rl_replay_problems
from strands_robots.training.rl import RLTrainSpec

# Values that survive ``value <= 0`` and cannot be a count. ``True`` is the silent
# half - a degenerate size of one - and the fractions / non-finite values reach a
# tensor allocation or a ``range()`` and raise there.
SILENTLY_DEGENERATE_OR_LATE: list[Any] = [True, 0.5, float("nan"), float("inf")]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(output_dir="/tmp/rl_replay_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


class TestTheDomainIsTheSharedCountRule:
    """The gate contributes no rule of its own beyond reading the three fields."""

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

    def test_a_non_count_survives_the_local_comparison(self) -> None:
        """Why a local ``<= 0`` is weaker: each of these passes it."""
        for value in SILENTLY_DEGENERATE_OR_LATE:
            assert not (value <= 0), f"{value!r}"

    def test_a_bool_is_a_degenerate_size_of_one(self) -> None:
        """The silent half: ``True`` is a one-slot buffer / a batch of one."""
        assert int(True) == 1
        assert len(range(True)) == 1

    def test_a_fractional_capacity_truncates_to_zero(self) -> None:
        """``int(0.5) == 0`` - the zero-capacity buffer the IndexError came from."""
        assert int(0.5) == 0

    def test_a_non_numeric_count_raises_out_of_the_comparison_itself(self) -> None:
        """Why a local ``<= 0`` cannot even report: it raises before appending."""
        non_numeric: list[Any] = ["256", None]
        for value in non_numeric:
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

    def test_fast_sac_still_refuses_learning_starts_below_batch_size(self, spec: RLTrainSpec) -> None:
        spec.learning_starts = 1
        spec.batch_size = 256
        assert [p for p in create_trainer("fast_sac").validate(spec) if "learning_starts" in p]
