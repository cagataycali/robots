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

``num_envs``, the third factor of ``steps_per_iter``, is outside this gate because
which *counts* are usable differs per backend - PPO parallelizes and accepts any
positive count, the MuJoCo-backed FastSAC requires exactly ``1`` - so that half is
not one shared rule. It is only that half: whether the value is a count at all is
the same domain, and each backend consults it before asking its own count rule, so
the third factor of this product is held to what its two siblings are held to.
:class:`TestNumEnvsIsNotInTheSharedDomain` pins the scope line;
``test_rl_env_count_domain.py`` pins the shared half.

Which backends refuse which values, in which words, is graded for every backend
in ``test_rl_spec_field_domains.py``, and that every backend inheriting the loop
routes the two fields through the one gate is a row of
``test_every_shared_domain_has_one_owner.py``; this file pins why the domain exists.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import rl_run_size_problems
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import positive_count_error

# The two values PPO's divisibility relation used to mishandle. Annotated ``Any``
# because they are deliberately outside the field's declared type - which is the
# property under test, so the annotation must not stand in the way of asserting it.
A_NON_FINITE_ROLLOUT: Any = float("nan")
A_STRING_ROLLOUT: Any = "24"


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(output_dir="/tmp/rl_run_size_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


class TestTheDomainIsTheSharedCountRule:
    """The gate contributes no rule of its own beyond reading the two fields."""

    def test_a_spec_without_the_rl_fields_reports_nothing(self) -> None:
        """The gate reads through ``getattr`` defaults, so a plain spec is quiet."""
        from strands_robots.training.base import TrainSpec

        assert rl_run_size_problems(TrainSpec(output_dir="/tmp/x"), context="mock") == []


# --- why the domain exists: the bound is derived, and the clamp hides that ----


class TestTheLoopBoundIsDerivedThroughAClamp:
    """The premise the domain rests on, asserted rather than described.

    If a later change drops the ``max(1, ...)`` clamp or stops deriving the bound
    from these two fields, these fail and the gate's stated reason gets
    re-examined instead of quietly becoming wrong. Every RL backend runs
    ``BaseRLAlgo.train`` (pinned in ``test_rl_base_lifecycle_contract.py``), so
    that one loop is the one to read.
    """

    def test_the_bound_is_the_clamped_quotient_of_the_two_factors(self) -> None:
        from strands_robots.training.rl.base_algo import BaseRLAlgo

        source = inspect.getsource(BaseRLAlgo.train)
        assert "num_iters = max(1, spec.total_timesteps // steps_per_iter)" in source
        assert "for it in range(num_iters):" in source


class TestNumEnvsIsNotInTheSharedDomain:
    """The scope line, pinned: the third factor is per-backend by necessity.

    ``num_envs`` is the other factor of ``steps_per_iter``, so it looks like it
    belongs beside the two above. It cannot: the backends disagree about which
    counts are usable, so there is no one domain to share.
    """

    def test_the_gate_reports_nothing_about_it(self, spec: RLTrainSpec) -> None:
        spec.num_envs = 0
        assert rl_run_size_problems(spec, context="ppo") == []


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
