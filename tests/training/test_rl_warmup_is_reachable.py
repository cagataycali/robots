# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run budget that never reaches ``learning_starts`` is refused.

The two off-policy backends take no gradient step until the replay buffer holds
``learning_starts`` transitions, and they collect a whole number of iterations of
``rollout_steps * num_envs`` env steps -
``max(1, total_timesteps // steps_per_iter)`` of them. Nothing related the budget
to the threshold, so a budget below it spent the entire run on the uniform-random
warmup: ``update()`` was never called, and the loop then wrote a checkpoint,
exported it and reported ``status="success"``.

Measured on the MuJoCo reach env used by ``tests/training/test_rl_fast_sac.py``
(``batch_size=16``, ``gradient_steps=2``, so ``learning_starts >= batch_size``
holds and the existing relation reports nothing), counting real ``update()``
calls and comparing the exported ``policy.pt`` against the weights ``setup()``
built:

===================================  ==========  ========  =======  ==========================
case                                 ``validate``  ``train``  updates  exported == initialization
===================================  ==========  ========  =======  ==========================
40 steps, 10/iter, warmup 16         ``[]``      success   3        no
40 steps, 10/iter, warmup 40         ``[]``      success   1        no
40 steps, 10/iter, warmup 64         ``[]``      success   **0**    **yes**
45 steps, 10/iter, warmup 45         ``[]``      success   **0**    **yes**
40 steps, 10/iter, warmup 64 (TD3)   ``[]``      success   **0**    **yes**
===================================  ==========  ========  =======  ==========================

So the loadable artifact the caller got back was the initialization, not a
trained policy, under a message that reports the iterations as complete. That is
the outcome :func:`~strands_robots.training._validate.rl_replay_problems` already
refuses for ``gradient_steps=0`` - its docstring records the same "zero gradient
updates, yet the run reported success" - and that
``tests/training/test_learning_starts_count_domain.py`` closes for a
``learning_starts`` that is not a count. This is the third route to it: every
operand is a usable count and the relation between them is what is wrong.

The relation is asked of the loop's own arithmetic rather than of
``total_timesteps`` alone, because the floor division is what the loop collects.
The fourth row above is the case that separates them: 45 steps at 10 per
iteration is four iterations of ten, so a warmup of 45 is out of reach even
though the budget is not below it, and a bare
``total_timesteps >= learning_starts`` test would admit it.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec

#: Both backends gate their update on the same warmup, so both are graded.
OFF_POLICY = ["fast_sac", "fast_td3"]

#: ``(total_timesteps, rollout_steps, learning_starts)`` the loop cannot reach.
#: The last pair is the one a bare ``total_timesteps >= learning_starts`` admits.
UNREACHABLE = [(40, 10, 64), (45, 10, 45), (10, 10, 11)]

#: The same shape, reachable. The middle pair is the exact boundary: the buffer
#: holds precisely ``learning_starts`` when the last iteration's steps land.
REACHABLE = [(40, 10, 16), (40, 10, 40), (100, 10, 100)]


def _spec(total_timesteps: int, rollout_steps: int, learning_starts: Any) -> RLTrainSpec:
    return RLTrainSpec(
        output_dir="/tmp/x",
        total_timesteps=total_timesteps,
        rollout_steps=rollout_steps,
        learning_starts=learning_starts,
        batch_size=16,
        gradient_steps=2,
    )


def _problems(provider: str, *case: Any) -> list[str]:
    return create_trainer(provider).validate(_spec(*case))


def _about_the_warmup(provider: str, *case: Any) -> list[str]:
    return [p for p in _problems(provider, *case) if "never reaches learning_starts" in p]


class TestABudgetThatCannotReachTheWarmupIsRefused:
    """The regression: no such run is admitted, on either backend."""

    @pytest.mark.parametrize("provider", OFF_POLICY)
    @pytest.mark.parametrize("case", UNREACHABLE, ids=str)
    def test_it_is_reported(self, provider: str, case: tuple[int, int, int]) -> None:
        assert _about_the_warmup(provider, *case)

    @pytest.mark.parametrize("provider", OFF_POLICY)
    @pytest.mark.parametrize("case", UNREACHABLE, ids=str)
    def test_the_report_names_both_operands_and_the_consequence(
        self, provider: str, case: tuple[int, int, int]
    ) -> None:
        """A caller must be able to see which two values disagree, and why."""
        problem = _about_the_warmup(provider, *case)[0]
        assert "total_timesteps" in problem and "learning_starts" in problem
        assert "zero gradient steps" in problem


class TestEveryRouteTheRefusalNamesIsOne:
    """A remedy is only a remedy if following it is accepted."""

    @pytest.mark.parametrize("case", UNREACHABLE, ids=str)
    def test_the_budget_it_names_is_reachable(self, case: tuple[int, int, int]) -> None:
        """Naming ``learning_starts`` itself would still floor below it."""
        total_timesteps, rollout_steps, learning_starts = case
        problem = _about_the_warmup("fast_sac", *case)[0]
        named = int(problem.split("raise total_timesteps to at least ")[1].split(" ")[0])
        assert not _about_the_warmup("fast_sac", named, rollout_steps, learning_starts)

    @pytest.mark.parametrize("case", UNREACHABLE, ids=str)
    def test_the_warmup_it_names_is_reachable(self, case: tuple[int, int, int]) -> None:
        total_timesteps, rollout_steps, learning_starts = case
        problem = _about_the_warmup("fast_sac", *case)[0]
        named = int(problem.split("lower learning_starts to at most ")[1].split(" ")[0])
        assert not _about_the_warmup("fast_sac", total_timesteps, rollout_steps, named)


class TestAReachableBudgetIsUntouched:
    """The over-reach control: the guard refuses only the unreachable relation."""

    @pytest.mark.parametrize("provider", OFF_POLICY)
    @pytest.mark.parametrize("case", REACHABLE, ids=str)
    def test_it_is_accepted(self, provider: str, case: tuple[int, int, int]) -> None:
        assert not _about_the_warmup(provider, *case)

    @pytest.mark.parametrize("provider", OFF_POLICY)
    def test_the_shipped_defaults_are_accepted(self, provider: str) -> None:
        """99984 steps collected against a warmup of 1000."""
        spec = RLTrainSpec(output_dir="/tmp/x")
        assert not [p for p in create_trainer(provider).validate(spec) if "never reaches" in p]


class TestTheGuardLeavesTheCountDomainToTheCountDomain:
    """An operand that is not a count is reported once, by the domain that owns it."""

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "10", None, True])
    def test_a_non_count_warmup_is_not_also_called_unreachable(self, value: Any) -> None:
        """Comparing against a non-count decides the relation silently, so it is not asked."""
        assert not _about_the_warmup("fast_sac", 40, 10, value)

    def test_a_warmup_below_the_batch_still_fails_its_own_relation(self) -> None:
        """The sibling relation on the same threshold is preserved, not replaced."""
        problems = _problems("fast_sac", 40, 10, 8)
        assert any("must be >= batch_size" in p for p in problems)


class TestOnPolicyDoesNotReadTheWarmup:
    """PPO has no warmup gate, so the relation must report nothing for it."""

    def test_ppo_is_silent_about_a_budget_below_the_warmup(self) -> None:
        spec = _spec(40, 10, 64)
        assert not [p for p in create_trainer("ppo").validate(spec) if "never reaches" in p]
