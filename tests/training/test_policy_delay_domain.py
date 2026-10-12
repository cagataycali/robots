"""FastTD3's delayed-update cadence is checked before it can silently skip the actor.

TD3's actor and target networks move only every ``policy_delay``-th gradient
step, and the field is consumed as the modulus of the one test that decides
whether they move at all::

    if self._update_count % spec.policy_delay == 0:  # actor + Polyak step

Nothing downstream judges that modulus. Measured on a CPU FastTD3 run of 6
gradient updates: the shipped ``2`` takes 3 actor updates; ``nan`` takes **0**
(``n % nan`` is ``nan``, which compares unequal to everything) while the
critics train normally, so the run reports success and checkpoints a
deployable actor that never took a gradient step; ``2.5`` takes 1 (only the
integer multiples of 2.5 satisfy the test) and ``True`` takes 6 - each a
silently different cadence from the one the caller named; ``0`` raises
``ZeroDivisionError`` and a string or ``None`` a bare ``TypeError``, from
inside the update loop after the env, the networks, the optimizers and the
replay buffer are built.

The domain is the shared strict-``int``
:func:`~strands_robots.utils.positive_count_error` - the rule this package
already applies to every count consumed as a modulus or ``range()`` bound -
with ``1`` first-class: a delay of one is TD3 with the delay disabled, a
configuration rather than a defect. The refusal through the real ``validate`` entry point is one row of
``tests/training/test_rl_spec_field_domains.py``; this module keeps what the
field does to the run that consumes it.
"""

from __future__ import annotations

import inspect

import pytest

from strands_robots.training.rl import RLTrainSpec


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(
        output_dir="/tmp/policy_delay_domain",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


class TestTheModulusSilentlySkipsTheActor:
    """The field is consumed as a modulus, so a cadence no count satisfies trains no actor."""

    def test_the_cadence_is_the_one_expression_the_field_reaches(self) -> None:
        """Executable premise: the modulus is where the field is consumed."""
        from strands_robots.training.rl.fast_td3 import FastTd3Trainer

        update = inspect.getsource(FastTd3Trainer.update)
        assert "self._update_count % spec.policy_delay == 0" in update
