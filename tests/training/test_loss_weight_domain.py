"""The on-policy loss weights are refused when they cannot be honored.

``value_loss_coef`` and ``entropy_coef`` are the two scalars that weight the
terms of the objective PPO's update descends, read in exactly one place - the
single expression that composes it::

    loss = surrogate_loss + spec.value_loss_coef * value_loss - spec.entropy_coef * entropy

Nothing judged either of them and the multiplication cannot, so every value a
caller cannot have meant reached the backward pass. Measured on a seeded 60-step
run whose honored checkpoint parameter sum is ``140.6023186540351162``:

* ``entropy_coef=True`` reports ``success`` and writes a checkpoint whose sum is
  ``140.6158002523716277`` - an entropy bonus at full weight where the field
  ships defaulting to ``0.0``, requested by a value that reads as a flag.
* ``nan`` / ``inf`` / ``"1.0"`` raise out of ``train()`` - documented to return a
  terminal ``TrainResult`` - from inside ``torch``, naming neither the field nor
  the value, after the env, the networks and a full rollout have been built.

The floor is deliberately **not** part of the domain: zero and negative are real
configurations for both fields, so this gate tests finiteness and numeric-ness
only. That is what distinguishes it from the sibling ``max_grad_norm`` gate,
whose endpoint ``clip_grad_norm_`` itself settles. The refusal through the real
``validate`` entry point is one row of ``tests/training/test_rl_spec_field_domains.py``;
this module keeps what the weights do to the objective.
"""

from __future__ import annotations

import inspect

import pytest

from strands_robots.training.rl import RLTrainSpec


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(output_dir="/tmp/loss_weight_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


def _ppo_update_source() -> str:
    """The body of the on-policy update, so the premises below are the real one."""
    from strands_robots.training.rl.ppo import PpoTrainer

    return inspect.getsource(PpoTrainer.update)


class TestTheConsumerHonorsTheDomain:
    """Both weights reach the loss in the one expression the update computes."""

    def test_both_weights_are_read_in_one_expression(self) -> None:
        source = _ppo_update_source()
        assert "spec.value_loss_coef * value_loss - spec.entropy_coef * entropy" in source
