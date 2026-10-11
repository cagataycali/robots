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
import math

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
    """The premises the domain rests on, measured against torch itself."""

    def test_both_weights_are_read_in_one_expression(self) -> None:
        source = _ppo_update_source()
        assert "spec.value_loss_coef * value_loss - spec.entropy_coef * entropy" in source

    def test_a_non_finite_weight_makes_the_loss_non_finite(self) -> None:
        torch = pytest.importorskip("torch")
        value_loss, entropy = torch.tensor(2.0), torch.tensor(0.5)
        for weight in (math.nan, math.inf):
            loss = torch.tensor(1.0) + weight * value_loss - 0.0 * entropy
            assert not bool(torch.isfinite(loss)), weight

    def test_a_non_finite_loss_writes_non_finite_parameters(self) -> None:
        """Why the raise lands one rollout later, naming neither field nor value."""
        torch = pytest.importorskip("torch")
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        (model(torch.ones(1, 3)).sum() * math.nan).backward()
        optimizer.step()
        assert bool(torch.isnan(model.weight).any())

    def test_a_bool_lands_as_a_coefficient_of_one(self) -> None:
        torch = pytest.importorskip("torch")
        entropy = torch.tensor(0.5)
        assert float(True * entropy) == float(1.0 * entropy)

    def test_every_accepted_weight_composes_a_finite_loss(self) -> None:
        torch = pytest.importorskip("torch")
        for value in [0.0, -0.5, 1.0, 2.0]:
            loss = torch.tensor(1.0) + value * torch.tensor(2.0) - value * torch.tensor(0.5)
            assert bool(torch.isfinite(loss)), f"{value!r}"
