"""The on-policy trust-region half-width is checked against its domain.

``RLTrainSpec.clip_param`` is the half-width of the trust region PPO is named
for. ``PpoTrainer.update`` reads it twice per mini-batch, in the two expressions
that clip::

    surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - spec.clip_param, 1.0 + spec.clip_param)
    value_clipped = old_values + (value - old_values).clamp(-spec.clip_param, spec.clip_param)

Nothing judged the value, and ``torch.clamp`` cannot: it is defined for every
value below, and each one produced a *finite, successful, deployable* run whose
objective was not the one the caller configured. Measured on this backend before
the gate existed, over a seeded 60-step run against a never-trained control whose
checkpoint parameter sum is ``139.8929914773252676``:

* ``nan`` **silently removed the trust region.** Both clipped terms become
  ``nan``, so ``torch.max(surrogate, surrogate_clipped)`` returns ``nan`` - but
  its gradient flows to the *unclipped* branch, because every comparison against
  ``nan`` is false. The run therefore descended the unclipped objective, and its
  checkpoint parameter sum was ``140.1735330768706262`` - bit-identical to the
  ``inf`` run - while ``surrogate_loss``, ``value_loss`` and ``latest_loss`` were
  all reported as ``nan``. PPO's defining mechanism was off, and the only signal
  that anything had happened was a metric a caller cannot act on.
* ``-0.2`` **inverted the clamp bounds.** ``1 - c`` exceeds ``1 + c``, so the
  clamp returns a constant regardless of the ratio, and the reported surrogate
  loss changed sign - ``+0.081992`` against the honored run's ``-0.008662``. The
  checkpoint sum moved to ``140.1913412402318500``.
* ``0`` is the same failure at the boundary. The value clip becomes
  ``clamp(-0, 0)``, so ``value_clipped`` is exactly ``old_values`` and the
  critic's clipped branch is a constant; the checkpoint sum moved to
  ``140.2282075245283863``.
* ``-inf`` trained on ``inf`` losses and still reported success
  (``140.0613218537641842``), and ``True`` was a silent half-width of one - five
  times the shipped ``0.2`` - written by a value that reads as a flag.
* ``"0.2"``, ``None`` and ``[0.2]`` raised ``TypeError`` from ``rl/ppo.py``
  mid-update, after the environment, the networks and a full rollout had been
  built.

The honored default's sum is ``140.1741519418580992``, so every row above is a
run that trained *differently* rather than harmlessly.

``inf`` is **inside** the domain: it is the field's only spelling of "do not
clip", and the consumer honors it - ``clamp(ratio, -inf, inf)`` returns the ratio
unchanged, which is a coherent unclipped run with finite losses. That is the same
endpoint, settled the same way, as the sibling clip bound ``max_grad_norm``, so
the two share one domain helper rather than carrying a copy each.

Scoped to the on-policy backend: ``spec.clip_param`` appears in ``rl/ppo.py`` and
nowhere else, so a backend that does not clip a policy ratio must not report on a
field it never reads. The refusal through the real ``validate`` entry point is one row of
``tests/training/test_rl_spec_field_domains.py``; this module keeps what the
field does to the run that consumes it.
"""

from __future__ import annotations

import ast
import inspect
import math
from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import (
    clip_range_problems,
    gradient_clip_problems,
)
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import positive_finite_number_error

# The one backend whose update clips a policy ratio.
ON_POLICY = "ppo"

# Half-widths the clip can honor.
USABLE: list[Any] = [0.2, 0.1, 0.5, 1.0, 3, np.float64(0.2), np.int64(2)]

# The only spelling of "do not clip" the field has.
NO_CLIPPING: list[Any] = [math.inf, float("inf"), np.float64(math.inf)]

# A window of zero width is not a window: the value clip pins ``value_clipped``
# to ``old_values`` exactly, so the critic's clipped branch is a constant.
NO_WINDOW: list[Any] = [0, 0.0]

# A negative half-width inverts the clamp bounds, which then return a constant
# regardless of the ratio.
INVERTED_WINDOW: list[Any] = [-0.2, -1.0, -100.0]

# Values the half-width cannot mean, or that read as a different width than the
# caller wrote. ``True`` is a half-width of one.
NOT_A_WIDTH: list[Any] = [
    True,
    False,
    float("nan"),
    float("-inf"),
    "0.2",
    "inf",
    None,
    [0.2],
    {"clip_param": 0.2},
    np.bool_(True),
]

UNUSABLE: list[Any] = [*NO_WINDOW, *INVERTED_WINDOW, *NOT_A_WIDTH]


def _spec(**overrides: Any) -> RLTrainSpec:
    """An otherwise-valid RL spec carrying *overrides*.

    Every value these tests set is deliberately outside its field's annotation -
    that is the property under test - so the overrides are applied through
    ``setattr``, and the one suppression the ``env_factory`` placeholder needs
    lives here rather than at each call site.
    """
    spec = RLTrainSpec(output_dir="/tmp/clip_range_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]
    for field, value in overrides.items():
        setattr(spec, field, value)
    return spec


@pytest.fixture
def spec() -> RLTrainSpec:
    """A spec the on-policy preflight otherwise accepts."""
    return _spec()


class TestTheTrustRegionRefusesAWidthItCannotHonor:
    """A half-width no reading makes usable is reported, not clipped with."""

    def test_two_unusable_optimization_fields_report_two_problems(self, spec: RLTrainSpec) -> None:
        """The clip-range gate is additional to the sibling bound, not a merge."""
        spec.clip_param = float("nan")
        spec.max_grad_norm = float("nan")
        problems = create_trainer(ON_POLICY).validate(spec)
        assert sum(1 for p in problems if ": clip_param " in p) == 1
        assert sum(1 for p in problems if ": max_grad_norm " in p) == 1


class TestTheUsableDomainIsUntouched:
    """Every half-width the clip can honor still passes."""

    def test_a_spec_without_the_field_is_silent(self) -> None:
        """A plain ``TrainSpec`` has no ``clip_param``, so there is nothing to judge."""
        from strands_robots.training.base import TrainSpec

        plain = TrainSpec(output_dir="/tmp/clip_range_domain")
        assert clip_range_problems(plain, context="ppo") == []


class TestTheTwoClipBoundsShareOneDomain:
    """One rule, one home - the two on-policy clip bounds cannot drift apart."""

    def test_both_bounds_give_the_same_verdict(self) -> None:
        for value in [*USABLE, *NO_CLIPPING, *UNUSABLE]:
            width = _spec(clip_param=value)
            norm = _spec(max_grad_norm=value)
            assert bool(clip_range_problems(width, context="ppo")) == bool(
                gradient_clip_problems(norm, context="ppo")
            ), f"the two clip bounds disagree on {value!r}"

    def test_both_gates_read_the_one_shared_helper(self) -> None:
        """Structural: neither gate carries its own copy of the rule."""
        for gate in (clip_range_problems, gradient_clip_problems):
            source = inspect.getsource(gate)
            called = {
                node.func.id
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            assert "_clip_bound_error" in called, f"{gate.__name__} does not delegate: {called}"

    def test_infinity_is_the_one_carve_out(self) -> None:
        for value in NO_CLIPPING:
            assert clip_range_problems(_spec(clip_param=value), context="ppo") == [], f"{value!r}"
            assert positive_finite_number_error(value, "clip_param", "ppo") is not None, f"{value!r}"
