"""FastTD3's three noise scales are checked where zero silently removes the mechanism.

A deterministic actor has no distribution to sample, so TD3 carries its noise
explicitly, in exactly two expressions::

    action = (action + torch.randn_like(action) * spec.exploration_noise_std).clamp(-1.0, 1.0)
    noise = (torch.randn_like(a) * spec.target_noise_std).clamp(-spec.target_noise_clip, spec.target_noise_clip)

The first is the only exploration the policy has once the random warmup ends;
the second is target policy smoothing, the mechanism that keeps the critic from
exploiting its own sharp errors. Neither expression judges its scalars:

* Zero silently removes the mechanism. ``randn * 0`` is exactly ``0``, so a
  zero exploration std collects the identical trajectory over and over, and a
  zero smoothing std (or clip - clamping every sample into ``[0, 0]``) trains
  plain clipped double-Q while reporting the smoothing knobs it was asked for.
* A negative scale is silently the identical run: Gaussian noise is symmetric,
  so ``randn * (-s)`` draws from the same distribution as ``randn * s``. A
  negative *clip* is worse - ``clamp`` with inverted bounds returns the
  constant upper bound, so every smoothing sample becomes the same negative
  offset and the target is biased rather than smoothed.
* A non-finite value poisons what the expression feeds. Measured on a CPU
  FastTD3 run: a ``nan`` smoothing std made the critic loss ``nan`` and left
  non-finite critic parameters behind while the run kept stepping, and an
  ``inf`` exploration std saturated the clamp so every action was one of the
  two bounds - bang-bang control, not a large noise. ``True`` is a silent
  scale of ``1.0``, ten times the shipped exploration default.

Only a positive finite number can be honored, so all three consult the shared
:func:`~strands_robots.utils.positive_finite_number_error` domain. There is no
"disable" spelling to carve out - a run that wants no smoothing is a different
algorithm, not a boundary value of this one. The refusal through the real ``validate`` entry point is one row of
``tests/training/test_rl_spec_field_domains.py``; this module keeps what the
field does to the run that consumes it.
"""

from __future__ import annotations

import pytest

from strands_robots.training.rl import RLTrainSpec


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(
        output_dir="/tmp/td3_noise_domain",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


class TestTheSilentReadingsAreReal:
    """The measured premises: zero, a negative, and inverted clamp bounds."""

    def test_zero_noise_is_exactly_no_noise(self) -> None:
        torch = pytest.importorskip("torch")
        noise = torch.randn(64, 4) * 0.0
        assert bool((noise == 0.0).all()), "premise: randn * 0 removes the mechanism outright"

    def test_a_negative_scale_draws_the_identical_distribution(self) -> None:
        """Symmetric noise: the sign of the scale changes nothing a run can see."""
        torch = pytest.importorskip("torch")
        torch.manual_seed(7)
        pos = torch.randn(64, 4) * 0.2
        torch.manual_seed(7)
        neg = torch.randn(64, 4) * -0.2
        assert torch.equal(neg, -pos), "premise: the two runs differ only by a reflection of a symmetric draw"

    def test_a_negative_clip_is_a_constant_bias(self) -> None:
        """Inverted clamp bounds return a constant, so the target is biased not smoothed."""
        torch = pytest.importorskip("torch")
        clipped = torch.randn(64).clamp(0.5, -0.5)
        assert bool((clipped == -0.5).all())
