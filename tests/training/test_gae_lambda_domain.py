"""The GAE trace-decay coefficient is checked against its interval.

``RLTrainSpec.lam`` is the second factor of the advantage trace's decay. The
recursion in ``compute_gae`` carries it forward as ``last_adv = delta + gamma *
lam * (1 - done) * last_adv``, so the trace decays by the **product**
``gamma * lam``. The discount-factor gate bounds ``gamma`` to the closed interval
[0, 1]; nothing bounded ``lam``, so the divergence that gate exists to refuse
stayed reachable through the other factor: with a ``gamma`` of ``0.99`` well
inside its domain, a ``lam`` of ``1.5`` decays by ``1.485`` and the largest
advantage grows without bound in the rollout horizon, under a run that reports
success and writes a checkpoint.

``lam`` below the interval fails two different ways - far enough below and the
trace diverges again (the decay is ``|gamma * lam|``), merely below zero and it
collapses to the immediate reward - a non-finite value poisons every advantage
and surfaces only as a torch constraint error from the distribution sample, and
``True`` is a silent ``lam`` of one, a Monte-Carlo estimator rather than the
bootstrapped trace the caller asked for.

Scoped to the on-policy backend: FastSAC has no advantage trace and must not
report on a field it never reads. The refusal through the real ``validate`` entry point is one row of
``tests/training/test_rl_spec_field_domains.py``; this module keeps what the
field does to the run that consumes it.
"""

from __future__ import annotations

import pytest

from strands_robots.training.rl import RLTrainSpec


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(output_dir="/tmp/gae_lambda_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


class TestBoundingTheDiscountFactorAloneDoesNotBoundTheTrace:
    """The executable premise: the decay is a product, so one factor is not enough.

    Measured on this backend\'s own ``compute_gae`` over a rollout of unit
    rewards. Every ``gamma`` here is inside the discount-factor gate\'s accepted
    interval, so each divergence below is reachable on a spec that gate passes.
    """

    @staticmethod
    def _largest_advantage(gamma: float, lam: float, horizon: int) -> float:
        torch = pytest.importorskip("torch")

        from strands_robots.training.rl.ppo import compute_gae

        zeros = torch.zeros(horizon)
        advantages, _ = compute_gae(torch.ones(horizon), zeros, zeros, zeros, zeros, gamma, lam)
        return float(advantages.abs().max())

    def test_an_accepted_gamma_with_an_out_of_interval_lam_diverges(self) -> None:
        """gamma=0.99 passes its own gate; lam=1.5 makes the product 1.485."""
        honored = self._largest_advantage(0.99, 0.95, 24)
        assert self._largest_advantage(0.99, 1.5, 24) > 1000 * honored
        # Unbounded, not merely large: doubling the horizon compounds again.
        assert self._largest_advantage(0.99, 1.5, 48) > 1000 * self._largest_advantage(0.99, 1.5, 24)

    def test_a_large_lam_overflows_the_advantage_entirely(self) -> None:
        assert self._largest_advantage(0.99, 1e6, 12) == float("inf")

    def test_a_lam_far_below_zero_diverges_too(self) -> None:
        """The decay is the magnitude of the product, so a sign flip does not save it."""
        assert self._largest_advantage(0.99, -2.0, 48) > 1e6

    def test_a_lam_just_below_zero_stops_accumulating_future_advantage(self) -> None:
        """Alternating signs cancel the trace down to the immediate reward."""
        assert self._largest_advantage(0.99, -0.5, 24) < self._largest_advantage(0.99, 0.95, 24)

    def test_a_non_finite_lam_poisons_every_advantage(self) -> None:
        torch = pytest.importorskip("torch")

        from strands_robots.training.rl.ppo import compute_gae

        zeros = torch.zeros(8)
        advantages, _ = compute_gae(torch.ones(8), zeros, zeros, zeros, zeros, 0.99, float("nan"))
        assert not bool(torch.isfinite(advantages).all())

    def test_a_boolean_lam_is_a_different_estimator(self) -> None:
        """``True`` is a silent lam of one: Monte-Carlo, not a bootstrapped trace."""
        assert self._largest_advantage(0.99, True, 24) == self._largest_advantage(0.99, 1.0, 24)
        assert self._largest_advantage(0.99, True, 24) != self._largest_advantage(0.99, 0.95, 24)

    @pytest.mark.parametrize("endpoint", [0.0, 1.0])
    def test_both_endpoints_compute_a_finite_trace(self, endpoint: float) -> None:
        """Neither accepted endpoint is a degenerate spelling of "broken"."""
        assert 0.0 < self._largest_advantage(0.99, endpoint, 24) < float("inf")
