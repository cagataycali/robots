"""The discount factor is checked against its interval, not left to the return.

``RLTrainSpec.gamma`` weights every future reward in the return an RL algorithm
optimizes, and it is the one coefficient both RL backends read: PPO discounts
the GAE recursion with it (on the single-env and the vectorized rollout path),
FastSAC discounts its target-Q bootstrap. Neither preflight bounded it, and the
arithmetic that consumes it never judges it, so an unusable value was accepted:
``gamma > 1`` makes the discounted return diverge in the rollout horizon and the
run still reports success and writes a checkpoint, ``gamma < 0`` alternates the
sign of each future reward so the trace stops accumulating return, ``bool`` is a
silent ``gamma`` of one, and a non-finite value surfaces only once the update
samples the action distribution - as a torch constraint error naming neither the
field nor the run, after the env, the networks and a full rollout are built.

The off-policy preflights already bounded their own interval coefficient this way
(``tau`` must be in ``(0, 1]``); this pins the same shape for the field both
on-policy backends share. That precedent is a shared gate too now rather than a
bare comparison inside each backend - see
``tests/training/test_polyak_coefficient_domain.py``, whose interval is half-open
where this one is closed. The refusal through the real ``validate`` entry point is one row of
``tests/training/test_rl_spec_field_domains.py``; this module keeps what the
field does to the run that consumes it.
"""

from __future__ import annotations

import pytest

from strands_robots.training.rl import RLTrainSpec


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised.

    A non-None ``env_factory`` and an ``output_dir`` keep the unrelated preflight
    guards quiet; every other field keeps its default. Mirrors the isolation the
    PPO preflight contract test uses.
    """
    return RLTrainSpec(output_dir="/tmp/discount_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


class TestTheDivergenceTheDomainPrevents:
    """The refused values really do break the return, measured on the real GAE.

    This is the executable premise behind the domain: an out-of-interval
    ``gamma`` is not merely unconventional, it changes what the recursion
    computes. Without it the interval would be an unexplained constant.
    """

    @staticmethod
    def _largest_advantage(gamma: float, horizon: int) -> float:
        torch = pytest.importorskip("torch")

        from strands_robots.training.rl.ppo import compute_gae

        zeros = torch.zeros(horizon)
        advantages, _ = compute_gae(torch.ones(horizon), zeros, zeros, zeros, zeros, gamma, 0.95)
        return float(advantages.abs().max())

    def test_a_gamma_above_one_diverges_in_the_horizon(self) -> None:
        """The discounted return is a geometric series: above 1 it is unbounded."""
        honored = self._largest_advantage(0.99, 24)
        assert self._largest_advantage(1.5, 24) > 100 * honored
        # Unbounded, not merely large: doubling the horizon compounds again.
        assert self._largest_advantage(1.5, 48) > 100 * self._largest_advantage(1.5, 24)

    def test_a_negative_gamma_stops_accumulating_future_return(self) -> None:
        """Alternating signs cancel the trace down to the immediate reward."""
        assert self._largest_advantage(-0.5, 24) < self._largest_advantage(0.99, 24)

    def test_a_non_finite_gamma_poisons_every_advantage(self) -> None:
        torch = pytest.importorskip("torch")

        from strands_robots.training.rl.ppo import compute_gae

        zeros = torch.zeros(8)
        advantages, _ = compute_gae(torch.ones(8), zeros, zeros, zeros, zeros, float("nan"), 0.95)
        assert not bool(torch.isfinite(advantages).all())

    @pytest.mark.parametrize("endpoint", [0.0, 1.0])
    def test_both_endpoints_compute_a_finite_return(self, endpoint: float) -> None:
        """Neither accepted endpoint is a degenerate spelling of "broken"."""
        assert 0.0 < self._largest_advantage(endpoint, 24) < float("inf")
