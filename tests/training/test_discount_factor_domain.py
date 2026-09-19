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
where this one is closed. Every test here reaches the real ``validate`` entry point, so it
covers the wiring as well as the domain.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import (
    _closed_unit_interval_error,
)
from strands_robots.training.rl import RLTrainSpec

# Every backend that discounts a return with the field.
RL_BACKENDS = ("ppo", "fast_sac", "fast_td3")

# Values outside the closed interval: the discounted return either diverges in
# the horizon (> 1) or stops accumulating future reward at all (< 0).
OUT_OF_INTERVAL: list[Any] = [1.5, 5.0, 1.0000001, -0.5, -1, 100.0]

# Values the shared numeric domain refuses before the interval is considered.
NOT_A_FINITE_NUMBER: list[Any] = [
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    False,
    "0.99",
    None,
    [0.99],
    {"gamma": 0.99},
]

UNUSABLE: list[Any] = [*OUT_OF_INTERVAL, *NOT_A_FINITE_NUMBER]

# Both endpoints are legitimate and standard: 1 is the undiscounted episodic
# return, 0 a myopic agent that optimizes the immediate reward only. Integral
# and NumPy spellings of an in-range value are usable too - the shared domain
# accepts any finite real, and float() reads them all.
USABLE: list[Any] = [0.0, 1.0, 0.99, 0.5, 0, 1, np.float64(0.97), np.float32(0.9)]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised.

    A non-None ``env_factory`` and an ``output_dir`` keep the unrelated preflight
    guards quiet; every other field keeps its default. Mirrors the isolation the
    PPO preflight contract test uses.
    """
    return RLTrainSpec(output_dir="/tmp/discount_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


def _gamma_problems(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``gamma``.

    Filtered on the shared domains' ``"{context}: {param} "`` message shape
    rather than on a bare ``"gamma"`` substring, so an unrelated problem can
    neither mask a missing refusal nor be mistaken for one.
    """
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(f"{provider}: gamma ")]


class TestEveryRLBackendRefusesAnUnusableDiscountFactor:
    """Both backends refuse every value the return cannot be discounted by."""

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: RLTrainSpec, provider: str, value: Any) -> None:
        spec.gamma = value
        assert _gamma_problems(provider, spec), f"{provider} accepted gamma={value!r}"

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("value", OUT_OF_INTERVAL)
    def test_an_out_of_interval_value_names_the_interval(self, spec: RLTrainSpec, provider: str, value: Any) -> None:
        """The message states the domain, so the fix needs no guesswork."""
        spec.gamma = value
        assert any("must be in [0, 1]" in p for p in _gamma_problems(provider, spec))

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("value", NOT_A_FINITE_NUMBER)
    def test_a_non_numeric_value_reads_like_every_other_numeric_field(
        self, spec: RLTrainSpec, provider: str, value: Any
    ) -> None:
        """Type / bool / finiteness refusals come from the shared domain."""
        spec.gamma = value
        assert any("must be a finite number" in p for p in _gamma_problems(provider, spec))

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    def test_the_problem_names_the_backend_that_refused_it(self, spec: RLTrainSpec, provider: str) -> None:
        spec.gamma = 1.5
        assert _gamma_problems(provider, spec)

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("value", NOT_A_FINITE_NUMBER)
    def test_an_unusable_value_is_a_problem_not_an_exception(
        self, spec: RLTrainSpec, provider: str, value: Any
    ) -> None:
        """``validate`` returns problems; it must not raise out of the check.

        A string or ``None`` would raise from a bare comparison against the
        interval bounds, which is the failure mode a read-only preflight exists
        to replace.
        """
        spec.gamma = value
        problems = create_trainer(provider).validate(spec)  # must not raise
        assert any(p.startswith(f"{provider}: gamma ") for p in problems), problems


class TestTheUsableDomainIsUntouched:
    """No value the return can be discounted by becomes a problem."""

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    @pytest.mark.parametrize("value", USABLE)
    def test_it_is_accepted(self, spec: RLTrainSpec, provider: str, value: Any) -> None:
        spec.gamma = value
        assert _gamma_problems(provider, spec) == [], f"{provider} refused gamma={value!r}"

    @pytest.mark.parametrize("provider", RL_BACKENDS)
    def test_the_default_spec_still_validates_clean(self, spec: RLTrainSpec, provider: str) -> None:
        """The shipped default (0.99) reports nothing at all."""
        assert create_trainer(provider).validate(spec) == []


class TestTheIntervalIsTheWholeLocalContribution:
    """The helper decides the interval and delegates everything else.

    Type, ``bool`` and finiteness are the shared numeric domain's job, so the two
    functions must agree on every value except the ones the interval alone
    excludes. Pinning that keeps the refusal text for a non-numeric ``gamma``
    identical to every other numeric field's.
    """

    @pytest.mark.parametrize("value", [*USABLE, *NOT_A_FINITE_NUMBER])
    def test_it_agrees_with_the_shared_domain(self, value: Any) -> None:
        from strands_robots.utils import finite_number_error

        local = _closed_unit_interval_error(value, "gamma", "ppo")
        shared = finite_number_error(value, "gamma", "ppo")
        assert (local is None) == (shared is None), f"diverged for gamma={value!r}"

    @pytest.mark.parametrize("value", OUT_OF_INTERVAL)
    def test_only_the_interval_diverges(self, value: Any) -> None:
        from strands_robots.utils import finite_number_error

        assert finite_number_error(value, "gamma", "ppo") is None
        assert _closed_unit_interval_error(value, "gamma", "ppo") is not None


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
