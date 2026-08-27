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
report on a field it never reads. Every test here reaches the real ``validate``
entry point, so it covers the wiring as well as the domain.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import (
    _closed_unit_interval_error,
    gae_lambda_problems,
)
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from tests.training._spec_field_reads import reads_spec_field

# The one backend that estimates an advantage trace.
ON_POLICY = "ppo"

# Backends that read ``gamma`` but never ``lam`` - they must stay quiet about it.
NO_TRACE_BACKENDS = ("fast_sac", "mock")

# Values outside the closed interval. Above 1 the trace diverges in the horizon;
# below 0 it either diverges again (|gamma * lam| > 1) or stops accumulating.
OUT_OF_INTERVAL: list[Any] = [1.5, 2.0, 1.0000001, 10.0, 1e6, -0.5, -1, -2.0]

# Values the shared numeric domain refuses before the interval is considered.
NOT_A_FINITE_NUMBER: list[Any] = [
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    False,
    "0.95",
    None,
    [0.95],
    {"lam": 0.95},
]

UNUSABLE: list[Any] = [*OUT_OF_INTERVAL, *NOT_A_FINITE_NUMBER]

# Both endpoints are legitimate and standard: 1 is the Monte-Carlo advantage (no
# bootstrapping) and 0 is TD(0), the one-step advantage. Integral and NumPy
# spellings of an in-range value are usable too - the shared domain accepts any
# finite real, and float() reads them all.
USABLE: list[Any] = [0.0, 1.0, 0.95, 0.5, 0, 1, np.float64(0.9), np.float32(0.8)]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(output_dir="/tmp/gae_lambda_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


def _lam_problems(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``lam``.

    Filtered on the shared domains\' ``"{context}: {param} "`` message shape
    rather than on a bare ``"lam"`` substring, so an unrelated problem can
    neither mask a missing refusal nor be mistaken for one.
    """
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(f"{provider}: lam ")]


class TestTheOnPolicyBackendRefusesAnUnusableTraceDecay:
    """PPO refuses every value the advantage trace cannot decay by."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: RLTrainSpec, value: Any) -> None:
        spec.lam = value
        assert _lam_problems(ON_POLICY, spec), f"ppo accepted lam={value!r}"

    @pytest.mark.parametrize("value", OUT_OF_INTERVAL)
    def test_an_out_of_interval_value_names_the_interval(self, spec: RLTrainSpec, value: Any) -> None:
        spec.lam = value
        assert "must be in [0, 1]" in _lam_problems(ON_POLICY, spec)[0]

    @pytest.mark.parametrize("value", NOT_A_FINITE_NUMBER)
    def test_a_non_numeric_value_reads_like_every_other_numeric_field(self, spec: RLTrainSpec, value: Any) -> None:
        spec.lam = value
        assert "must be a finite number" in _lam_problems(ON_POLICY, spec)[0]

    def test_the_problem_names_the_backend_that_refused_it(self, spec: RLTrainSpec) -> None:
        spec.lam = 1.5
        assert _lam_problems(ON_POLICY, spec)[0].startswith("ppo: lam ")

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_unusable_value_is_a_problem_not_an_exception(self, spec: RLTrainSpec, value: Any) -> None:
        """``validate`` is documented pure and read-only: it reports, never raises."""
        spec.lam = value
        assert isinstance(create_trainer(ON_POLICY).validate(spec), list)


class TestTheUsableDomainIsUntouched:
    """No value the trace can actually decay by becomes a problem."""

    @pytest.mark.parametrize("value", USABLE)
    def test_it_is_accepted(self, spec: RLTrainSpec, value: Any) -> None:
        spec.lam = value
        assert _lam_problems(ON_POLICY, spec) == []

    def test_the_default_spec_still_validates_clean(self, spec: RLTrainSpec) -> None:
        assert _lam_problems(ON_POLICY, spec) == []


class TestABackendWithNoAdvantageTraceStaysQuiet:
    """A backend that never reads the field must not report on it.

    :class:`~strands_robots.training.base.TrainSpec` documents that a backend
    reads the fields it supports and ignores the rest, so FastSAC - which
    bootstraps a target-Q rather than estimating a trace - refusing an unusable
    ``lam`` would be a false rejection. That is the whole reason this is a
    separate gate from the discount factor's, which every RL backend does read.
    """

    @pytest.mark.parametrize("provider", NO_TRACE_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_reports_nothing_about_lam(self, spec: RLTrainSpec, provider: str, value: Any) -> None:
        spec.lam = value
        assert _lam_problems(provider, spec) == []

    @pytest.mark.parametrize("value", [1.5, float("nan")])
    def test_but_the_shared_discount_factor_is_still_refused(self, spec: RLTrainSpec, value: Any) -> None:
        """Non-vacuity: the quiet backend is not simply ignoring the whole spec."""
        spec.gamma = value
        assert [p for p in create_trainer("fast_sac").validate(spec) if p.startswith("fast_sac: gamma ")]


class TestTheIntervalIsTheWholeLocalContribution:
    """The helper decides the interval and delegates everything else.

    Type, ``bool`` and finiteness are the shared numeric domain\'s job, so the two
    functions must agree on every value except the ones the interval alone
    excludes. Pinning that keeps the refusal text for a non-numeric ``lam``
    identical to every other numeric field\'s.
    """

    @pytest.mark.parametrize("value", [*USABLE, *NOT_A_FINITE_NUMBER])
    def test_it_agrees_with_the_shared_domain(self, value: Any) -> None:
        from strands_robots.utils import finite_number_error

        local = _closed_unit_interval_error(value, "lam", "ppo")
        shared = finite_number_error(value, "lam", "ppo")
        assert (local is None) == (shared is None), f"diverged for lam={value!r}"

    @pytest.mark.parametrize("value", OUT_OF_INTERVAL)
    def test_only_the_interval_diverges(self, value: Any) -> None:
        from strands_robots.utils import finite_number_error

        assert finite_number_error(value, "lam", "ppo") is None
        assert _closed_unit_interval_error(value, "lam", "ppo") is not None


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


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(gae_lambda_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_trace_decay(source: str) -> bool:
    """Does *source* read ``spec.lam``, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, ("lam",))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_gae_lambda_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheTraceDecayDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* ``spec.lam`` must route it through the shared gate, so a
    second on-policy backend that starts decaying a trace with the field fails
    this test until it does.
    """

    def test_every_module_that_reads_it_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name
            for p in _training_modules()
            if _reads_the_trace_decay(p.read_text()) and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"modules read spec.lam without the shared gate: {adrift}"

    def test_the_reader_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        readers = {p.name for p in _training_modules() if _reads_the_trace_decay(p.read_text())}
        assert readers == {"ppo.py"}, readers

    def test_the_scanner_detects_a_planted_reader(self) -> None:
        """A module reading the field without the gate is really reported."""
        planted = "def validate(self, spec):\n    return [] if spec.lam else []\n"
        assert _reads_the_trace_decay(planted)
        assert not _calls_the_gate(planted)

    def test_no_backend_re_implements_the_interval(self) -> None:
        """A local copy of the bounds would drift from the shared rule."""
        offenders = [
            p.name for p in _training_modules() if "spec.lam <" in p.read_text() or "spec.lam >" in p.read_text()
        ]
        assert offenders == [], f"modules compare spec.lam locally: {offenders}"
