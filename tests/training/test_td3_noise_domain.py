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
algorithm, not a boundary value of this one. Every test reaches the real
``validate`` entry point, so the wiring is covered as well as the domain.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import positive_finite_number_error
from tests.training._spec_field_reads import reads_spec_field

# The three noise scalars, with the defaults the shipped spec carries.
NOISE_FIELDS = ("exploration_noise_std", "target_noise_std", "target_noise_clip")

# The one backend built on a deterministic actor.
DETERMINISTIC_BACKEND = "fast_td3"

# Backends that never read the noise fields: SAC explores through its
# stochastic actor and smooths nothing, PPO likewise.
NO_NOISE_BACKENDS = ("ppo", "fast_sac", "mock")

# Values with no usable reading as a noise scale: zero and negatives are
# silent (the mechanism vanishes or nothing changes), ``True`` a silent 1.0,
# non-finite values poison the actions or the TD target, and the rest raise.
UNUSABLE: list[Any] = [
    0,
    0.0,
    -0.1,
    -1,
    True,
    False,
    float("nan"),
    float("inf"),
    float("-inf"),
    "0.1",
    None,
    [0.1],
    {},
]

# Scales the two expressions honor, including config spellings.
USABLE: list[Any] = [0.1, 0.2, 0.5, 1.0, 1e-3, np.float64(0.3), np.float32(0.5)]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(
        output_dir="/tmp/td3_noise_domain",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


def _noise_reports(provider: str, spec: RLTrainSpec, field: str) -> list[str]:
    """Problems the real ``validate`` entry point reports about *field*.

    Filtered on the shared domains' ``"{context}: {param} "`` message shape rather
    than on a bare substring, so an unrelated problem can neither mask a missing
    refusal nor be mistaken for one.
    """
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(f"{provider}: {field} ")]


class TestTheDeterministicBackendRefusesAnUnusableScale:
    """FastTD3 refuses every value neither noise expression can honor."""

    @pytest.mark.parametrize("field", NOISE_FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE, ids=repr)
    def test_it_is_reported_as_a_problem(self, spec: RLTrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        assert _noise_reports(DETERMINISTIC_BACKEND, spec, field), f"fast_td3 accepted {field}={value!r}"

    @pytest.mark.parametrize("field", NOISE_FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE, ids=repr)
    def test_the_problem_names_the_field_and_the_value(self, spec: RLTrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        (problem,) = _noise_reports(DETERMINISTIC_BACKEND, spec, field)
        assert problem.startswith(f"fast_td3: {field} must be > 0"), problem
        assert repr(value) in problem, problem

    def test_every_unusable_scale_is_reported_at_once(self, spec: RLTrainSpec) -> None:
        """Three bad scalars come back as three problems, not one per round."""
        for field in NOISE_FIELDS:
            setattr(spec, field, 0.0)
        problems = create_trainer(DETERMINISTIC_BACKEND).validate(spec)
        for field in NOISE_FIELDS:
            assert any(p.startswith(f"fast_td3: {field} ") for p in problems), (field, problems)


class TestTheUsableDomainIsUntouched:
    """A scale the expressions honor is not newly refused."""

    @pytest.mark.parametrize("field", NOISE_FIELDS)
    @pytest.mark.parametrize("value", USABLE, ids=repr)
    def test_a_usable_scale_reports_nothing(self, spec: RLTrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        assert _noise_reports(DETERMINISTIC_BACKEND, spec, field) == []

    def test_the_default_spec_reports_nothing(self, spec: RLTrainSpec) -> None:
        """The shipped defaults must not trip the new gate."""
        for field in NOISE_FIELDS:
            assert _noise_reports(DETERMINISTIC_BACKEND, spec, field) == []
        assert (spec.exploration_noise_std, spec.target_noise_std, spec.target_noise_clip) == (0.1, 0.2, 0.5)


class TestABackendWithNoNoiseStaysQuiet:
    """A backend that never reads the fields must not report on them."""

    @pytest.mark.parametrize("provider", NO_NOISE_BACKENDS)
    @pytest.mark.parametrize("field", NOISE_FIELDS)
    def test_it_reports_nothing_about_the_noise(self, provider: str, spec: RLTrainSpec, field: str) -> None:
        setattr(spec, field, 0.0)
        assert _noise_reports(provider, spec, field) == []

    @pytest.mark.parametrize("provider", NO_NOISE_BACKENDS)
    def test_silence_is_scoping_rather_than_an_empty_preflight(self, provider: str, spec: RLTrainSpec) -> None:
        """The same spec's *own* learning rate is still refused by those backends."""
        spec.learning_rate = 0.0
        problems = create_trainer(provider).validate(spec)
        assert any(p.startswith(f"{provider}: learning_rate ") for p in problems), problems


class TestTheGateAddsNothingToTheSharedDomain:
    """The verdict is the shared rule's, so the two cannot drift apart."""

    @pytest.mark.parametrize("field", NOISE_FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE + USABLE, ids=repr)
    def test_the_verdict_matches_the_shared_domain(self, spec: RLTrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        shared = positive_finite_number_error(value, field, DETERMINISTIC_BACKEND)
        assert _noise_reports(DETERMINISTIC_BACKEND, spec, field) == ([shared] if shared is not None else [])


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


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = (root / "_validate.py").resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_noise(source: str) -> bool:
    """Does *source* read a TD3 noise field, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, NOISE_FIELDS)


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_td3_noise_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheNoiseScaleDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* a noise field must route it through the shared gate, so
    a second deterministic-actor backend fails this test until it does.
    """

    def test_every_module_that_reads_them_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name
            for p in _training_modules()
            if _reads_the_noise(p.read_text()) and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"modules read the TD3 noise fields without the shared gate: {adrift}"

    def test_the_reader_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        readers = {p.name for p in _training_modules() if _reads_the_noise(p.read_text())}
        assert readers == {"fast_td3.py"}, readers

    def test_the_scanner_detects_a_planted_reader(self) -> None:
        """A module reading a field without the gate is really reported."""
        planted = "def collect(self):\n    return spec.exploration_noise_std\n"
        assert _reads_the_noise(planted)
        assert not _calls_the_gate(planted)

    def test_no_backend_re_implements_the_domain(self) -> None:
        """A local comparison would drift from the shared rule."""
        offenders = [
            p.name
            for p in _training_modules()
            if any(f"spec.{field} <" in p.read_text() or f"spec.{field} >" in p.read_text() for field in NOISE_FIELDS)
        ]
        assert offenders == [], f"modules compare a noise field locally: {offenders}"
