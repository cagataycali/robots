"""FastSAC's entropy temperature is checked where it starts, not only where it moves.

FastSAC does not store the temperature: it stores the temperature's *logarithm*,
so ``init_alpha`` reaches ``torch.log`` on both temperature branches::

    self.log_alpha = torch.tensor(float(torch.log(torch.tensor(spec.init_alpha))), ...)

and ``alpha = log_alpha.exp()`` scales the entropy term in the critic's TD target
and in the actor loss from there on. The temperature's *learning rate* went
through the shared positive-finite gate; the value that rate moves reached
``torch.log`` unchecked - and that gate's own reasoning depends on this one, since
it refuses ``alpha_lr=0`` on the grounds that "the temperature stays at
``init_alpha`` for the whole run".

Measured on a 40-timestep run. ``init_alpha=0`` (and ``0.0``, and ``False``) makes
``log(0) == -inf``, so ``alpha`` is exactly ``0``: the entropy term is gone from
both losses, and the run is no longer the maximum-entropy algorithm that was
asked for. Automatic tuning cannot lift it back - ``log_alpha`` was still ``-inf``
after further gradient steps, because no finite update moves an infinity - and the
run reported ``status="success"`` while checkpointing ``log_alpha == -inf``, so the
unusable temperature outlives the run. ``True`` is a silent temperature of exactly
``1.0``. A negative value, ``nan`` or ``inf`` poisons the actor loss instead and
the first update raises ``ValueError`` from ``torch.distributions.Normal`` about a
tensor of ``nan`` policy means - naming that distribution's parameter rather than
the field, and only inside ``train``, after the env and both networks are built.

Scoped once, not twice: only the off-policy backend holds a temperature, but
within it ``init_alpha`` is read on *both* branches, so unlike the rate this check
must not be conditioned on ``autotune_alpha`` - with tuning off the field is the
temperature for the whole run and nothing can move it afterwards. Every test
reaches the real ``validate`` entry point, so the wiring is covered as well as the
domain.
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

# The one backend that holds an entropy temperature.
OFF_POLICY = "fast_sac"

# Backends that never read ``init_alpha``.
NO_TEMPERATURE_BACKENDS = ("ppo", "fast_td3", "mock")

# Values with no usable logarithm. ``0`` / ``0.0`` / ``False`` give ``-inf`` and a
# temperature of exactly zero that nothing can move; ``True`` is a silent ``1.0``;
# the rest give ``nan`` or ``inf`` and poison the actor loss.
UNUSABLE: list[Any] = [
    0,
    0.0,
    -1e-3,
    -1,
    False,
    True,
    float("nan"),
    float("inf"),
    float("-inf"),
    "1.0",
    None,
    [1.0],
    {},
]

# Temperatures with a finite logarithm, including the spellings a config or a
# probed value arrives as.
USABLE: list[Any] = [1.0, 0.2, 5.0, 1e-3, np.float64(1.0), np.float32(0.5)]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(  # type: ignore[return-value]
        output_dir="/tmp/initial_temperature_domain",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


def _init_alpha_problems(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``init_alpha``.

    Filtered on the shared domains' ``"{context}: {param} "`` message shape rather
    than on a bare ``"init_alpha"`` substring, so an unrelated problem can neither
    mask a missing refusal nor be mistaken for one.
    """
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(f"{provider}: init_alpha ")]


class TestTheOffPolicyBackendRefusesATemperatureWithNoLogarithm:
    """FastSAC refuses every value ``torch.log`` cannot turn into a temperature."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: RLTrainSpec, value: Any) -> None:
        spec.init_alpha = value
        assert _init_alpha_problems(OFF_POLICY, spec), f"fast_sac accepted init_alpha={value!r}"

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_problem_names_the_field_and_the_value(self, spec: RLTrainSpec, value: Any) -> None:
        spec.init_alpha = value
        (problem,) = _init_alpha_problems(OFF_POLICY, spec)
        assert problem.startswith("fast_sac: init_alpha must be > 0"), problem
        assert repr(value) in problem, problem

    def test_a_refusal_does_not_hide_the_rate_that_moves_it(self, spec: RLTrainSpec) -> None:
        """The temperature and its rate are reported at once, not one round at a time."""
        spec.init_alpha = 0.0
        spec.alpha_lr = 0.0
        problems = create_trainer(OFF_POLICY).validate(spec)
        assert any(p.startswith("fast_sac: init_alpha ") for p in problems), problems
        assert any(p.startswith("fast_sac: alpha_lr ") for p in problems), problems


class TestTheUsableDomainIsUntouched:
    """A temperature with a finite logarithm is not newly refused."""

    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_temperature_reports_nothing(self, spec: RLTrainSpec, value: Any) -> None:
        spec.init_alpha = value
        assert _init_alpha_problems(OFF_POLICY, spec) == []

    def test_the_default_spec_reports_nothing(self, spec: RLTrainSpec) -> None:
        """The shipped ``1.0`` default must not trip the new gate."""
        assert _init_alpha_problems(OFF_POLICY, spec) == []
        assert spec.init_alpha == 1.0


class TestBothTemperatureBranchesAreCovered:
    """``init_alpha`` is read whether or not the temperature is tuned.

    This is the one scoping difference from the rate: ``alpha_lr`` guards an
    optimizer only the ``autotune_alpha`` branch constructs, so that gate is
    conditioned on the flag. ``init_alpha`` is read on both branches, and with
    tuning off it is the temperature for the entire run, so conditioning this
    check on the flag would leave the more permanent case unguarded.
    """

    @pytest.mark.parametrize("autotune", [True, False])
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_on_either_branch(self, spec: RLTrainSpec, value: Any, autotune: bool) -> None:
        spec.autotune_alpha = autotune
        spec.init_alpha = value
        assert _init_alpha_problems(OFF_POLICY, spec), f"fast_sac accepted init_alpha={value!r} (autotune={autotune})"

    def test_both_branches_read_the_field(self) -> None:
        """Executable premise: the un-tuned branch is not a path that ignores it."""
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        setup = inspect.getsource(FastSacTrainer.setup)
        assert setup.count("spec.init_alpha") == 2, setup.count("spec.init_alpha")


class TestABackendWithNoTemperatureStaysQuiet:
    """A backend that never reads the field must not report on it."""

    @pytest.mark.parametrize("provider", NO_TEMPERATURE_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_reports_nothing_about_the_temperature(self, provider: str, spec: RLTrainSpec, value: Any) -> None:
        spec.init_alpha = value
        assert _init_alpha_problems(provider, spec) == []

    @pytest.mark.parametrize("provider", NO_TEMPERATURE_BACKENDS)
    def test_silence_is_scoping_rather_than_an_empty_preflight(self, provider: str, spec: RLTrainSpec) -> None:
        """The same spec's *own* learning rate is still refused by those backends."""
        spec.learning_rate = 0.0
        problems = create_trainer(provider).validate(spec)
        assert any(p.startswith(f"{provider}: learning_rate ") for p in problems), problems


class TestNoneIsAValueRatherThanASentinel:
    """``init_alpha`` is annotated ``float`` with a concrete default.

    Contrast ``learning_rate``, where ``None`` asks the backend for its own
    default. Here there is nothing to fall back to: ``torch.log(None)`` raises.
    """

    def test_none_is_refused(self, spec: RLTrainSpec) -> None:
        spec.init_alpha = None  # type: ignore[assignment]
        assert _init_alpha_problems(OFF_POLICY, spec)

    def test_the_field_declares_a_concrete_default(self) -> None:
        import dataclasses

        (field,) = [f for f in dataclasses.fields(RLTrainSpec) if f.name == "init_alpha"]
        assert field.default == 1.0
        assert "None" not in str(field.type)


class TestTheGateAddsNothingToTheSharedDomain:
    """The verdict is the shared rule's, so the two cannot drift apart."""

    @pytest.mark.parametrize("value", UNUSABLE + USABLE)
    def test_the_verdict_matches_the_shared_domain(self, spec: RLTrainSpec, value: Any) -> None:
        spec.init_alpha = value
        shared = positive_finite_number_error(value, "init_alpha", OFF_POLICY)
        assert _init_alpha_problems(OFF_POLICY, spec) == ([shared] if shared is not None else [])


class TestGuardingTheRateDoesNotGuardTheTemperature:
    """The measured consequence: a temperature of zero that nothing can move.

    This is why the starting value needed its own call rather than being covered
    by the rate. ``log(0)`` is ``-inf``, so ``alpha`` is exactly zero and the
    entropy term is absent from both losses; and because no finite gradient step
    moves an infinity, the automatic tuning the spec asked for cannot recover it.
    """

    def test_a_zero_temperature_is_zero_and_stays_zero(self) -> None:
        torch = pytest.importorskip("torch")

        log_alpha = torch.tensor(float(torch.log(torch.tensor(0.0))), requires_grad=True)
        assert float(log_alpha.detach()) == float("-inf")
        assert float(log_alpha.detach().exp()) == 0.0, "premise: the temperature starts at exactly zero"

        opt = torch.optim.Adam([log_alpha], lr=3e-4)
        for _ in range(5):
            opt.zero_grad()
            (-(log_alpha * torch.tensor(-2.0))).backward()
            opt.step()
        assert float(log_alpha.detach().exp()) == 0.0, (
            f"a usable temperature rate lifted alpha to {float(log_alpha.detach().exp())}, "
            "so the starting value would not need its own check"
        )

    def test_a_usable_temperature_does_move(self) -> None:
        """Control: the same loop moves a temperature that started with a logarithm."""
        torch = pytest.importorskip("torch")

        log_alpha = torch.tensor(float(torch.log(torch.tensor(1.0))), requires_grad=True)
        before = float(log_alpha.detach().exp())
        opt = torch.optim.Adam([log_alpha], lr=3e-4)
        opt.zero_grad()
        (-(log_alpha * torch.tensor(-2.0))).backward()
        opt.step()
        assert float(log_alpha.detach().exp()) != before

    def test_the_temperature_reaches_a_logarithm(self) -> None:
        """Executable premise: the field is not stored as the temperature itself."""
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        setup = inspect.getsource(FastSacTrainer.setup)
        assert "torch.log(torch.tensor(spec.init_alpha))" in setup


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = (root / "_validate.py").resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_temperature(source: str) -> bool:
    """Does *source* read ``spec.init_alpha``, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, ("init_alpha",))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_initial_temperature_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheInitialTemperatureDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* ``spec.init_alpha`` must route it through the shared gate,
    so a second off-policy backend that starts holding a temperature fails this
    test until it does.
    """

    def test_every_module_that_reads_it_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name
            for p in _training_modules()
            if _reads_the_temperature(p.read_text()) and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"modules read spec.init_alpha without the shared gate: {adrift}"

    def test_the_reader_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        readers = {p.name for p in _training_modules() if _reads_the_temperature(p.read_text())}
        assert readers == {"fast_sac.py"}, readers

    def test_the_scanner_detects_a_planted_reader(self) -> None:
        """A module reading the field without the gate is really reported."""
        planted = "def setup(self, spec):\n    return log(spec.init_alpha)\n"
        assert _reads_the_temperature(planted)
        assert not _calls_the_gate(planted)

    def test_no_backend_re_implements_the_domain(self) -> None:
        """A local comparison would drift from the shared rule."""
        offenders = [
            p.name
            for p in _training_modules()
            if "spec.init_alpha <" in p.read_text() or "spec.init_alpha >" in p.read_text()
        ]
        assert offenders == [], f"modules compare spec.init_alpha locally: {offenders}"
