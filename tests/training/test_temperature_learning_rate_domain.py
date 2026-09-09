"""FastSAC's second learning rate is checked against the same domain as its first.

FastSAC builds two optimizers from two separate learning-rate fields::

    self.actor_optimizer = torch.optim.Adam(actor_params, lr=spec.learning_rate)
    self.critic_optimizer = torch.optim.Adam(critic_params, lr=spec.learning_rate)
    ...
    self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=spec.alpha_lr)

``learning_rate`` went through the shared optimization gate; ``alpha_lr`` - which
``RLTrainSpec`` documents as the "Learning rate for the temperature optimizer
(SAC)" - reached the same constructor unchecked, so every failure mode that gate
documents stayed reachable through the second field. Measured on a 40-timestep
run starting from ``init_alpha=1.0``: ``alpha_lr=0`` builds the optimizer and
moves the temperature by nothing, so ``autotune_alpha=True`` silently behaves
like ``autotune_alpha=False``; ``inf`` sends it to an infinity on the first step
and, because the temperature multiplies the log-probability in the *actor* loss,
the run finished ``status="success"`` with a checkpoint whose largest parameter
magnitude was ``inf``; ``True`` was a silent rate of ``1.0``. A negative value
and ``nan`` are refused by ``torch.optim.Adam``, and a ``str`` / ``None`` /
``list`` raises a bare ``TypeError``, but all four only in ``setup`` - after the
env and both networks are built.

Scoped twice over: only the off-policy backend tunes a temperature, and within it
only the ``autotune_alpha`` branch builds the optimizer, so both the other
backends and a spec that tunes nothing must stay quiet. Every test reaches the
real ``validate`` entry point, so the wiring is covered as well as the domain.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import temperature_learning_rate_problems
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import positive_finite_number_error
from tests.training._spec_field_reads import reads_spec_field

# The one backend that tunes an entropy temperature.
OFF_POLICY = "fast_sac"

# Backends that read ``learning_rate`` but never ``alpha_lr``.
NO_TEMPERATURE_BACKENDS = ("ppo", "fast_td3", "mock")

# Values the temperature optimizer cannot be driven by. ``0`` and ``False`` build
# it and never move it; ``inf`` poisons it and the actor with it; ``True`` is a
# silent rate of 1.0; the rest raise, but only once ``setup`` runs.
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
    "3e-4",
    None,
    [3e-4],
    {},
]

# Rates a temperature optimizer can actually be driven by, including the spellings
# a config or a probed value arrives as.
USABLE: list[Any] = [3e-4, 1e-5, 1.0, 0.5, np.float64(3e-4), np.float32(1e-4)]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(  # type: ignore[return-value]
        output_dir="/tmp/temperature_learning_rate_domain",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


def _alpha_lr_problems(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``alpha_lr``.

    Filtered on the shared domains' ``"{context}: {param} "`` message shape rather
    than on a bare ``"alpha_lr"`` substring, so an unrelated problem can neither
    mask a missing refusal nor be mistaken for one.
    """
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(f"{provider}: alpha_lr ")]


class TestTheOffPolicyBackendRefusesAnUnusableTemperatureRate:
    """FastSAC refuses every value its temperature optimizer cannot be driven by."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: RLTrainSpec, value: Any) -> None:
        spec.alpha_lr = value
        assert _alpha_lr_problems(OFF_POLICY, spec), f"fast_sac accepted alpha_lr={value!r}"

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_problem_names_the_field_and_the_value(self, spec: RLTrainSpec, value: Any) -> None:
        spec.alpha_lr = value
        (problem,) = _alpha_lr_problems(OFF_POLICY, spec)
        assert problem.startswith("fast_sac: alpha_lr must be > 0"), problem
        assert repr(value) in problem, problem

    def test_a_refusal_does_not_hide_the_first_learning_rate(self, spec: RLTrainSpec) -> None:
        """Both optimizers' rates are reported at once, not one round at a time."""
        spec.alpha_lr = 0.0
        spec.learning_rate = 0.0
        problems = create_trainer(OFF_POLICY).validate(spec)
        assert any(p.startswith("fast_sac: alpha_lr ") for p in problems), problems
        assert any(p.startswith("fast_sac: learning_rate ") for p in problems), problems


class TestTheUsableDomainIsUntouched:
    """A rate the optimizer can be driven by is not newly refused."""

    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_rate_reports_nothing(self, spec: RLTrainSpec, value: Any) -> None:
        spec.alpha_lr = value
        assert _alpha_lr_problems(OFF_POLICY, spec) == []

    def test_the_default_spec_reports_nothing(self, spec: RLTrainSpec) -> None:
        """The shipped ``3e-4`` default must not trip the new gate."""
        assert _alpha_lr_problems(OFF_POLICY, spec) == []


class TestTheCheckAppliesOnlyWhenATemperatureIsTuned:
    """``autotune_alpha`` selects whether the field is read at all.

    ``setup`` builds a temperature optimizer only on that branch, so refusing the
    field when it is unset would reject a value the run never reads. Both
    directions are pinned, or "drop the condition" and "drop the check" would each
    pass.
    """

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_a_spec_that_tunes_nothing_is_not_reported_on(self, spec: RLTrainSpec, value: Any) -> None:
        spec.autotune_alpha = False
        spec.alpha_lr = value
        assert _alpha_lr_problems(OFF_POLICY, spec) == []

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_a_spec_that_tunes_a_temperature_is_reported_on(self, spec: RLTrainSpec, value: Any) -> None:
        spec.autotune_alpha = True
        spec.alpha_lr = value
        assert _alpha_lr_problems(OFF_POLICY, spec), f"fast_sac accepted alpha_lr={value!r} while tuning"

    def test_tuning_is_on_by_default(self, spec: RLTrainSpec) -> None:
        """Non-vacuity: the guarded branch is the one the default spec takes."""
        assert spec.autotune_alpha is True


class TestABackendWithNoTemperatureStaysQuiet:
    """A backend that never reads the field must not report on it."""

    @pytest.mark.parametrize("provider", NO_TEMPERATURE_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_reports_nothing_about_the_temperature_rate(self, provider: str, spec: RLTrainSpec, value: Any) -> None:
        spec.alpha_lr = value
        assert _alpha_lr_problems(provider, spec) == []

    @pytest.mark.parametrize("provider", NO_TEMPERATURE_BACKENDS)
    def test_silence_is_scoping_rather_than_an_empty_preflight(self, provider: str, spec: RLTrainSpec) -> None:
        """The same spec's *own* learning rate is still refused by those backends."""
        spec.learning_rate = 0.0
        problems = create_trainer(provider).validate(spec)
        assert any(p.startswith(f"{provider}: learning_rate ") for p in problems), problems


class TestNoneIsAValueRatherThanASentinel:
    """``alpha_lr`` has a concrete default, so ``None`` is not a request for one.

    ``TrainSpec.learning_rate`` is annotated ``float | None`` and documents ``None``
    as "use the backend's own default", so its gate exempts it. ``alpha_lr`` is
    annotated ``float`` with no such sentinel anywhere in the hierarchy, so
    ``None`` is a value the temperature optimizer cannot take rather than a request
    for a default. The asymmetry is deliberate and pinned here so it is not
    smoothed away in either direction.
    """

    def test_a_none_temperature_rate_is_refused(self, spec: RLTrainSpec) -> None:
        spec.alpha_lr = None  # type: ignore[assignment]
        assert _alpha_lr_problems(OFF_POLICY, spec)

    def test_a_none_learning_rate_is_still_accepted(self, spec: RLTrainSpec) -> None:
        # ``RLTrainSpec`` narrows the annotation to ``float``, but the gate still
        # honours the sentinel the base contract documents - which is the whole
        # asymmetry under test, so the assignment is deliberate.
        spec.learning_rate = None  # type: ignore[assignment]
        problems = [p for p in create_trainer(OFF_POLICY).validate(spec) if "learning_rate" in p]
        assert problems == []

    def test_only_the_first_rate_declares_the_sentinel(self) -> None:
        """Executable premise for the asymmetry above: the annotations differ.

        The basis is the declared type rather than the default - ``RLTrainSpec``
        narrows ``learning_rate`` to a concrete ``1e-4``, so comparing defaults
        would find no asymmetry at all while the gates still, correctly, differ.
        """
        import dataclasses

        from strands_robots.training.base import TrainSpec

        base = {f.name: f for f in dataclasses.fields(TrainSpec)}
        rl = {f.name: f for f in dataclasses.fields(RLTrainSpec)}
        assert base["learning_rate"].type == "float | None"
        assert rl["alpha_lr"].type == "float"
        assert rl["alpha_lr"].default == pytest.approx(3e-4)


class TestTheGateAddsNothingToTheSharedDomain:
    """The gate contributes the scoping, not a second numeric rule.

    Every verdict is the shared ``positive_finite_number_error`` domain's, so the
    two cannot drift on what counts as a usable rate.
    """

    @pytest.mark.parametrize("value", UNUSABLE + USABLE)
    def test_the_verdict_is_the_shared_domains(self, spec: RLTrainSpec, value: Any) -> None:
        spec.alpha_lr = value
        shared = positive_finite_number_error(value, "alpha_lr", OFF_POLICY)
        assert bool(_alpha_lr_problems(OFF_POLICY, spec)) is (shared is not None)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_message_is_the_shared_domains_verbatim(self, spec: RLTrainSpec, value: Any) -> None:
        spec.alpha_lr = value
        (problem,) = _alpha_lr_problems(OFF_POLICY, spec)
        assert problem == positive_finite_number_error(value, "alpha_lr", OFF_POLICY)


class TestGuardingTheFirstRateDoesNotGuardTheSecond:
    """The measured consequence: a run whose temperature never adapts.

    This is why the second field needed its own call rather than being covered by
    the first. ``alpha_lr=0`` leaves ``log_alpha`` where it started, so the
    temperature the spec asked to have tuned stays at ``init_alpha`` - while a
    usable rate moves it - and the pre-fix ``validate`` reported neither.
    """

    def test_a_zero_temperature_rate_freezes_the_temperature(self) -> None:
        torch = pytest.importorskip("torch")
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        def _alpha_after_one_step(alpha_lr: float) -> tuple[float, float]:
            log_alpha = torch.tensor(0.0, requires_grad=True)  # init_alpha = 1.0
            opt = torch.optim.Adam([log_alpha], lr=alpha_lr)
            before = float(log_alpha.detach().exp())
            (-(log_alpha * torch.tensor(-2.0))).backward()
            opt.step()
            return before, float(log_alpha.detach().exp())

        frozen_before, frozen_after = _alpha_after_one_step(0.0)
        tuned_before, tuned_after = _alpha_after_one_step(3e-4)
        assert frozen_before == frozen_after == 1.0, (frozen_before, frozen_after)
        assert tuned_after != tuned_before, (tuned_before, tuned_after)
        # And the field that *was* guarded cannot express this: it drives a
        # different optimizer entirely.
        assert "alpha_lr" in inspect.getsource(FastSacTrainer.setup)

    def test_the_two_rates_drive_different_optimizers(self) -> None:
        """Executable premise: the guarded field cannot stand in for this one."""
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        source = inspect.getsource(FastSacTrainer.setup)
        assert "lr=spec.learning_rate" in source
        assert "lr=spec.alpha_lr" in source


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(temperature_learning_rate_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_temperature_rate(source: str) -> bool:
    """Does *source* read ``spec.alpha_lr``, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, ("alpha_lr",))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_temperature_learning_rate_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheTemperatureRateDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* ``spec.alpha_lr`` must route it through the shared gate, so
    a second off-policy backend that starts driving a temperature optimizer with
    the field fails this test until it does.
    """

    def test_every_module_that_reads_it_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name
            for p in _training_modules()
            if _reads_the_temperature_rate(p.read_text()) and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"modules read spec.alpha_lr without the shared gate: {adrift}"

    def test_the_reader_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        readers = {p.name for p in _training_modules() if _reads_the_temperature_rate(p.read_text())}
        assert readers == {"fast_sac.py"}, readers

    def test_the_scanner_detects_a_planted_reader(self) -> None:
        """A module reading the field without the gate is really reported."""
        planted = "def setup(self, spec):\n    return Adam([x], lr=spec.alpha_lr)\n"
        assert _reads_the_temperature_rate(planted)
        assert not _calls_the_gate(planted)

    def test_no_backend_re_implements_the_domain(self) -> None:
        """A local comparison would drift from the shared rule."""
        offenders = [
            p.name
            for p in _training_modules()
            if "spec.alpha_lr <" in p.read_text() or "spec.alpha_lr >" in p.read_text()
        ]
        assert offenders == [], f"modules compare spec.alpha_lr locally: {offenders}"
