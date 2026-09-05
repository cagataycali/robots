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
whose endpoint ``clip_grad_norm_`` itself settles.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import loss_weight_problems
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import finite_number_error
from tests.training._spec_field_reads import reads_spec_field

# The backend that composes the weighted objective.
ON_POLICY = "ppo"

# Backends that read neither weight. Per ``TrainSpec`` a backend ignores the
# fields it does not support, so reporting on one would be a false rejection.
NON_COMPOSING_BACKENDS = ("fast_sac", "fast_td3", "mock")

# The two weights, with the default each ships.
WEIGHTS: tuple[tuple[str, float], ...] = (("value_loss_coef", 1.0), ("entropy_coef", 0.0))

# A non-finite weight makes the loss non-finite, so the optimizer writes it into
# every parameter and the next rollout raises from inside torch.
NON_FINITE: list[Any] = [math.nan, math.inf, -math.inf, np.float64(np.nan), np.float64(np.inf)]

# A bool reads as a flag and lands as a coefficient of one.
BOOLEANS: list[Any] = [True, False, np.bool_(True)]

# Not a number at all: these raise from inside torch, naming nothing.
NON_NUMERIC: list[Any] = ["1.0", None, [1.0], {}]

# Finite in decimal but outside a 64-bit float, so the conversion itself fails.
BEYOND_FLOAT_RANGE: list[Any] = [10**400]

UNUSABLE: list[Any] = [*NON_FINITE, *BOOLEANS, *NON_NUMERIC, *BEYOND_FLOAT_RANGE]

# Any finite real. The floor is not this gate's question: zero disables the term
# and a negative weight reverses its sign, and both are configurations rather
# than mistakes.
USABLE: list[Any] = [1.0, 0.0, -0.5, 0.5, 2, 1e-8, np.float64(1.0), np.float32(0.5), np.int64(2)]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(output_dir="/tmp/loss_weight_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


def _weight_problems(provider: str, spec: RLTrainSpec, param: str) -> list[str]:
    """Problems the real ``validate`` entry point reports about *param*.

    Filtered on the shared domains' ``"{context}: {param} "`` message shape
    rather than on a bare substring, so an unrelated problem can neither mask a
    missing refusal nor be mistaken for one.
    """
    prefix = f"{provider}: {param} "
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(prefix)]


class TestTheOnPolicyBackendRefusesAnUnusableLossWeight:
    """PPO refuses every weight the composed objective cannot honor."""

    @pytest.mark.parametrize("param,default", WEIGHTS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: RLTrainSpec, param: str, default: float, value: Any) -> None:
        setattr(spec, param, value)
        assert _weight_problems(ON_POLICY, spec, param), f"ppo accepted {param}={value!r}"

    @pytest.mark.parametrize("param,default", WEIGHTS)
    @pytest.mark.parametrize("value", [*NON_FINITE, *BOOLEANS, *NON_NUMERIC])
    def test_the_problem_names_the_field_the_domain_and_the_value(
        self, spec: RLTrainSpec, param: str, default: float, value: Any
    ) -> None:
        setattr(spec, param, value)
        (problem,) = _weight_problems(ON_POLICY, spec, param)
        assert param in problem
        assert "must be a finite number" in problem
        assert repr(value) in problem, problem

    def test_each_weight_is_reported_independently(self, spec: RLTrainSpec) -> None:
        """Two unusable weights produce two problems, not one."""
        spec.value_loss_coef = math.nan
        spec.entropy_coef = math.inf
        problems = create_trainer(ON_POLICY).validate(spec)
        assert len([p for p in problems if p.startswith("ppo: value_loss_coef ")]) == 1
        assert len([p for p in problems if p.startswith("ppo: entropy_coef ")]) == 1

    @pytest.mark.parametrize("param,default", WEIGHTS)
    def test_the_refusal_is_read_only_and_train_fails_closed(
        self, tmp_path: pathlib.Path, param: str, default: float
    ) -> None:
        """``train`` reports the problem rather than raising from inside torch."""
        spec = RLTrainSpec(output_dir=str(tmp_path / "run"), env_factory=lambda: None)  # type: ignore[arg-type,return-value]
        setattr(spec, param, math.nan)
        result = create_trainer(ON_POLICY).train(spec)
        assert result.status == "error"
        assert param in result.message
        assert not list(tmp_path.rglob("*.pt")), "a refused spec must not write a checkpoint"


class TestTheUsableDomainIsUntouched:
    """Every finite real weight is still accepted, including the endpoints."""

    @pytest.mark.parametrize("param,default", WEIGHTS)
    @pytest.mark.parametrize("value", USABLE)
    def test_it_reports_no_problem(self, spec: RLTrainSpec, param: str, default: float, value: Any) -> None:
        setattr(spec, param, value)
        assert _weight_problems(ON_POLICY, spec, param) == []

    @pytest.mark.parametrize("param,default", WEIGHTS)
    def test_the_shipped_default_is_accepted(self, spec: RLTrainSpec, param: str, default: float) -> None:
        assert getattr(spec, param) == default
        assert _weight_problems(ON_POLICY, spec, param) == []


class TestTheFloorIsDeliberatelyNotDecided:
    """Zero and negative are configurations, so this gate must not refuse them.

    Recorded as its own contract because it is the difference between this gate
    and the sibling ``max_grad_norm`` one: there the endpoint is settled by
    ``clip_grad_norm_``, here both readings are live - ``entropy_coef=0.0`` is the
    shipped default, ``value_loss_coef=0`` stops training the critic, and a
    negative entropy weight penalizes entropy rather than rewarding it.
    """

    @pytest.mark.parametrize("param,default", WEIGHTS)
    @pytest.mark.parametrize("value", [0.0, 0, -0.5, -1.0, -1e-8])
    def test_zero_and_negative_stay_inside_the_domain(
        self, spec: RLTrainSpec, param: str, default: float, value: Any
    ) -> None:
        setattr(spec, param, value)
        assert _weight_problems(ON_POLICY, spec, param) == []


class TestTheGateIsExactlyTheSharedRule:
    """The gate adds nothing to the shared finite-number domain.

    Looped over both the usable and the unusable sets, so a local difference
    would have to show up as a disagreement rather than as absent coverage.
    """

    @pytest.mark.parametrize("param,default", WEIGHTS)
    @pytest.mark.parametrize("value", [*USABLE, *UNUSABLE])
    def test_it_agrees_with_the_shared_domain_everywhere(
        self, spec: RLTrainSpec, param: str, default: float, value: Any
    ) -> None:
        setattr(spec, param, value)
        gate_refuses = bool(loss_weight_problems(spec, context="ppo"))
        shared_refuses = finite_number_error(value, param, "ppo") is not None
        assert gate_refuses is shared_refuses, f"{param}={value!r}"

    def test_the_message_is_the_shared_one_verbatim(self, spec: RLTrainSpec) -> None:
        spec.value_loss_coef = math.nan
        (problem,) = [p for p in loss_weight_problems(spec, context="ppo") if "value_loss_coef" in p]
        assert problem == finite_number_error(math.nan, "value_loss_coef", "ppo")


class TestTheBackendsThatDoNotComposeItStaySilent:
    """A backend that reads neither weight must not report on either."""

    @pytest.mark.parametrize("provider", NON_COMPOSING_BACKENDS)
    @pytest.mark.parametrize("param,default", WEIGHTS)
    def test_no_problem_is_reported(self, provider: str, param: str, default: float, tmp_path: pathlib.Path) -> None:
        spec = RLTrainSpec(output_dir=str(tmp_path), env_factory=lambda: None)  # type: ignore[arg-type,return-value]
        setattr(spec, param, math.nan)
        assert _weight_problems(provider, spec, param) == []

    @pytest.mark.parametrize("provider", NON_COMPOSING_BACKENDS)
    def test_the_silence_is_scoping_rather_than_an_empty_preflight(self, provider: str, tmp_path: pathlib.Path) -> None:
        """Non-vacuity: the same spec is still refused on a field they do read."""
        spec = RLTrainSpec(output_dir=str(tmp_path), env_factory=lambda: None)  # type: ignore[arg-type,return-value]
        spec.learning_rate = math.nan
        assert _weight_problems(provider, spec, "learning_rate")


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

    @pytest.mark.parametrize("value", [0.0, -0.5, 1.0, 2.0])
    def test_every_accepted_weight_composes_a_finite_loss(self, value: float) -> None:
        torch = pytest.importorskip("torch")
        loss = torch.tensor(1.0) + value * torch.tensor(2.0) - value * torch.tensor(0.5)
        assert bool(torch.isfinite(loss))


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(loss_weight_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_a_weight(source: str) -> bool:
    """Does *source* read either ``spec.value_loss_coef`` or ``spec.entropy_coef``, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, ("value_loss_coef", "entropy_coef"))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_loss_weight_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheLossWeightDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* either weight must route it through the shared gate, so a
    second backend that starts composing a weighted objective with these fields
    fails this test until it does.
    """

    def test_every_module_that_reads_one_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name for p in _training_modules() if _reads_a_weight(p.read_text()) and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"modules read a loss weight without the shared gate: {adrift}"

    def test_the_reader_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        readers = {p.name for p in _training_modules() if _reads_a_weight(p.read_text())}
        assert readers == {"ppo.py"}, readers

    def test_the_scanner_detects_a_planted_reader(self) -> None:
        """A module reading a weight without the gate is really reported."""
        planted = "def validate(self, spec):\n    return [] if spec.entropy_coef else []\n"
        assert _reads_a_weight(planted)
        assert not _calls_the_gate(planted)

    def test_no_backend_re_implements_the_domain(self) -> None:
        """A hand-rolled comparison agrees with the shared rule until it drifts."""
        offenders = [
            p.name
            for p in _training_modules()
            if any(
                f"{field} {op}" in p.read_text()
                for field in ("value_loss_coef", "entropy_coef")
                for op in ("<= 0", "< 0", "> 0", ">= 0")
            )
        ]
        assert offenders == [], f"modules compare a loss weight themselves: {offenders}"
