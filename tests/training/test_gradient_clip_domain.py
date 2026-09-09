"""The on-policy gradient-norm clip is checked against its domain.

``RLTrainSpec.max_grad_norm`` is the last thing that touches a gradient before
PPO steps: ``PpoTrainer.update`` ends every mini-batch with
``clip_grad_norm_(self.actor_critic.parameters(), spec.max_grad_norm)``. Nothing
judged the value, and ``clip_grad_norm_`` does not either - it multiplies every
gradient by ``max_norm / total_norm`` whenever that ratio is below one, and that
expression is defined for values no caller can have meant.

Measured on this backend before the gate existed, over a seeded 60-step run
whose checkpoint parameter sum starts at ``17.9251941755865118``:

* ``0`` and ``0.0`` reported ``status="success"`` and wrote a deployable
  checkpoint whose parameters were **bit-identical to a never-trained control** -
  the parameter delta was exactly ``0.0000000000``. Every gradient had been
  scaled to zero, so the optimizer stepped with no information.
* ``-1.0`` and ``-0.5`` also reported success, and moved the parameter sum to
  ``17.8211606460`` while the honored run moved it to ``17.9833114612`` - i.e.
  **away** from the objective. A negative bound negates the scaling ratio, so
  the same parameter whose gradient is ``[3.0, 4.0]`` comes out of
  ``clip_grad_norm_(-1.0)`` as ``[-0.6, -0.8]`` and the update becomes gradient
  *ascent* on the loss.
* ``True`` was a silent clip of one and ``"1.0"`` was silently accepted, because
  ``clip_grad_norm_`` coerces its bound through ``float()``.
* ``nan``, ``None`` and ``[1.0]`` raised from inside ``torch`` mid-update, after
  the environment, the networks and a full rollout had been built.

``inf`` is **inside** the domain: it is the field's only spelling of "do not
clip", and the consumer honors it by leaving every gradient untouched. That is
the reading which, had it belonged to zero, would have made this a contract
question rather than a defect - so the domain is a positive real, finite or
infinite, with no undecided endpoint.

The preflight is read-only, so it reports rather than raises for every value -
including a real no float64 stands for (``10**400``, ``Fraction(10**400, 3)``)
and a :class:`numbers.Real` registration with no working ``__float__``. Those
reach the shared domain, which answers each with a reason of its own.

Scoped to the on-policy backend: ``clip_grad_norm_`` appears in ``rl/ppo.py``
and nowhere else, so a backend that does not clip must not report on a field it
never reads. Every domain test here reaches the real ``validate`` entry point,
so it covers the wiring as well as the domain.
"""

from __future__ import annotations

import ast
import inspect
import math
import numbers
import pathlib
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import _clip_bound_error, gradient_clip_problems
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import positive_finite_number_error
from tests.training._spec_field_reads import reads_spec_field

# The one backend whose update clips a gradient.
ON_POLICY = "ppo"

# Backends that never clip - they must stay quiet about the field.
NO_CLIP_BACKENDS = ("fast_sac", "fast_td3", "mock")

# Zero scales every gradient to zero, so the run takes no informed step at all.
NO_GRADIENT_STEP: list[Any] = [0, 0.0]

# A negative bound negates the scaling ratio: the update inverts.
INVERTED_GRADIENT: list[Any] = [-1.0, -0.5, -100.0]

# Values the clip cannot mean, or reads as a different bound than the caller
# wrote. ``True`` is a bound of one; ``"1.0"`` is coerced through ``float()``.
NOT_A_CLIP: list[Any] = [
    True,
    False,
    float("nan"),
    float("-inf"),
    "1.0",
    "inf",
    None,
    [1.0],
    {"max_grad_norm": 1.0},
    np.bool_(True),
]


class _RealWithNoFloat:
    """A :class:`numbers.Real` registration whose conversion refuses.

    ``numbers.Real`` is a registration rather than an inheritance, so a value
    that satisfies the type test owes a guard no working ``__float__``. The
    carve-out converts, so this is the shape that reaches its wrapper: without
    one, the conversion's own exception escapes ``validate``.

    The exception is supplied per instance rather than fixed, because the
    handler is deliberately broader than any single type - a probe that raised
    only one would let an ``except`` clause narrowed to it keep passing. Each
    type used is one the data model prescribes for a conversion that cannot be
    performed, which is also what keeps these probes out of a merge gate: the
    ``py/unexpected-raise-in-special-method`` CodeQL rule reports a special
    method that always raises an exception unexpected for it, an alert opens a
    review thread, and thread resolution is required to merge - so a
    ``RuntimeError`` here would block the PR it is testing. Suppression is not
    the alternative: the filter set is deliberately exactly two rule ids, pinned
    by ``tests/test_codeql_query_filters.py``.
    """

    def __init__(self, error: Exception) -> None:
        self._error = error

    def __float__(self) -> float:
        raise self._error

    def __repr__(self) -> str:
        return f"<a real whose float raises {type(self._error).__name__}>"


numbers.Real.register(_RealWithNoFloat)

#: The exceptions the data model prescribes for a conversion that cannot be
#: performed: ``TypeError`` for an unsupported operand, and the two ``float()``
#: itself raises for a value it cannot represent.
PRESCRIBED_CONVERSION_ERRORS = (TypeError, ValueError, OverflowError)

# A registration the conversion cannot read at all. Neither reason the shared
# domain has for a *number* is true of it, so it earns the positivity message,
# the same as every other value that is not a usable positive real.
UNREADABLE_REAL: list[Any] = [
    _RealWithNoFloat(TypeError("this real cannot be read as a float")),
    _RealWithNoFloat(ValueError("this real has no float value")),
]

# A registration whose conversion *overflows* is beyond the float range by the
# shared helper's own definition - ``_beyond_float_range`` asks the conversion,
# not the type - so it earns the range reason and belongs with ``10**400``
# rather than with the values above. Keeping it in the right family is what
# makes the two refusal-reason tests below mean anything.
OVERFLOWING_REAL: list[Any] = [_RealWithNoFloat(OverflowError("this real is past the float range"))]

# Reals no float64 stands for. Positive and finite, so ``must be > 0`` would be a
# false statement about them - the shared domain answers them with a reason of its
# own, and the carve-out has to reach it rather than raising on the conversion.
BEYOND_FLOAT_RANGE: list[Any] = [10**400, -(10**400), Fraction(10**400, 3), *OVERFLOWING_REAL]

# Every unusable value whose refusal is the shared domain's ``must be > 0``.
PLAIN_REFUSAL: list[Any] = [*NO_GRADIENT_STEP, *INVERTED_GRADIENT, *NOT_A_CLIP, *UNREADABLE_REAL]

UNUSABLE: list[Any] = [*PLAIN_REFUSAL, *BEYOND_FLOAT_RANGE]

# The silent half: these reach a full run and report success.
SILENT: list[Any] = [*NO_GRADIENT_STEP, *INVERTED_GRADIENT, True, "1.0"]

# Any positive real. Unlike the count domains this one is continuous, so an
# integral float and a NumPy real are both usable - the consumer divides by the
# gradient norm rather than indexing with the value.
USABLE: list[Any] = [1.0, 0.5, 10.0, 1e-8, 3, np.float64(1.0), np.float32(0.5), np.int64(2)]

# The one value this gate accepts that the shared positive-finite domain does
# not: the consumer applies it coherently as "do not clip".
NO_CLIPPING: list[Any] = [math.inf, np.float64(np.inf)]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(output_dir="/tmp/gradient_clip_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


def _clip_problems(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about the field.

    Filtered on the shared domains' ``"{context}: {param} "`` message shape
    rather than on a bare substring, so an unrelated problem can neither mask a
    missing refusal nor be mistaken for one.
    """
    prefix = f"{provider}: max_grad_norm "
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(prefix)]


class TestTheOnPolicyBackendRefusesAnUnusableGradientClip:
    """PPO refuses every value ``clip_grad_norm_`` cannot honor."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: RLTrainSpec, value: Any) -> None:
        spec.max_grad_norm = value
        assert _clip_problems(ON_POLICY, spec), f"ppo accepted max_grad_norm={value!r}"

    @pytest.mark.parametrize("value", PLAIN_REFUSAL)
    def test_the_problem_names_the_field_the_domain_and_the_value(self, spec: RLTrainSpec, value: Any) -> None:
        spec.max_grad_norm = value
        (problem,) = _clip_problems(ON_POLICY, spec)
        assert "max_grad_norm" in problem
        assert "must be > 0" in problem
        assert repr(value) in problem, problem

    @pytest.mark.parametrize("value", BEYOND_FLOAT_RANGE)
    def test_a_value_past_the_float64_range_is_refused_with_its_own_reason(self, spec: RLTrainSpec, value: Any) -> None:
        """It is positive and finite, so ``must be > 0`` would be false of it."""
        spec.max_grad_norm = value
        (problem,) = _clip_problems(ON_POLICY, spec)
        assert "max_grad_norm" in problem
        assert "must be within the range of a 64-bit float" in problem
        assert "must be > 0" not in problem
        assert repr(value) in problem

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_problem_names_the_backend_that_refused_it(self, spec: RLTrainSpec, value: Any) -> None:
        spec.max_grad_norm = value
        (problem,) = _clip_problems(ON_POLICY, spec)
        assert problem.startswith(f"{ON_POLICY}: ")

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_preflight_reports_rather_than_raises(self, spec: RLTrainSpec, value: Any) -> None:
        """``validate`` is read-only, so even a value ``float()`` chokes on reports."""
        spec.max_grad_norm = value
        assert isinstance(create_trainer(ON_POLICY).validate(spec), list)

    @pytest.mark.parametrize("value", SILENT)
    def test_a_silently_honored_value_never_reaches_a_run(self, spec: RLTrainSpec, value: Any) -> None:
        """``train`` is fail-closed on ``validate``, so no rollout is collected."""
        spec.max_grad_norm = value
        result = create_trainer(ON_POLICY).train(spec)
        assert result.status == "error"
        assert "max_grad_norm" in result.message


class TestTheUsableDomainIsUntouched:
    """Every bound the clip can honor still passes."""

    @pytest.mark.parametrize("value", [*USABLE, *NO_CLIPPING])
    def test_a_positive_real_is_accepted(self, spec: RLTrainSpec, value: Any) -> None:
        spec.max_grad_norm = value
        assert _clip_problems(ON_POLICY, spec) == []

    def test_the_shipped_default_is_inside_the_domain(self, spec: RLTrainSpec) -> None:
        assert spec.max_grad_norm == 1.0
        assert _clip_problems(ON_POLICY, spec) == []

    def test_a_spec_without_the_field_is_silent(self) -> None:
        """A plain ``TrainSpec`` has no ``max_grad_norm``, so there is nothing to judge."""
        from strands_robots.training.base import TrainSpec

        plain = TrainSpec(output_dir="/tmp/gradient_clip_domain")
        assert gradient_clip_problems(plain, context="ppo") == []


class TestInfinityIsTheOnlyDifferenceFromTheSharedRule:
    """The local domain's whole contribution is accepting "do not clip"."""

    @pytest.mark.parametrize("value", [*USABLE, *UNUSABLE])
    def test_it_agrees_with_the_shared_positive_finite_rule(self, value: Any) -> None:
        mine = _clip_bound_error(value, "max_grad_norm", "ppo")
        shared = positive_finite_number_error(value, "max_grad_norm", "ppo")
        assert mine == shared, f"diverged on {value!r}: {mine!r} vs {shared!r}"

    @pytest.mark.parametrize("value", NO_CLIPPING)
    def test_infinity_is_the_carve_out(self, value: Any) -> None:
        assert _clip_bound_error(value, "max_grad_norm", "ppo") is None
        assert positive_finite_number_error(value, "max_grad_norm", "ppo") is not None

    def test_negative_infinity_is_not_carved_out(self) -> None:
        assert _clip_bound_error(float("-inf"), "max_grad_norm", "ppo") is not None

    @pytest.mark.parametrize("value", [*BEYOND_FLOAT_RANGE, *UNREADABLE_REAL])
    def test_the_carve_out_declines_rather_than_raising(self, value: Any) -> None:
        """A value the carve-out cannot read is delegated, never raised on."""
        assert isinstance(_clip_bound_error(value, "max_grad_norm", "ppo"), str)

    def test_the_unreadable_probes_span_the_prescribed_conversion_errors(self) -> None:
        """Non-vacuity for the wrapper, and a guard against the probes drifting.

        The wrapper is broad on purpose, so the probes have to be broader than
        one exception or an ``except`` clause narrowed to it would still pass.
        They also have to stay *prescribed*: a special method that always raises
        an exception the data model does not expect for it is a CodeQL alert, and
        an alert opens a review thread that blocks the merge.
        """
        raised = []
        for probe in [*UNREADABLE_REAL, *OVERFLOWING_REAL]:
            with pytest.raises(PRESCRIBED_CONVERSION_ERRORS) as excinfo:
                float(probe)
            raised.append(excinfo.type)
        assert set(raised) == set(PRESCRIBED_CONVERSION_ERRORS)

    def test_a_string_spelling_of_infinity_is_not_carved_out(self) -> None:
        """``float("inf")`` is the carve-out's value, so the type test carries it."""
        assert float("inf") == math.inf  # the premise the type test guards
        assert _clip_bound_error("inf", "max_grad_norm", "ppo") is not None


class TestTheBackendsThatDoNotClipStaySilent:
    """A backend that never reads the field must not report on it."""

    @pytest.mark.parametrize("provider", NO_CLIP_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_no_problem_is_reported(self, spec: RLTrainSpec, provider: str, value: Any) -> None:
        spec.max_grad_norm = value
        assert _clip_problems(provider, spec) == []

    @pytest.mark.parametrize("provider", NO_CLIP_BACKENDS)
    def test_the_silence_is_scoping_and_not_an_empty_preflight(self, spec: RLTrainSpec, provider: str) -> None:
        """Non-vacuity: these backends do refuse a field they *do* read."""
        spec.max_grad_norm = 0.0
        spec.learning_rate = -1.0
        problems = create_trainer(provider).validate(spec)
        assert any(p.startswith(f"{provider}: learning_rate ") for p in problems), problems
        assert not any(p.startswith(f"{provider}: max_grad_norm ") for p in problems), problems


def _ppo_update_source() -> str:
    """The source of the on-policy update, for premise checks."""
    from strands_robots.training.rl.ppo import PpoTrainer

    return inspect.getsource(PpoTrainer.update)


class TestTheConsumerHonorsTheDomain:
    """The premises the domain rests on, measured against ``torch`` itself."""

    def test_the_update_clips_with_the_field(self) -> None:
        source = _ppo_update_source()
        assert "clip_grad_norm_" in source
        assert "spec.max_grad_norm" in source

    def test_the_clip_is_the_last_thing_before_the_step(self) -> None:
        """So the bound decides the whole update, not a fraction of it."""
        source = _ppo_update_source()
        clip = source.index("clip_grad_norm_")
        step = source.index("self.optimizer.step()")
        backward = source.index("loss.backward()")
        assert backward < clip < step

    @pytest.mark.parametrize(
        ("bound", "expected"),
        [
            (1.0, [0.6, 0.8]),  # clipped: scaled by max_norm / total_norm
            (math.inf, [3.0, 4.0]),  # honored as "do not clip"
            (0.0, [0.0, 0.0]),  # every gradient scaled to zero
            (-1.0, [-0.6, -0.8]),  # the ratio is negated: the update inverts
        ],
    )
    def test_clip_grad_norm_applies_the_bound_verbatim(self, bound: float, expected: list[float]) -> None:
        torch = pytest.importorskip("torch")
        param = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        param.grad = torch.tensor([3.0, 4.0])  # norm 5
        torch.nn.utils.clip_grad_norm_([param], bound)
        assert param.grad is not None
        assert param.grad.tolist() == pytest.approx(expected, abs=1e-5)

    def test_a_bool_bound_is_a_silent_clip_of_one(self) -> None:
        """Which is why ``bool`` must be refused rather than read as one."""
        torch = pytest.importorskip("torch")
        params = []
        for bound in (True, 1.0):
            param = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
            param.grad = torch.tensor([3.0, 4.0])
            torch.nn.utils.clip_grad_norm_([param], bound)
            assert param.grad is not None
            params.append(param.grad.tolist())
        assert params[0] == params[1]

    def test_a_string_bound_is_silently_accepted(self) -> None:
        """So the consumer cannot be relied on to reject a non-numeric bound."""
        torch = pytest.importorskip("torch")
        param = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        param.grad = torch.tensor([3.0, 4.0])
        torch.nn.utils.clip_grad_norm_([param], "1.0")  # type: ignore[arg-type]
        assert param.grad is not None
        assert param.grad.tolist() == pytest.approx([0.6, 0.8], abs=1e-5)

    def test_a_non_finite_bound_poisons_every_gradient(self) -> None:
        torch = pytest.importorskip("torch")
        param = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        param.grad = torch.tensor([3.0, 4.0])
        torch.nn.utils.clip_grad_norm_([param], float("nan"))
        assert param.grad is not None
        assert not bool(torch.isfinite(param.grad).all())


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(gradient_clip_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_clip(source: str) -> bool:
    """Does *source* read ``spec.max_grad_norm``, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, ("max_grad_norm",))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_gradient_clip_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheGradientClipDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* ``spec.max_grad_norm`` must route it through the shared
    gate, so a second backend that starts clipping with the field fails this
    test until it does.
    """

    def test_every_module_that_reads_it_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name for p in _training_modules() if _reads_the_clip(p.read_text()) and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"modules read spec.max_grad_norm without the shared gate: {adrift}"

    def test_the_reader_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        readers = {p.name for p in _training_modules() if _reads_the_clip(p.read_text())}
        assert readers == {"ppo.py"}, readers

    def test_the_scanner_detects_a_planted_reader(self) -> None:
        """A module reading the field without the gate is really reported."""
        planted = "def validate(self, spec):\n    return [] if spec.max_grad_norm else []\n"
        assert _reads_the_clip(planted)
        assert not _calls_the_gate(planted)

    def test_no_backend_re_implements_the_bound(self) -> None:
        """A hand-rolled comparison agrees with the shared rule until it drifts."""
        offenders = [
            p.name
            for p in _training_modules()
            if "max_grad_norm <= 0" in p.read_text() or "max_grad_norm < 0" in p.read_text()
        ]
        assert offenders == [], f"modules compare max_grad_norm themselves: {offenders}"
