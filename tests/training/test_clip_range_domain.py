"""The on-policy trust-region half-width is checked against its domain.

``RLTrainSpec.clip_param`` is the half-width of the trust region PPO is named
for. ``PpoTrainer.update`` reads it twice per mini-batch, in the two expressions
that clip::

    surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - spec.clip_param, 1.0 + spec.clip_param)
    value_clipped = old_values + (value - old_values).clamp(-spec.clip_param, spec.clip_param)

Nothing judged the value, and ``torch.clamp`` cannot: it is defined for every
value below, and each one produced a *finite, successful, deployable* run whose
objective was not the one the caller configured. Measured on this backend before
the gate existed, over a seeded 60-step run against a never-trained control whose
checkpoint parameter sum is ``139.8929914773252676``:

* ``nan`` **silently removed the trust region.** Both clipped terms become
  ``nan``, so ``torch.max(surrogate, surrogate_clipped)`` returns ``nan`` - but
  its gradient flows to the *unclipped* branch, because every comparison against
  ``nan`` is false. The run therefore descended the unclipped objective, and its
  checkpoint parameter sum was ``140.1735330768706262`` - bit-identical to the
  ``inf`` run - while ``surrogate_loss``, ``value_loss`` and ``latest_loss`` were
  all reported as ``nan``. PPO's defining mechanism was off, and the only signal
  that anything had happened was a metric a caller cannot act on.
* ``-0.2`` **inverted the clamp bounds.** ``1 - c`` exceeds ``1 + c``, so the
  clamp returns a constant regardless of the ratio, and the reported surrogate
  loss changed sign - ``+0.081992`` against the honored run's ``-0.008662``. The
  checkpoint sum moved to ``140.1913412402318500``.
* ``0`` is the same failure at the boundary. The value clip becomes
  ``clamp(-0, 0)``, so ``value_clipped`` is exactly ``old_values`` and the
  critic's clipped branch is a constant; the checkpoint sum moved to
  ``140.2282075245283863``.
* ``-inf`` trained on ``inf`` losses and still reported success
  (``140.0613218537641842``), and ``True`` was a silent half-width of one - five
  times the shipped ``0.2`` - written by a value that reads as a flag.
* ``"0.2"``, ``None`` and ``[0.2]`` raised ``TypeError`` from ``rl/ppo.py``
  mid-update, after the environment, the networks and a full rollout had been
  built.

The honored default's sum is ``140.1741519418580992``, so every row above is a
run that trained *differently* rather than harmlessly.

``inf`` is **inside** the domain: it is the field's only spelling of "do not
clip", and the consumer honors it - ``clamp(ratio, -inf, inf)`` returns the ratio
unchanged, which is a coherent unclipped run with finite losses. That is the same
endpoint, settled the same way, as the sibling clip bound ``max_grad_norm``, so
the two share one domain helper rather than carrying a copy each.

Scoped to the on-policy backend: ``spec.clip_param`` appears in ``rl/ppo.py`` and
nowhere else, so a backend that does not clip a policy ratio must not report on a
field it never reads. Every domain test here reaches the real ``validate`` entry
point, so it covers the wiring as well as the domain.
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
from strands_robots.training._validate import (
    _clip_bound_error,
    clip_range_problems,
    gradient_clip_problems,
)
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import positive_finite_number_error
from tests.training._spec_field_reads import reads_spec_field

# The one backend whose update clips a policy ratio.
ON_POLICY = "ppo"

# Backends that never clip a ratio - they must stay quiet about the field.
NO_CLIP_BACKENDS = ("fast_sac", "fast_td3", "mock")

# Half-widths the clip can honor.
USABLE: list[Any] = [0.2, 0.1, 0.5, 1.0, 3, np.float64(0.2), np.int64(2)]

# The only spelling of "do not clip" the field has.
NO_CLIPPING: list[Any] = [math.inf, float("inf"), np.float64(math.inf)]

# A window of zero width is not a window: the value clip pins ``value_clipped``
# to ``old_values`` exactly, so the critic's clipped branch is a constant.
NO_WINDOW: list[Any] = [0, 0.0]

# A negative half-width inverts the clamp bounds, which then return a constant
# regardless of the ratio.
INVERTED_WINDOW: list[Any] = [-0.2, -1.0, -100.0]

# Values the half-width cannot mean, or that read as a different width than the
# caller wrote. ``True`` is a half-width of one.
NOT_A_WIDTH: list[Any] = [
    True,
    False,
    float("nan"),
    float("-inf"),
    "0.2",
    "inf",
    None,
    [0.2],
    {"clip_param": 0.2},
    np.bool_(True),
]

UNUSABLE: list[Any] = [*NO_WINDOW, *INVERTED_WINDOW, *NOT_A_WIDTH]


def _spec(**overrides: Any) -> RLTrainSpec:
    """An otherwise-valid RL spec carrying *overrides*.

    Every value these tests set is deliberately outside its field's annotation -
    that is the property under test - so the overrides are applied through
    ``setattr``, and the one suppression the ``env_factory`` placeholder needs
    lives here rather than at each call site.
    """
    spec = RLTrainSpec(output_dir="/tmp/clip_range_domain", env_factory=lambda: None)  # type: ignore[arg-type,return-value]
    for field, value in overrides.items():
        setattr(spec, field, value)
    return spec


@pytest.fixture
def spec() -> RLTrainSpec:
    """A spec the on-policy preflight otherwise accepts."""
    return _spec()


def _clip_problems(provider: str, spec: RLTrainSpec) -> list[str]:
    """Every problem the backend's real ``validate`` reports about the width."""
    return [p for p in create_trainer(provider).validate(spec) if ": clip_param " in p]


class TestTheTrustRegionRefusesAWidthItCannotHonor:
    """A half-width no reading makes usable is reported, not clipped with."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_unusable_width_is_refused(self, spec: RLTrainSpec, value: Any) -> None:
        spec.clip_param = value
        assert _clip_problems(ON_POLICY, spec) != [], f"{value!r} was accepted"

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_problem_names_the_field_the_domain_and_the_value(self, spec: RLTrainSpec, value: Any) -> None:
        spec.clip_param = value
        (problem,) = _clip_problems(ON_POLICY, spec)
        assert problem.startswith("ppo: clip_param "), problem
        assert "must be > 0" in problem, problem

    def test_the_message_is_the_shared_domain_verbatim(self, spec: RLTrainSpec) -> None:
        """The gate adds no wording of its own; it delegates."""
        spec.clip_param = -0.2
        (problem,) = _clip_problems(ON_POLICY, spec)
        assert problem == _clip_bound_error(-0.2, "clip_param", "ppo")

    def test_the_preflight_reports_rather_than_raises(self) -> None:
        """``validate`` is read-only, so an unusable width never escapes it."""
        assert isinstance(create_trainer(ON_POLICY).validate(_spec(clip_param=None)), list)

    def test_two_unusable_optimization_fields_report_two_problems(self, spec: RLTrainSpec) -> None:
        """The clip-range gate is additional to the sibling bound, not a merge."""
        spec.clip_param = float("nan")
        spec.max_grad_norm = float("nan")
        problems = create_trainer(ON_POLICY).validate(spec)
        assert sum(1 for p in problems if ": clip_param " in p) == 1
        assert sum(1 for p in problems if ": max_grad_norm " in p) == 1


class TestTheUsableDomainIsUntouched:
    """Every half-width the clip can honor still passes."""

    @pytest.mark.parametrize("value", [*USABLE, *NO_CLIPPING])
    def test_a_positive_real_is_accepted(self, spec: RLTrainSpec, value: Any) -> None:
        spec.clip_param = value
        assert _clip_problems(ON_POLICY, spec) == []

    def test_the_shipped_default_is_inside_the_domain(self, spec: RLTrainSpec) -> None:
        assert spec.clip_param == 0.2
        assert _clip_problems(ON_POLICY, spec) == []

    def test_a_spec_without_the_field_is_silent(self) -> None:
        """A plain ``TrainSpec`` has no ``clip_param``, so there is nothing to judge."""
        from strands_robots.training.base import TrainSpec

        plain = TrainSpec(output_dir="/tmp/clip_range_domain")
        assert clip_range_problems(plain, context="ppo") == []


class TestTheTwoClipBoundsShareOneDomain:
    """One rule, one home - the two on-policy clip bounds cannot drift apart."""

    @pytest.mark.parametrize("value", [*USABLE, *NO_CLIPPING, *UNUSABLE])
    def test_both_bounds_give_the_same_verdict(self, value: Any) -> None:
        width = _spec(clip_param=value)
        norm = _spec(max_grad_norm=value)
        assert bool(clip_range_problems(width, context="ppo")) == bool(gradient_clip_problems(norm, context="ppo")), (
            f"the two clip bounds disagree on {value!r}"
        )

    def test_both_gates_read_the_one_shared_helper(self) -> None:
        """Structural: neither gate carries its own copy of the rule."""
        for gate in (clip_range_problems, gradient_clip_problems):
            source = inspect.getsource(gate)
            called = {
                node.func.id
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            assert "_clip_bound_error" in called, f"{gate.__name__} does not delegate: {called}"

    @pytest.mark.parametrize("value", [*USABLE, *UNUSABLE])
    def test_the_domain_is_the_shared_positive_finite_rule(self, value: Any) -> None:
        mine = clip_range_problems(_spec(clip_param=value), context="ppo")
        shared = positive_finite_number_error(value, "clip_param", "ppo")
        assert mine == ([shared] if shared is not None else []), f"diverged on {value!r}"

    @pytest.mark.parametrize("value", NO_CLIPPING)
    def test_infinity_is_the_one_carve_out(self, value: Any) -> None:
        assert clip_range_problems(_spec(clip_param=value), context="ppo") == []
        assert positive_finite_number_error(value, "clip_param", "ppo") is not None

    def test_negative_infinity_is_not_carved_out(self) -> None:
        assert clip_range_problems(_spec(clip_param=float("-inf")), context="ppo") != []


class TestTheBackendsThatDoNotClipStaySilent:
    """A backend that never reads the field must not report on it."""

    @pytest.mark.parametrize("provider", NO_CLIP_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_a_non_clipping_backend_says_nothing_about_the_width(
        self, provider: str, spec: RLTrainSpec, value: Any
    ) -> None:
        spec.clip_param = value
        assert _clip_problems(provider, spec) == []

    @pytest.mark.parametrize("provider", NO_CLIP_BACKENDS)
    def test_the_silence_is_scoping_rather_than_an_empty_preflight(self, provider: str) -> None:
        """Non-vacuity: the same backends do refuse a field they *do* read."""
        spec = _spec(learning_rate=float("nan"))
        problems = create_trainer(provider).validate(spec)
        assert any(": learning_rate " in p for p in problems), problems


class TestTheConsumerCannotJudgeTheWidth:
    """The premises the domain rests on, measured against ``torch``."""

    def test_a_nan_width_routes_the_gradient_to_the_unclipped_branch(self) -> None:
        """The crux: the loss reads ``nan`` while the update descends unclipped.

        ``torch.max`` propagates the ``nan`` forward, so every reported loss is
        ``nan``; backward, every comparison against ``nan`` is false, so the
        gradient is the *unclipped* surrogate's. That is why a ``nan`` half-width
        trains bit-identically to an unclipped run instead of poisoning it.
        """
        torch = pytest.importorskip("torch")
        adv = torch.tensor([0.5, -0.3, 0.9])

        def surrogate_grad(width: float) -> tuple[list[float], bool]:
            ratio = torch.tensor([0.7, 1.0, 1.4], requires_grad=True)
            clipped = -adv * torch.clamp(ratio, 1.0 - width, 1.0 + width)
            loss = torch.max(-adv * ratio, clipped).mean()
            loss.backward()
            assert ratio.grad is not None
            return [round(v, 6) for v in ratio.grad.tolist()], bool(torch.isnan(loss))

        nan_grad, nan_loss_is_nan = surrogate_grad(float("nan"))
        unclipped_grad, inf_loss_is_nan = surrogate_grad(math.inf)
        clipped_grad, default_loss_is_nan = surrogate_grad(0.2)

        assert nan_loss_is_nan, "a nan width must show up in the reported loss"
        assert not inf_loss_is_nan and not default_loss_is_nan
        assert nan_grad == unclipped_grad, "a nan width must not be the unclipped gradient"
        assert nan_grad != clipped_grad, "the clipped gradient must differ, or the probe is vacuous"

    def test_a_negative_width_inverts_the_bounds_into_a_constant(self) -> None:
        torch = pytest.importorskip("torch")
        ratio = torch.tensor([0.7, 1.0, 1.4])
        clamped = torch.clamp(ratio, 1.0 - (-0.2), 1.0 + (-0.2)).tolist()
        assert len(set(clamped)) == 1, f"expected a constant, got {clamped}"
        assert torch.clamp(ratio, 0.8, 1.2).tolist() != clamped

    def test_a_zero_width_pins_the_value_clip_to_the_old_values(self) -> None:
        torch = pytest.importorskip("torch")
        old_values = torch.tensor([0.4, -1.2, 3.0])
        delta = torch.tensor([-0.9, 0.5, 0.9])
        assert torch.equal(old_values + delta.clamp(-0.0, 0.0), old_values)
        assert not torch.equal(old_values + delta.clamp(-0.2, 0.2), old_values)

    def test_infinity_leaves_the_ratio_unchanged(self) -> None:
        """The measured basis for the carve-out."""
        torch = pytest.importorskip("torch")
        ratio = torch.tensor([0.7, 1.0, 1.4])
        assert torch.equal(torch.clamp(ratio, 1.0 - math.inf, 1.0 + math.inf), ratio)


class TestTheRefusalPrecedesTheUpdate:
    """A refused width reaches no optimizer."""

    def test_the_on_policy_train_fails_closed_before_building_anything(self, tmp_path: Any) -> None:
        pytest.importorskip("torch")
        called: list[str] = []

        def env_factory() -> None:
            called.append("env")
            raise AssertionError("the refused width reached the environment")

        spec = _spec(env_factory=env_factory, output_dir=str(tmp_path), clip_param=float("nan"))
        result = create_trainer(ON_POLICY).train(spec)
        assert result.status == "error"
        assert "clip_param" in (result.message or ""), result.message
        assert called == [], "the preflight must precede the environment"


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(clip_range_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_width(source: str) -> bool:
    """Does *source* read ``spec.clip_param``, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, ("clip_param",))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_clip_range_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheClipRangeDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* ``spec.clip_param`` must route it through the shared
    gate, so a second backend that starts clipping with the field fails this
    test until it does.
    """

    def test_every_module_that_reads_it_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name
            for p in _training_modules()
            if _reads_the_width(p.read_text()) and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"modules read spec.clip_param without the shared gate: {adrift}"

    def test_the_reader_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        readers = {p.name for p in _training_modules() if _reads_the_width(p.read_text())}
        assert readers == {"ppo.py"}, readers

    def test_the_scanner_detects_a_planted_reader(self) -> None:
        """A module reading the field without the gate is really reported."""
        planted = "def validate(self, spec):\n    return [] if spec.clip_param else []\n"
        assert _reads_the_width(planted)
        assert not _calls_the_gate(planted)

    def test_no_backend_re_implements_the_bound(self) -> None:
        """A hand-rolled comparison agrees with the shared rule until it drifts."""
        offenders = [
            p.name
            for p in _training_modules()
            if "clip_param <= 0" in p.read_text() or "clip_param < 0" in p.read_text()
        ]
        assert offenders == [], f"modules compare clip_param themselves: {offenders}"
