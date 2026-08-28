"""Value-domain contracts for the Microduck policy surface's two scalars.

Both classes in this package validate their *structural* arguments carefully.
:class:`MicroduckPolicyBundle` refuses an empty mapping, refuses a value that is
not a :class:`MicroduckPolicy` (by name and by type), and refuses an ``active``
skill that is not one of its keys; :meth:`MicroduckPolicy.set_robot_state_keys`
routes its list through the shared ``name_list_error`` domain. The two
caller-supplied *numbers* reached their consumer through a bare ``float()``.

Neither number fails when it is unusable - each one silently changes what the
policy commands, which is why the guard belongs where the value arrives:

* ``motor_target = default_pose + raw_action * action_scale``, so a scale of
  ``0`` (or ``False``, an ``int`` subclass) makes every target exactly
  ``default_pose``: the network's decision is discarded and the biped holds its
  nominal stance while the rollout reports success. A non-finite scale makes all
  fourteen targets ``nan``. It reaches the decode two ways - the constructor
  kwarg and the ONNX ``action_scale`` metadata - and a guard on one route only
  would let the same value in through the other.
* the bundle's velocity gate is ``|twist| >= switch_on_velocity``, and a
  magnitude is never negative, so a threshold of ``0`` or below can never select
  the idle skill (a biped told to stop keeps walking) and a non-finite one can
  never select the move skill (a biped told to walk stands still). Both are
  reported as a successful tick.

The domain member is not a new judgement. ``WBCConfig`` - the other ONNX
locomotion provider, decoding with the same
``default_angles + action_scale * raw_action`` formula - already holds its
identically-named ``action_scale`` to ``positive_finite_number_error``, and
:meth:`Policy.set_control_frequency` in the base class states the same reason
for its own rate ("``nan`` and ``inf`` both survive a bare ``hz <= 0`` test").
These tests pin that domain through the public surfaces a caller reaches, and
pin the values that stay first-class so the guard cannot creep into refusing a
controller a caller may legitimately ask for.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import math
import pathlib
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.microduck import MicroduckPolicy, MicroduckPolicyBundle
from strands_robots.policies.microduck.policy import MICRODUCK_DEFAULT_POSE, MICRODUCK_JOINT_NAMES
from tests.policies.microduck.test_microduck_policy import _obs_dict, _StubSession

#: The values neither knob can use. Every one of them was accepted before, and
#: every one of them changes what the policy commands rather than raising.
_UNUSABLE = [
    pytest.param(0.0, id="zero"),
    pytest.param(-0.25, id="negative"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="neg-inf"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param("0.25", id="numeric-string"),
    pytest.param([0.25], id="one-element-list"),
]

#: Values that must keep working, so the guard is a domain and not a narrowing.
_USABLE = [
    pytest.param(0.25, id="float"),
    pytest.param(1, id="int"),
    pytest.param(np.float64(0.25), id="numpy-float"),
    pytest.param(1e-9, id="tiny-but-positive"),
]

_PACKAGE = pathlib.Path(MicroduckPolicy.__module__.replace(".", "/")).parent


def _scaled_session(scale: str) -> _StubSession:
    """A stub session whose ONNX metadata declares ``action_scale=scale``."""
    return _StubSession(
        meta={
            "joint_names": ",".join(MICRODUCK_JOINT_NAMES),
            "default_joint_pos": ",".join(f"{v}" for v in MICRODUCK_DEFAULT_POSE),
            "action_scale": scale,
            "command_names": "twist,head_pose,body_pose",
        }
    )


def _targets(policy: MicroduckPolicy) -> np.ndarray:
    """One tick's motor targets, in joint order."""
    out = asyncio.run(policy.get_actions(_obs_dict(), ""))[0]
    return np.array([out[name] for name in MICRODUCK_JOINT_NAMES], dtype=float)


def _bundle(**kwargs: Any) -> MicroduckPolicyBundle:
    """A two-skill bundle whose children are stub-backed."""
    return MicroduckPolicyBundle(
        {"walk": MicroduckPolicy(session=_StubSession()), "stand": MicroduckPolicy(session=_StubSession())},
        active="stand",
        **kwargs,
    )


class TestTheArithmeticThatMakesItSilent:
    """Why a bare comparison cannot stand in for the domain.

    Premises: they hold on either version of the code, and they are the reason
    the guard has to be at the constructor rather than at the comparison.
    """

    @pytest.mark.parametrize("magnitude", [0.0, 0.5, 1e9])
    def test_a_non_finite_threshold_loses_every_comparison(self, magnitude: float) -> None:
        # The gate can never select the move skill: no magnitude clears it.
        for threshold in (math.nan, math.inf):
            assert not (magnitude >= threshold)

    @pytest.mark.parametrize("threshold", [0.0, -1.0, -math.inf])
    def test_a_non_positive_threshold_wins_every_comparison(self, threshold: float) -> None:
        # A magnitude is never negative, so the gate can never select idle:
        # these are constants, not thresholds.
        for magnitude in (0.0, 0.5, 1e9):
            assert magnitude >= threshold

    def test_a_bool_is_an_int_subclass_so_a_flag_reads_as_a_number(self) -> None:
        assert isinstance(True, int) and float(True) == 1.0
        assert float(False) == 0.0

    def test_the_sibling_provider_already_holds_this_knob_to_this_domain(self) -> None:
        from strands_robots.policies.wbc import config as wbc_config

        source = inspect.getsource(wbc_config)
        assert 'positive_finite_number_error(self.action_scale, "action_scale"' in source


class TestTheActionScaleConstructorRoute:
    """The scale a caller hands the constructor is checked where it arrives."""

    @pytest.mark.parametrize("value", _UNUSABLE)
    def test_an_unusable_scale_is_refused_at_construction(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"action_scale"):
            MicroduckPolicy(session=_StubSession(), action_scale=value)

    def test_the_refusal_names_the_class_the_field_and_the_route(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            MicroduckPolicy(session=_StubSession(), action_scale=0.0)
        text = str(excinfo.value)
        assert text.startswith("MicroduckPolicy (constructor): action_scale")
        assert "must be > 0" in text

    def test_the_scale_is_refused_before_it_is_stored(self) -> None:
        # A guard after the coercion would leave a nan on the instance.
        with pytest.raises(ValueError):
            MicroduckPolicy(session=_StubSession(), action_scale=float("nan"))


class TestTheActionScaleMetadataRoute:
    """The same scale arriving from the ONNX file meets the same domain.

    The metadata route is resolved lazily on the first inference, so the refusal
    surfaces from ``get_actions`` rather than from the constructor - but it is
    the same domain, and a caller cannot route around the constructor's check by
    exporting the value into the file instead.
    """

    @pytest.mark.parametrize("declared", ["0", "0.0", "-0.25", "nan", "inf", "-inf"])
    def test_an_unusable_declared_scale_is_refused(self, declared: str) -> None:
        policy = MicroduckPolicy(session=_scaled_session(declared))
        with pytest.raises(ValueError, match=r"action_scale"):
            asyncio.run(policy.get_actions(_obs_dict(), ""))

    def test_the_refusal_names_the_route_it_came_from(self) -> None:
        policy = MicroduckPolicy(session=_scaled_session("nan"))
        with pytest.raises(ValueError) as excinfo:
            asyncio.run(policy.get_actions(_obs_dict(), ""))
        assert str(excinfo.value).startswith("MicroduckPolicy (ONNX metadata): action_scale")

    def test_a_declared_scale_that_is_not_a_number_names_the_field(self) -> None:
        policy = MicroduckPolicy(session=_scaled_session("fast"))
        with pytest.raises(ValueError) as excinfo:
            asyncio.run(policy.get_actions(_obs_dict(), ""))
        text = str(excinfo.value)
        assert "MicroduckPolicy" in text and "action_scale" in text and "'fast'" in text

    def test_an_explicit_scale_still_wins_over_the_file(self) -> None:
        policy = MicroduckPolicy(session=_scaled_session("0.5"), action_scale=0.25)
        assert policy._action_scale == 0.25
        assert np.all(np.isfinite(_targets(policy)))


class TestTheVelocityGateThreshold:
    """The gate's threshold is a magnitude bound, so it must be positive."""

    @pytest.mark.parametrize("value", _UNUSABLE)
    def test_an_unusable_threshold_is_refused_at_construction(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"switch_on_velocity"):
            _bundle(switch_on_velocity=value)

    def test_the_refusal_names_the_class_and_the_field(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _bundle(switch_on_velocity=float("nan"))
        text = str(excinfo.value)
        assert text.startswith("MicroduckPolicyBundle: switch_on_velocity")
        assert "must be > 0" in text

    def test_the_structural_checks_still_run_first(self) -> None:
        # An unusable threshold must not mask the identity error a caller would
        # rather hear about: the active-skill check precedes it.
        with pytest.raises(ValueError, match=r"active skill"):
            MicroduckPolicyBundle(
                {"walk": MicroduckPolicy(session=_StubSession())},
                active="nope",
                switch_on_velocity=float("nan"),
            )


class TestOneOwnerForBothActionScaleRoutes:
    """Both routes to the decode consult one domain helper, not two copies."""

    def test_both_routes_call_the_shared_helper(self) -> None:
        from strands_robots.policies.microduck import policy as policy_module

        tree = ast.parse(inspect.getsource(policy_module))
        callers = {
            fn.name
            for fn in ast.walk(tree)
            if isinstance(fn, ast.FunctionDef)
            for call in ast.walk(fn)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_action_scale_error"
        }
        assert callers == {"__init__", "_ensure_config"}, callers

    def test_the_module_states_the_domain_in_exactly_one_place(self) -> None:
        from strands_robots.policies.microduck import policy as policy_module

        source = inspect.getsource(policy_module)
        assert source.count('positive_finite_number_error(value, "action_scale"') == 1


class TestEveryCallerSuppliedScalarConsultsADomain:
    """Derived: a scalar knob added later is held to the same rule.

    The inventory is read off the annotations rather than listed, so a third
    number arriving in this package is graded the hour it lands instead of
    inheriting an exemption by being absent from a tuple. A parameter is a
    *scalar* here when its annotation mentions ``float`` and does not also name
    an array or sequence type - the vector parameters (``command``,
    ``default_pose``) are a different domain member and are out of scope.

    The rule is scoped to the two public CLASSES, which is where a caller of the
    provider hands values in. The module-level primitives in
    :mod:`~strands_robots.policies.microduck.observation` are called per tick by
    the policy with values it has already checked, so a second check there would
    be a per-tick cost with no producer - the same split the WBC provider makes
    between ``WBCConfig`` (guarded) and ``compute_targets`` (not).
    """

    @staticmethod
    def _scalar_knobs() -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for path in sorted(_PACKAGE.glob("*.py")):
            tree = ast.parse(path.read_text())
            for klass in ast.walk(tree):
                if not isinstance(klass, ast.ClassDef) or klass.name.startswith("_"):
                    continue
                for fn in klass.body:
                    if not isinstance(fn, ast.FunctionDef):
                        continue
                    if fn.name.startswith("_") and fn.name != "__init__":
                        continue
                    for arg in fn.args.args + fn.args.kwonlyargs:
                        if arg.arg in ("self", "cls") or arg.annotation is None:
                            continue
                        annotation = ast.unparse(arg.annotation)
                        if "float" not in annotation:
                            continue
                        if any(t in annotation for t in ("NDArray", "list", "Sequence", "tuple", "dict")):
                            continue
                        found.setdefault(f"{path.name}::{fn.name}", []).append(arg.arg)
        return found

    @staticmethod
    def _consults_a_domain(path: pathlib.Path, fn_name: str, param: str) -> bool:
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef) or fn.name != fn_name:
                continue
            for call in ast.walk(fn):
                if not isinstance(call, ast.Call):
                    continue
                rendered = ast.unparse(call)
                if param not in rendered:
                    continue
                target = call.func
                name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
                # Either a domain call here, or delegation to the base class
                # that owns the check (``super().set_control_frequency(hz)``).
                if name.endswith(("_error", "_problems")) or name == fn_name:
                    return True
        return False

    def test_the_inventory_is_not_empty(self) -> None:
        knobs = self._scalar_knobs()
        flat = {param for params in knobs.values() for param in params}
        assert {"action_scale", "switch_on_velocity"} <= flat, knobs

    def test_every_scalar_knob_consults_a_domain(self) -> None:
        unguarded = [
            f"{where}:{param}"
            for where, params in self._scalar_knobs().items()
            for param in params
            if not self._consults_a_domain(_PACKAGE / where.split("::")[0], where.split("::")[1], param)
        ]
        assert unguarded == [], (
            f"caller-supplied scalar(s) reaching a consumer with no value domain: {unguarded}. "
            "Route the value through a shared domain in strands_robots.utils where it arrives."
        )


class TestWhatStaysFirstClass:
    """The guard is a domain, not a narrowing: these must keep working."""

    @pytest.mark.parametrize("value", _USABLE)
    def test_a_usable_scale_still_builds_and_offsets_from_the_default_pose(self, value: Any) -> None:
        policy = MicroduckPolicy(session=_StubSession(), action_scale=value)
        targets = _targets(policy)
        assert np.all(np.isfinite(targets))
        # A discarded decision is EXACT equality with the default pose; any
        # positive scale moves at least one joint off it, however slightly.
        assert not np.array_equal(targets, np.array(MICRODUCK_DEFAULT_POSE, dtype=float))

    @pytest.mark.parametrize("value", _USABLE)
    def test_a_usable_threshold_still_gates_both_ways(self, value: Any) -> None:
        bundle = _bundle(switch_on_velocity=value)
        asyncio.run(bundle.get_actions(_obs_dict(), "", target_velocity=[float(value) * 10.0, 0.0, 0.0]))
        assert bundle.active == "walk"
        asyncio.run(bundle.get_actions(_obs_dict(), "", target_velocity=[0.0, 0.0, 0.0]))
        assert bundle.active == "stand"

    def test_omitting_the_threshold_leaves_the_gate_off(self) -> None:
        bundle = _bundle()
        assert bundle._switch_on_velocity is None
        asyncio.run(bundle.get_actions(_obs_dict(), "", target_velocity=[9.0, 0.0, 0.0]))
        assert bundle.active == "stand"

    def test_omitting_the_scale_still_reads_it_from_the_file(self) -> None:
        policy = MicroduckPolicy(session=_scaled_session("0.5"))
        assert np.all(np.isfinite(_targets(policy)))
        assert policy._action_scale == 0.5

    def test_the_free_decode_helper_is_unchanged(self) -> None:
        # ``decode_action`` is reached from the policy, which has already
        # checked the scale; it keeps taking whatever it is handed.
        from strands_robots.policies.microduck.observation import decode_action

        raw = np.arange(14, dtype=np.float32) * 0.01
        default = np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32)
        out = decode_action(raw, default_pose=default, action_scale=2.0)
        assert np.allclose(out, default + raw * 2.0)
