"""The VERA IK path's two numeric knobs are refused when they cannot be honored.

``VeraPolicy`` converts the server's end-effector delta chunk into joint targets
inside :meth:`get_actions`, and two caller-supplied numbers shape every target it
produces:

* ``translation_scale`` multiplies every translation delta before the IK solve
  (:func:`~strands_robots.policies.vera.sim_ik.decode_vera_delta_chunk_to_targets`,
  also settable on :meth:`VeraPolicy.set_ik_target`), and
* ``ik_smoothing`` blends each solved target with the previous one
  (``VeraPolicy(ik_smoothing=...)``).

Both were read with a bare ``float()``. Neither is refused by anything
downstream, because both are *applied* rather than forwarded: the scale
multiplies a delta and the coefficient weights a blend, so an unusable value
produces a plausible-looking chunk of joint targets rather than an error. The
classes below measure what those chunks are - the arm frozen, diverging,
undamped, translating the wrong way, or non-finite throughout - and then pin the
domains that refuse them.

The behavioural classes drive the real conversion against a pure-numpy stub
bridge (``decode_vera_delta_chunk_to_targets`` touches only ``solve``,
``ee_pose`` and ``model.nq``), so nothing here needs ``mink`` or ``mujoco``.
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

from strands_robots.policies.vera import provider as provider_mod
from strands_robots.policies.vera.provider import MAX_IK_SMOOTHING, VeraPolicy
from strands_robots.policies.vera.sim_ik import decode_vera_delta_chunk_to_targets
from strands_robots.utils import finite_number_error, positive_finite_number_error

# Values no caller can have meant for either knob: a wrong type, a non-finite
# number, or a bool (an ``int`` subclass whose ``True`` acts as a silent 1.0).
UNUSABLE_FOR_BOTH: list[Any] = [
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    False,
    "0.5",
    [0.5],
    {},
]
#: Additionally unusable as a positive multiplier.
UNUSABLE_SCALES: list[Any] = [*UNUSABLE_FOR_BOTH, 0.0, -0.0, -1.0]
#: Additionally outside the smoothing interval ``[0, 1)``.
UNUSABLE_SMOOTHINGS: list[Any] = [*UNUSABLE_FOR_BOTH, 1.0, 1.5, 2.0, -0.5]


class _CartesianStubBridge:
    """A perfect 3-DoF cartesian arm: ``q[:3]`` *is* the end-effector position.

    Makes the accumulated translation the IK path produces exactly readable, so
    a test can assert on metres of end-effector travel rather than on joint
    angles. ``decode_vera_delta_chunk_to_targets`` needs nothing else from a
    bridge.
    """

    class _Model:
        nq = 3

    model = _Model()

    def ee_pose(self, qpos: Any) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray(qpos, dtype=np.float64)[:3]
        return pose

    def solve(self, target_pose: Any, q_init: Any) -> np.ndarray:
        del q_init
        return np.asarray(target_pose, dtype=np.float64)[:3, 3].copy()


def _descend_chunk(steps: int = 8) -> np.ndarray:
    """A chunk asking for a saturated straight-down translation each step."""
    chunk = np.zeros((steps, 7), dtype=np.float64)
    chunk[:, 2] = -1.0  # saturated -z translation delta
    chunk[:, 6] = 1.0  # gripper open
    return chunk


def _decode(bridge: Any, translation_scale: Any, q_init: Any = None) -> Any:
    """Call the decoder through one deliberately untyped funnel.

    The bridges above are duck-typed on purpose - the decoder touches only
    ``solve``, ``ee_pose`` and ``model.nq``, so a stub needs neither ``mink`` nor
    ``mujoco`` - while the production signature names the concrete
    ``MinkIKBridge``. Routing every call through one ``Any``-typed helper states
    that once rather than a suppression per call site, and does the same for the
    scales these tests deliberately supply out of domain.
    """
    return decode_vera_delta_chunk_to_targets(
        _descend_chunk(),
        bridge,
        np.zeros(3, dtype=np.float64) if q_init is None else q_init,
        rotation_dim=3,
        has_gripper=True,
        translation_scale=translation_scale,
    )


def _ee_travel(translation_scale: Any) -> np.ndarray:
    """End-effector displacement (m) the IK path produces for a scale."""
    q0 = np.zeros(3, dtype=np.float64)
    out = _decode(_CartesianStubBridge(), translation_scale, q0)
    qpos = np.asarray(out["qpos"], dtype=np.float64)
    return qpos[-1] - q0


class TestWhyTheTranslationScaleDomainIsPositiveFinite:
    """Measure what an out-of-domain scale does to the joint targets.

    Not a restatement of the guard: each test drives the real conversion and
    asserts the chunk it returns, which is the evidence that the value is
    applied rather than rejected somewhere downstream.
    """

    def test_a_usable_scale_descends_by_the_scaled_delta(self) -> None:
        """The reference: 8 saturated -z steps at the OSC scale descend 8 * 0.05 m."""
        travel = _ee_travel(1.0)
        assert travel[2] == pytest.approx(-0.40, abs=1e-9)
        assert travel[0] == pytest.approx(0.0)
        assert travel[1] == pytest.approx(0.0)

    def test_a_zero_scale_discards_the_translation_half_of_every_action(self) -> None:
        """``0`` returns a chunk that only rotates - the translation is gone.

        Not a refusal and not an error: the caller receives ``T`` joint targets
        and a ``tracking_error`` reporting a perfect solve, because the target it
        tracked perfectly was the arm's current position.
        """
        with pytest.raises(ValueError):
            _decode(_CartesianStubBridge(), 0.0)
        # The pre-fix behaviour this refusal replaces, reproduced from the
        # arithmetic the function performs: the scale multiplies the delta, so a
        # zero one leaves the seed pose as every solved target.
        saturated_delta = np.array([0.0, 0.0, -1.0])
        assert np.allclose(saturated_delta * (0.05 * 0.0), 0.0)
        assert not np.allclose(saturated_delta * (0.05 * 1.0), 0.0)

    def test_a_negative_scale_inverts_every_commanded_translation(self) -> None:
        """A sign error would send the arm up for a command to descend."""
        with pytest.raises(ValueError):
            _ee_travel(-1.0)
        # What the arithmetic would have produced: the mirror of the reference.
        assert _ee_travel(1.0)[2] == pytest.approx(-0.40, abs=1e-9)

    def test_a_non_finite_scale_makes_every_joint_target_non_finite(self) -> None:
        """The sharpest case: nothing in the returned chunk is usable.

        Pre-fix this returned normally, so the caller's next step handed
        ``send_action`` a full set of ``nan`` targets - refused there for being
        non-finite, which reads as a wrong-embodiment action-key mismatch rather
        than as the scale that caused it. ``tracking_error`` was ``nan`` too, so
        even the diagnostic the function returns could not report it.
        """
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError):
                _decode(_CartesianStubBridge(), bad)
        # The arithmetic, to show the guard is not merely tightening a bound:
        # a non-finite factor poisons the delta and therefore every target.
        poisoned = np.array([0.0, 0.0, -1.0]) * (0.05 * float("nan"))
        assert not np.all(np.isfinite(poisoned))


def _smoothed_targets(alpha: float, solved: list[float]) -> list[float]:
    """Run the provider's EMA over one joint's IK solutions.

    Mirrors the blend ``get_actions`` applies per step so the effect of a
    coefficient is measurable without a server, a model or an event loop.
    """
    prev: float | None = None
    out: list[float] = []
    for value in solved:
        target = value
        if alpha > 0.0:
            if prev is not None:
                target = (1.0 - alpha) * target + alpha * prev
            prev = target
        out.append(target)
    return out


class TestWhyTheSmoothingIntervalIsTheDomain:
    """Measure what a coefficient outside ``[0, 1)`` does to the joint targets."""

    SOLVED = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    def test_a_usable_coefficient_damps_without_stopping_the_arm(self) -> None:
        """The reference: 0.6 damps the travel but the arm still gets there."""
        travel = _smoothed_targets(0.6, self.SOLVED)[-1] - self.SOLVED[0]
        ideal = self.SOLVED[-1] - self.SOLVED[0]
        assert 0.0 < travel < ideal

    def test_one_freezes_the_arm_at_its_first_solved_pose(self) -> None:
        """``alpha = 1`` weights only the previous target, so nothing moves."""
        targets = _smoothed_targets(1.0, self.SOLVED)
        assert targets == [self.SOLVED[0]] * len(self.SOLVED)
        assert MAX_IK_SMOOTHING == 1.0
        assert _ik_refusal(1.0) is not None, "the value that freezes the arm must be refused"

    def test_above_one_the_targets_diverge_away_from_the_solution(self) -> None:
        """A negative weight on the IK solution extrapolates, unbounded."""
        ideal = self.SOLVED[-1] - self.SOLVED[0]
        for alpha, at_least in ((1.5, 5.0), (2.0, 30.0)):
            travel = _smoothed_targets(alpha, self.SOLVED)[-1] - self.SOLVED[0]
            assert travel < 0.0, f"alpha={alpha} should move opposite the solution"
            assert abs(travel) > at_least * ideal, f"alpha={alpha} should diverge"
            assert _ik_refusal(alpha) is not None

    def test_a_negative_or_nan_coefficient_silently_applies_no_smoothing(self) -> None:
        """Both fail the ``alpha > 0`` test the blend is gated on.

        So the caller asked for damping and got none, with nothing logged - the
        mode-comparison hole that makes this a silent defect rather than a loud
        one.
        """
        undamped = _smoothed_targets(0.0, self.SOLVED)
        for alpha in (-0.5, float("nan")):
            assert not alpha > 0.0, "the blend is skipped entirely for this value"
            assert _smoothed_targets(alpha, self.SOLVED) == undamped
            assert _ik_refusal(alpha) is not None

    def test_infinity_makes_every_target_after_the_first_non_finite(self) -> None:
        targets = _smoothed_targets(float("inf"), self.SOLVED)
        assert targets[0] == self.SOLVED[0]
        assert all(not math.isfinite(t) for t in targets[1:])
        assert _ik_refusal(float("inf")) is not None


def _ik_refusal(value: Any) -> str | None:
    return provider_mod._ik_smoothing_error(value, "ik_smoothing", "VeraPolicy")


class _StubClient:
    """The minimum a ``VeraPolicy`` needs at construction time."""

    def get_server_metadata(self) -> dict[str, Any]:
        return {}

    def reset(self, info: Any) -> None:
        _ = info

    def configure(self, params: Any) -> dict[str, Any]:
        _ = params
        return {}

    def close(self) -> None:
        return None


def _policy(client: Any = None, **kwargs: Any) -> VeraPolicy:
    """Build a policy against a duck-typed client, for the same reason as ``_decode``."""
    stub: Any = _StubClient() if client is None else client
    return VeraPolicy(client=stub, auto_launch_server=False, **kwargs)


class TestTheSmoothingCoefficientIsRefusedAtConstruction:
    @pytest.mark.parametrize("value", UNUSABLE_SMOOTHINGS, ids=repr)
    def test_an_unusable_coefficient_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError, match="ik_smoothing"):
            _policy(ik_smoothing=value)

    @pytest.mark.parametrize("value", [0.0, 0.3, 0.6, 0.9999, np.float32(0.25), np.float64(0.5)], ids=repr)
    def test_a_usable_coefficient_is_stored(self, value: Any) -> None:
        assert _policy(ik_smoothing=value)._ik_smoothing == pytest.approx(float(value))

    def test_the_default_disables_the_smoothing(self) -> None:
        """``0`` is the documented "no smoothing" default and must stay valid."""
        assert _policy()._ik_smoothing == 0.0
        assert _ik_refusal(0.0) is None


class TestTheTranslationScaleIsRefusedAtBothSurfaces:
    """One value, two public entry points, one domain."""

    @pytest.mark.parametrize("value", UNUSABLE_SCALES, ids=repr)
    def test_set_ik_target_refuses_an_unusable_scale(self, value: Any) -> None:
        policy = _policy()
        with pytest.raises(ValueError, match="translation_scale"):
            policy.set_ik_target(object(), "hand", "body", translation_scale=value)

    @pytest.mark.parametrize("value", UNUSABLE_SCALES, ids=repr)
    def test_the_decoder_refuses_an_unusable_scale(self, value: Any) -> None:
        with pytest.raises(ValueError, match="translation_scale"):
            _decode(_CartesianStubBridge(), value)

    @pytest.mark.parametrize("value", UNUSABLE_SCALES, ids=repr)
    def test_the_two_surfaces_agree(self, value: Any) -> None:
        """Neither surface may accept what the other refuses."""
        policy = _policy()
        try:
            policy.set_ik_target(object(), "hand", "body", translation_scale=value)
            setter_refused = False
        except ValueError:
            setter_refused = True
        try:
            _decode(_CartesianStubBridge(), value)
            decoder_refused = False
        except ValueError:
            decoder_refused = True
        assert setter_refused == decoder_refused, f"verdicts differ for translation_scale={value!r}"

    @pytest.mark.parametrize("value", [0.5, 1.0, 2.5, np.float32(1.5)], ids=repr)
    def test_a_usable_scale_is_stored_and_honored(self, value: Any) -> None:
        policy = _policy()
        policy.set_ik_target(object(), "hand", "body", translation_scale=value)
        assert policy._translation_scale == pytest.approx(float(value))
        travel = _ee_travel(value)
        assert travel[2] == pytest.approx(-0.05 * 8 * float(value), rel=1e-6)

    def test_none_still_means_leave_the_current_value(self) -> None:
        """``None`` is the documented "no override" spelling, not a bad value.

        The decoder has no such spelling - its default is ``1.0`` - so the same
        value is legitimately accepted by one surface and refused by the other.
        """
        policy = _policy()
        policy.set_ik_target(object(), "hand", "body", translation_scale=2.5)
        policy.set_ik_target(object(), "hand", "body")
        assert policy._translation_scale == 2.5
        assert positive_finite_number_error(None, "translation_scale", "set_ik_target") is not None


class TestTheIntervalIsTheWholeLocalContribution:
    """``_ik_smoothing_error`` decides the bounds and delegates everything else."""

    @pytest.mark.parametrize(
        "value",
        [*UNUSABLE_FOR_BOTH, 0.0, 0.6, 0.9999, np.float32(0.5), 10**400],
        ids=repr,
    )
    def test_it_agrees_with_the_shared_domain_inside_the_interval(self, value: Any) -> None:
        shared = finite_number_error(value, "p", "C") is None
        local = _ik_refusal(value) is None
        assert local == shared, f"the two must only differ outside [0, {MAX_IK_SMOOTHING})"

    @pytest.mark.parametrize("value", [-0.5, 1.0, 1.5, 2.0], ids=repr)
    def test_it_refuses_exactly_the_finite_numbers_outside_the_interval(self, value: Any) -> None:
        assert finite_number_error(value, "p", "C") is None, "the shared domain accepts it"
        assert _ik_refusal(value) is not None, "the interval must refuse it"

    def test_the_bound_is_named_in_the_message(self) -> None:
        message = _ik_refusal(1.5)
        assert message is not None
        assert f"[0, {MAX_IK_SMOOTHING})" in message


class TestNothingIsConfiguredByARefusedValue:
    def test_a_refused_scale_leaves_the_previous_one_in_place(self) -> None:
        policy = _policy()
        original_model = object()
        policy.set_ik_target(original_model, "hand", "body", translation_scale=2.5)
        original_bridge = policy._ik_bridge
        with pytest.raises(ValueError):
            policy.set_ik_target(object(), "elbow", "site", translation_scale=float("nan"))
        assert policy._translation_scale == 2.5
        # Guard-before-mutation: nothing else changed either.
        assert policy._mj_model is original_model
        assert policy._ee_frame_name == "hand"
        assert policy._ee_frame_type == "body"
        assert policy._ik_bridge is original_bridge

    def test_a_refused_coefficient_precedes_the_config(self) -> None:
        """The guard runs before ``VeraConfig`` is built, so nothing is left half-configured."""
        with pytest.raises(ValueError, match="ik_smoothing"):
            _policy(ik_smoothing=1.0, embodiment="mimicgen")

    def test_a_refused_scale_reaches_no_ik_solve(self) -> None:
        """The decoder's guard runs before the first ``solve``."""

        class _CountingBridge(_CartesianStubBridge):
            solves = 0

            def solve(self, target_pose: Any, q_init: Any) -> np.ndarray:
                type(self).solves += 1
                return super().solve(target_pose, q_init)

        with pytest.raises(ValueError):
            _decode(_CountingBridge(), 0.0)
        assert _CountingBridge.solves == 0


class TestTheGuardsSurviveARealRollout:
    """A usable pair still drives ``get_actions`` end to end."""

    def test_a_smoothed_rollout_still_produces_joint_targets(self) -> None:
        class _Client:
            def get_server_metadata(self) -> dict[str, Any]:
                return {
                    "action_space": "eef_delta",
                    "context_frames": 1,
                    "gripper_dim_index": 6,
                    "gripper_is_raw": True,
                    "view_keys": ["image"],
                }

            def reset(self, info: Any) -> None:
                _ = info

            def configure(self, params: Any) -> dict[str, Any]:
                _ = params
                return {}

            def infer(self, request: Any) -> dict[str, Any]:
                del request
                return {"action": np.asarray([[0.1, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0]] * 4, np.float32)}

            def close(self) -> None:
                return None

        class _Bridge(_CartesianStubBridge):
            class _SevenDofModel:
                nq = 7

            model: Any = _SevenDofModel()

            def solve(self, target_pose: Any, q_init: Any) -> np.ndarray:
                q = np.asarray(q_init, dtype=np.float64).copy()
                q[:3] = np.asarray(target_pose, dtype=np.float64)[:3, 3]
                return q

        joints = [f"joint_{i}" for i in range(6)] + ["gripper"]
        policy = _policy(client=_Client(), ik_smoothing=0.6)
        policy._runner = None
        policy.set_robot_state_keys(joints)
        policy.set_ik_target(_Bridge.model, "hand", "body", translation_scale=1.5)
        # Injected last: set_ik_target clears the lazily-built bridge so a later
        # model change rebuilds it, which would need the real mink stack.
        policy._mj_model = _Bridge.model
        policy._ee_frame_name = "hand"
        policy._ik_bridge = _Bridge()
        # The cache serves a bridge only for the model and frame it was built
        # from, so the injected one states which those are.
        policy._ik_bridge_binding = (_Bridge.model, "hand", "body")
        observation = {"image": np.zeros((8, 8, 3), np.uint8), **dict.fromkeys(joints, 0.0)}
        actions = asyncio.run(policy.get_actions(observation, "pick"))
        assert actions, "a usable pair must still produce actions"
        assert any(key.startswith("joint_") for key in actions[0]), actions[0]
        assert all(math.isfinite(v) for v in actions[0].values()), actions[0]


class TestNoIkNumericSurfaceDrifts:
    """Every public VERA surface taking these knobs routes through a domain."""

    KNOBS = {"translation_scale", "ik_smoothing"}
    GUARDS = {"positive_finite_number_error", "_ik_smoothing_error"}
    EXPECTED = {
        ("provider.py", "__init__"),
        ("provider.py", "set_ik_target"),
        ("sim_ik.py", "decode_vera_delta_chunk_to_targets"),
    }

    @staticmethod
    def _surfaces(sources: dict[str, str]) -> dict[tuple[str, str], set[str]]:
        """Map every public function taking a knob to the guards it calls."""
        found: dict[tuple[str, str], set[str]] = {}
        for name, src in sources.items():
            for node in ast.walk(ast.parse(src)):
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if node.name.startswith("_") and node.name != "__init__":
                    continue
                params = {a.arg for a in node.args.args + node.args.kwonlyargs}
                if not params & TestNoIkNumericSurfaceDrifts.KNOBS:
                    continue
                calls = {
                    call.func.id
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                }
                found[(name, node.name)] = calls & TestNoIkNumericSurfaceDrifts.GUARDS
        return found

    @staticmethod
    def _package_sources() -> dict[str, str]:
        root = pathlib.Path(inspect.getfile(VeraPolicy)).parent
        return {path.name: path.read_text() for path in sorted(root.glob("*.py"))}

    def test_the_scan_finds_exactly_the_known_surfaces(self) -> None:
        """Non-vacuity: a scan rooted elsewhere would find nothing."""
        assert set(self._surfaces(self._package_sources())) == self.EXPECTED

    def test_every_surface_calls_a_domain(self) -> None:
        adrift = {key for key, guards in self._surfaces(self._package_sources()).items() if not guards}
        assert not adrift, f"these read an IK knob without a domain: {sorted(adrift)}"

    def test_the_scanner_detects_a_planted_surface(self) -> None:
        """A guard that matched nothing would pass the test above vacuously."""
        planted = "def configure_scale(translation_scale: float) -> None:\n    pass\n"
        found = self._surfaces({"planted.py": planted})
        assert found == {("planted.py", "configure_scale"): set()}
