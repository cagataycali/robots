"""The turn envelope lookup tools name what ``LocoClient.SetVelocity`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes
``SetVelocity(vx, vy, vyaw, duration_s)`` without clamps of its own;
the neon bundle
(``cagataycali/neon-the-g1/tools/g1_locomotion.py::g1_turn``) narrows
a two-argument ``(angle_rad, yaw_rate)`` surface to
``angle_rad = max(-2*pi, min(2*pi, float(angle_rad)))`` and
``yaw_rate = max(0.1, min(0.6, abs(float(yaw_rate))))`` before
dispatch. The :mod:`strands_robots.tools.g1.g1_turn_envelope` module
snapshots that clamp pair into module-level constants and exposes
two agent-facing verbs -- :func:`g1_list_turn_envelope` (name the
whole envelope) and :func:`g1_turn_admits` (decide one query) -- so
a caller can decide the refusal decidably before a future locomotion
write path is attempted. The tests here fix that contract without
pulling the SDK: the module is loadable on a host without
``unitree_sdk2py`` (the same SDK-load-hygiene rule every other file
under :mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off
the module's own snapshot rather than restated in the tests, so a
widen or narrow to the observed range surfaces here as a shape
change rather than as a diverging table this file would need to
manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The envelope is the neon
  bundle's observed range, not the SDK's own clamps (the SDK has
  none). A driver-side wrapper for the turn surface that lands later
  will re-check the envelope at wire time and its refusal string
  will quote the ``7404`` gate-refusal code the driver's
  ``_check_motion_gates`` also quotes.
* Whether the driver's live ``fsm_id`` sits inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That is a
  live driver-instance read and belongs on
  :mod:`~strands_robots.tools.g1.g1_state` /
  :mod:`~strands_robots.tools.g1.g1_motion_gates`; the verb surfaces
  the set as a snapshot so a caller comparing an intended write
  against both conditions has the FSM set on hand.
"""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

import pytest

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS
from strands_robots.tools.g1.g1_turn_envelope import (
    _ANGLE_RAD_MAX,
    _ANGLE_RAD_MIN,
    _ANGLE_SIGN_THRESHOLD,
    _GATE_REFUSAL_CODE,
    _MAX_COMPOSED_DURATION,
    _MIN_COMPOSED_DURATION,
    _SDK_METHOD,
    _YAW_RATE_MAX,
    _YAW_RATE_MIN,
    g1_list_turn_envelope,
    g1_turn_admits,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process; this helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent (refs strands-labs/robots#358); a module that
    pulled a submodule at import time would break every headless CI
    runner and Thor before an office bring-up.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_turn_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_turn_envelope imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        f"loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_envelope_bounds_are_finite_and_ordered() -> None:
    """Every clamp is a finite float and each min/max pair is ordered.

    A non-finite bound would let :func:`g1_turn_admits` admit every
    value on that dimension; an inverted min/max pair (min > max)
    would reject every finite value. Pins the invariant on both the
    angle and yaw-rate axes so a widen or narrow that inverts either
    pair surfaces here rather than as a silently unreachable envelope
    in production.
    """
    for name, value in (
        ("_ANGLE_RAD_MIN", _ANGLE_RAD_MIN),
        ("_ANGLE_RAD_MAX", _ANGLE_RAD_MAX),
        ("_YAW_RATE_MIN", _YAW_RATE_MIN),
        ("_YAW_RATE_MAX", _YAW_RATE_MAX),
        ("_ANGLE_SIGN_THRESHOLD", _ANGLE_SIGN_THRESHOLD),
        ("_MIN_COMPOSED_DURATION", _MIN_COMPOSED_DURATION),
        ("_MAX_COMPOSED_DURATION", _MAX_COMPOSED_DURATION),
    ):
        assert math.isfinite(value), f"{name} is not finite: {value!r}"

    assert _ANGLE_RAD_MIN <= _ANGLE_RAD_MAX, (
        f"angle bounds inverted: min={_ANGLE_RAD_MIN} > max={_ANGLE_RAD_MAX}. "
        f"g1_turn_admits would refuse every finite angle."
    )
    assert _YAW_RATE_MIN <= _YAW_RATE_MAX, (
        f"yaw-rate bounds inverted: min={_YAW_RATE_MIN} > max={_YAW_RATE_MAX}. "
        f"g1_turn_admits would refuse every finite yaw rate."
    )
    assert _MIN_COMPOSED_DURATION <= _MAX_COMPOSED_DURATION, (
        f"composed-duration bounds inverted: "
        f"min={_MIN_COMPOSED_DURATION} > max={_MAX_COMPOSED_DURATION}. "
        f"The advertised composed range would be empty."
    )


def test_the_yaw_rate_clamp_stays_strictly_positive() -> None:
    """The yaw-rate lower clamp is strictly positive.

    The neon wrapper takes ``abs(yaw_rate)`` before clamping, and its
    minimum ``0.1`` rad/s is the neon-bundle-observed minimum
    turnable rate. A future revision that dropped the floor to zero
    (or below) would either divide-by-zero in the composed-duration
    computation or route a stall-rate argument through to the
    controller. Pinned so that shift lands here rather than in
    production.
    """
    assert _YAW_RATE_MIN > 0.0, (
        f"_YAW_RATE_MIN must be strictly positive to avoid a divide-by-zero in composed-duration; got {_YAW_RATE_MIN!r}"
    )


def test_the_angle_clamp_is_symmetric_two_pi() -> None:
    """The angle clamp is exactly ``[-2*pi, 2*pi]``.

    The neon wrapper's ``max(-2*pi, min(2*pi, float(angle_rad)))``
    admits one full revolution in either direction; a caller passing
    ``4*pi`` (two full revolutions) reaches the outer clamp and the
    composed duration doubles. Pinned so a future revision that
    widened the clamp to ``4*pi`` (multiple turns) or narrowed it to
    ``pi`` (half turn) lands here rather than in production.
    """
    assert _ANGLE_RAD_MIN == pytest.approx(-2.0 * math.pi)
    assert _ANGLE_RAD_MAX == pytest.approx(2.0 * math.pi)
    assert _ANGLE_RAD_MIN == pytest.approx(-_ANGLE_RAD_MAX)


def test_the_composed_duration_bounds_match_the_clamp_pair() -> None:
    """The advertised composed-duration range matches ``[0 / yaw_max, angle_max / yaw_min]``.

    A caller comparing this envelope against
    :mod:`~strands_robots.tools.g1.g1_locomotion_duration_envelope`
    reads ``composed_duration_max`` to see whether the turn surface
    can compose a duration the duration envelope would clamp. Pinned
    so a widen of either the angle or yaw-rate clamp that shifted
    the composed range surfaces here rather than as a silent
    divergence.
    """
    assert _MIN_COMPOSED_DURATION == 0.0
    assert _MAX_COMPOSED_DURATION == pytest.approx(_ANGLE_RAD_MAX / _YAW_RATE_MIN)


def test_the_gate_refusal_code_matches_the_driver_constant() -> None:
    """The envelope's refusal code names the driver's gate refusal.

    The driver's ``_check_motion_gates`` refuses a locomotion-shaped
    write on an FSM outside :data:`WALK_FSMS` with rc=7404, and the
    ``7404`` entry in
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries the
    text a driver-side turn wrapper would surface. Pinned here so a
    re-wording of that message lands in one place, not one in the
    driver and a diverging copy in this envelope.
    """
    assert _GATE_REFUSAL_CODE == 7404
    assert _GATE_REFUSAL_CODE in ERR_CODES, (
        f"envelope quotes rc={_GATE_REFUSAL_CODE} but that code is not in "
        f"ERR_CODES. The refusal string would render as 'unknown'."
    )


def test_the_sdk_method_name_is_set_velocity() -> None:
    """The SDK method the envelope names is ``SetVelocity``.

    The neon bundle's ``g1_turn`` verb composes into a
    ``g1_move_velocity`` call which fronts ``LocoClient.SetVelocity``;
    a driver-side turn wrapper would target the same SDK method.
    Pinned here so a firmware release that renamed the SDK entry
    surfaces on this constant rather than as a diverging copy in
    production.
    """
    assert _SDK_METHOD == "SetVelocity"


def test_g1_list_turn_envelope_returns_the_full_envelope() -> None:
    """The verb's payload names every clamp, the gate set, and the refusal.

    ``envelope`` carries every clamp constant plus the sign threshold,
    the composed-duration range, and the SDK method name;
    ``walk_ready_fsm_ids`` quotes :data:`WALK_FSMS`; and ``refusals``
    names the ``7404`` gate-refusal code with the decoded text
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.
    """
    result = _call(g1_list_turn_envelope)
    assert result["status"] == "success"
    env = result["envelope"]
    assert env["angle_rad_min"] == _ANGLE_RAD_MIN
    assert env["angle_rad_max"] == _ANGLE_RAD_MAX
    assert env["yaw_rate_min"] == _YAW_RATE_MIN
    assert env["yaw_rate_max"] == _YAW_RATE_MAX
    assert env["angle_sign_threshold"] == _ANGLE_SIGN_THRESHOLD
    assert env["composed_duration_min"] == _MIN_COMPOSED_DURATION
    assert env["composed_duration_max"] == _MAX_COMPOSED_DURATION
    assert env["sdk_method"] == _SDK_METHOD
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)
    assert result["refusals"] == [
        {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
    ]


def test_g1_turn_admits_a_pair_inside_both_envelopes() -> None:
    """An ``(angle_rad, yaw_rate)`` pair strictly inside both clamps is admitted.

    The identity case (``angle_rad=0.5``, ``yaw_rate=0.3``) is the
    neon bundle's own default and sits strictly inside both clamps,
    so a driver-side turn wrapper would not refuse it on envelope
    grounds (whether the FSM gate admits it is a separate live-read
    decision the verb does not answer). ``composed_duration`` names
    what ``abs(angle_rad) / yaw_rate`` would evaluate to at those
    arguments.
    """
    result = _call(g1_turn_admits, angle_rad=0.5, yaw_rate=0.3)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []
    assert result["composed_duration"] == pytest.approx(0.5 / 0.3)
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)


def test_g1_turn_admits_at_the_exact_clamp_boundaries() -> None:
    """Both clamps are inclusive on both axes.

    The neon wrapper's ``max(-2*pi, min(2*pi, ...))`` and
    ``max(0.1, min(0.6, ...))`` are both inclusive at the boundary
    (a saturated command is a command); a strict-inequality refusal
    would silently drop the neon-bundle-admitted saturated case.
    """
    for a, y in (
        (_ANGLE_RAD_MIN, _YAW_RATE_MIN),
        (_ANGLE_RAD_MIN, _YAW_RATE_MAX),
        (_ANGLE_RAD_MAX, _YAW_RATE_MIN),
        (_ANGLE_RAD_MAX, _YAW_RATE_MAX),
        (_ANGLE_SIGN_THRESHOLD, _YAW_RATE_MIN),
    ):
        result = _call(g1_turn_admits, angle_rad=a, yaw_rate=y)
        assert result["admits"] is True, f"boundary ({a}, {y}) refused: {result['refusals']!r}"
        assert result["refusals"] == []


def test_g1_turn_admits_a_strictly_negative_angle_as_cw_turn() -> None:
    """A strictly-negative angle is admitted; the composed duration is on ``abs(angle_rad)``.

    The neon wrapper picks the sign of ``vyaw`` from the sign of
    ``angle_rad`` (``vyaw = yaw_rate if angle_rad >= 0 else -yaw_rate``);
    the magnitude the composed duration reads is ``abs(angle_rad)``.
    So an ``angle_rad=-math.pi, yaw_rate=0.3`` call is admitted and
    composes the same duration a ``angle_rad=math.pi`` call would;
    the caller who wants to distinguish CW-turn from CCW-turn reads
    the sign of the argument directly.
    """
    result = _call(g1_turn_admits, angle_rad=-math.pi, yaw_rate=0.3)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []
    assert result["composed_duration"] == pytest.approx(math.pi / 0.3)


def test_g1_turn_admits_an_angle_beyond_the_ceiling() -> None:
    """An angle above ``angle_rad_max`` refuses on that bound.

    The refusal descriptor names ``dimension="angle_rad"``, the
    offending value, the bound it violated, and the ``7404``
    gate-refusal code. ``composed_duration`` is ``None`` because a
    rejected pair does not yield a duration a caller could pass to
    the duration envelope.
    """
    over = _ANGLE_RAD_MAX + 0.1
    result = _call(g1_turn_admits, angle_rad=over, yaw_rate=0.3)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["composed_duration"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "angle_rad"
    assert r["value"] == over
    assert r["bound_key"] == "angle_rad_max"
    assert r["bound"] == _ANGLE_RAD_MAX
    assert r["comparison"] == "value > bound"
    assert r["code"] == _GATE_REFUSAL_CODE
    assert r["text"] == ERR_CODES[_GATE_REFUSAL_CODE]


def test_g1_turn_admits_a_yaw_rate_below_the_floor() -> None:
    """A yaw rate below ``yaw_rate_min`` refuses on that bound.

    The neon wrapper's ``max(0.1, ...)`` clamps a stall-rate input
    up to the turnable minimum; this verb refuses instead so the
    caller sees the bound rather than the silent clamp. A negative
    ``yaw_rate`` also lands here (the admits helper reads
    ``yaw_rate`` directly, not through ``abs()`` -- the neon-
    wrapper's sign fold is documented on the module-level constant).
    """
    result = _call(g1_turn_admits, angle_rad=0.5, yaw_rate=0.05)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["composed_duration"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "yaw_rate"
    assert r["bound_key"] == "yaw_rate_min"
    assert r["bound"] == _YAW_RATE_MIN
    assert r["comparison"] == "value < bound"


def test_g1_turn_admits_names_both_violated_bounds() -> None:
    """A pair violating both axes surfaces both refusals.

    A caller passing an out-of-range angle and an out-of-range
    yaw-rate sees two refusal descriptors, one per axis; the shape
    lets a caller decide the refusal message without composing
    which axis's bound was hit first.
    """
    result = _call(g1_turn_admits, angle_rad=_ANGLE_RAD_MAX + 0.1, yaw_rate=_YAW_RATE_MAX + 0.1)
    assert result["admits"] is False
    assert result["composed_duration"] is None
    dims = {r["dimension"] for r in result["refusals"]}
    assert dims == {"angle_rad", "yaw_rate"}


@pytest.mark.parametrize("bad_angle", [math.inf, -math.inf, math.nan])
def test_g1_turn_admits_refuses_non_finite_angle(bad_angle: float) -> None:
    """``math.inf`` / ``-math.inf`` / ``math.nan`` on the angle axis refuse.

    A NaN cannot be compared decidably against the clamps
    (``nan < min`` is ``False`` but ``nan >= min`` is also ``False``,
    so neither branch admits it), and an infinity would overrun the
    outer angle bound and compose an unbounded duration; both are
    shape violations rather than value ones. Named on the refusal
    descriptor with ``comparison="non-finite"``.
    """
    result = _call(g1_turn_admits, angle_rad=bad_angle, yaw_rate=0.3)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["composed_duration"] is None
    dims = {r["dimension"]: r for r in result["refusals"]}
    assert "angle_rad" in dims
    assert dims["angle_rad"]["comparison"] == "non-finite"
    assert dims["angle_rad"]["code"] == _GATE_REFUSAL_CODE


@pytest.mark.parametrize("bad_yaw", [math.inf, -math.inf, math.nan])
def test_g1_turn_admits_refuses_non_finite_yaw_rate(bad_yaw: float) -> None:
    """``math.inf`` / ``-math.inf`` / ``math.nan`` on the yaw-rate axis refuse.

    Symmetric partner to the angle-side non-finite check. A non-
    finite yaw rate would either divide-by-zero (``nan`` and ``inf``
    both propagate through the composed-duration division in shape-
    violating ways) or route the controller a bogus yaw command;
    both are shape violations rather than value ones.
    """
    result = _call(g1_turn_admits, angle_rad=0.5, yaw_rate=bad_yaw)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["composed_duration"] is None
    dims = {r["dimension"]: r for r in result["refusals"]}
    assert "yaw_rate" in dims
    assert dims["yaw_rate"]["comparison"] == "non-finite"


def test_g1_turn_admits_default_call_is_neon_shape() -> None:
    """Calling with no arguments admits a conservative default pair.

    The verb's defaults ``angle_rad=0.5, yaw_rate=0.3`` sit strictly
    inside both clamps, so a zero-arg invocation ("what would a
    conservative turn do?") returns ``admits=True`` with a finite
    ``composed_duration``. Pinned so a change to the defaults that
    silently shifted the conservative-query answer surfaces here.
    """
    result = _call(g1_turn_admits)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []
    assert result["composed_duration"] == pytest.approx(0.5 / 0.3)


def test_g1_turn_admits_composed_duration_overhangs_the_duration_envelope() -> None:
    """The maximum-composed-duration case overhangs the duration envelope's clamp.

    A turn call at ``angle_rad=2*pi, yaw_rate=0.1`` composes to a
    ``~62.83`` s duration, which is above the locomotion-duration
    envelope's ``10.0`` s upper clamp (port #2972). The pair is
    admitted at the turn envelope layer (both arguments sit at their
    own clamps' inner boundaries), and the ``composed_duration`` the
    verb returns names the overhang so a caller comparing this
    envelope's admission against the duration envelope's admission
    sees where the two disagree. Pinned so a widen of either the
    turn or the duration clamps that closed the overhang surfaces
    here.
    """
    result = _call(g1_turn_admits, angle_rad=_ANGLE_RAD_MAX, yaw_rate=_YAW_RATE_MIN)
    assert result["admits"] is True
    assert result["composed_duration"] == pytest.approx(_MAX_COMPOSED_DURATION)
    assert result["composed_duration"] == pytest.approx((2.0 * math.pi) / 0.1)
