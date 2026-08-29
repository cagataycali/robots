"""Agent-facing lookup for the turn envelope ``LocoClient.SetVelocity`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes
``SetVelocity(vx, vy, vyaw, duration_s)`` as the bounded write for a
yaw-only turn-in-place, and the neon bundle
(``cagataycali/neon-the-g1/tools/g1_locomotion.py::g1_turn``) fronts
it with a two-argument surface -- ``angle_rad`` in radians and
``yaw_rate`` in rad/s -- that composes to a ``(0.0, 0.0, vyaw)``
velocity command with a computed ``duration = abs(angle_rad) / yaw_rate``
and the sign of ``vyaw`` picked from the sign of ``angle_rad``. The
SDK itself places *no* clamps on either input: a caller that passes
``angle_rad=100.0`` or ``yaw_rate=5.0`` reaches the controller
unchanged, and the controller's behaviour above the neon-bundle-
observed range is undefined -- the G1 has no runaway guard on that
write path. The neon bundle's own wrapper narrows the two arguments
to ``angle_rad = max(-2*pi, min(2*pi, float(angle_rad)))`` and
``yaw_rate = max(0.1, min(0.6, abs(float(yaw_rate))))`` before
dispatch, so this module snapshots that clamp pair into module-level
constants and exposes two agent-facing verbs --
:func:`g1_list_turn_envelope` (name the whole envelope) and
:func:`g1_turn_admits` (decide one query) -- so a caller can decide
the refusal decidably before a future driver-side wrapper fires.
Refs strands-labs/robots#358.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_turn`` verb composes
  ``(angle_rad, yaw_rate)`` into a ``SetVelocity`` call under the
  same DDS singleton :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
  the driver holds; that write is the same locomotion topic
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` gates, which
  today's :class:`~strands_robots.drivers.g1.G1Driver` refuses through
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` on
  any locomotion-shaped write while ``_fsm_id`` is outside that set.
  A future driver method that fronts the ``angle_rad/yaw_rate``
  surface will land the write verb; refs strands-labs/robots#358 for
  the SDK-facing gate work that write belongs on. This module ports
  the read-only envelope half without also introducing a second
  locomotion writer path the driver does not yet own.
* An SDK re-import. The clamp table is captured here as module-level
  constants so ``import strands_robots.tools.g1.g1_turn_envelope``
  pulls no ``unitree_sdk2py`` submodule -- the import-hygiene
  contract every other file in this package carries, refs
  strands-labs/robots#358. A revision of the neon bundle's observed
  bounds is a driver-side update; when the driver's turn method
  lands, its refusal will quote the same ``7404`` code the entry in
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.

What this module does not decide.

* Whether the driver's live ``_fsm_id`` is currently inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That is a
  driver-instance read carried on the driver's ``get_status``
  envelope; a caller planning a turn command compares the driver's
  live fsm against :func:`g1_list_turn_envelope`'s
  ``walk_ready_fsm_ids`` to see whether the write gate is currently
  open. The three membership tests together -- envelope for angle,
  envelope for yaw_rate, walk_ready for the gate -- are the
  conditions a future write verb would refuse on.
* Whether the composed ``(0.0, 0.0, vyaw)`` yaw magnitude sits inside
  the velocity envelope. That is a separate lookup
  (:mod:`~strands_robots.tools.g1.g1_velocity_envelope`, port #2965,
  refs strands-labs/robots#358). The neon wrapper's ``yaw_rate``
  clamp (``[0.1, 0.6]``) already sits strictly inside the velocity
  envelope's ``vyaw`` clamp, so a value admitted here is also
  admitted there; a caller that wants to reach the velocity
  envelope's wider ``vyaw`` range must call
  :mod:`~strands_robots.tools.g1.g1_velocity_envelope` directly
  rather than the turn surface.
* Whether the composed ``duration = abs(angle_rad) / yaw_rate`` sits
  inside the duration envelope. That is a separate lookup
  (``g1_locomotion_duration_envelope``, port #2972, refs
  strands-labs/robots#358). At the turn clamps the
  composed duration is bounded to ``[0.0 / 0.6, 2*pi / 0.1]``
  = ``[0.0, ~62.83]`` seconds, which overhangs the duration
  envelope's ``[0.0, 10.0]`` upper clamp: a turn call at
  ``angle_rad=2*pi, yaw_rate=0.1`` composes to a ~63-second turn
  the duration envelope would clamp down to 10.0 s at the
  ``g1_move_velocity`` layer. The overhang is named on the returned
  envelope so a caller comparing the two admission layers sees it
  without composing the arithmetic itself.
"""

from __future__ import annotations

import math
from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS

#: The lower clamp the neon ``g1_turn`` verb places on the
#: ``angle_rad`` argument before dispatch. The value is the one the
#: neon bundle's own wrapper
#: (``max(-2*pi, min(2*pi, float(angle_rad)))``) rounds up to for
#: any value below ``-2*pi``. Named as an inclusive lower bound
#: because a saturated clockwise-turn command is still a command;
#: refusing it would drop the neon bundle's own admitted range. A
#: negative ``angle_rad`` is admitted because the neon wrapper picks
#: the sign of ``vyaw`` from the sign of ``angle_rad``
#: (``vyaw = yaw_rate if angle_rad >= 0 else -yaw_rate``), so a
#: strictly-negative ``angle_rad`` reaches the controller as a
#: clockwise (CW) turn rather than as a shape violation.
_ANGLE_RAD_MIN: float = -2.0 * math.pi

#: The upper clamp the neon ``g1_turn`` verb places on the
#: ``angle_rad`` argument before dispatch. The value is the one the
#: neon bundle's own wrapper rounds down to for any value above
#: ``2*pi``. Above this the controller's response is undefined (the
#: SDK does not clamp), and the composed duration
#: (``abs(angle_rad) / yaw_rate``) grows without bound -- a caller
#: passing ``angle_rad=100.0`` would turn the robot for many minutes
#: at the neon bundle's minimum yaw rate. Named as an inclusive
#: upper bound because a saturated counter-clockwise-turn command is
#: a command; the neon bundle picks positive-angle as CCW per the
#: G1's right-handed body frame.
_ANGLE_RAD_MAX: float = 2.0 * math.pi

#: The lower clamp the neon ``g1_turn`` verb places on the
#: ``yaw_rate`` argument before dispatch. The value is the one the
#: neon bundle's own wrapper
#: (``max(0.1, min(0.6, abs(float(yaw_rate))))``) rounds up to for
#: any strictly-positive value below ``0.1``. The neon wrapper takes
#: the absolute value first, so a negative ``yaw_rate`` is folded to
#: positive before the clamp; the sign of the composed ``vyaw``
#: comes from the sign of ``angle_rad``, not from the sign of
#: ``yaw_rate``. Named as an inclusive lower bound because ``0.1``
#: rad/s is the minimum turnable yaw rate the neon bundle observed
#: on the G1; below this the controller stalls rather than turns.
_YAW_RATE_MIN: float = 0.1

#: The upper clamp the neon ``g1_turn`` verb places on the
#: ``yaw_rate`` argument before dispatch. The value is the one the
#: neon bundle's own wrapper rounds down to for any value above
#: ``0.6``. Above this the controller's response is undefined (the
#: SDK does not clamp), and the neon bundle never observed a stable
#: turn-in-place above this rate on a walkable surface. Named as an
#: inclusive upper bound because a saturated yaw-rate command is a
#: command.
_YAW_RATE_MAX: float = 0.6

#: The zero of the sign axis the neon wrapper reads to pick the
#: direction of ``vyaw``. The neon bundle's conditional is
#: ``vyaw = yaw_rate if angle_rad >= 0 else -yaw_rate``, so
#: ``angle_rad == 0.0`` (and ``angle_rad == -0.0``, which Python's
#: ``>=`` comparison reads as non-negative) routes to CCW-turn at
#: ``vyaw = +yaw_rate`` and composes to a zero-duration no-op the
#: SDK does not refuse. Named here so a future revision of the neon
#: wrapper that changed the sign convention (for example to
#: ``angle_rad > 0`` strict, which would route ``0.0`` to CW-turn)
#: lands as a shape change on this constant rather than as a silent
#: divergence in the tests.
_ANGLE_SIGN_THRESHOLD: float = 0.0

#: The upper bound of the composed ``duration = abs(angle_rad) / yaw_rate``
#: at the turn clamps: ``_ANGLE_RAD_MAX / _YAW_RATE_MIN``
#: = ``2*pi / 0.1`` ≈ ``62.832`` seconds. Above the neon bundle's
#: locomotion-duration envelope's own ``10.0`` s upper clamp
#: (``g1_locomotion_duration_envelope``, port #2972), so a turn call at ``angle_rad=2*pi, yaw_rate=0.1``
#: composes to a duration the duration envelope would clamp down to
#: ``10.0`` s at the ``g1_move_velocity`` layer. Named on the
#: returned envelope so a caller comparing the two admission layers
#: sees the overhang without composing the arithmetic itself.
_MAX_COMPOSED_DURATION: float = (2.0 * math.pi) / 0.1

#: The lower bound of the composed ``duration = abs(angle_rad) / yaw_rate``
#: at the turn clamps: ``0.0 / _YAW_RATE_MAX`` = ``0.0`` seconds.
#: The neon bundle's ``g1_move_velocity`` verb refuses on
#: ``duration <= 0`` before the SDK is touched, but the turn surface
#: passes the composed value through untransformed, so a caller at
#: ``angle_rad=0.0`` reaches that refusal at the underlying layer
#: rather than at the turn surface. Named so a caller comparing the
#: two admission layers sees the shape of the composed duration
#: without composing it itself.
_MIN_COMPOSED_DURATION: float = 0.0

#: The SDK method the neon bundle's ``g1_turn`` dispatches through.
#: The neon wrapper calls ``g1_move_velocity`` internally, which
#: fronts ``LocoClient.SetVelocity`` under the same single-writer
#: DDS singleton the driver holds. Named here so the returned
#: envelope carries the exact SDK entry a driver-side wrapper would
#: target, and so a firmware release that renamed the SDK method
#: lands in one place instead of drifting between the neon bundle
#: and this lookup.
_SDK_METHOD: str = "SetVelocity"

#: The error-table entry the driver's own ``_check_motion_gates``
#: quotes when it refuses a locomotion-shaped write on an FSM outside
#: :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. Named here
#: so the returned envelope carries the exact refusal string a future
#: driver-side turn wrapper would surface, and so a re-wording of it
#: lands in one place instead of drifting between the driver's log
#: and this lookup. The write path and this lookup share the constant.
_GATE_REFUSAL_CODE: int = 7404


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_turn_envelope`
    so :func:`g1_turn_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands in
    one place. Every field is a snapshot read; no bus is touched.
    """
    return {
        "angle_rad_min": _ANGLE_RAD_MIN,
        "angle_rad_max": _ANGLE_RAD_MAX,
        "yaw_rate_min": _YAW_RATE_MIN,
        "yaw_rate_max": _YAW_RATE_MAX,
        "angle_sign_threshold": _ANGLE_SIGN_THRESHOLD,
        "composed_duration_min": _MIN_COMPOSED_DURATION,
        "composed_duration_max": _MAX_COMPOSED_DURATION,
        "sdk_method": _SDK_METHOD,
    }


@tool
def g1_list_turn_envelope() -> dict[str, Any]:
    """Return the turn envelope the neon bundle clamps to.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for the turn surface is called, so a caller can compare an
    intended ``(angle_rad, yaw_rate)`` pair against the clamps the
    neon bundle's ``g1_turn`` narrows to, and can also compare the
    driver's live ``fsm_id`` (from ``G1Driver.get_status``) against
    ``walk_ready_fsm_ids`` to decide whether the locomotion write
    gate is currently open.

    The envelope names two argument clamps the neon bundle stacks
    on top of the SDK: ``[angle_rad_min, angle_rad_max]`` =
    ``[-2*pi, 2*pi]`` rad (a negative ``angle_rad`` routes to CW
    turn; the sign of the composed ``vyaw`` is picked from the sign
    of ``angle_rad`` against ``angle_sign_threshold``), and
    ``[yaw_rate_min, yaw_rate_max]`` = ``[0.1, 0.6]`` rad/s (the neon
    wrapper takes ``abs(yaw_rate)`` before clamping, so a negative
    ``yaw_rate`` is folded to positive; the direction comes from
    ``angle_rad``, not from ``yaw_rate``). The composed
    ``duration = abs(angle_rad) / yaw_rate`` sits in
    ``[composed_duration_min, composed_duration_max]`` =
    ``[0.0, 2*pi/0.1]`` seconds at the turn clamps, which overhangs
    the locomotion-duration envelope's own ``10.0`` s upper clamp --
    a turn call at ``angle_rad=2*pi, yaw_rate=0.1`` composes to a
    ~62.83 s duration the duration envelope would clamp down to
    ``10.0`` s at the ``g1_move_velocity`` layer.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        both argument clamps, the sign threshold, the composed-
        duration range, and the SDK method name
        (``angle_rad_min``, ``angle_rad_max``, ``yaw_rate_min``,
        ``yaw_rate_max``, ``angle_sign_threshold``,
        ``composed_duration_min``, ``composed_duration_max``,
        ``sdk_method``); a ``walk_ready_fsm_ids`` list quoting
        :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, the
        set the driver's motion gate admits locomotion-shaped writes
        on; and a ``refusals`` list carrying the ``7404`` gate-
        refused code and its decoded text, the one a future write
        verb would surface. Every field is a snapshot of an observed
        bound or a driver constant; no dynamic decode runs here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
        "refusals": [
            {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
        ],
    }


def _finite(value: float) -> bool:
    """Return whether ``value`` is a finite float.

    Kept here rather than pulled from ``strands_robots.utils`` because
    the envelope check needs only the finiteness half of the shared
    validator; the positivity half does not apply on the
    ``angle_rad`` axis (a strictly-negative ``angle_rad`` is a
    legitimate CW-turn command the neon wrapper admits), and on the
    ``yaw_rate`` axis the neon wrapper takes ``abs(yaw_rate)`` before
    clamping so a shared positive-finite validator would refuse a
    negative ``yaw_rate`` the neon wrapper would silently fold to
    positive. A future consolidation with the shared validator lands
    when the driver-side write verb reuses this admits function and
    the sign-handling contract is settled with #358.
    """
    return math.isfinite(float(value))


@tool
def g1_turn_admits(angle_rad: float = 0.5, yaw_rate: float = 0.3) -> dict[str, Any]:
    """Decide whether an ``(angle_rad, yaw_rate)`` pair sits inside the turn envelope.

    Read-only. Compares each argument against the clamps
    :func:`g1_list_turn_envelope` returns and reports whether the
    neon bundle's wrapper would dispatch the pair unchanged (it
    clamps silently for out-of-range values, but this verb surfaces
    the refusal so the caller sees which bound would be hit). No
    driver instance, no DDS, no SDK: the decision reads only module-
    level constants and the two arguments themselves.

    A pair inside both clamps is *not* the same as an admitted write:
    the driver's motion gate (``_check_motion_gates``) also refuses
    on ``_fsm_id`` outside
    :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, which this
    verb does not read (that is a live driver-instance query answered
    by :mod:`~strands_robots.tools.g1.g1_state` and
    :mod:`~strands_robots.tools.g1.g1_motion_gates`). The returned
    envelope names ``walk_ready_fsm_ids`` so a caller comparing an
    intended write against both conditions has the FSM set on hand.

    The routing conventions:

    * ``angle_rad`` inside ``[angle_rad_min, angle_rad_max]`` and
      ``yaw_rate`` inside ``[yaw_rate_min, yaw_rate_max]`` ->
      admitted. Both bounds are inclusive on both axes (a saturated
      command is a command); the composed ``vyaw`` sign is picked
      from ``angle_rad >= angle_sign_threshold`` (``+yaw_rate`` on
      non-negative angle, ``-yaw_rate`` on strictly-negative).
    * ``angle_rad`` outside the angle clamp OR ``yaw_rate`` outside
      the yaw-rate clamp -> refused. Every violated bound is named
      on its own refusal descriptor so a caller sees which axis
      (and which bound on that axis) blocked the pair.
    * Either argument non-finite (``math.inf``, ``math.nan``) ->
      refused with ``comparison="non-finite"`` on that axis's
      refusal. A NaN cannot be compared decidably against the
      clamps (``nan < min`` is ``False`` but so is ``nan >= min``),
      and an infinity would overrun the outer bound on either axis;
      both are shape violations rather than value ones.

    Args:
        angle_rad: The radians-to-turn argument the neon ``g1_turn``
            verb clamps to ``[angle_rad_min, angle_rad_max]`` before
            dispatch. A strictly-negative ``angle_rad`` routes to
            CW-turn (``vyaw = -yaw_rate``) rather than to a shape
            violation.
        yaw_rate: The yaw-rate argument the neon ``g1_turn`` verb
            clamps to ``[yaw_rate_min, yaw_rate_max]`` after taking
            ``abs(yaw_rate)``. This admits helper reads ``yaw_rate``
            directly rather than through ``abs()``, so a negative
            ``yaw_rate`` refuses on the ``yaw_rate_min`` bound; a
            caller who means to fold the sign should pass
            ``abs(yaw_rate)`` at the call site to match the neon
            wrapper's behaviour.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        both arguments sit inside their clamps; a ``refusals`` list
        carrying one refusal descriptor per violated bound (each
        with the offending dimension ``"angle_rad"`` or
        ``"yaw_rate"``, the offending value, the bound it violated,
        the comparison, and the ``7404`` gate-refusal code the
        driver would quote); a ``composed_duration`` float naming
        what ``abs(angle_rad) / yaw_rate`` would evaluate to at the
        arguments as-passed (``None`` on a rejected pair, and
        ``None`` on ``yaw_rate == 0.0`` to avoid a divide-by-zero --
        a caller can distinguish the two by inspecting
        ``refusals``); the same ``envelope`` sub-dict
        :func:`g1_list_turn_envelope` returns; and
        ``walk_ready_fsm_ids`` for the follow-on gate decision. On
        an admitted pair the ``refusals`` list is empty and
        ``composed_duration`` is a finite float; on a rejected pair
        every violated bound is named.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    def _reject(dimension: str, value: float, bound_key: str, bound: float, cmp: str) -> None:
        refusals.append(
            {
                "dimension": dimension,
                "value": float(value),
                "bound_key": bound_key,
                "bound": bound,
                "comparison": cmp,
                "code": _GATE_REFUSAL_CODE,
                "text": ERR_CODES[_GATE_REFUSAL_CODE],
            }
        )

    if not _finite(angle_rad):
        _reject("angle_rad", angle_rad, "angle_rad_max", _ANGLE_RAD_MAX, "non-finite")
    else:
        a = float(angle_rad)
        if a < _ANGLE_RAD_MIN:
            _reject("angle_rad", a, "angle_rad_min", _ANGLE_RAD_MIN, "value < bound")
        elif a > _ANGLE_RAD_MAX:
            _reject("angle_rad", a, "angle_rad_max", _ANGLE_RAD_MAX, "value > bound")

    if not _finite(yaw_rate):
        _reject("yaw_rate", yaw_rate, "yaw_rate_max", _YAW_RATE_MAX, "non-finite")
    else:
        y = float(yaw_rate)
        if y < _YAW_RATE_MIN:
            _reject("yaw_rate", y, "yaw_rate_min", _YAW_RATE_MIN, "value < bound")
        elif y > _YAW_RATE_MAX:
            _reject("yaw_rate", y, "yaw_rate_max", _YAW_RATE_MAX, "value > bound")

    # Composed duration reported only on an admitted pair with a
    # non-zero yaw_rate; a caller who wants to reach the duration
    # envelope with a rejected pair reads the refusals list first.
    composed_duration: float | None = None
    if not refusals:
        y = float(yaw_rate)
        if y != 0.0:
            composed_duration = abs(float(angle_rad)) / y

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "composed_duration": composed_duration,
        "envelope": envelope,
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
    }
