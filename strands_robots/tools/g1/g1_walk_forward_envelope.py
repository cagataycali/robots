"""Agent-facing lookup for the walk-forward envelope ``LocoClient.SetVelocity`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes
``SetVelocity(vx, vy, vyaw, duration_s)`` as the bounded write for a
straight-line walk, and the neon bundle
(``cagataycali/neon-the-g1/tools/g1_locomotion.py::g1_walk_forward``)
fronts it with a two-argument surface -- ``distance`` in metres and
``speed`` in m/s -- that composes to a ``(vx, 0.0, 0.0)`` velocity
command with a computed ``duration = abs(distance) / speed`` and the
sign of ``vx`` picked from the sign of ``distance``. The SDK itself
places *no* clamps on either input: a caller that passes
``distance=100.0`` or ``speed=5.0`` reaches the controller unchanged,
and the controller's behaviour above the neon-bundle-observed range
is undefined -- the G1 has no runaway guard on that write path. The
neon bundle's own wrapper narrows the two arguments to
``distance = max(-1.0, min(1.0, float(distance)))`` and
``speed = max(0.05, min(0.5, abs(float(speed))))`` before dispatch,
so this module snapshots that clamp pair into module-level constants
and exposes two agent-facing verbs --
:func:`g1_list_walk_forward_envelope` (name the whole envelope) and
:func:`g1_walk_forward_admits` (decide one query) -- so a caller can
decide the refusal decidably before a future driver-side wrapper
fires. Refs strands-labs/robots#358.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_walk_forward`` verb
  composes ``(distance, speed)`` into a ``SetVelocity`` call under
  the same DDS singleton :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
  the driver holds; that write is the same locomotion topic
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` gates, which
  today's :class:`~strands_robots.drivers.g1.G1Driver` refuses through
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` on
  any locomotion-shaped write while ``_fsm_id`` is outside that set.
  A future driver method that fronts the ``distance/speed`` surface
  will land the write verb; refs strands-labs/robots#358 for the
  SDK-facing gate work that write belongs on. This module ports the
  read-only envelope half without also introducing a second
  locomotion writer path the driver does not yet own.
* An SDK re-import. The clamp table is captured here as module-level
  constants so
  ``import strands_robots.tools.g1.g1_walk_forward_envelope`` pulls
  no ``unitree_sdk2py`` submodule -- the import-hygiene contract
  every other file in this package carries, refs
  strands-labs/robots#358. A revision of the neon bundle's observed
  bounds is a driver-side update; when the driver's walk-forward
  method lands, its refusal will quote the same ``7404`` code the
  entry in :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
  carries.

What this module does not decide.

* Whether the driver's live ``_fsm_id`` is currently inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That is a
  driver-instance read carried on the driver's ``get_status``
  envelope; a caller planning a walk-forward command compares the
  driver's live fsm against
  :func:`g1_list_walk_forward_envelope`'s ``walk_ready_fsm_ids`` to
  see whether the write gate is currently open. The three membership
  tests together -- envelope for distance, envelope for speed,
  walk_ready for the gate -- are the conditions a future write verb
  would refuse on.
* Whether the composed ``(vx, 0.0, 0.0)`` velocity magnitude sits
  inside the velocity envelope. That is a separate lookup
  (:mod:`~strands_robots.tools.g1.g1_velocity_envelope`, port #2965,
  refs strands-labs/robots#358). The neon wrapper's ``speed`` clamp
  (``[0.05, 0.5]``) already sits strictly inside the velocity
  envelope's ``vx`` clamp, so a value admitted here is also admitted
  there; a caller that wants to reach the velocity envelope's wider
  ``vx`` range must call :mod:`~strands_robots.tools.g1.g1_velocity_envelope`
  directly rather than the walk-forward surface.
* Whether the composed ``duration = abs(distance) / speed`` sits
  inside the duration envelope. That is a separate lookup
  (``g1_locomotion_duration_envelope``, port #2972,
  refs strands-labs/robots#358). At the walk-forward
  clamps the composed duration is bounded to ``[0.0 / 0.5, 1.0 / 0.05]``
  = ``[0.0, 20.0]`` seconds, which overhangs the duration envelope's
  ``[0.0, 10.0]`` upper clamp: a walk-forward call at ``distance=1.0,
  speed=0.05`` composes to a 20-second walk the duration envelope
  would clamp down to 10.0 s at the ``g1_move_velocity`` layer. The
  overhang is named on the returned envelope so a caller comparing
  the two admission layers sees it without composing the arithmetic
  itself.
"""

from __future__ import annotations

import math
from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS

#: The lower clamp the neon ``g1_walk_forward`` verb places on the
#: ``distance`` argument before dispatch. The value is the one the
#: neon bundle's own wrapper (``max(-1.0, min(1.0, float(distance)))``)
#: rounds up to for any value below ``-1.0``. Named as an inclusive
#: lower bound because a saturated backwards-walk command is still a
#: command; refusing it would drop the neon bundle's own admitted
#: range. A negative ``distance`` is admitted because the neon
#: wrapper picks the sign of ``vx`` from the sign of ``distance``
#: (``vx = speed if distance >= 0 else -speed``), so a strictly-negative
#: ``distance`` reaches the controller as a backwards walk rather than
#: as a shape violation.
_DISTANCE_MIN: float = -1.0

#: The upper clamp the neon ``g1_walk_forward`` verb places on the
#: ``distance`` argument before dispatch. The value is the one the
#: neon bundle's own wrapper rounds down to for any value above
#: ``1.0``. Above this the controller's response is undefined (the
#: SDK does not clamp), and the composed duration
#: (``abs(distance) / speed``) grows without bound -- a caller passing
#: ``distance=100.0`` would walk the robot for hours at the neon
#: bundle's minimum speed. Named as an inclusive upper bound because
#: a saturated forward-walk command is a command.
_DISTANCE_MAX: float = 1.0

#: The lower clamp the neon ``g1_walk_forward`` verb places on the
#: ``speed`` argument before dispatch. The value is the one the neon
#: bundle's own wrapper (``max(0.05, min(0.5, abs(float(speed))))``)
#: rounds up to for any strictly-positive value below ``0.05``. The
#: neon wrapper takes the absolute value first, so a negative
#: ``speed`` is folded to positive before the clamp; the sign of the
#: composed ``vx`` comes from the sign of ``distance``, not from the
#: sign of ``speed``. Named as an inclusive lower bound because
#: ``0.05`` m/s is the minimum walkable speed the neon bundle
#: observed on the G1; below this the controller stalls rather than
#: walks.
_SPEED_MIN: float = 0.05

#: The upper clamp the neon ``g1_walk_forward`` verb places on the
#: ``speed`` argument before dispatch. The value is the one the neon
#: bundle's own wrapper rounds down to for any value above ``0.5``.
#: Above this the controller's response is undefined (the SDK does
#: not clamp), and the neon bundle never observed a stable gait
#: above this speed on a walkable surface. Named as an inclusive
#: upper bound because a saturated speed command is a command.
_SPEED_MAX: float = 0.5

#: The zero of the sign axis the neon wrapper reads to pick the
#: direction of ``vx``. The neon bundle's conditional is
#: ``vx = speed if distance >= 0 else -speed``, so ``distance == 0.0``
#: (and ``distance == -0.0``, which Python's ``>=`` comparison reads
#: as non-negative) routes to forward-walk at ``vx = +speed`` and
#: composes to a zero-duration no-op the SDK does not refuse.
#: Named here so a future revision of the neon wrapper that changed
#: the sign convention (for example to ``distance > 0`` strict, which
#: would route ``0.0`` to backwards-walk) lands as a shape change on
#: this constant rather than as a silent divergence in the tests.
_DISTANCE_SIGN_THRESHOLD: float = 0.0

#: The upper bound of the composed ``duration = abs(distance) / speed``
#: at the walk-forward clamps: ``_DISTANCE_MAX / _SPEED_MIN``
#: = ``1.0 / 0.05`` = ``20.0`` seconds. Above the neon bundle's
#: locomotion-duration envelope's own ``10.0`` s upper clamp
#: (``g1_locomotion_duration_envelope``, port #2972), so a
#: walk-forward call at ``distance=1.0, speed=0.05``
#: composes to a duration the duration envelope would clamp down to
#: ``10.0`` s at the ``g1_move_velocity`` layer. Named on the
#: returned envelope so a caller comparing the two admission layers
#: sees the overhang without composing the arithmetic itself.
_MAX_COMPOSED_DURATION: float = 20.0

#: The lower bound of the composed ``duration = abs(distance) / speed``
#: at the walk-forward clamps: ``0.0 / _SPEED_MAX`` = ``0.0`` seconds.
#: The neon bundle's ``g1_move_velocity`` verb refuses on
#: ``duration <= 0`` before the SDK is touched, but the walk-forward
#: surface passes the composed value through untransformed, so a
#: caller at ``distance=0.0`` reaches that refusal at the underlying
#: layer rather than at the walk-forward surface. Named so a caller
#: comparing the two admission layers sees the shape of the composed
#: duration without composing it itself.
_MIN_COMPOSED_DURATION: float = 0.0

#: The SDK method the neon bundle's ``g1_walk_forward`` dispatches
#: through. The neon wrapper calls ``g1_move_velocity`` internally,
#: which fronts ``LocoClient.SetVelocity`` under the same
#: single-writer DDS singleton the driver holds. Named here so the
#: returned envelope carries the exact SDK entry a driver-side
#: wrapper would target, and so a firmware release that renamed the
#: SDK method lands in one place instead of drifting between the
#: neon bundle and this lookup.
_SDK_METHOD: str = "SetVelocity"

#: The error-table entry the driver's own ``_check_motion_gates``
#: quotes when it refuses a locomotion-shaped write on an FSM outside
#: :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. Named here
#: so the returned envelope carries the exact refusal string a future
#: driver-side walk-forward wrapper would surface, and so a
#: re-wording of it lands in one place instead of drifting between
#: the driver's log and this lookup. The write path and this lookup
#: share the constant.
_GATE_REFUSAL_CODE: int = 7404


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_walk_forward_envelope`
    so :func:`g1_walk_forward_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands in
    one place. Every field is a snapshot read; no bus is touched.
    """
    return {
        "distance_min": _DISTANCE_MIN,
        "distance_max": _DISTANCE_MAX,
        "speed_min": _SPEED_MIN,
        "speed_max": _SPEED_MAX,
        "distance_sign_threshold": _DISTANCE_SIGN_THRESHOLD,
        "composed_duration_min": _MIN_COMPOSED_DURATION,
        "composed_duration_max": _MAX_COMPOSED_DURATION,
        "sdk_method": _SDK_METHOD,
    }


@tool
def g1_list_walk_forward_envelope() -> dict[str, Any]:
    """Return the walk-forward envelope the neon bundle clamps to.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for the walk-forward surface is called, so a caller can compare
    an intended ``(distance, speed)`` pair against the clamps the
    neon bundle's ``g1_walk_forward`` narrows to, and can also
    compare the driver's live ``fsm_id`` (from
    ``G1Driver.get_status``) against ``walk_ready_fsm_ids`` to decide
    whether the locomotion write gate is currently open.

    The envelope names two argument clamps the neon bundle stacks on
    top of the SDK: ``[distance_min, distance_max]`` = ``[-1.0, 1.0]``
    metres (a negative ``distance`` routes to backwards-walk; the sign
    of the composed ``vx`` is picked from the sign of ``distance``
    against ``distance_sign_threshold``), and
    ``[speed_min, speed_max]`` = ``[0.05, 0.5]`` m/s (the neon wrapper
    takes ``abs(speed)`` before clamping, so a negative ``speed`` is
    folded to positive; the direction comes from ``distance``, not
    from ``speed``). The composed
    ``duration = abs(distance) / speed`` sits in
    ``[composed_duration_min, composed_duration_max]`` = ``[0.0, 20.0]``
    seconds at the walk-forward clamps, which overhangs the
    locomotion-duration envelope's own ``10.0`` s upper clamp -- a
    walk-forward call at ``distance=1.0, speed=0.05`` composes to a
    ``20.0`` s duration the duration envelope would clamp down to
    ``10.0`` s at the ``g1_move_velocity`` layer.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        both argument clamps, the sign threshold, the composed-
        duration range, and the SDK method name
        (``distance_min``, ``distance_max``, ``speed_min``,
        ``speed_max``, ``distance_sign_threshold``,
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
    validator; the positivity half does not apply on the ``distance``
    axis (a strictly-negative ``distance`` is a legitimate
    backwards-walk command the neon wrapper admits), and on the
    ``speed`` axis the neon wrapper takes ``abs(speed)`` before
    clamping so a shared positive-finite validator would refuse a
    negative ``speed`` the neon wrapper would silently fold to
    positive. A future consolidation with the shared validator lands
    when the driver-side write verb reuses this admits function and
    the sign-handling contract is settled with #358.
    """
    return math.isfinite(float(value))


@tool
def g1_walk_forward_admits(distance: float = 0.3, speed: float = 0.2) -> dict[str, Any]:
    """Decide whether a ``(distance, speed)`` pair sits inside the walk-forward envelope.

    Read-only. Compares each argument against the clamps
    :func:`g1_list_walk_forward_envelope` returns and reports whether
    the neon bundle's wrapper would dispatch the pair unchanged (it
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

    * ``distance`` inside ``[distance_min, distance_max]`` and
      ``speed`` inside ``[speed_min, speed_max]`` -> admitted. Both
      bounds are inclusive on both axes (a saturated command is a
      command); the composed ``vx`` sign is picked from
      ``distance >= distance_sign_threshold`` (``+speed`` on
      non-negative distance, ``-speed`` on strictly-negative).
    * ``distance`` outside the distance clamp OR ``speed`` outside
      the speed clamp -> refused. Every violated bound is named on
      its own refusal descriptor so a caller sees which axis (and
      which bound on that axis) blocked the pair; a single call can
      surface up to four refusals if both arguments violate both of
      their bounds (which cannot happen with finite floats but the
      shape is preserved for the non-finite case).
    * Either argument non-finite (``math.inf``, ``math.nan``) ->
      refused with ``comparison="non-finite"`` on that axis's
      refusal. A NaN cannot be compared decidably against the
      clamps (``nan < min`` is ``False`` but so is ``nan >= min``),
      and an infinity would overrun the outer bound on either axis;
      both are shape violations rather than value ones.

    Args:
        distance: The metres-to-travel argument the neon
            ``g1_walk_forward`` verb clamps to
            ``[distance_min, distance_max]`` before dispatch. A
            strictly-negative ``distance`` routes to backwards-walk
            (``vx = -speed``) rather than to a shape violation.
        speed: The forward-velocity argument the neon
            ``g1_walk_forward`` verb clamps to ``[speed_min, speed_max]``
            after taking ``abs(speed)``. This admits helper reads
            ``speed`` directly rather than through ``abs()``, so a
            negative ``speed`` refuses on the ``speed_min`` bound;
            a caller who means to fold the sign should pass
            ``abs(speed)`` at the call site to match the neon
            wrapper's behaviour.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        both arguments sit inside their clamps; a ``refusals`` list
        carrying one refusal descriptor per violated bound (each
        with the offending dimension ``"distance"`` or ``"speed"``,
        the offending value, the bound it violated, the comparison,
        and the ``7404`` gate-refusal code the driver would quote);
        a ``composed_duration`` float naming what
        ``abs(distance) / speed`` would evaluate to at the arguments
        as-passed (``None`` on a rejected pair, and ``None`` on
        ``speed == 0.0`` to avoid a divide-by-zero -- a caller can
        distinguish the two by inspecting ``refusals``); the same
        ``envelope`` sub-dict :func:`g1_list_walk_forward_envelope`
        returns; and ``walk_ready_fsm_ids`` for the follow-on gate
        decision. On an admitted pair the ``refusals`` list is empty
        and ``composed_duration`` is a finite float; on a rejected
        pair every violated bound is named.
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

    if not _finite(distance):
        _reject("distance", distance, "distance_max", _DISTANCE_MAX, "non-finite")
    else:
        d = float(distance)
        if d < _DISTANCE_MIN:
            _reject("distance", d, "distance_min", _DISTANCE_MIN, "value < bound")
        elif d > _DISTANCE_MAX:
            _reject("distance", d, "distance_max", _DISTANCE_MAX, "value > bound")

    if not _finite(speed):
        _reject("speed", speed, "speed_max", _SPEED_MAX, "non-finite")
    else:
        s = float(speed)
        if s < _SPEED_MIN:
            _reject("speed", s, "speed_min", _SPEED_MIN, "value < bound")
        elif s > _SPEED_MAX:
            _reject("speed", s, "speed_max", _SPEED_MAX, "value > bound")

    # Composed duration reported only on an admitted pair with a
    # non-zero speed; a caller who wants to reach the duration
    # envelope with a rejected pair reads the refusals list first.
    composed_duration: float | None = None
    if not refusals:
        s = float(speed)
        if s != 0.0:
            composed_duration = abs(float(distance)) / s

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "composed_duration": composed_duration,
        "envelope": envelope,
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
    }
