"""Agent-facing lookup for the velocity envelope ``LocoClient.SetVelocity`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes a
continuous velocity setter via ``SetVelocity(vx, vy, vyaw, duration)`` and
its ``Move(vx, vy, vyaw, continous_move=True)`` cousin. The SDK itself
places *no* clamps on those arguments: a caller that passes ``vx=10.0`` or
``duration=86400`` reaches the controller unchanged, and the controller's
own behaviour above those numbers is undefined - the G1 has no runaway
guard on that write path. The neon bundle's ``g1_move_velocity`` verb
(``cagataycali/neon-the-g1/tools/g1_locomotion.py``) fronts the same call
under a fixed set of clamps observed against the real robot on a gantry:
``vx``/``vy`` treated as speed magnitudes bounded by
:data:`_VX_ABS_MAX`/:data:`_VY_ABS_MAX`, ``vyaw`` by
:data:`_VYAW_ABS_MAX`, and ``duration`` bounded by
:data:`_DURATION_MAX_SECONDS`. This module surfaces the envelope to an
agent so a caller can decide the refusal decidably before a future
driver-side wrapper fires, rather than pinning it inside the write path
where the refusal is invisible to the planner.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_move_velocity`` verb wrapped
  ``LocoClient.SetVelocity`` under a single-writer lock; that write is
  the same ``rt/lowcmd``-adjacent locomotion topic
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` gates, which
  today's :class:`~strands_robots.drivers.g1.G1Driver` refuses through
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` on any
  arm-SDK-shaped write while ``_fsm_id`` is outside that set. A future
  driver method that fronts ``SetVelocity`` will land the write verb;
  refs strands-labs/robots#358 for the SDK-facing gate work that write
  belongs on. This module ports the read-only envelope half without also
  introducing a second locomotion writer path the driver does not yet
  own.
* An SDK re-import. The clamp table is captured here as module-level
  constants so ``import strands_robots.tools.g1.g1_velocity_envelope``
  pulls no ``unitree_sdk2py`` submodule - the import-hygiene contract
  every other file in this package carries, refs strands-labs/robots#358.
  A revision of the neon bundle's observed bounds is a driver-side
  update; when the driver's velocity method lands, its refusal will
  quote the same code the ``7404`` entry in
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.

What this module does not decide.

* Whether the driver's live ``_fsm_id`` is currently inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That is a
  driver-instance read carried on the driver's ``get_status`` envelope;
  a caller planning a velocity command compares the driver's live fsm
  against :func:`g1_list_velocity_envelope`'s ``walk_ready_fsm_ids`` to
  see whether the write gate is currently open. The two membership
  tests together - envelope for the vector, walk_ready for the gate -
  are the two conditions a future write verb would refuse on.
* Whether ``rt/lowcmd`` is currently held by another writer. The
  driver's single-writer lock reports that at wire time; a caller
  planning a velocity write cannot decide it without opening the
  topic itself, and this module opens no channel.
"""

from __future__ import annotations

import math
from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS

#: The absolute-value clamp the neon ``g1_move_velocity`` verb places on
#: ``vx`` before dispatch. The value is the one the neon bundle observed
#: as walkable on a gantry; above it the controller's response is
#: undefined and the robot may skate. Named as an absolute bound because
#: the neon clamp is symmetric (``vx`` reversed becomes reverse-walk of
#: the same magnitude).
_VX_ABS_MAX: float = 0.8

#: The absolute-value clamp on ``vy`` (lateral velocity). Kept smaller
#: than ``vx`` because the G1's balance controller has less headroom on
#: sideways strafe than on forward walk; the neon bundle's docstring
#: names the safe range as ``|vy| <= 0.2`` and this envelope quotes that.
_VY_ABS_MAX: float = 0.2

#: The absolute-value clamp on ``vyaw`` (in-place rotation rate). The
#: neon bundle's ``g1_turn`` verb further clamps its own ``yaw_rate``
#: input to ``[0.1, 0.6]``; this envelope quotes the higher ceiling that
#: ``g1_move_velocity`` observed as walkable, and a caller wanting a
#: turn-specific ceiling reaches :data:`_YAW_RATE_MAX` below.
_VYAW_ABS_MAX: float = 0.6

#: The upper clamp on ``duration`` (seconds) a single non-continuous
#: velocity write may keep the leg controller commanded for. Larger
#: values on the SDK path reach the controller as-is and would walk the
#: robot for hours if the caller times out; the neon bundle caps at
#: :data:`_DURATION_MAX_SECONDS` on the single-shot path and switches
#: to ``Move(continous_move=True)`` for anything longer, so this bound
#: is the boundary between the two modes rather than the SDK's own
#: limit (the SDK does not have one).
_DURATION_MAX_SECONDS: float = 10.0

#: The lower clamp on ``duration``. Zero-or-negative durations reach the
#: SDK as no-op-shaped writes today (the controller ignores them without
#: raising), so the neon bundle refuses them here rather than letting a
#: silently-dropped command look like a successful walk to the planner.
#: This bound is **exclusive** and is the one asymmetry in this envelope:
#: the abs-max clamps admit their boundary (``|value| > bound``) because a
#: saturated command is a command, while the value *at* this bound is the
#: no-op the refusal exists to catch, so it refuses on ``value <= bound``.
_DURATION_MIN_SECONDS: float = 0.0

#: Additional bounds for the ``g1_walk_forward`` and ``g1_turn`` sugar
#: verbs the neon bundle layers on top of ``g1_move_velocity``. Named
#: alongside the vector clamp so a caller planning a distance-shaped
#: walk sees both the sugar-verb bound and the underlying vector bound
#: in the same envelope.
_DISTANCE_ABS_MAX: float = 1.0
_SPEED_MIN: float = 0.05
_SPEED_MAX: float = 0.5
_ANGLE_ABS_MAX: float = 2.0 * math.pi
_YAW_RATE_MIN: float = 0.1
_YAW_RATE_MAX: float = 0.6

#: The error-table entry the driver's own ``_check_motion_gates`` quotes
#: when it refuses a locomotion-shaped write on an FSM outside
#: :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. Named here so
#: the returned envelope carries the exact refusal string a future
#: driver-side velocity wrapper would surface, and so a re-wording of
#: it lands in one place instead of drifting between the driver's log
#: and this lookup. The write path and this lookup share the constant.
_GATE_REFUSAL_CODE: int = 7404


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_velocity_envelope`
    so :func:`g1_velocity_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands in
    one place. Every field is a snapshot read; no bus is touched.
    """
    return {
        "vx_abs_max": _VX_ABS_MAX,
        "vy_abs_max": _VY_ABS_MAX,
        "vyaw_abs_max": _VYAW_ABS_MAX,
        "duration_min_seconds": _DURATION_MIN_SECONDS,
        "duration_max_seconds": _DURATION_MAX_SECONDS,
        "distance_abs_max": _DISTANCE_ABS_MAX,
        "speed_min": _SPEED_MIN,
        "speed_max": _SPEED_MAX,
        "angle_abs_max": _ANGLE_ABS_MAX,
        "yaw_rate_min": _YAW_RATE_MIN,
        "yaw_rate_max": _YAW_RATE_MAX,
    }


@tool
def g1_list_velocity_envelope() -> dict[str, Any]:
    """Return the velocity clamp envelope the neon bundle observed as walkable.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for ``LocoClient.SetVelocity`` is called, so a caller can compare
    an intended ``(vx, vy, vyaw, duration)`` vector against the
    envelope the neon bundle's ``g1_move_velocity`` refused outside of,
    and can also compare the driver's live ``fsm_id`` (from
    ``G1Driver.get_status``) against ``walk_ready_fsm_ids`` to decide
    whether the locomotion write gate is currently open.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying every
        clamp the neon bundle applied (``vx_abs_max``, ``vy_abs_max``,
        ``vyaw_abs_max``, ``duration_min_seconds``,
        ``duration_max_seconds``) plus the sugar-verb clamps
        (``distance_abs_max``, ``speed_min``/``speed_max``,
        ``angle_abs_max``, ``yaw_rate_min``/``yaw_rate_max``); a
        ``walk_ready_fsm_ids`` list quoting
        :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, the set
        the driver's motion gate admits locomotion-shaped writes on;
        and a ``refusals`` list carrying the ``7404`` gate-refused
        code and its decoded text, the one a future write verb would
        surface. Every field is a snapshot of an observed bound or a
        driver constant; no dynamic decode runs here.
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
    the envelope check needs only the finiteness half of
    :func:`~strands_robots.utils.positive_finite_number_error`; the
    positivity half does not apply (``vx``/``vy``/``vyaw`` may be
    negative). A future consolidation with the shared validator lands
    when the driver-side write verb reuses this admits function.
    """
    return math.isfinite(float(value))


@tool
def g1_velocity_admits(
    vx: float = 0.0,
    vy: float = 0.0,
    vyaw: float = 0.0,
    duration: float = 1.0,
) -> dict[str, Any]:
    """Decide whether a ``(vx, vy, vyaw, duration)`` vector sits inside the envelope.

    Read-only. Compares the four arguments against the clamps
    :func:`g1_list_velocity_envelope` returns and reports every
    dimension that would be refused, so a caller sees the full set of
    boundary violations in one call rather than one per dispatch
    attempt. No driver instance, no DDS, no SDK: the decision reads
    only module-level constants and the arguments themselves.

    A vector inside the envelope is *not* the same as an admitted
    write: the driver's motion gate (``_check_motion_gates``) also
    refuses on ``_fsm_id`` outside
    :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, which this
    verb does not read (that is a live driver-instance query answered
    by :mod:`~strands_robots.tools.g1.g1_state` and
    :mod:`~strands_robots.tools.g1.g1_motion_gates`). The returned
    envelope names ``walk_ready_fsm_ids`` so a caller comparing an
    intended write against both conditions has the FSM set on hand.

    Args:
        vx: forward velocity in metres per second. Any finite float;
            an ``|vx|`` above ``vx_abs_max`` reads as refused.
        vy: lateral velocity in metres per second. Same shape as
            ``vx`` against ``vy_abs_max``.
        vyaw: yaw rate in radians per second. Same shape against
            ``vyaw_abs_max``.
        duration: seconds to keep the velocity commanded on the
            single-shot path. Refused at or below
            ``duration_min_seconds`` - that bound is exclusive, so a
            ``duration`` of exactly zero is refused rather than
            admitted - and refused above ``duration_max_seconds``,
            which is inclusive like the three abs-max clamps.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        every dimension is inside its clamp; a ``refusals`` list of
        per-dimension refusal descriptors, each carrying the dimension
        name, the offending value, the clamp it violated, and the
        ``7404`` gate-refusal code the driver would quote if the
        write were attempted while the vector is outside the
        envelope; the same ``envelope`` sub-dict
        :func:`g1_list_velocity_envelope` returns; and
        ``walk_ready_fsm_ids`` for the follow-on gate decision. On
        an admitted vector the ``refusals`` list is empty; on a
        rejected vector every violated dimension is named.
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

    for name, value, bound_key, bound in (
        ("vx", vx, "vx_abs_max", _VX_ABS_MAX),
        ("vy", vy, "vy_abs_max", _VY_ABS_MAX),
        ("vyaw", vyaw, "vyaw_abs_max", _VYAW_ABS_MAX),
    ):
        if not _finite(value):
            _reject(name, value, bound_key, bound, "non-finite")
            continue
        if abs(float(value)) > bound:
            _reject(name, value, bound_key, bound, "|value| > bound")

    if not _finite(duration):
        _reject("duration", duration, "duration_max_seconds", _DURATION_MAX_SECONDS, "non-finite")
    else:
        d = float(duration)
        if d <= _DURATION_MIN_SECONDS:
            _reject(
                "duration",
                duration,
                "duration_min_seconds",
                _DURATION_MIN_SECONDS,
                "value <= bound",
            )
        elif d > _DURATION_MAX_SECONDS:
            _reject(
                "duration",
                duration,
                "duration_max_seconds",
                _DURATION_MAX_SECONDS,
                "value > bound",
            )

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "envelope": envelope,
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
    }
