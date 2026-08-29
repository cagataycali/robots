"""Agent-facing lookup for the locomotion-duration envelope ``LocoClient.SetVelocity`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes
``SetVelocity(vx, vy, vyaw, duration_s)`` as the bounded write for a
velocity command that terminates on its own after ``duration_s``
seconds, and ``Move(vx, vy, vyaw, continous_move=True)`` as the
unbounded cousin the neon bundle
(``cagataycali/neon-the-g1/tools/g1_locomotion.py::g1_move_velocity``)
routes to when the caller passes ``continuous=True``. The SDK itself
places *no* clamps on the ``duration_s`` argument: a caller that
passes ``duration=3600.0`` reaches the controller unchanged and walks
the robot for an hour, and a caller that passes ``duration=0.0``
reaches it as a no-op the SDK does not refuse. The neon bundle's
``g1_move_velocity`` verb narrows the argument to
``max(0.0, min(10.0, float(duration)))`` before dispatch, then refuses
outright on ``duration <= 0`` with the message
``"duration<=0 (non-continuous), refusing"`` before the SDK is
touched. This module snapshots that clamp pair (plus the neon-bundle-
observed zero-refusal shape) into module-level constants and exposes
two agent-facing verbs - :func:`g1_list_locomotion_duration_envelope`
(name the whole envelope) and :func:`g1_locomotion_duration_admits`
(decide one query) - so a caller can decide the refusal decidably
before a future driver-side wrapper fires. Refs
strands-labs/robots#358.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_move_velocity`` verb
  wraps ``LocoClient.SetVelocity`` and ``LocoClient.Move`` under the
  same DDS singleton :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
  the driver holds; those writes are the same locomotion topic
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` gates, which
  today's :class:`~strands_robots.drivers.g1.G1Driver` refuses through
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` on
  any locomotion-shaped write while ``_fsm_id`` is outside that set.
  A future driver method that fronts ``SetVelocity`` will land the
  write verb; refs strands-labs/robots#358 for the SDK-facing gate
  work that write belongs on. This module ports the read-only
  envelope half without also introducing a second locomotion writer
  path the driver does not yet own.
* An SDK re-import. The clamp table is captured here as module-level
  constants so
  ``import strands_robots.tools.g1.g1_locomotion_duration_envelope``
  pulls no ``unitree_sdk2py`` submodule - the import-hygiene contract
  every other file in this package carries, refs
  strands-labs/robots#358. A revision of the neon bundle's observed
  bounds is a driver-side update; when the driver's velocity method
  lands, its refusal will quote the same ``7404`` code the entry in
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.

What this module does not decide.

* Whether the driver's live ``_fsm_id`` is currently inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That is a
  driver-instance read carried on the driver's ``get_status``
  envelope; a caller planning a velocity command compares the
  driver's live fsm against
  :func:`g1_list_locomotion_duration_envelope`'s
  ``walk_ready_fsm_ids`` to see whether the write gate is currently
  open. The two membership tests together - envelope for the
  duration value, walk_ready for the gate - are the two conditions
  a future write verb would refuse on.
* Whether the caller means to send a ``continuous=True`` command
  (the neon bundle's route to ``LocoClient.Move(..., continous_move=True)``
  rather than ``LocoClient.SetVelocity``). ``continuous=True`` bypasses
  the ``duration`` argument entirely at the neon-wrapper layer - the
  SDK's ``Move`` call reads only the three velocity components and
  runs until an explicit ``StopMove`` arrives - so this envelope
  applies only to the bounded ``SetVelocity`` path. A caller who
  wants an unbounded walk queries the envelope for the *bounded*
  route the driver-side wrapper would gate on and then decides at
  its own layer whether to take the unbounded branch instead.
* Whether the velocity components (``vx``, ``vy``, ``vyaw``) sit
  inside their own envelope. That is a separate lookup
  (:mod:`~strands_robots.tools.g1.g1_velocity_envelope`, port #2965,
  refs strands-labs/robots#358) so a caller composes two admission
  decisions - velocity magnitudes and duration - before the write
  gate reads the FSM. Keeping the two envelopes in separate modules
  matches the per-argument envelope pattern the neon bundle already
  uses on the driver side.
"""

from __future__ import annotations

import math
from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS

#: The lower clamp the neon ``g1_move_velocity`` verb places on the
#: ``duration`` argument before dispatch. The neon wrapper's own
#: expression ``max(0.0, min(10.0, float(duration)))`` names ``0.0``
#: as the floor a negative input rounds up to, but the neon verb's
#: subsequent conditional ``if duration <= 0: refuse`` makes the
#: floor **exclusive** at admission time: a duration equal to the
#: floor value never reaches the SDK because the neon wrapper
#: refuses it with the message ``"duration<=0 (non-continuous),
#: refusing"``. Named as a strict lower bound so
#: :func:`g1_locomotion_duration_admits` reads
#: ``duration > _LOCOMOTION_DURATION_MIN`` to admit, matching the
#: neon wrapper's own conditional. A caller who wants a zero-length
#: no-op writes should not pass ``duration=0.0``; that is a no-op
#: shape the SDK does not admit through ``SetVelocity`` and the neon
#: wrapper refuses before the SDK is touched.
_LOCOMOTION_DURATION_MIN: float = 0.0

#: The upper clamp the neon ``g1_move_velocity`` verb places on the
#: ``duration`` argument before dispatch. The value is the one the
#: neon bundle's own wrapper (``max(0.0, min(10.0, float(duration)))``)
#: rounds down to for any input above ``10.0`` s. Above this the
#: neon bundle's own safety comment reads "a runaway value walks the
#: robot for hours"; the SDK does not clamp, so a caller passing
#: ``duration=3600.0`` unfronted would walk the robot for an hour
#: on a single write. Named as an inclusive upper bound because
#: ``10.0`` s is a value the neon wrapper does dispatch (``min(10.0,
#: 10.0) == 10.0``) and refusing it would drop a saturated command.
_LOCOMOTION_DURATION_MAX: float = 10.0

#: The SDK method name the neon bundle's ``g1_move_velocity`` fronts
#: on the bounded-duration branch. Named here so the returned
#: envelope carries the exact dispatch identifier a driver-side
#: wrapper would target, and so a firmware release that renamed the
#: method lands in one place instead of drifting between the neon
#: bundle and this lookup. The SDK's own method takes four positional
#: floats (``vx``, ``vy``, ``vyaw``, ``duration_s``); this envelope
#: quotes only the duration side because that is the argument the
#: neon wrapper clamps and this module envelopes.
_LOCOMOTION_SET_VELOCITY_METHOD: str = "SetVelocity"

#: The SDK method name the neon bundle's ``g1_move_velocity`` fronts
#: on the unbounded-duration branch (``continuous=True``). Named on
#: the envelope so a caller who is not sure which route to take
#: sees both SDK method names alongside the duration clamp. The
#: neon wrapper passes ``continous_move=True`` (spelled the way the
#: SDK spells it, with a single ``u``) to distinguish this from the
#: bounded call; a caller reading the envelope sees the two method
#: names and can pick the branch before the write.
_LOCOMOTION_CONTINUOUS_MOVE_METHOD: str = "Move"

#: The error-table entry the driver's own ``_check_motion_gates``
#: quotes when it refuses a locomotion-shaped write on an FSM
#: outside :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`.
#: Named here so the returned envelope carries the exact refusal
#: string a future driver-side velocity wrapper would surface, and
#: so a re-wording of it lands in one place instead of drifting
#: between the driver's log and this lookup. The write path and
#: this lookup share the constant.
_GATE_REFUSAL_CODE: int = 7404


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_locomotion_duration_envelope` so
    :func:`g1_locomotion_duration_admits` names the same fields on
    its admitted-path payload and so a widen to the descriptor lands
    in one place. Every field is a snapshot read; no bus is touched.
    """
    return {
        "duration_min": _LOCOMOTION_DURATION_MIN,
        "duration_max": _LOCOMOTION_DURATION_MAX,
        "set_velocity_method": _LOCOMOTION_SET_VELOCITY_METHOD,
        "continuous_move_method": _LOCOMOTION_CONTINUOUS_MOVE_METHOD,
    }


@tool
def g1_list_locomotion_duration_envelope() -> dict[str, Any]:
    """Return the locomotion-duration envelope the neon bundle clamps to.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side
    wrapper for ``LocoClient.SetVelocity`` is called, so a caller
    can compare an intended ``duration`` value against the clamps
    the neon bundle's ``g1_move_velocity`` narrows to, and can also
    compare the driver's live ``fsm_id`` (from
    ``G1Driver.get_status``) against ``walk_ready_fsm_ids`` to
    decide whether the locomotion write gate is currently open.

    The envelope names the hard clamp
    ``(duration_min, duration_max]`` = ``(0.0, 10.0]`` s the neon
    bundle enforces on the ``duration`` argument to
    :data:`_LOCOMOTION_SET_VELOCITY_METHOD` (the SDK's bounded
    write). The lower bound is **strict** because the neon wrapper
    refuses ``duration <= 0`` outright with a named refusal string
    before the SDK is touched; the upper bound is **inclusive**
    because a saturated 10-second command is a command the neon
    wrapper does dispatch. The envelope also names the alternative
    unbounded-duration branch (:data:`_LOCOMOTION_CONTINUOUS_MOVE_METHOD`,
    ``LocoClient.Move`` with ``continous_move=True``) so a caller
    who wants an unbounded walk sees the SDK method name alongside
    the clamp.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        both clamp bounds
        (``duration_min``, ``duration_max``) plus the two SDK
        method names (``set_velocity_method``,
        ``continuous_move_method``); a ``walk_ready_fsm_ids`` list
        quoting :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`,
        the set the driver's motion gate admits locomotion-shaped
        writes on; and a ``refusals`` list carrying the ``7404``
        gate-refused code and its decoded text, the one a future
        write verb would surface. Every field is a snapshot of an
        observed bound or a driver constant; no dynamic decode
        runs here.
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
    the envelope check needs only the finiteness half of a validator;
    the positivity half is answered by the explicit
    :data:`_LOCOMOTION_DURATION_MIN` clamp (``0.0``, checked strictly
    below). A future consolidation with the shared validator lands
    when the driver-side write verb reuses this admits function.
    """
    return math.isfinite(float(value))


@tool
def g1_locomotion_duration_admits(duration: float = 1.0) -> dict[str, Any]:
    """Decide whether a ``duration`` value sits inside the locomotion-duration envelope.

    Read-only. Compares the argument against the clamps
    :func:`g1_list_locomotion_duration_envelope` returns and reports
    whether the neon bundle's wrapper would dispatch it unchanged.
    The neon wrapper clamps silently for above-ceiling values before
    refusing zero-or-negative outright, but this verb surfaces the
    refusal so the caller sees which bound would be hit rather than
    which value the neon wrapper would silently clamp to. No driver
    instance, no DDS, no SDK: the decision reads only module-level
    constants and the argument itself.

    A value inside the envelope is *not* the same as an admitted
    write: the driver's motion gate (``_check_motion_gates``) also
    refuses on ``_fsm_id`` outside
    :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, which
    this verb does not read (that is a live driver-instance query
    answered by :mod:`~strands_robots.tools.g1.g1_state` and
    :mod:`~strands_robots.tools.g1.g1_motion_gates`). The returned
    envelope names ``walk_ready_fsm_ids`` so a caller comparing an
    intended write against both conditions has the FSM set on hand.

    The routing conventions:

    * ``duration_min < duration <= duration_max`` (i.e.
      ``0.0 < duration <= 10.0`` s) -> admitted; the route the
      verb reports is ``"set_velocity"`` (the SDK's bounded
      ``LocoClient.SetVelocity`` call). The lower bound is
      **strict** (``duration <= _LOCOMOTION_DURATION_MIN``
      refuses) because the neon wrapper's own conditional
      ``if duration <= 0`` refuses ``0.0`` before dispatch; the
      upper bound is **inclusive** (``duration ==
      _LOCOMOTION_DURATION_MAX`` admits) because a saturated
      10-second command is a command the neon wrapper does
      dispatch.
    * ``duration > _LOCOMOTION_DURATION_MAX`` -> refused,
      ``comparison="value > bound"``. The neon wrapper clamps
      silently, but a caller who wants to send a longer walk
      should route to the unbounded ``Move(..., continous_move=True)``
      branch (named on the envelope as
      :data:`_LOCOMOTION_CONTINUOUS_MOVE_METHOD`) rather than
      have the neon wrapper drop the excess.
    * ``duration <= _LOCOMOTION_DURATION_MIN`` -> refused,
      ``comparison="value <= bound"``. This includes
      ``duration == 0.0`` (a no-op the SDK does not honour on the
      bounded path) and every strictly-negative value. The neon
      wrapper's own refusal string quotes
      ``"duration<=0 (non-continuous), refusing"``; this envelope
      reproduces the same refusal shape.
    * ``duration`` non-finite (``math.inf``, ``math.nan``) -> refused,
      ``comparison="non-finite"``. A NaN cannot be compared
      decidably (``nan > 0`` is ``False`` but so is
      ``nan <= 10.0``), and an infinity would either overrun
      ``duration_max`` or, on the negative branch, underrun
      ``duration_min`` - both are shape violations rather than
      value ones.

    Args:
        duration: Target velocity command duration in seconds. The
            default ``1.0`` s matches the neon bundle's own
            ``g1_move_velocity`` default (a "single 1-second impulse
            for safety") so a caller who does not pass an explicit
            argument lands on the same admitted value the neon
            wrapper defaults to.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the value would reach the neon wrapper without being
        clamped or refused; a ``route`` string naming the SDK
        dispatch path a future write verb would take
        (``"set_velocity"``) or ``None`` on a rejected value; a
        ``refusals`` list carrying the refusal descriptors on a
        rejected value, each with the offending value, the bound
        it violated, the comparison, and the ``7404`` gate-refusal
        code the driver would quote if the write were attempted
        while the value is outside the envelope; the same
        ``envelope`` sub-dict
        :func:`g1_list_locomotion_duration_envelope` returns; and
        ``walk_ready_fsm_ids`` for the follow-on gate decision.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []
    route: str | None = None

    def _reject(value: float, bound_key: str, bound: float, cmp: str) -> None:
        refusals.append(
            {
                "dimension": "duration",
                "value": float(value),
                "bound_key": bound_key,
                "bound": bound,
                "comparison": cmp,
                "code": _GATE_REFUSAL_CODE,
                "text": ERR_CODES[_GATE_REFUSAL_CODE],
            }
        )

    if not _finite(duration):
        _reject(duration, "duration_max", _LOCOMOTION_DURATION_MAX, "non-finite")
    else:
        d = float(duration)
        if d <= _LOCOMOTION_DURATION_MIN:
            _reject(
                duration,
                "duration_min",
                _LOCOMOTION_DURATION_MIN,
                "value <= bound",
            )
        elif d > _LOCOMOTION_DURATION_MAX:
            _reject(
                duration,
                "duration_max",
                _LOCOMOTION_DURATION_MAX,
                "value > bound",
            )
        else:
            route = "set_velocity"

    return {
        "status": "success",
        "admits": not refusals,
        "route": route,
        "refusals": refusals,
        "envelope": envelope,
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
    }
