"""Agent-facing lookup for the swing-height envelope ``LocoClient`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) does not
expose a public ``SetSwingHeight`` method - the setter is reachable
only through the SDK's raw ``_Call`` on API id ``7103``, which the
neon bundle's ``g1_set_swing_height`` verb
(``cagataycali/neon-the-g1/tools/g1_posture.py`` and the shared
``_g1_common.set_swing_height`` helper) fronts under a single-writer
lock. The SDK itself places *no* clamps on the ``7103`` argument: a
caller that passes ``height=10.0`` or ``height=-3.5`` reaches the
controller unchanged, and the controller's behaviour above the
neon-bundle-observed step-clearance range is undefined - the G1 has
no runaway guard on that write path. The neon bundle's own wrapper
narrows the argument to ``max(0.0, min(0.2, float(height)))`` before
dispatch, so this module snapshots that clamp pair into module-level
constants and exposes two agent-facing verbs -
:func:`g1_list_swing_height_envelope` (name the whole envelope) and
:func:`g1_swing_height_admits` (decide one query) - so a caller can
decide the refusal decidably before a future driver-side wrapper
fires. Refs strands-labs/robots#358.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_set_swing_height`` verb
  wrapped the raw ``_Call(7103, ...)`` under a single-writer lock;
  that write is the same locomotion topic
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` gates, which
  today's :class:`~strands_robots.drivers.g1.G1Driver` refuses
  through :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
  on any locomotion-shaped write while ``_fsm_id`` is outside that
  set. A future driver method that fronts the ``7103`` setter will
  land the write verb; refs strands-labs/robots#358 for the
  SDK-facing gate work that write belongs on. This module ports the
  read-only envelope half without also introducing a second
  locomotion writer path the driver does not yet own.
* An SDK re-import. The clamp table is captured here as module-level
  constants so ``import strands_robots.tools.g1.g1_swing_height_envelope``
  pulls no ``unitree_sdk2py`` submodule - the import-hygiene contract
  every other file in this package carries, refs
  strands-labs/robots#358. A revision of the neon bundle's observed
  bounds is a driver-side update; when the driver's swing-height
  method lands, its refusal will quote the same ``7404`` code the
  entry in :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
  carries.

What this module does not decide.

* Whether the driver's live ``_fsm_id`` is currently inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That is a
  driver-instance read carried on the driver's ``get_status``
  envelope; a caller planning a swing-height command compares the
  driver's live fsm against :func:`g1_list_swing_height_envelope`'s
  ``walk_ready_fsm_ids`` to see whether the write gate is currently
  open. The two membership tests together - envelope for the value,
  walk_ready for the gate - are the two conditions a future write
  verb would refuse on.
* Whether ``rt/lowcmd`` (or the ``7103`` RPC channel) is currently
  held by another writer. The driver's single-writer lock reports
  that at wire time; a caller planning a swing-height write cannot
  decide it without opening the channel itself, and this module
  opens no channel.
* Which API id the underlying dispatch uses. The neon bundle's
  ``_g1_common.set_swing_height`` fronts API ``7103`` via raw
  ``_Call`` rather than a public SDK method (the SDK does not
  export ``SetSwingHeight``); that dispatch detail is a driver-side
  concern named in the neon bundle's own docstring, and this
  envelope quotes it as :data:`_SWING_HEIGHT_API_ID` so a caller
  comparing this lookup to the neon bundle sees the same integer
  on both sides.
"""

from __future__ import annotations

import math
from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS

#: The lower clamp the neon ``g1_set_swing_height`` verb places on the
#: ``height`` argument before dispatch. The value is the one the neon
#: bundle's own wrapper (``max(0.0, min(0.2, float(height)))``) rounds
#: up to for any strictly-negative input, and matches the SDK's
#: implicit floor: a negative swing height would ask the leg to lift
#: *below* the ground plane, a shape violation rather than a value
#: one. Named as an inclusive lower bound because ``0.0`` is a
#: legitimate command the neon wrapper does not reject (it is the
#: minimum-clearance "shuffle" gait); refusing it would drop a
#: caller's most conservative locomotion command.
_SWING_HEIGHT_MIN: float = 0.0

#: The upper clamp the neon ``g1_set_swing_height`` verb places on the
#: ``height`` argument before dispatch. The value is the one the neon
#: bundle's own wrapper (``max(0.0, min(0.2, float(height)))``) rounds
#: down to for any input above ``0.2`` m. Above this the controller's
#: response is undefined (the SDK does not clamp), the leg lifts more
#: than the neon bundle ever observed on a walkable surface, and the
#: energy cost rises with no benefit. Named as an inclusive upper
#: bound because a saturated command is still a command.
_SWING_HEIGHT_MAX: float = 0.2

#: The lower bound of the neon-bundle-documented *recommended* swing
#: range - the interval the neon bundle's ``g1_set_swing_height``
#: docstring names as "Typical safe range: 0.05-0.15 m". A value
#: below this admits at the envelope level (the neon wrapper does
#: not reject it), but a caller comparing an intended write against
#: the neon bundle's own safety guidance sees the recommendation
#: here. Not enforced by :func:`g1_swing_height_admits` - refusing
#: a value between ``_SWING_HEIGHT_MIN`` and this bound would drop
#: the neon bundle's own admitted range - but surfaced on the
#: envelope so a caller can decide the softer refusal itself.
_SWING_HEIGHT_RECOMMENDED_MIN: float = 0.05

#: The upper bound of the neon-bundle-documented *recommended* swing
#: range. Symmetric partner of :data:`_SWING_HEIGHT_RECOMMENDED_MIN`;
#: named for the same reason. A value between this and
#: :data:`_SWING_HEIGHT_MAX` is admitted by :func:`g1_swing_height_admits`
#: (the neon wrapper does not refuse it) but sits above the neon
#: bundle's own recommended clearance.
_SWING_HEIGHT_RECOMMENDED_MAX: float = 0.15

#: The SDK RPC API id the neon bundle's
#: ``_g1_common.set_swing_height`` invokes via raw ``_Call``. The
#: Unitree SDK does not expose a public ``SetSwingHeight`` method:
#: the setter is reachable only through the raw ``_Call(7103, ...)``
#: path, which the neon bundle wraps under a single-writer lock.
#: Named here so the returned envelope carries the exact API id a
#: driver-side wrapper would target, and so a firmware release that
#: renumbered the setter lands in one place instead of drifting
#: between the neon bundle and this lookup.
_SWING_HEIGHT_API_ID: int = 7103

#: The error-table entry the driver's own ``_check_motion_gates``
#: quotes when it refuses a locomotion-shaped write on an FSM
#: outside :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`.
#: Named here so the returned envelope carries the exact refusal
#: string a future driver-side swing-height wrapper would surface,
#: and so a re-wording of it lands in one place instead of drifting
#: between the driver's log and this lookup. The write path and
#: this lookup share the constant.
_GATE_REFUSAL_CODE: int = 7404


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_swing_height_envelope`
    so :func:`g1_swing_height_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands in
    one place. Every field is a snapshot read; no bus is touched.
    """
    return {
        "swing_height_min": _SWING_HEIGHT_MIN,
        "swing_height_max": _SWING_HEIGHT_MAX,
        "swing_height_recommended_min": _SWING_HEIGHT_RECOMMENDED_MIN,
        "swing_height_recommended_max": _SWING_HEIGHT_RECOMMENDED_MAX,
        "swing_height_api_id": _SWING_HEIGHT_API_ID,
    }


@tool
def g1_list_swing_height_envelope() -> dict[str, Any]:
    """Return the swing-height envelope the neon bundle clamps to.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for the swing-height setter is called, so a caller can compare an
    intended ``height`` value against the clamps the neon bundle's
    ``g1_set_swing_height`` narrows to, and can also compare the
    driver's live ``fsm_id`` (from ``G1Driver.get_status``) against
    ``walk_ready_fsm_ids`` to decide whether the locomotion write
    gate is currently open.

    The envelope names two ranges the neon bundle stacks on top of
    the SDK: the hard clamp ``[swing_height_min, swing_height_max]``
    the neon wrapper enforces (``max(0.0, min(0.2, height))``), and
    the softer recommended interval
    ``[swing_height_recommended_min, swing_height_recommended_max]``
    the neon bundle's docstring names as "Typical safe range: 0.05
    -0.15 m". Values inside the hard clamp are admitted by
    :func:`g1_swing_height_admits`; values inside the recommended
    interval are named on the returned envelope so a caller can
    decide the softer refusal itself. The underlying SDK RPC id
    (``swing_height_api_id = 7103``) is quoted for
    caller-to-neon-bundle comparison because the SDK does not
    expose a public ``SetSwingHeight`` method.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        both ranges plus the SDK RPC api id
        (``swing_height_min``, ``swing_height_max``,
        ``swing_height_recommended_min``,
        ``swing_height_recommended_max``, ``swing_height_api_id``);
        a ``walk_ready_fsm_ids`` list quoting
        :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, the
        set the driver's motion gate admits locomotion-shaped
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
    the positivity half is already answered by the explicit
    ``_SWING_HEIGHT_MIN`` clamp (``0.0``), and a shared positive-finite
    validator would refuse ``0.0`` which the neon bundle admits as the
    minimum-clearance shuffle gait. A future consolidation with the
    shared validator lands when the driver-side write verb reuses
    this admits function.
    """
    return math.isfinite(float(value))


@tool
def g1_swing_height_admits(height: float = 0.1) -> dict[str, Any]:
    """Decide whether a ``height`` value sits inside the swing-height envelope.

    Read-only. Compares the argument against the clamps
    :func:`g1_list_swing_height_envelope` returns and reports whether
    the neon bundle's wrapper would dispatch it unchanged (it clamps
    silently for out-of-range values, but this verb surfaces the
    refusal so the caller sees which bound would be hit). No driver
    instance, no DDS, no SDK: the decision reads only module-level
    constants and the argument itself.

    A value inside the hard clamp is *not* the same as an admitted
    write: the driver's motion gate (``_check_motion_gates``) also
    refuses on ``_fsm_id`` outside
    :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, which this
    verb does not read (that is a live driver-instance query
    answered by :mod:`~strands_robots.tools.g1.g1_state` and
    :mod:`~strands_robots.tools.g1.g1_motion_gates`). The returned
    envelope names ``walk_ready_fsm_ids`` so a caller comparing an
    intended write against both conditions has the FSM set on hand.

    Two ranges are named on the returned payload:

    * Hard clamp: ``[swing_height_min, swing_height_max]`` =
      ``[0.0, 0.2]`` m. Values inside this range are admitted; the
      route the verb reports is ``"call_7103"`` (the raw
      ``LocoClient._Call(7103, ...)`` path the neon bundle's
      ``_g1_common.set_swing_height`` fronts). Both bounds are
      inclusive (``height < min`` refuses, ``height > max``
      refuses) because a saturated command is a command the neon
      wrapper would dispatch at the clamped boundary.
    * Recommendation interval: ``[swing_height_recommended_min,
      swing_height_recommended_max]`` = ``[0.05, 0.15]`` m. Values
      inside the hard clamp but outside this interval are still
      admitted (the neon wrapper does not refuse them); the
      returned payload names ``inside_recommended`` so a caller
      that wants to enforce the softer bound sees the answer without
      having to re-check it. Refusing outside the recommendation
      would drop the neon bundle's own admitted range.
    * ``height`` non-finite (``math.inf``, ``math.nan``) -> refused,
      ``comparison="non-finite"``. A NaN cannot be compared
      decidably (``nan < 0`` is ``False`` but so is ``nan > 0.2``),
      and an infinity would either overrun ``swing_height_max`` or
      underrun ``swing_height_min`` - both are shape violations
      rather than value ones.

    Args:
        height: Target swing height in metres. The default ``0.1`` m
            sits at the centre of the neon bundle's recommended
            interval so a caller who does not pass an explicit
            argument lands on an admitted mid-range command.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether the
        value would reach the neon wrapper without being clamped or
        refused; a ``route`` string naming the SDK dispatch path a
        future write verb would take (``"call_7103"``) or ``None``
        on a rejected value; an ``inside_recommended`` bool naming
        whether the value sits inside the softer recommendation
        interval (``True`` even when the value equals a
        recommendation boundary); a ``refusals`` list carrying the
        refusal descriptors on a rejected value, each with the
        offending value, the bound it violated, the comparison, and
        the ``7404`` gate-refusal code the driver would quote if
        the write were attempted while the value is outside the
        envelope; the same ``envelope`` sub-dict
        :func:`g1_list_swing_height_envelope` returns; and
        ``walk_ready_fsm_ids`` for the follow-on gate decision.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []
    route: str | None = None
    inside_recommended = False

    def _reject(value: float, bound_key: str, bound: float, cmp: str) -> None:
        refusals.append(
            {
                "dimension": "height",
                "value": float(value),
                "bound_key": bound_key,
                "bound": bound,
                "comparison": cmp,
                "code": _GATE_REFUSAL_CODE,
                "text": ERR_CODES[_GATE_REFUSAL_CODE],
            }
        )

    if not _finite(height):
        _reject(height, "swing_height_max", _SWING_HEIGHT_MAX, "non-finite")
    else:
        h = float(height)
        if h < _SWING_HEIGHT_MIN:
            _reject(height, "swing_height_min", _SWING_HEIGHT_MIN, "value < bound")
        elif h > _SWING_HEIGHT_MAX:
            _reject(height, "swing_height_max", _SWING_HEIGHT_MAX, "value > bound")
        else:
            route = "call_7103"
            inside_recommended = _SWING_HEIGHT_RECOMMENDED_MIN <= h <= _SWING_HEIGHT_RECOMMENDED_MAX

    return {
        "status": "success",
        "admits": not refusals,
        "route": route,
        "inside_recommended": inside_recommended,
        "refusals": refusals,
        "envelope": envelope,
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
    }
