"""Agent-facing lookup for the stand-height envelope ``LocoClient.SetStandHeight`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes a
continuous stand-height setter via ``SetStandHeight(height_m)`` and a
one-shot ``HighStand()`` cousin the SDK dispatches by passing a
``UINT32_MAX`` sentinel through the same handler. The SDK itself
places *no* clamps on the ``SetStandHeight`` argument: a caller that
passes ``height=10.0`` or ``height=-3.5`` reaches the controller
unchanged, and the controller's own behaviour above the neon-bundle-
observed walkable range is undefined - the G1 has no runaway guard
on that write path. The neon bundle's ``g1_set_stand_height`` verb
(``cagataycali/neon-the-g1/tools/g1_posture.py``) fronts the same
call under two conventions observed against the real robot on a
gantry: a non-negative ``height`` in metres is passed through as a
``SetStandHeight`` argument bounded by :data:`_STAND_HEIGHT_MAX`,
and a negative ``height`` is translated to ``HighStand()`` (the
sentinel-driven "max stand" path the SDK exposes as a separate
method). This module surfaces both halves to an agent so a caller
can decide the refusal decidably before a future driver-side wrapper
fires, rather than pinning it inside the write path where the
refusal is invisible to the planner.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_set_stand_height`` verb
  wrapped ``LocoClient.SetStandHeight`` / ``LocoClient.HighStand``
  under a single-writer lock; that write is the same locomotion
  topic :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` gates,
  which today's :class:`~strands_robots.drivers.g1.G1Driver` refuses
  through :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
  on any locomotion-shaped write while ``_fsm_id`` is outside that
  set. A future driver method that fronts ``SetStandHeight`` will
  land the write verb; refs strands-labs/robots#358 for the
  SDK-facing gate work that write belongs on. This module ports the
  read-only envelope half without also introducing a second
  locomotion writer path the driver does not yet own.
* An SDK re-import. The clamp table is captured here as module-level
  constants so ``import strands_robots.tools.g1.g1_stand_height_envelope``
  pulls no ``unitree_sdk2py`` submodule - the import-hygiene contract
  every other file in this package carries, refs strands-labs/robots#358.
  A revision of the neon bundle's observed bounds is a driver-side
  update; when the driver's stand-height method lands, its refusal
  will quote the same code the ``7404`` entry in
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.

What this module does not decide.

* Whether the driver's live ``_fsm_id`` is currently inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That is a
  driver-instance read carried on the driver's ``get_status``
  envelope; a caller planning a stand-height command compares the
  driver's live fsm against :func:`g1_list_stand_height_envelope`'s
  ``walk_ready_fsm_ids`` to see whether the write gate is currently
  open. The two membership tests together - envelope for the value,
  walk_ready for the gate - are the two conditions a future write
  verb would refuse on.
* Whether ``rt/lowcmd`` is currently held by another writer. The
  driver's single-writer lock reports that at wire time; a caller
  planning a stand-height write cannot decide it without opening the
  topic itself, and this module opens no channel.
* What ``HighStand()`` actually reaches on the wire. The SDK
  translates its call to a ``SetStandHeight`` handler invocation
  with a ``UINT32_MAX`` sentinel value; the neon bundle's convention
  of routing a negative ``height`` argument to ``HighStand()`` is
  the *agent-facing sugar*, not the SDK's own boundary. The verbs
  here name both the sugar (negative ``height`` routes to
  ``HighStand``) and the underlying SDK method (``SetStandHeight``
  with a sentinel) so a caller sees which path a query would take
  and can compare that against the driver's own future refusal.
"""

from __future__ import annotations

import math
from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS

#: The upper clamp the neon ``g1_set_stand_height`` verb places on a
#: non-negative ``height`` argument before dispatch. The value is the
#: one the neon bundle's docstring quotes as the walkable ceiling
#: (``0.0..~0.8`` metres); above it the controller's response is
#: undefined and the robot may over-extend. Named as an inclusive
#: upper bound because a saturated command is still a command and
#: rejecting it would silently drop a caller's intended write.
_STAND_HEIGHT_MAX: float = 0.8

#: The lower clamp on a non-negative ``height`` argument. Zero is
#: admitted because the neon bundle documents ``0.0`` as the "LOW
#: stand" pose (a crouched-but-standing height the controller accepts);
#: refusing it would drop the LLM's most conservative posture. This
#: bound is **inclusive** on the low side (``value >= _STAND_HEIGHT_MIN``
#: admits, ``value < _STAND_HEIGHT_MIN`` refuses) so the boundary
#: value itself lands as an admitted low-stand command.
_STAND_HEIGHT_MIN: float = 0.0

#: The sentinel the SDK's ``HighStand`` method passes to the same
#: ``SetStandHeight`` handler under the hood: ``UINT32_MAX`` reinterpreted
#: as a magic marker for "go to the controller's built-in max stand
#: pose". Named here so the returned envelope carries the exact wire
#: value a future driver-side ``HighStand`` wrapper would surface, and
#: so a caller comparing this lookup's answer to the SDK's own
#: constant sees the same integer on both sides. The value is
#: ``2**32 - 1`` (4294967295); the SDK carries it as an ``int`` in
#: the handler's argument slot.
_HIGH_STAND_SENTINEL: int = 2**32 - 1

#: The neon-bundle-observed sentinel convention for the ``height``
#: argument the agent surface accepts: any negative float routes to
#: ``HighStand()`` rather than ``SetStandHeight(height)``. The bound is
#: **strict** (``height < 0.0`` routes to HighStand; ``height == 0.0``
#: is the LOW-stand path, not the HighStand path) because ``-0.0`` and
#: ``0.0`` compare equal to Python but carry different sign bits, and
#: the neon bundle's own conditional (``if height < 0``) reads ``-0.0``
#: as non-negative (``-0.0 < 0`` is ``False``). This bound documents
#: that: ``-0.0`` is admitted as a low-stand, ``-1e-300`` routes to
#: HighStand. A caller who wants HighStand should pass ``-1.0`` (any
#: strictly-negative sentinel) rather than a value near zero.
_HIGH_STAND_ROUTE_THRESHOLD: float = 0.0

#: The error-table entry the driver's own ``_check_motion_gates`` quotes
#: when it refuses a locomotion-shaped write on an FSM outside
#: :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. Named here so
#: the returned envelope carries the exact refusal string a future
#: driver-side stand-height wrapper would surface, and so a re-wording
#: of it lands in one place instead of drifting between the driver's
#: log and this lookup. The write path and this lookup share the
#: constant.
_GATE_REFUSAL_CODE: int = 7404


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_stand_height_envelope`
    so :func:`g1_stand_height_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands in
    one place. Every field is a snapshot read; no bus is touched.
    """
    return {
        "stand_height_min": _STAND_HEIGHT_MIN,
        "stand_height_max": _STAND_HEIGHT_MAX,
        "high_stand_route_threshold": _HIGH_STAND_ROUTE_THRESHOLD,
        "high_stand_sentinel": _HIGH_STAND_SENTINEL,
    }


@tool
def g1_list_stand_height_envelope() -> dict[str, Any]:
    """Return the stand-height envelope the neon bundle observed as walkable.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for ``LocoClient.SetStandHeight`` / ``LocoClient.HighStand`` is
    called, so a caller can compare an intended ``height`` value
    against the envelope the neon bundle's ``g1_set_stand_height``
    refused outside of, and can also compare the driver's live
    ``fsm_id`` (from ``G1Driver.get_status``) against
    ``walk_ready_fsm_ids`` to decide whether the locomotion write
    gate is currently open.

    The envelope names two conventions the neon bundle stacks on top
    of the SDK: a bounded ``SetStandHeight`` range on non-negative
    ``height`` (``[stand_height_min, stand_height_max]``, both
    inclusive - a saturated command is a command), and a sentinel
    route from any strictly-negative ``height`` to the SDK's
    ``HighStand()`` method (which itself dispatches a
    ``SetStandHeight`` handler call with the ``UINT32_MAX``
    sentinel the SDK reserves for "max stand"). The ``HighStand``
    branch is admitted for every strictly-negative float; the
    boundary is at ``high_stand_route_threshold = 0.0`` and is
    strict, so ``-0.0`` routes as a low-stand and ``-1e-300`` (or any
    other strictly-negative sentinel) routes as ``HighStand``.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        both conventions (``stand_height_min``, ``stand_height_max``,
        ``high_stand_route_threshold``, ``high_stand_sentinel``); a
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
    the envelope check needs only the finiteness half of the shared
    validator; the positivity half does not apply (a strictly-negative
    ``height`` is a legitimate agent-facing sentinel that routes to
    ``HighStand`` rather than a bounds violation). A future
    consolidation with the shared validator lands when the
    driver-side write verb reuses this admits function.
    """
    return math.isfinite(float(value))


@tool
def g1_stand_height_admits(height: float = 0.0) -> dict[str, Any]:
    """Decide whether a ``height`` value sits inside the stand-height envelope.

    Read-only. Compares the argument against the clamps
    :func:`g1_list_stand_height_envelope` returns and reports which
    SDK path a would-be dispatch would take (``SetStandHeight`` or
    ``HighStand``) plus any bound the value violates. No driver
    instance, no DDS, no SDK: the decision reads only module-level
    constants and the argument itself.

    A value inside the envelope is *not* the same as an admitted
    write: the driver's motion gate (``_check_motion_gates``) also
    refuses on ``_fsm_id`` outside
    :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, which this
    verb does not read (that is a live driver-instance query answered
    by :mod:`~strands_robots.tools.g1.g1_state` and
    :mod:`~strands_robots.tools.g1.g1_motion_gates`). The returned
    envelope names ``walk_ready_fsm_ids`` so a caller comparing an
    intended write against both conditions has the FSM set on hand.

    The routing conventions:

    * ``height >= 0.0`` and inside ``[stand_height_min,
      stand_height_max]`` -> admitted, ``route="set_stand_height"``.
      The SDK call a future write verb would fire is
      ``LocoClient.SetStandHeight(height)``. Both bounds are
      inclusive (``height < min`` refuses, ``height > max`` refuses)
      because a saturated posture is a legitimate posture command.
    * ``height < 0.0`` (strictly) -> admitted, ``route="high_stand"``.
      The SDK call a future write verb would fire is
      ``LocoClient.HighStand()``, which itself passes the
      ``UINT32_MAX`` sentinel through the same
      ``SetStandHeight`` handler. Every strictly-negative float
      routes here; there is no upper or lower magnitude bound on
      the sentinel because the SDK never reads it (only the sign is
      consulted at the agent-surface layer).
    * ``height`` non-finite (``math.inf``, ``math.nan``) -> refused,
      ``comparison="non-finite"``. A NaN cannot be routed decidably
      (``nan < 0`` is ``False`` but so is ``nan >= 0``), and an
      infinity would either overrun ``stand_height_max`` or, on
      the negative branch, route as an unbounded HighStand
      sentinel - both are shape violations rather than value ones.

    Args:
        height: Target stand height in metres, with the neon-bundle
            sentinel convention: ``height >= 0.0`` is a
            ``SetStandHeight`` argument in metres,
            ``height < 0.0`` (strictly) is a ``HighStand`` sentinel.
            ``-0.0`` is admitted as a low-stand (not HighStand) to
            match the neon bundle's own ``if height < 0`` conditional,
            which reads ``-0.0`` as non-negative.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether the
        value would reach the controller (as either a set-stand-height
        or a high-stand write); a ``route`` string naming which SDK
        method a future write verb would call
        (``"set_stand_height"`` or ``"high_stand"``); a ``refusals``
        list carrying the refusal descriptors on a rejected value,
        each with the offending value, the bound it violated, the
        comparison, and the ``7404`` gate-refusal code the driver
        would quote if the write were attempted while the value is
        outside the envelope; the same ``envelope`` sub-dict
        :func:`g1_list_stand_height_envelope` returns; and
        ``walk_ready_fsm_ids`` for the follow-on gate decision. On an
        admitted value the ``refusals`` list is empty and ``route``
        names which SDK path a would-be write would take; on a
        rejected value ``route`` is ``None`` and every violated
        bound is named.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []
    route: str | None = None

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
        _reject(height, "stand_height_max", _STAND_HEIGHT_MAX, "non-finite")
    else:
        h = float(height)
        if h < _HIGH_STAND_ROUTE_THRESHOLD:
            # Strictly-negative height routes to HighStand; every finite
            # negative value is admitted because the SDK's HighStand
            # method reads only the sentinel it passes internally, not
            # the caller's own magnitude. -0.0 does not reach here
            # (-0.0 < 0.0 is False), matching the neon bundle's
            # conditional.
            route = "high_stand"
        elif h < _STAND_HEIGHT_MIN:
            # Unreachable with the current constants
            # (_STAND_HEIGHT_MIN == _HIGH_STAND_ROUTE_THRESHOLD == 0.0)
            # but named explicitly so a future revision that widens
            # the LOW-stand floor upward - so a caller passing
            # 0 <= height < min lands on a refusal rather than on
            # a silently-admitted below-floor stand - falls through
            # this branch without a code change.
            _reject(height, "stand_height_min", _STAND_HEIGHT_MIN, "value < bound")
        elif h > _STAND_HEIGHT_MAX:
            _reject(height, "stand_height_max", _STAND_HEIGHT_MAX, "value > bound")
        else:
            route = "set_stand_height"

    return {
        "status": "success",
        "admits": not refusals,
        "route": route,
        "refusals": refusals,
        "envelope": envelope,
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
    }
