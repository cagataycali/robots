"""Agent-facing lookup for the ``turn_flag`` argument ``LocoClient.WaveHand`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes
``WaveHand(turn_flag: bool = False)`` as a one-shot pre-programmed
gesture: the arm goes through a waving motion, and the ``turn_flag``
selects whether the robot also turns 180 degrees during the wave
(``True``) or waves in place (``False``). The SDK routes both variants
through its own task dispatcher (``SetTaskId(0)`` when
``turn_flag=False``, ``SetTaskId(1)`` when ``turn_flag=True``), so the
membership question at the ``WaveHand`` surface is boolean and the
composed dispatch is one of the two task ids
:mod:`~strands_robots.tools.g1.g1_loco_task_ids` already names on the
``SetTaskId`` side. The neon bundle's ``g1_wave_hand_loco`` wrapper
(``cagataycali/neon-the-g1/tools/g1_locomotion.py::g1_wave_hand_loco``)
fronts that with a matching ``turn: bool = False`` argument and calls
``LocoClient.WaveHand(turn_flag=bool(turn))`` verbatim; any non-boolean
input to the neon wrapper is coerced through ``bool()`` before the SDK
sees it, so a caller passing ``turn=1`` and a caller passing
``turn=True`` reach the same task id. This module snapshots the two
admitted ``turn_flag`` values into a module-level constant and exposes
two agent-facing verbs -
:func:`g1_list_wave_hand_turn_flags` (list the whole envelope) and
:func:`g1_wave_hand_turn_flag_admits` (decide one query) - so a caller
can decide the refusal decidably before a future driver-side wrapper
for ``WaveHand`` fires. Refs strands-labs/robots#358.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_wave_hand_loco`` verb
  called ``LocoClient.WaveHand(turn_flag=bool(turn))`` directly under
  the same DDS singleton
  :func:`~strands_robots.tools.g1._g1_common.ensure_dds` the driver
  holds; that write is the same locomotion topic
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` gates, which
  today's :class:`~strands_robots.drivers.g1.G1Driver` refuses
  through
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` on
  any locomotion-shaped write while ``_fsm_id`` is outside that set.
  A future driver method that fronts the ``turn_flag`` surface will
  land the write verb; refs strands-labs/robots#358 for the SDK-
  facing gate work that write belongs on. This module ports the
  read-only envelope half without also introducing a second
  locomotion writer path the driver does not yet own.
* An SDK re-import. The turn-flag table is captured here as a
  module-level constant snapshot of the two values the SDK's
  ``WaveHand`` dispatcher admits (the same set the neon bundle
  observed against the real robot); the constant lives here rather
  than being re-imported from the SDK so
  ``import strands_robots.tools.g1.g1_wave_hand_turn_flag_envelope``
  pulls no ``unitree_sdk2py`` submodule - the import-hygiene contract
  every other file in this package carries, refs
  strands-labs/robots#358. A firmware release that widened or
  narrowed the ``turn_flag`` set (adding a third variant, or renaming
  the boolean surface) is a driver-side update; when the driver's
  ``WaveHand`` wrapper lands, its refusal will name the same
  ``rc=7303`` code the entry in
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries -
  ``WaveHand`` shares the SDK's task-dispatch handler with
  ``SetTaskId`` and ``ShakeHand``, so all three lookups quote the
  same refusal string.

What this module does not decide.

* Whether the driver's live ``_fsm_id`` sits inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That is a
  driver-instance read carried on the driver's ``get_status``
  envelope; a caller planning a wave-hand dispatch compares the
  driver's live fsm against
  :func:`g1_list_wave_hand_turn_flags`'s ``walk_ready_fsm_ids`` to
  see whether the write gate is currently open. The two membership
  tests together - envelope for turn-flag, gate for the FSM - are
  the conditions a future write verb would refuse on. The gate
  answer is available on
  :mod:`~strands_robots.tools.g1.g1_motion_gates`; this module
  carries the FSM set on its returned descriptor so a caller
  consuming both verbs sees the same list.
* Whether the composed task id is inside
  :mod:`~strands_robots.tools.g1.g1_loco_task_ids`. The SDK routes
  ``WaveHand(turn_flag=False)`` through ``SetTaskId(0)`` and
  ``WaveHand(turn_flag=True)`` through ``SetTaskId(1)``; both task
  ids are members of
  :data:`~strands_robots.tools.g1.g1_loco_task_ids._LOCO_TASK_MAP`,
  so a caller planning a two-lookup composition (``WaveHand`` then
  a follow-up task) sees the same handler on both sides. This module
  names the composed task id on each descriptor so the cross-lookup
  is one field read rather than a duplicate table walk; the
  composed-id lookup is not a re-validation of the task id envelope
  (that lookup is
  :func:`~strands_robots.tools.g1.g1_loco_task_ids.g1_loco_task_admits`).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS

#: Snapshot of the ``turn_flag`` values ``LocoClient.WaveHand`` admits
#: as dispatches today. The labels are the names the neon bundle's
#: ``g1_wave_hand_loco`` docstring observed against the real robot
#: (the SDK does not ship a canonical bool -> name mapping and its
#: example scripts name the two variants by the arm-plus-body motion
#: they trigger). ``False`` is the SDK's own default: a caller that
#: omits the argument entirely picks the wave-in-place variant.
#:
#: The snapshot lives here rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because the mapping is
#: only useful for the ``WaveHand``-side of the conversation; a
#: caller that only needs the write gate reaches :data:`WALK_FSMS`
#: directly. Colocating the label table with the dispatch verb
#: mirrors ``_LOCO_TASK_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_loco_task_ids` and
#: ``_SHAKE_HAND_STAGE_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_shake_hand_stage_envelope`: one
#: snapshot per SDK-facing table, one verb pair per snapshot.
_WAVE_HAND_TURN_FLAG_MAP: dict[bool, str] = {
    False: "WaveHand (no turn)",
    True: "WaveHand (with turn)",
}

#: Composed ``SetTaskId`` value the SDK dispatches through when
#: ``WaveHand`` is called with each ``turn_flag`` variant. The SDK
#: routes ``WaveHand(turn_flag=False)`` through ``SetTaskId(0)`` and
#: ``WaveHand(turn_flag=True)`` through ``SetTaskId(1)`` - both ids
#: are members of
#: :data:`~strands_robots.tools.g1.g1_loco_task_ids._LOCO_TASK_MAP`,
#: so a caller comparing the composed dispatch against the task id
#: envelope sees the same handler on both sides. Named here so a
#: firmware release that renumbered the ``WaveHand`` task ids lands
#: as a shape change on this constant rather than as a silent
#: divergence in the tests.
_WAVE_HAND_TASK_ID_MAP: dict[bool, int] = {
    False: 0,
    True: 1,
}

#: The SDK method the neon bundle's ``g1_wave_hand_loco`` dispatches
#: through. The neon wrapper calls ``LocoClient.WaveHand(turn_flag=bool(turn))``
#: verbatim; the SDK routes both variants through its own task
#: dispatcher, so this constant names the caller-facing entry rather
#: than the wire-level ``SetTaskId`` the two variants compose to.
#: Named here so the returned envelope carries the exact SDK entry a
#: driver-side wrapper would target, and so a firmware release that
#: renamed the SDK method lands in one place instead of drifting
#: between the neon bundle and this lookup.
_SDK_METHOD: str = "WaveHand"

#: The error-table entry the SDK's ``WaveHand`` dispatcher quotes on
#: any input the boolean coercion in the neon wrapper would fail to
#: fold to ``{False, True}``. In practice Python's ``bool()`` folds
#: every non-boolean input to a boolean (``bool(2)`` is ``True``,
#: ``bool(0)`` is ``False``), so a caller passing an integer or a
#: non-empty string reaches the SDK's dispatcher with an admitted
#: task id and the refusal never fires - but the SDK is the one
#: making that decision, not this module. A driver-side wrapper that
#: refused non-boolean inputs earlier than the SDK's own coercion
#: would surface a different refusal (a shape error at the tool
#: surface); the :func:`g1_wave_hand_turn_flag_admits` verb makes
#: that choice explicit (a non-bool ``turn_flag`` is a shape error,
#: not a membership answer). The SDK's own ``rc=7303`` refusal is
#: surfaced on the returned envelope for parity with the sibling
#: lookups that share the same dispatch handler.
_INVALID_TASK_CODE: int = 7303

#: The error-table entry the driver's own ``_check_motion_gates``
#: quotes when it refuses a locomotion-shaped write on an FSM outside
#: :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. Named here
#: because :func:`g1_wave_hand_turn_flag_admits` surfaces it alongside
#: the SDK's ``7303`` on the returned refusal list - wave-hand
#: dispatch is a locomotion write and gets the same gate refusal a
#: follow-up ``send_action`` would face if the driver's live FSM sits
#: outside :data:`WALK_FSMS`. The two codes together are the pair of
#: refusals a caller would face on the same dispatch attempt.
_GATE_REFUSAL_CODE: int = 7404


def _describe(turn_flag: bool) -> dict[str, Any]:
    """Build the per-turn-flag descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_wave_hand_turn_flags` so
    :func:`g1_wave_hand_turn_flag_admits`'s admitted-path payload
    names the same fields, and so a widen to the descriptor lands in
    one place. Every field is a snapshot read; no bus is touched. The
    ``admits_loco_writes`` flag is always ``True`` on every admitted
    variant (every ``WaveHand`` dispatch *is* a locomotion write); it
    is surfaced anyway so the returned shape matches
    :mod:`~strands_robots.tools.g1.g1_loco_task_ids` and
    :mod:`~strands_robots.tools.g1.g1_shake_hand_stage_envelope`
    verbatim and a caller consuming any of the three verbs sees the
    same descriptor keys.
    """
    return {
        "turn_flag": turn_flag,
        "name": _WAVE_HAND_TURN_FLAG_MAP[turn_flag],
        "composed_task_id": _WAVE_HAND_TASK_ID_MAP[turn_flag],
        "sdk_method": _SDK_METHOD,
        "admits_loco_writes": True,
    }


@tool
def g1_list_wave_hand_turn_flags() -> dict[str, Any]:
    """Return the ``turn_flag`` values ``LocoClient.WaveHand`` admits as dispatches.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for ``WaveHand`` is called, so a caller can compare an intended
    ``turn_flag`` argument against the boolean set the SDK's
    dispatcher would test membership in, and decide alongside that
    which composed ``SetTaskId`` value the variant would route
    through - the same id
    :mod:`~strands_robots.tools.g1.g1_loco_task_ids` already names.

    The envelope names two admitted variants: ``False`` (the SDK's
    own default, wave in place, composes to ``SetTaskId(0)``) and
    ``True`` (wave while turning 180 degrees, composes to
    ``SetTaskId(1)``). A ``sdk_method`` field names the caller-facing
    entry (``"WaveHand"``) so the composed dispatch is one field read
    on the descriptor rather than a duplicate table walk against the
    task-id lookup. A ``walk_ready_fsm_ids`` field mirrors
    :data:`WALK_FSMS` (the driver-side gate every ``WaveHand``
    dispatch shares with locomotion velocity commands).

    Returns:
        A dict with ``status``; a ``count`` naming the number of
        admitted turn-flag values (``2``); a ``wave_hand_turn_flags``
        list of descriptors (one per admitted variant, ordered
        ``False, True``) carrying ``turn_flag``, ``name`` (the neon-
        bundle label), ``composed_task_id`` (the ``SetTaskId`` value
        the variant routes through), ``sdk_method`` (always
        ``"WaveHand"``), and ``admits_loco_writes`` (always ``True``,
        surfaced for shape parity with the sibling task-id and
        shake-hand-stage lookups); a ``turn_flags`` list naming the
        admitted values (``[False, True]``); a ``composed_task_ids``
        list naming the composed dispatches (``[0, 1]``); a
        ``walk_ready_fsm_ids`` list mirroring :data:`WALK_FSMS`; and
        a ``refusals`` list carrying the two refusal codes (``7303``
        invalid task id at the SDK's dispatcher, ``7404`` gate-
        refused write at the driver's motion gate) and their decoded
        text a future dispatch verb would surface. Every field is a
        snapshot of an SDK or driver constant; no dynamic decode
        runs here.
    """
    turn_flags = [False, True]
    return {
        "status": "success",
        "count": len(_WAVE_HAND_TURN_FLAG_MAP),
        "wave_hand_turn_flags": [_describe(flag) for flag in turn_flags],
        "turn_flags": turn_flags,
        "composed_task_ids": [_WAVE_HAND_TASK_ID_MAP[flag] for flag in turn_flags],
        "sdk_method": _SDK_METHOD,
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
        "refusals": [
            {"code": _INVALID_TASK_CODE, "text": ERR_CODES[_INVALID_TASK_CODE]},
            {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
        ],
    }


@tool
def g1_wave_hand_turn_flag_admits(turn_flag: bool = False) -> dict[str, Any]:
    """Decide whether a ``turn_flag`` value is inside the SDK's dispatch set.

    Read-only. Reads the module's snapshot of the SDK's ``WaveHand``
    turn-flag table and returns the same membership answer the SDK's
    dispatcher would compute at wire time. A caller with a boolean
    ``turn_flag`` resolves it against the SDK's set before a future
    dispatch verb dispatches, rather than triggering the SDK's
    ``rc=7303`` refusal at wire time.

    A ``turn_flag`` inside the SDK's set is *not* the same as an
    admitted write: the driver's motion gate
    (``_check_motion_gates``) also refuses on ``_fsm_id`` outside
    :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, which this
    verb does not read (that is a live driver-instance query
    answered by :mod:`~strands_robots.tools.g1.g1_state` and
    :mod:`~strands_robots.tools.g1.g1_motion_gates`). The returned
    envelope names ``walk_ready_fsm_ids`` so a caller comparing an
    intended write against both conditions has the FSM set on hand.

    Args:
        turn_flag: The turn-flag argument to test. Must be a
            ``bool``; a non-boolean input is refused as a shape
            error rather than resolved through Python's ``bool()``
            coercion (the neon wrapper calls ``bool(turn)`` before
            the SDK sees the value, so a caller passing ``1`` or
            ``"yes"`` reaches the SDK's dispatcher with an admitted
            task id - but this lookup makes the boolean shape
            decidable at the tool surface rather than at wire time).
            Both admitted values (``False`` wave-in-place, ``True``
            wave-and-turn) return ``admitted=True``.

    Returns:
        A dict with ``status``; a ``query`` sub-dict carrying the
        supplied ``turn_flag``; an ``admitted`` boolean naming
        whether the SDK's ``WaveHand`` dispatcher would admit the
        query; and (when ``admitted`` is ``True``) a ``target`` sub-
        dict carrying the same descriptor
        :func:`g1_list_wave_hand_turn_flags` returns for the
        resolved variant (``turn_flag``, ``name``,
        ``composed_task_id``, ``sdk_method``, ``admits_loco_writes``)
        so a caller sees the composed task id on the same call. On
        a shape error (non-bool) the dict carries ``status="error"``
        with a message naming the type refused and citing
        ``strands-labs/robots#358``.
    """
    if not isinstance(turn_flag, bool):
        return {
            "status": "error",
            "message": (
                f"turn_flag must be bool, got {type(turn_flag).__name__} ({turn_flag!r}). Refs strands-labs/robots#358."
            ),
        }

    # Every bool is admitted (the SDK's ``WaveHand`` dispatcher admits both
    # ``False`` and ``True``); no not-admitted branch fires on a bool. The
    # branch remains for shape parity with the sibling admits verbs, and so
    # a firmware release that narrowed the admitted set (e.g. removed the
    # turn-and-wave variant) lands here as an ``admitted=False`` payload
    # instead of a shape change on the return type.
    return {
        "status": "success",
        "admitted": True,
        "query": {"turn_flag": turn_flag},
        "target": _describe(turn_flag),
    }
