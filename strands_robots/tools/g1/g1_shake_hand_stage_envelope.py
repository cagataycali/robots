"""Agent-facing lookup for the stage argument ``LocoClient.ShakeHand`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes
``ShakeHand(stage: int = -1)`` as a two-stage gesture dispatch: stage
``0`` sends the arm out (reach), stage ``1`` shakes the extended hand,
and ``-1`` asks the SDK to toggle the internal stage counter (a
sentinel the SDK's own default reads through). The neon bundle's
``g1_shake_hand_loco`` wrapper
(``cagataycali/neon-the-g1/tools/g1_locomotion.py::g1_shake_hand_loco``)
fronts that with a matching three-valued ``stage`` argument and calls
``LocoClient.ShakeHand(stage=int(stage))`` verbatim; every value outside
``{-1, 0, 1}`` reaches the SDK's own dispatcher, which decodes it
against a fixed internal table and returns ``rc=7303``
("Invalid task id (loco)") - the same refusal
:mod:`~strands_robots.tools.g1.g1_loco_task_ids` names on ``SetTaskId``
because the SDK routes ``ShakeHand`` through the same task-dispatch
handler at the wire. This module snapshots the three admitted stages
into a module-level constant and exposes two agent-facing verbs -
:func:`g1_list_shake_hand_stages` (list the whole envelope) and
:func:`g1_shake_hand_stage_admits` (decide one query) - so a caller can
decide the refusal decidably before a future driver-side wrapper for
``ShakeHand`` fires.

Refs strands-labs/robots#358.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_shake_hand_loco`` verb
  called ``LocoClient.ShakeHand(stage=int(stage))`` directly under the
  same DDS singleton
  :func:`~strands_robots.tools.g1._g1_common.ensure_dds` the driver
  holds; that write is the same locomotion topic
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` gates, which
  today's :class:`~strands_robots.drivers.g1.G1Driver` refuses through
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` on
  any locomotion-shaped write while ``_fsm_id`` is outside that set.
  A future driver method that fronts the ``stage`` surface will land
  the write verb; refs strands-labs/robots#358 for the SDK-facing
  gate work that write belongs on. This module ports the read-only
  envelope half without also introducing a second locomotion writer
  path the driver does not yet own.
* An SDK re-import. The stage table is captured here as a module-level
  constant snapshot of the three values the SDK's ``ShakeHand``
  dispatcher admits today (the same set the neon bundle observed
  against the real robot); the constant lives here rather than being
  re-imported from the SDK so
  ``import strands_robots.tools.g1.g1_shake_hand_stage_envelope`` pulls
  no ``unitree_sdk2py`` submodule - the import-hygiene contract every
  other file in this package carries, refs strands-labs/robots#358. A
  firmware release that widened or narrowed the stage set is a
  driver-side update; when the driver's ``ShakeHand`` wrapper lands,
  its refusal will name the same ``rc=7303`` code the entry in
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.

What this module does not decide.

* Whether the driver's live ``_fsm_id`` sits inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That is a
  driver-instance read carried on the driver's ``get_status``
  envelope; a caller planning a shake-hand dispatch compares the
  driver's live fsm against :func:`g1_list_shake_hand_stages`'s
  ``walk_ready_fsm_ids`` to see whether the write gate is currently
  open. The two membership tests together - envelope for stage, gate
  for the FSM - are the conditions a future write verb would refuse
  on. The gate answer is available on
  :mod:`~strands_robots.tools.g1.g1_motion_gates`; this module carries
  the FSM set on its returned descriptor so a caller consuming both
  verbs sees the same list.
* The stage ordering rule. Stage ``0`` (reach) is the preamble the
  neon bundle observed as the required predecessor to stage ``1``
  (shake); dispatching ``1`` without a prior ``0`` leaves the arm
  reaching a hand it has not extended, which the SDK admits (returns
  ``rc=0``) but the controller does not honour. The sentinel ``-1``
  is the SDK's own "toggle" value: it advances the internal counter
  from whatever it last was, so two ``-1`` calls in a row execute
  the reach-and-shake pair. The stage table names each of the three
  in its label and marks ``0`` / ``1`` as ``sequenced=True`` (the
  same field :mod:`~strands_robots.tools.g1.g1_loco_task_ids` uses
  for the ``2`` / ``3`` reach-and-shake pair on ``SetTaskId``, which
  is the same handler behind the same table); ``-1`` is
  ``sequenced=False`` because a caller that means to fire the whole
  sequence with one wrapper uses the toggle rather than the ordered
  pair. Enforcing the ordering is the caller's decision and the
  driver's refusal string domain, not this lookup's.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS

#: Snapshot of the ``stage`` values ``LocoClient.ShakeHand`` admits as
#: dispatches today. The labels are the names the neon bundle's
#: ``g1_shake_hand_loco`` docstring observed against the real robot
#: (the SDK does not ship a canonical stage -> name mapping and its
#: example scripts name the two ordered stages by the arm motion they
#: trigger). The sentinel ``-1`` is the SDK's own default: it advances
#: the internal stage counter from whatever it last was, so two ``-1``
#: calls in a row execute the reach-and-shake pair without the caller
#: naming either stage id explicitly.
#:
#: The snapshot lives here rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because the mapping is
#: only useful for the ``ShakeHand``-side of the conversation; a
#: caller that only needs the write gate reaches :data:`WALK_FSMS`
#: directly. Colocating the label table with the dispatch verb mirrors
#: ``_LOCO_TASK_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_loco_task_ids`: one snapshot per
#: SDK-facing table, one verb pair per snapshot.
_SHAKE_HAND_STAGE_MAP: dict[int, str] = {
    -1: "Toggle (advance internal stage counter)",
    0: "ShakeHand stage 1 (reach out)",
    1: "ShakeHand stage 2 (shake)",
}

#: Stage values that are members of an ordered pair rather than a
#: standalone dispatch. Stage ``0`` (reach) and stage ``1`` (shake)
#: are only useful in order - dispatching ``1`` without a prior ``0``
#: leaves the arm reaching for a hand it has not extended, which the
#: SDK admits (``rc=0``) but the controller does not honour. Called
#: out separately so a caller planning a one-shot toggle filters
#: against this set before dispatch: a caller using ``-1`` skips the
#: ordering question because the SDK advances the counter itself.
_SEQUENCED_STAGES: frozenset[int] = frozenset({0, 1})

#: The stage argument value that means "advance the internal counter"
#: to the SDK's ``ShakeHand`` dispatcher. Named here so a caller can
#: pick the toggle explicitly without also knowing the ordered-pair
#: values, and so a firmware release that renamed the sentinel lands
#: as a shape change on this constant rather than as a silent
#: divergence in the tests.
_TOGGLE_STAGE: int = -1

#: The error-table entry the SDK's ``ShakeHand`` dispatcher quotes on
#: a stage outside :data:`_SHAKE_HAND_STAGE_MAP`. Named here so the
#: returned envelope carries the exact refusal string a future
#: driver-side wrapper would surface, and so a re-wording of it lands
#: in one place instead of drifting between the SDK-side log and this
#: lookup. The SDK routes ``ShakeHand`` through the same task-dispatch
#: handler that :mod:`~strands_robots.tools.g1.g1_loco_task_ids` names
#: on ``SetTaskId``, so both lookups quote the same ``7303`` code.
_INVALID_STAGE_CODE: int = 7303

#: The error-table entry the driver's own ``_check_motion_gates``
#: quotes when it refuses a locomotion-shaped write on an FSM outside
#: :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. Named here
#: because :func:`g1_shake_hand_stage_admits` surfaces it alongside
#: the SDK's ``7303`` on the returned refusal list - shake-hand
#: dispatch is a locomotion write and gets the same gate refusal a
#: follow-up ``send_action`` would face if the driver's live FSM sits
#: outside :data:`WALK_FSMS`. The two codes together are the pair of
#: refusals a caller would face on the same dispatch attempt.
_GATE_REFUSAL_CODE: int = 7404


def _describe(stage: int) -> dict[str, Any]:
    """Build the per-stage descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_shake_hand_stages`
    so :func:`g1_shake_hand_stage_admits`'s admitted-path payload names
    the same fields, and so a widen to the descriptor lands in one
    place. Every field is a snapshot read; no bus is touched. The
    ``admits_loco_writes`` flag is always ``True`` on every admitted
    stage (every ``ShakeHand`` dispatch *is* a locomotion write); it
    is surfaced anyway so the returned shape matches
    :mod:`~strands_robots.tools.g1.g1_loco_task_ids` verbatim and a
    caller consuming both verbs sees the same descriptor keys.
    """
    return {
        "stage": stage,
        "name": _SHAKE_HAND_STAGE_MAP[stage],
        "sequenced": stage in _SEQUENCED_STAGES,
        "toggle": stage == _TOGGLE_STAGE,
        "admits_loco_writes": True,
    }


@tool
def g1_list_shake_hand_stages() -> dict[str, Any]:
    """Return the stage values ``LocoClient.ShakeHand`` admits as dispatches.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for ``ShakeHand`` is called, so a caller can compare an intended
    ``stage`` argument against the set the SDK's dispatcher would test
    membership in, and decide alongside that whether the stage is a
    member of the ordered reach-and-shake pair (dispatching stage 1
    without stage 0 leaves the arm reaching for a hand it has not
    extended, which the SDK admits with ``rc=0`` but the controller
    does not honour).

    The envelope names three admitted stages: ``-1`` (the SDK's own
    toggle sentinel, which advances the internal stage counter and
    executes the next pair member without the caller naming either id
    explicitly), ``0`` (reach out) and ``1`` (shake). A separate
    ``sequenced_stages`` field lists the ordered-pair values, so a
    caller filtering for a one-shot toggle compares against that set
    directly rather than walking the descriptors. A
    ``walk_ready_fsm_ids`` field mirrors :data:`WALK_FSMS` (the
    driver-side gate every ``ShakeHand`` dispatch shares with
    locomotion velocity commands).

    Returns:
        A dict with ``status``; a ``count`` naming the number of
        admitted stage values; a ``shake_hand_stages`` list of
        descriptors (one per admitted stage, sorted ascending)
        carrying ``stage``, ``name`` (the neon-bundle label), a
        ``sequenced`` flag naming whether the stage is a member of
        the ordered reach-and-shake pair, a ``toggle`` flag naming
        whether the stage is the SDK's own advance-the-counter
        sentinel, and ``admits_loco_writes`` (always ``True`` here,
        surfaced for shape parity with
        :mod:`~strands_robots.tools.g1.g1_loco_task_ids`); a
        ``stages`` field listing the admitted values sorted
        ascending; a ``sequenced_stages`` list carrying the ordered-
        pair values; a ``toggle_stage`` field naming the SDK's own
        advance-the-counter sentinel; a ``walk_ready_fsm_ids`` list
        mirroring :data:`WALK_FSMS`; and a ``refusals`` list carrying
        the two refusal codes (``7303`` invalid stage, ``7404`` gate-
        refused write) and their decoded text a future dispatch verb
        would surface. Every field is a snapshot of an SDK or driver
        constant; no dynamic decode runs here.
    """
    stages = sorted(_SHAKE_HAND_STAGE_MAP)
    return {
        "status": "success",
        "count": len(_SHAKE_HAND_STAGE_MAP),
        "shake_hand_stages": [_describe(stage) for stage in stages],
        "stages": stages,
        "sequenced_stages": sorted(_SEQUENCED_STAGES),
        "toggle_stage": _TOGGLE_STAGE,
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
        "refusals": [
            {"code": _INVALID_STAGE_CODE, "text": ERR_CODES[_INVALID_STAGE_CODE]},
            {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
        ],
    }


@tool
def g1_shake_hand_stage_admits(stage: int = -1) -> dict[str, Any]:
    """Decide whether a ``stage`` value is inside the SDK's dispatch set.

    Read-only. Reads the module's snapshot of the SDK's stage table
    and returns the same membership answer the SDK's ``ShakeHand``
    dispatcher would compute at wire time. A caller with an integer
    stage resolves it against the SDK's set before a future dispatch
    verb dispatches, rather than triggering the SDK's ``rc=7303``
    refusal at wire time.

    A stage inside the SDK's set is *not* the same as an admitted
    write: the driver's motion gate (``_check_motion_gates``) also
    refuses on ``_fsm_id`` outside
    :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, which this
    verb does not read (that is a live driver-instance query answered
    by :mod:`~strands_robots.tools.g1.g1_state` and
    :mod:`~strands_robots.tools.g1.g1_motion_gates`). The returned
    envelope names ``walk_ready_fsm_ids`` so a caller comparing an
    intended write against both conditions has the FSM set on hand.

    Args:
        stage: The stage argument to test. Must be an ``int``;
            ``bool`` is refused (``True`` is ``int(1)`` but a
            passed-through boolean is a caller mistake, not a valid
            dispatch query). The three admitted values are ``-1``
            (the SDK's own toggle sentinel), ``0`` (reach out) and
            ``1`` (shake).

    Returns:
        A dict with ``status``; a ``query`` sub-dict carrying the
        supplied ``stage``; an ``admitted`` boolean naming whether the
        SDK's ``ShakeHand`` dispatcher would admit the query; and
        (when ``admitted`` is ``True``) a ``target`` sub-dict carrying
        the same descriptor :func:`g1_list_shake_hand_stages` returns
        for the resolved stage (``stage``, ``name``, ``sequenced``,
        ``toggle``, ``admits_loco_writes``) so a caller sees the
        sequenced flag on the same call. On a not-admitted query the
        dict carries ``refusal_code`` / ``refusal_text`` naming the
        ``rc=7303`` refusal the SDK would return. On a shape error
        (``bool``, non-int) the dict carries ``status="error"`` with
        a message naming the type refused.
    """
    if isinstance(stage, bool):
        return {
            "status": "error",
            "message": (f"stage must be int, got bool ({stage!r}). Refs strands-labs/robots#358."),
        }
    if not isinstance(stage, int):
        return {
            "status": "error",
            "message": (f"stage must be int, got {type(stage).__name__} ({stage!r}). Refs strands-labs/robots#358."),
        }

    admitted = stage in _SHAKE_HAND_STAGE_MAP
    if not admitted:
        return {
            "status": "success",
            "admitted": False,
            "query": {"stage": stage},
            "refusal_code": _INVALID_STAGE_CODE,
            "refusal_text": ERR_CODES[_INVALID_STAGE_CODE],
        }

    return {
        "status": "success",
        "admitted": True,
        "query": {"stage": stage},
        "target": _describe(stage),
    }
