"""Agent-facing lookup for the task ids the G1 locomotion SDK admits dispatches on.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes a
task dispatcher via ``SetTaskId(int)`` that admits a fixed set of
pre-programmed gesture tasks (``0`` WaveHand-no-turn,
``1`` WaveHand-with-turn, ``2`` ShakeHand-reach, ``3`` ShakeHand-shake);
its handler returns ``rc=7303`` ("Invalid task id (loco)") on every
integer outside that set. This module surfaces the id table to an agent
so a caller can decide the refusal decidably before a future dispatch
path is attempted, rather than triggering it from the SDK at wire time.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_set_task_id`` verb wrapped
  ``LocoClient.SetTaskId(id)`` directly; that call is a locomotion-SDK
  write, which today's :class:`~strands_robots.drivers.g1.G1Driver`
  gates through
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
  before ``send_action`` / ``run_policy`` accept a joint payload. A
  future driver method that fronts ``SetTaskId`` will land the
  dispatch verb; refs strands-labs/robots#358 for the SDK-facing
  gate work that write belongs on. This module ports the read-only
  lookup half without also introducing a second locomotion writer
  path the driver does not yet own.
* An SDK re-import. The id table is captured here as a module-level
  constant snapshot of the four task ids the SDK's ``SetTaskId``
  handler admits today (the same set the neon bundle observed against
  the real robot); the constant lives here rather than being
  re-imported from the SDK so
  ``import strands_robots.tools.g1.g1_loco_task_ids`` pulls no
  ``unitree_sdk2py`` submodule - the import-hygiene contract every
  other file in this package carries, refs strands-labs/robots#358.
  An SDK release that widens or narrows the id set is a driver-side
  update; when the driver's dispatch method lands, its refusal will
  name the ``rc=7303`` error the same
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` entry this
  lookup returns, so both sides quote the same text.

What this module does not decide.

* Whether the FSM currently admits the write. Task dispatch is
  locomotion-SDK-shaped: the SDK's ``SetTaskId`` is refused by the
  driver's gate outside :data:`WALK_FSMS` (which is narrower than
  :data:`HANDSHAKE_FSMS` because sitting accepts arm gestures but not
  walking-track dispatch). Gate membership is answered by
  :mod:`~strands_robots.tools.g1.g1_motion_gates`; this lookup is the
  ``SetTaskId``-side of the same conversation, so the returned
  descriptor carries an ``admits_loco_writes`` flag but the flag is
  always ``True`` here (every admitted task is a locomotion write and
  every locomotion write needs the same gate). The flag is surfaced
  anyway so the payload shape matches
  :mod:`~strands_robots.tools.g1.g1_fsm_targets` verbatim.
* Which FSM ids the SDK admits as transition targets. That is the
  ``SetFsmId``-side lookup, answered by
  :mod:`~strands_robots.tools.g1.g1_fsm_targets`. Tasks are dispatched
  *while inside* a walking-ready FSM; a caller planning a task-and-
  transition chain compares the driver's live ``fsm_id`` against
  ``g1_fsm_targets`` (to pick a reachable walk FSM) and against this
  verb (to pick a task the SDK admits at that FSM).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS

#: Snapshot of the task ids the Unitree G1 locomotion SDK's
#: ``SetTaskId`` admits as gesture dispatches today. The labels are the
#: names the neon bundle's ``g1_set_task_id`` docstring observed
#: against the real robot (the SDK does not ship a canonical
#: id -> name mapping and its example scripts name them by
#: locomotion-behaviour rather than by id). Two of the four labels
#: (``0`` and ``1``) resolve to the same neon-bundle wrapper
#: ``g1_wave_hand_loco``; the two-entry difference is whether the
#: robot turns while waving. Two others (``2`` and ``3``) are the two
#: stages of a shake-hand sequence: reach out, then shake.
#:
#: The snapshot lives here rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because the mapping is
#: only useful for the ``SetTaskId``-side of the conversation; a
#: caller that only needs the write gate reaches :data:`WALK_FSMS`
#: directly. Colocating the label table with the dispatch verb mirrors
#: ``_ARM_ACTION_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_arm_actions` and ``_FSM_NAME_MAP``
#: in :mod:`~strands_robots.tools.g1.g1_fsm_targets`: one snapshot per
#: SDK-facing table, one verb pair per snapshot.
_LOCO_TASK_MAP: dict[int, str] = {
    0: "WaveHand (no turn)",
    1: "WaveHand (with turn)",
    2: "ShakeHand stage 1 (reach out)",
    3: "ShakeHand stage 2 (shake)",
}

#: Task ids that are stages of a multi-step sequence rather than a
#: standalone gesture. ``2`` (ShakeHand reach) and ``3`` (ShakeHand
#: shake) are only useful in order; the neon bundle's
#: ``g1_shake_hand_loco`` wrapper drove them sequentially. Called out
#: separately so a caller planning a one-shot task filters against
#: this set before dispatch: dispatching ``3`` without ``2`` leaves
#: the robot mid-gesture with no reach-out preamble.
_SEQUENCED_TASK_IDS: frozenset[int] = frozenset({2, 3})

#: The error-table entry the SDK's ``SetTaskId`` quotes on an id
#: outside :data:`_LOCO_TASK_MAP`. Named here so the returned envelope
#: carries the exact refusal string a future driver-side wrapper would
#: surface, and so a re-wording of it lands in one place instead of
#: drifting between the SDK-side log and this lookup.
_INVALID_TASK_CODE: int = 7303

#: The error-table entry the driver's own ``_check_motion_gates``
#: quotes when it refuses a write on an FSM outside its admitted gate.
#: Named here because :func:`g1_loco_task_admits` surfaces it alongside
#: the SDK's ``7303`` on the returned refusal list - task dispatch is a
#: locomotion write and gets the same gate refusal a follow-up
#: ``send_action`` would face if the driver's live FSM sits outside
#: :data:`WALK_FSMS`. The two codes together are the pair of refusals
#: a caller would face on the same dispatch attempt.
_GATE_REFUSAL_CODE: int = 7404


def _describe(task_id: int) -> dict[str, Any]:
    """Build the per-id descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_loco_tasks` so
    :func:`g1_loco_task_admits`'s admitted-path payload names the same
    fields, and so a widen to the descriptor lands in one place. Every
    field is a snapshot read; no bus is touched. The
    ``admits_loco_writes`` flag is always ``True`` on every admitted
    id (task dispatch *is* a locomotion write); it is surfaced anyway
    so the returned shape matches
    :mod:`~strands_robots.tools.g1.g1_fsm_targets` verbatim and a
    caller consuming both verbs sees the same descriptor keys.
    """
    return {
        "task_id": task_id,
        "name": _LOCO_TASK_MAP[task_id],
        "sequenced": task_id in _SEQUENCED_TASK_IDS,
        "admits_loco_writes": True,
    }


@tool
def g1_list_loco_tasks() -> dict[str, Any]:
    """Return the task ids ``LocoClient.SetTaskId`` admits as dispatches.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for ``SetTaskId`` is called, so a caller can compare an intended
    task id against the set the SDK's dispatcher would test membership
    in, and decide alongside that whether the task is a stage of a
    multi-step sequence (dispatching stage 2 without stage 1 leaves
    the robot mid-gesture).

    Returns:
        A dict with ``status``, a ``count`` naming the number of
        admitted task ids, a ``loco_tasks`` list of descriptors
        (one per admitted id, sorted ascending) carrying ``task_id``,
        ``name`` (the neon-bundle label), a ``sequenced`` flag naming
        whether the task is a stage of a multi-step sequence
        (``2`` ShakeHand reach, ``3`` ShakeHand shake), and
        ``admits_loco_writes`` (always ``True`` here, surfaced for
        shape parity with
        :mod:`~strands_robots.tools.g1.g1_fsm_targets`). A separate
        ``sequenced_ids`` field lists the stage-ids as a set, so a
        caller filtering for one-shot tasks compares against that set
        directly rather than walking the descriptors. A
        ``loco_ready_fsm_ids`` field mirrors :data:`WALK_FSMS` (the
        driver-side gate every task dispatch shares with locomotion
        velocity commands); a ``refusals`` list carries the two
        refusal codes (``7303`` invalid task id, ``7404`` gate-refused
        write) and their decoded text that a future dispatch verb
        would surface. Every field is a snapshot of an SDK or driver
        constant; no dynamic decode runs here.
    """
    task_ids = sorted(_LOCO_TASK_MAP)
    return {
        "status": "success",
        "count": len(_LOCO_TASK_MAP),
        "loco_tasks": [_describe(tid) for tid in task_ids],
        "task_ids": task_ids,
        "sequenced_ids": sorted(_SEQUENCED_TASK_IDS),
        "loco_ready_fsm_ids": sorted(WALK_FSMS),
        "refusals": [
            {"code": _INVALID_TASK_CODE, "text": ERR_CODES[_INVALID_TASK_CODE]},
            {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
        ],
    }


@tool
def g1_loco_task_admits(
    task_id: int | None = None,
    name: str = "",
) -> dict[str, Any]:
    """Decide whether a task id or name is inside the SDK's dispatch set.

    Read-only. Reads the module's snapshot of the SDK's task table and
    returns the same membership answer the SDK's ``SetTaskId`` would
    compute at wire time. A caller with either an integer id or a
    neon-bundle label resolves it against the SDK's set before a
    future dispatch verb dispatches, rather than triggering the SDK's
    ``rc=7303`` refusal at wire time.

    Exactly one of ``task_id`` (an int) or ``name`` (a string) must be
    supplied. Supplying both, or neither, carries ``status="error"``:
    the ambiguous case is a caller mistake, not a lookup this verb
    should resolve arbitrarily.

    Args:
        task_id: The task dispatch id to test. Must be an ``int``;
            ``bool`` is refused (``True`` is ``int(1)`` but a
            passed-through boolean is a caller mistake, not a valid
            dispatch query).
        name: The neon-bundle label to test. Case-sensitive to match
            the snapshot's own keys (a caller writing ``"wavehand"``
            gets a key-not-found; the snapshot ships
            ``"WaveHand (no turn)"``). Empty string means "no name
            supplied".

    Returns:
        A dict with ``status`` (``"success"`` on any decidable
        answer, ``"error"`` on the both-supplied / neither-supplied
        ambiguity), a ``query`` sub-dict carrying whichever of
        ``task_id`` / ``name`` was supplied, an ``admitted`` boolean
        naming whether the SDK's ``SetTaskId`` would admit the query,
        and (when ``admitted`` is ``True``) a ``target`` sub-dict
        carrying the same descriptor :func:`g1_list_loco_tasks`
        returns for the resolved id (``task_id``, ``name``,
        ``sequenced``, ``admits_loco_writes``) so a caller sees the
        sequenced flag on the same call. On a not-admitted query the
        dict carries ``refusal_code`` / ``refusal_text`` naming the
        ``rc=7303`` refusal the SDK would return.
    """
    supplied_id = task_id is not None
    supplied_name = bool(name)
    if supplied_id == supplied_name:
        return {
            "status": "error",
            "message": (
                "supply exactly one of task_id= (int) or name= (str); "
                f"got task_id={task_id!r}, name={name!r}. "
                "Refs strands-labs/robots#358."
            ),
        }
    if supplied_id and isinstance(task_id, bool):
        return {
            "status": "error",
            "message": (f"task_id must be int, got bool ({task_id!r}). Refs strands-labs/robots#358."),
        }
    if supplied_id and not isinstance(task_id, int):
        return {
            "status": "error",
            "message": (
                f"task_id must be int, got {type(task_id).__name__} ({task_id!r}). Refs strands-labs/robots#358."
            ),
        }

    if supplied_id:
        admitted = task_id in _LOCO_TASK_MAP
        resolved_id = task_id if admitted else None
        query: dict[str, Any] = {"task_id": task_id}
    else:
        # Reverse-lookup: snapshot is small (4 entries) so a linear
        # scan is fine and avoids maintaining a name->id dict on the
        # side.
        resolved_id = next(
            (tid for tid, label in _LOCO_TASK_MAP.items() if label == name),
            None,
        )
        admitted = resolved_id is not None
        query = {"name": name}

    if not admitted:
        return {
            "status": "success",
            "admitted": False,
            "query": query,
            "refusal_code": _INVALID_TASK_CODE,
            "refusal_text": ERR_CODES[_INVALID_TASK_CODE],
        }

    assert resolved_id is not None  # mypy narrowing: admitted implies resolved
    return {
        "status": "success",
        "admitted": True,
        "query": query,
        "target": _describe(resolved_id),
    }
