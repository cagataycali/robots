"""Agent-facing lookup for the FSM ids the G1 locomotion SDK admits transitions to.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes an FSM
selector via ``SetFsmId(int)`` that admits a fixed set of pre-programmed
target states (``1`` Damp, ``500`` Start, ``501`` Walk, ...); its handler
returns ``rc=7302`` ("Invalid FSM id (loco)") on every integer outside
that set. This module surfaces the id table to an agent so a caller can
decide the refusal decidably before a future transition path is
attempted, rather than triggering it from the SDK at wire time.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_set_fsm`` verb wrapped
  ``LocoClient.SetFsmId(id)`` under a single-writer lock; that write
  is the ``rt/lowcmd``-adjacent locomotion topic, which today's
  :class:`~strands_robots.drivers.g1.G1Driver` gates through
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
  before ``send_action`` / ``run_policy`` accept a joint payload. A
  future driver method that fronts ``SetFsmId`` will land the
  transition verb; refs strands-labs/robots#358 for the SDK-facing
  gate work that write belongs on. This module ports the read-only
  lookup half without also introducing a second locomotion writer
  path the driver does not yet own.
* An SDK re-import. The id table is captured here as a module-level
  constant snapshot of the ten FSM ids the SDK's ``SetFsmId`` handler
  admits today (the same set the neon bundle's ``FSM_NAMES`` dict
  ships); the constant lives here rather than being re-imported from
  the SDK so ``import strands_robots.tools.g1.g1_fsm_targets`` pulls
  no ``unitree_sdk2py`` submodule - the import-hygiene contract every
  other file in this package carries, refs strands-labs/robots#358.
  An SDK release that widens or narrows the id set is a driver-side
  update; when the driver's transition method lands, its refusal
  will name the ``rc=7302`` error the same
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` entry this
  lookup returns, so both sides quote the same text.

What this module does not decide.

* Whether the FSM currently admits a *write* on either the arm-SDK or
  the locomotion gate. Gate membership is
  :data:`~strands_robots.tools.g1._g1_common.HANDSHAKE_FSMS` /
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` and is
  answered by :mod:`~strands_robots.tools.g1.g1_motion_gates` /
  :mod:`~strands_robots.tools.g1.g1_state`. This lookup is the
  ``SetFsmId``-side of the same conversation: which ids the SDK
  admits as *transition targets*, not which ids admit joint writes.
  The two sets overlap (``500``, ``501``, ``801`` are transition
  targets and also arm-write gates) but are not the same question.
* Whether the driver's live ``_fsm_id`` is currently inside the
  gate. That is a driver-instance read carried on the driver's
  ``get_status`` envelope; a caller planning a transition compares
  the driver's live fsm against this lookup's set to see whether the
  target is reachable at all.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import (
    ERR_CODES,
    HANDSHAKE_FSMS,
    WALK_FSMS,
)

#: Snapshot of the FSM ids the Unitree G1 locomotion SDK's ``SetFsmId``
#: admits as transition targets today. The names are the neon bundle's
#: ``FSM_NAMES`` labels (the SDK does not ship a canonical id -> name
#: mapping; these labels are the ones the neon bundle observed against
#: the real robot and the ones the driver's motion-gate refusal
#: strings would quote). ``0`` (ZeroTorque) is included even though
#: the SDK admits it: the robot *collapses* off-gantry on that target,
#: so the lookup surfaces the id but the returned envelope carries a
#: separate ``dangerous_ids`` field so a caller can decide the safety
#: refusal decidably before dispatch.
#:
#: The snapshot lives here rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because the mapping is
#: only useful for the ``SetFsmId``-side of the conversation; a
#: caller that only needs the gate set reaches :data:`HANDSHAKE_FSMS`
#: / :data:`WALK_FSMS` directly. Colocating the name table with the
#: transition verb mirrors ``_ARM_ACTION_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_arm_actions`: one snapshot per
#: SDK-facing table, one verb pair per snapshot.
_FSM_NAME_MAP: dict[int, str] = {
    0: "ZeroTorque",
    1: "Damp",
    2: "Squat",
    3: "Sit",
    4: "StandUp",
    500: "Start",
    501: "Walk",
    702: "Lie2StandUp",
    706: "Squat2StandUp",
    801: "BalanceExpert",
}

#: FSM ids that will drop the robot off its balance controller. ``0``
#: (ZeroTorque) fully limps every joint; ``1`` (Damp) leaves gravity
#: doing the work with only soft limits. Called out separately because
#: the neon bundle's ``g1_set_fsm`` docstring flags them as gantry-only
#: and a future driver-side transition wrapper's refusal string will
#: name the same pair; a caller planning an off-gantry run compares
#: an intended target against this set before dispatch.
_DANGEROUS_FSM_IDS: frozenset[int] = frozenset({0, 1})

#: The error-table entry the SDK's ``SetFsmId`` quotes on an id outside
#: :data:`_FSM_NAME_MAP`. Named here so the returned envelope carries
#: the exact refusal string a future driver-side wrapper would surface,
#: and so a re-wording of it lands in one place instead of drifting
#: between the SDK-side log and this lookup.
_INVALID_FSM_CODE: int = 7302

#: The error-table entry the driver's own ``_check_motion_gates`` quotes
#: when it refuses a write on an FSM outside its admitted gate. Named
#: here because :func:`g1_fsm_target_admits` surfaces it alongside the
#: SDK's ``7302`` on a query whose target is inside the SDK's set but
#: outside the arm-SDK write gate - so a caller planning
#: ``SetFsmId(3)`` (Sit) sees that the transition itself is admitted
#: while a follow-up ``send_action`` would still be refused. The two
#: codes together are the pair of refusals a caller would face on the
#: same transition attempt.
_GATE_REFUSAL_CODE: int = 7404


def _describe(fsm_id: int) -> dict[str, Any]:
    """Build the per-id descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_fsm_targets` so
    :func:`g1_fsm_target_admits`'s admitted-path payload names the
    same fields, and so a widen to the descriptor lands in one place.
    Every field is a snapshot read; no bus is touched.
    """
    return {
        "fsm_id": fsm_id,
        "name": _FSM_NAME_MAP[fsm_id],
        "dangerous": fsm_id in _DANGEROUS_FSM_IDS,
        "admits_arm_writes": fsm_id in HANDSHAKE_FSMS,
        "admits_loco_writes": fsm_id in WALK_FSMS,
    }


@tool
def g1_list_fsm_targets() -> dict[str, Any]:
    """Return the FSM ids ``LocoClient.SetFsmId`` admits as transition targets.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for ``SetFsmId`` is called, so a caller can compare an intended
    target id against the set the SDK's transition path would test
    membership in, and decide alongside that whether the target sits
    inside the arm-SDK or locomotion write gate (a target the SDK
    admits at transition time may still refuse a follow-up
    ``send_action`` because the driver's gate is narrower).

    Returns:
        A dict with ``status``, a ``count`` naming the number of
        transition targets, an ``fsm_targets`` list of descriptors
        (one per admitted id, sorted ascending) carrying ``fsm_id``,
        ``name`` (the neon-bundle label), a ``dangerous`` flag naming
        whether the target drops balance control (``0`` ZeroTorque,
        ``1`` Damp), and ``admits_arm_writes`` /
        ``admits_loco_writes`` flags naming whether the target sits
        inside the two write gates the driver's
        :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
        answers on. A separate ``dangerous_ids`` field lists the
        gantry-only ids as a set, so a caller filtering an off-gantry
        run compares against that set directly rather than walking
        the descriptors. A ``refusals`` list carries the two refusal
        codes and their decoded text (``7302`` invalid transition id,
        ``7404`` gate-refused write) that a future transition verb
        would surface. Every field is a snapshot of an SDK or driver
        constant; no dynamic decode runs here.
    """
    fsm_ids = sorted(_FSM_NAME_MAP)
    return {
        "status": "success",
        "count": len(_FSM_NAME_MAP),
        "fsm_targets": [_describe(fid) for fid in fsm_ids],
        "fsm_ids": fsm_ids,
        "dangerous_ids": sorted(_DANGEROUS_FSM_IDS),
        "arm_ready_fsm_ids": sorted(HANDSHAKE_FSMS),
        "loco_ready_fsm_ids": sorted(WALK_FSMS),
        "refusals": [
            {"code": _INVALID_FSM_CODE, "text": ERR_CODES[_INVALID_FSM_CODE]},
            {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
        ],
    }


@tool
def g1_fsm_target_admits(
    fsm_id: int | None = None,
    name: str = "",
) -> dict[str, Any]:
    """Decide whether an FSM target id or name is inside the SDK's transition set.

    Read-only. Reads the module's snapshot of the SDK's transition
    table and returns the same membership answer the SDK's
    ``SetFsmId`` would compute at wire time. A caller with either an
    integer id or a neon-bundle label resolves it against the SDK's
    set before a future transition verb dispatches, rather than
    triggering the SDK's ``rc=7302`` refusal at wire time.

    Exactly one of ``fsm_id`` (an int) or ``name`` (a string) must be
    supplied. Supplying both, or neither, carries ``status="error"``:
    the ambiguous case is a caller mistake, not a lookup this verb
    should resolve arbitrarily.

    Args:
        fsm_id: The transition target id to test. Must be an ``int``;
            ``bool`` is refused (``True`` is ``int(1)`` but a
            passed-through boolean is a caller mistake, not a valid
            transition query).
        name: The neon-bundle label to test. Case-sensitive to match
            the snapshot's own keys (a caller writing ``"walk"`` gets
            a key-not-found; the snapshot ships ``"Walk"``). Empty
            string means "no name supplied".

    Returns:
        A dict with ``status`` (``"success"`` on any decidable
        answer, ``"error"`` on the both-supplied / neither-supplied
        ambiguity), a ``query`` sub-dict carrying whichever of
        ``fsm_id`` / ``name`` was supplied, an ``admitted`` boolean
        naming whether the SDK's ``SetFsmId`` would admit the query,
        and (when ``admitted`` is ``True``) a ``target`` sub-dict
        carrying the same descriptor :func:`g1_list_fsm_targets`
        returns for the resolved id (``fsm_id``, ``name``,
        ``dangerous``, ``admits_arm_writes``, ``admits_loco_writes``)
        so a caller sees the danger flag and gate membership on the
        same call. On a not-admitted query the dict carries
        ``refusal_code`` / ``refusal_text`` naming the ``rc=7302``
        refusal the SDK would return.
    """
    supplied_id = fsm_id is not None
    supplied_name = bool(name)
    if supplied_id == supplied_name:
        return {
            "status": "error",
            "message": (
                "supply exactly one of fsm_id= (int) or name= (str); "
                f"got fsm_id={fsm_id!r}, name={name!r}. "
                "Refs strands-labs/robots#358."
            ),
        }
    if supplied_id and isinstance(fsm_id, bool):
        return {
            "status": "error",
            "message": (f"fsm_id must be int, got bool ({fsm_id!r}). Refs strands-labs/robots#358."),
        }
    if supplied_id and not isinstance(fsm_id, int):
        return {
            "status": "error",
            "message": (f"fsm_id must be int, got {type(fsm_id).__name__} ({fsm_id!r}). Refs strands-labs/robots#358."),
        }

    if supplied_id:
        admitted = fsm_id in _FSM_NAME_MAP
        resolved_id = fsm_id if admitted else None
        query: dict[str, Any] = {"fsm_id": fsm_id}
    else:
        # Reverse-lookup: snapshot is small (10 entries) so a linear
        # scan is fine and avoids maintaining a name->id dict on the side.
        resolved_id = next(
            (fid for fid, label in _FSM_NAME_MAP.items() if label == name),
            None,
        )
        admitted = resolved_id is not None
        query = {"name": name}

    result: dict[str, Any] = {
        "status": "success",
        "query": query,
        "admitted": admitted,
    }
    if admitted:
        assert resolved_id is not None  # for the type checker; admitted implies resolved
        result["target"] = _describe(resolved_id)
    else:
        result["refusal_code"] = _INVALID_FSM_CODE
        result["refusal_text"] = ERR_CODES[_INVALID_FSM_CODE]
    return result
