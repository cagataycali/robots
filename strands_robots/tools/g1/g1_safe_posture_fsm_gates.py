"""Agent-facing lookup for the FSM-set preconditions the neon safe-posture verbs gate on.

The neon bundle's ``g1_safe_posture.py``
(``cagataycali/neon-the-g1/tools/g1_safe_posture.py``) carries three
compound-posture verbs (``g1_safe_squat_to_stand``,
``g1_safe_lie_to_stand``, ``g1_safe_stand_to_squat``) that each issue a
``LocoClient.Damp()`` preamble before their target transition. Every one
of them guards the Damp preamble behind an ``_assert_safe_for_damp``
call that refuses unless the driver's live FSM sits inside a
verb-specific whitelist:

* ``g1_safe_squat_to_stand`` requires the FSM in ``{3, 4, 706}`` -
  ``3`` Sit, ``4`` StandUp, ``706`` Squat2StandUp - so the controller
  is one that already carries the robot's weight before the Damp fires.
* ``g1_safe_lie_to_stand`` requires the FSM in ``{1, 702}`` -
  ``1`` Damp, ``702`` Lie2StandUp - so the robot is already flat on the
  floor with a controller aware of the pose.
* ``g1_safe_stand_to_squat`` requires the FSM in ``{500, 501, 801}`` -
  ``500`` Start, ``501`` Walk, ``801`` BalanceExpert - so the robot is
  actively balancing before the transition to Squat is issued.

The neon comment on those sets is that Damp is a
controller-to-controller handoff smoother, not a wake-from-limp helper:
firing Damp against an unheld robot leaves it slumping toward the
floor. This module surfaces the three whitelist sets as a module-level
constant snapshot and two agent-facing verbs
(:func:`g1_list_safe_posture_fsm_gates` returns the whole envelope;
:func:`g1_safe_posture_fsm_admits` decides one membership query) so a
caller planning a compound-posture rollout can name the precondition
decidably before any Damp-preamble verb lands on the driver side.
Refs strands-labs/robots#358.

Two things this module is deliberately *not*:

* An execution path. The neon verbs actually call
  ``LocoClient.Damp()`` then ``Squat2StandUp`` / ``Lie2StandUp`` /
  ``SetFsmId(2)``; those writes are the ``rt/lowcmd``-adjacent
  locomotion path :class:`~strands_robots.drivers.g1.G1Driver` gates
  through :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
  (refs the merged strands-labs/robots#2916). A future driver method
  that fronts a compound-posture transition will land the actuation
  half; this module ports the read-only precondition table without
  also introducing a second locomotion writer path.
* An SDK re-import. The FSM ids are captured here as
  module-level ``frozenset`` constants snapshotted from the neon
  ``_assert_safe_for_damp`` whitelists; the sets live here rather than
  being re-imported from the SDK so ``import
  strands_robots.tools.g1.g1_safe_posture_fsm_gates`` pulls no
  ``unitree_sdk2py`` submodule (the import-hygiene contract every
  other file in this package carries, refs strands-labs/robots#358).
  An SDK release that widens or narrows the FSM name table
  (:mod:`~strands_robots.tools.g1.g1_fsm_targets` names the full set)
  is a driver-side update; the whitelist rows this module carries are
  bounded by that name table and the sibling test asserts every id
  in every whitelist is also in the SDK-admitted name map, so a
  neon-side widen that named an id the SDK does not admit would
  surface at CI.

What this module does not decide.

* Whether the robot is *actually* held by a controller at the moment
  of the query. The FSM-set precondition is a necessary but not
  sufficient check: the neon verbs also read
  ``avg_knee`` off the LowState cache to refuse a deep-squat pose that
  the controller has already lost, and pass ``force=True`` to bypass
  both. That live-pose check is a driver-instance read
  (:mod:`~strands_robots.tools.g1.g1_state`); this lookup answers the
  FSM-set half without also taking a driver handle.
* Whether the Damp preamble is sound on the target FSM. The three
  whitelists are the neon bundle's chosen preconditions today; a
  future safe-posture verb (e.g. ``g1_safe_kneel_to_stand``) would
  ship its own whitelist row, and this table would grow by one entry.
  The verb-to-set mapping here is the one contract callers can rely
  on before a driver-side wrapper exists; the actuation contract
  belongs on the driver's ``_check_motion_gates`` (refs the merged
  strands-labs/robots#2916).
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: Snapshot of the FSM-id whitelists the neon ``_assert_safe_for_damp``
#: gate refuses Damp preambles outside of. Each key names the neon
#: verb the row gates, each value is a ``frozenset`` of admitted FSM
#: ids (matching the neon ``expected_fsms`` argument byte-for-byte).
#:
#: The verb names carry the ``g1_safe_`` prefix the neon module ships
#: with, so a caller reading a refusal that names the verb can find
#: this row by string match. The FSM ids are integers, matching the
#: SDK's own ``SetFsmId(int)`` API and the name map surfaced by
#: :mod:`~strands_robots.tools.g1.g1_fsm_targets` (an SDK-side widen
#: of that name map would leave the whitelist rows here unchanged
#: unless the neon bundle also widened; a sibling test asserts every
#: id here is admitted by the SDK).
_SAFE_POSTURE_FSM_GATES: dict[str, frozenset[int]] = {
    "g1_safe_squat_to_stand": frozenset({3, 4, 706}),
    "g1_safe_lie_to_stand": frozenset({1, 702}),
    "g1_safe_stand_to_squat": frozenset({500, 501, 801}),
}

#: Human-readable description of what each safe-posture verb transitions
#: from-to, mirrored from the neon docstrings so a returned envelope
#: names the verb's intent alongside the FSM set. Kept here rather than
#: in the neon docstrings because the neon file is not importable from
#: this package (import-hygiene contract, refs strands-labs/robots#358)
#: and a caller reading the envelope wants both the FSM set and the
#: transition it gates in one place.
_SAFE_POSTURE_DESCRIPTIONS: dict[str, str] = {
    "g1_safe_squat_to_stand": (
        "Damp preamble then Squat2StandUp; safe only when a controller "
        "already carries the robot (Sit, StandUp, or in-progress "
        "Squat2StandUp)."
    ),
    "g1_safe_lie_to_stand": (
        "Damp preamble then Lie2StandUp; safe only when the robot is flat "
        "on the floor with a controller aware of the pose (Damp or "
        "in-progress Lie2StandUp)."
    ),
    "g1_safe_stand_to_squat": (
        "Damp preamble then SetFsmId(2) into Squat; safe only when the "
        "robot is actively balancing (Start, Walk, or BalanceExpert)."
    ),
}


@tool
def g1_list_safe_posture_fsm_gates(verb: str = "") -> dict[str, Any]:
    """Return the FSM-id whitelists the neon safe-posture verbs gate Damp on.

    Read-only. Every field is a module-level constant snapshotted from
    the neon ``_assert_safe_for_damp`` gate; no bus is touched, no
    driver instance is required. Useful before a compound-posture
    rollout is planned, so a caller can compare the driver's live
    ``fsm_id`` (from a status read) against the whitelist a future
    driver-side safe-posture verb would refuse outside of.

    Args:
        verb: Optional verb-name filter. One of the three neon safe
            posture verb names (``"g1_safe_squat_to_stand"``,
            ``"g1_safe_lie_to_stand"``, ``"g1_safe_stand_to_squat"``);
            empty returns all three rows so a caller can see the whole
            gate at once.

    Returns:
        A dict with ``status`` and a ``gates`` list of records, one per
        returned verb (sorted lexicographically by verb name). Each
        record carries ``verb``, an ``fsm_ids`` list (sorted ascending,
        integers only), a ``description`` naming the transition the row
        gates, and the ``fsm_count`` so a caller can see the whitelist
        cardinality without walking the ids. On an unknown verb name
        the returned dict carries ``status="error"`` and a ``message``
        naming the valid verbs as a resolvable domain.
    """
    if verb and verb not in _SAFE_POSTURE_FSM_GATES:
        valid = sorted(_SAFE_POSTURE_FSM_GATES)
        return {
            "status": "error",
            "message": (f"unknown verb {verb!r}. Valid verbs are {valid}. Refs strands-labs/robots#358."),
        }
    verbs = (verb,) if verb else tuple(sorted(_SAFE_POSTURE_FSM_GATES))
    gates = [
        {
            "verb": name,
            "fsm_ids": sorted(_SAFE_POSTURE_FSM_GATES[name]),
            "fsm_count": len(_SAFE_POSTURE_FSM_GATES[name]),
            "description": _SAFE_POSTURE_DESCRIPTIONS[name],
        }
        for name in verbs
    ]
    return {
        "status": "success",
        "count": len(gates),
        "verb": verb or None,
        "verbs": sorted(_SAFE_POSTURE_FSM_GATES),
        "gates": gates,
    }


@tool
def g1_safe_posture_fsm_admits(fsm_id: int, verb: str = "") -> dict[str, Any]:
    """Decide whether a given FSM id is inside a safe-posture verb's whitelist.

    Read-only. Reads the module-level whitelist for the named verb and
    returns the same membership answer the neon
    ``_assert_safe_for_damp`` gate would compute. A caller with a live
    ``fsm_id`` from a G1 status read uses this to phrase a refusal in
    its own voice, rather than triggering the neon gate's refusal at
    wire time. When ``verb`` is empty the query is refused as a shape
    error because the whitelist is verb-specific (the three neon verbs
    have different admitted FSM sets); a caller that wants to see all
    three rows uses :func:`g1_list_safe_posture_fsm_gates` instead.

    Args:
        fsm_id: The FSM id to test. Must be an int; ``bool`` is refused
            (``True`` is ``int(1)`` but a dict-key typo of ``True`` for
            an FSM id is a caller mistake, not a valid gate query).
        verb: Which safe-posture verb's whitelist to test. One of
            ``"g1_safe_squat_to_stand"`` / ``"g1_safe_lie_to_stand"`` /
            ``"g1_safe_stand_to_squat"``; empty is refused with the
            valid verb list because the whitelist is verb-specific.

    Returns:
        A dict with ``status``, the requested ``verb``, the tested
        ``fsm_id``, an ``admitted`` boolean naming whether the neon
        gate would open, the ``fsm_ids`` list the answer was computed
        against, and the verb's ``description``. On an unknown verb
        or a non-int ``fsm_id``, ``status="error"`` and a ``message``
        names the resolution.
    """
    if not isinstance(verb, str):
        return {
            "status": "error",
            "message": (f"verb must be a str; got {type(verb).__name__} {verb!r}. Refs strands-labs/robots#358."),
        }
    if not verb:
        valid = sorted(_SAFE_POSTURE_FSM_GATES)
        return {
            "status": "error",
            "message": (
                f"verb is required; the safe-posture whitelist is verb-specific. "
                f"Valid verbs are {valid}. "
                "Use g1_list_safe_posture_fsm_gates() to see all rows at once. "
                "Refs strands-labs/robots#358."
            ),
        }
    if verb not in _SAFE_POSTURE_FSM_GATES:
        valid = sorted(_SAFE_POSTURE_FSM_GATES)
        return {
            "status": "error",
            "message": (f"unknown verb {verb!r}. Valid verbs are {valid}. Refs strands-labs/robots#358."),
        }
    if isinstance(fsm_id, bool) or not isinstance(fsm_id, int):
        return {
            "status": "error",
            "message": (
                f"fsm_id must be an int; got {type(fsm_id).__name__} {fsm_id!r}. Refs strands-labs/robots#358."
            ),
        }
    admitted = fsm_id in _SAFE_POSTURE_FSM_GATES[verb]
    return {
        "status": "success",
        "verb": verb,
        "fsm_id": fsm_id,
        "admitted": admitted,
        "fsm_ids": sorted(_SAFE_POSTURE_FSM_GATES[verb]),
        "description": _SAFE_POSTURE_DESCRIPTIONS[verb],
    }
