"""Agent-facing lookup for the safe-damp transition envelope.

The neon bundle's ``g1_safe_posture.py`` verbs
(``cagataycali/neon-the-g1/tools/g1_safe_posture.py``) each front a
``LocoClient.Damp()`` preamble followed by an FSM transition. The
preamble is only *safe* when the loco controller is actively holding
the robot: ``Damp()`` on an uncontrolled robot removes what little
holding torque the joint drives had and the robot collapses further.
The neon bundle observed against the real robot that each transition
is safe from a small set of controller-managed FSMs, and that a
deep-squat pose (``avg_knee > 1.4 rad``) means the controller has
already let go and no damp-preamble path will recover it. Those two
membership tests - expected FSM set and knee-angle threshold - are
the transition envelope; this module surfaces them so a caller can
decide the refusal decidably before a future driver-side wrapper for
the damp-preamble path is attempted, rather than pinning it inside
the write path where the refusal is invisible to the planner.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_safe_squat_to_stand``,
  ``g1_safe_lie_to_stand``, and ``g1_safe_stand_to_squat`` verbs each
  wrap ``LocoClient.Damp()`` followed by a locomotion-shaped write
  (``Squat2StandUp``, ``Lie2StandUp``, or ``SetFsmId(2)``); that pair
  of writes is the ``rt/lowcmd``-adjacent locomotion topic today's
  :class:`~strands_robots.drivers.g1.G1Driver` refuses through
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` on
  any locomotion-shaped write while ``_fsm_id`` is outside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. A future
  driver method that fronts the damp-preamble path will land the
  write verb; refs strands-labs/robots#358 for the SDK-facing gate
  work that write belongs on. This module ports the read-only
  envelope half without also introducing a second locomotion writer
  path the driver does not yet own.
* An SDK re-import. The transition table is captured here as
  module-level constants so
  ``import strands_robots.tools.g1.g1_damp_transition_envelope``
  pulls no ``unitree_sdk2py`` submodule - the import-hygiene contract
  every other file in this package carries, refs
  strands-labs/robots#358. A revision of the neon bundle's observed
  preconditions is a driver-side update; when the driver's
  damp-preamble method lands, its refusal will quote the same
  ``7404`` gate-refusal code
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.

What this module does not decide.

* Whether the driver's live ``_fsm_id`` currently sits inside a
  transition's ``expected_fsm_ids``. That is a live driver-instance
  read carried on the driver's ``get_status`` envelope; a caller
  planning a damp-preamble compares the driver's live fsm against
  the transition's ``expected_fsm_ids`` this verb surfaces to decide
  whether the damp gate is currently open.
* Whether the driver's live average knee angle is below the knee
  threshold. That is a live LowState read carried on the driver's
  ``get_status`` envelope; the threshold this verb surfaces is the
  refusal boundary, not a live sample.
* Whether ``rt/lowcmd`` is currently held by another writer. The
  driver's single-writer lock reports that at wire time; a caller
  planning a damp-preamble write cannot decide it without opening
  the topic itself, and this module opens no channel.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS

#: The upper bound on the robot's average knee angle (radians) above
#: which the neon bundle's ``_assert_safe_for_damp`` refuses the
#: damp-preamble. The neon bundle observed that at ``avg_knee > 1.4``
#: the robot is in a deep squat or is sagging: the loco controller
#: has already released holding torque, so ``Damp()`` will remove the
#: last resistance and the robot will collapse further. Named as an
#: absolute upper bound because a knee angle below this value is
#: within the controller-holdable range, and a value above means the
#: physical intervention path (gantry, remote) is the safe answer,
#: not another SDK write.
_AVG_KNEE_MAX_RAD: float = 1.4

#: The pair of fields the SDK's own return-code table names when the
#: driver's motion gate refuses a locomotion-shaped write on an FSM
#: outside :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. Named
#: here so the returned envelope carries the exact refusal string a
#: future driver-side damp-preamble wrapper would surface, and so a
#: re-wording of it lands in one place instead of drifting between
#: the driver's log and this lookup.
_GATE_REFUSAL_CODE: int = 7404

#: The refusal code a future driver-side wrapper would quote on a
#: transition name outside :data:`_DAMP_TRANSITIONS`. The SDK does not
#: ship a distinct rc for "unknown damp-preamble transition" (the
#: neon bundle owned the transition names, not the SDK); the neon
#: bundle refused unknown names at the verb boundary, so this lookup
#: uses the same ``7404`` gate-refusal shape a future driver-side
#: wrapper would quote when refusing at the same boundary. Named
#: separately from :data:`_GATE_REFUSAL_CODE` so a future SDK release
#: that adds a dedicated "invalid transition" code lands here without
#: also renaming the gate-refusal constant.
_INVALID_TRANSITION_CODE: int = 7404


#: Snapshot of the safe damp-preamble transitions the neon bundle's
#: ``g1_safe_posture.py`` verbs each front. Each entry captures the
#: three fields the neon bundle's ``_assert_safe_for_damp`` observes
#: at refusal time:
#:
#: * ``expected_fsm_ids`` - the FSM ids the controller must be in for
#:   the damp-preamble to be safe. Sourced from the neon bundle's own
#:   ``expected_fsms`` argument on each verb's call to
#:   ``_assert_safe_for_damp``.
#: * ``pose_check`` - whether the neon bundle's ``avg_knee`` refusal
#:   is enforced for this transition. Sourced from the same verb's
#:   ``pose_check`` argument. Only ``squat_to_stand`` uses the pose
#:   check today; ``lie_to_stand`` and ``stand_to_squat`` skip it (a
#:   lying robot can be at any knee angle; a stand-to-squat caller
#:   asks for a lower knee angle by definition).
#: * ``description`` - the neon bundle's own docstring summary of
#:   what the transition does, so an agent-facing planner has the
#:   one-line semantic on hand without re-reading the neon docs.
#:
#: The snapshot lives here rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because the mapping is
#: only useful for the damp-preamble-side of the conversation; a
#: caller that only needs the FSM set reaches
#: :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` or the
#: transition-specific ``expected_fsm_ids`` directly. Colocating the
#: map with the verb mirrors ``_BALANCE_MODE_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_balance_modes` and
#: ``_ARM_ACTION_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_arm_actions`: one snapshot per
#: SDK-facing transition table, one verb pair per snapshot.
_DAMP_TRANSITIONS: dict[str, dict[str, Any]] = {
    "squat_to_stand": {
        "expected_fsm_ids": frozenset({3, 4, 706}),
        "pose_check": True,
        "description": (
            "Damp preamble then Squat2StandUp. Safe from Sit (3), "
            "StandUp (4), or Squat2StandUp (706). Refuses when the "
            "average knee angle exceeds the threshold, because that "
            "means the controller has already let go."
        ),
    },
    "lie_to_stand": {
        "expected_fsm_ids": frozenset({1, 702}),
        "pose_check": False,
        "description": (
            "Damp preamble then Lie2StandUp. Safe from Damp (1) or "
            "Lie2StandUp (702) - the face-up lying poses the "
            "controller can lift from. Pose check skipped because a "
            "lying robot's knee angle is not the deciding factor."
        ),
    },
    "stand_to_squat": {
        "expected_fsm_ids": frozenset({500, 501, 801}),
        "pose_check": False,
        "description": (
            "Damp preamble then SetFsmId(2=Squat) - works around the "
            "SDK bug where LocoClient.StandUp2Squat() dispatches to "
            "FSM 706 (stand-up) instead of FSM 2 (squat). Safe from "
            "Start (500), Walk (501), or BalanceExpert (801) - the "
            "upright, actively-balancing FSMs. Pose check skipped "
            "because a caller asking for a stand-to-squat expects a "
            "lower knee angle by definition."
        ),
    },
}


def _describe(name: str) -> dict[str, Any]:
    """Build the per-transition descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_damp_transitions` so
    :func:`g1_damp_transition_admits`'s admitted-path payload names the
    same fields, and so a widen to the descriptor lands in one place.
    Every field is a snapshot read; no bus is touched.
    """
    entry = _DAMP_TRANSITIONS[name]
    return {
        "name": name,
        "expected_fsm_ids": sorted(entry["expected_fsm_ids"]),
        "pose_check": entry["pose_check"],
        "avg_knee_max_rad": _AVG_KNEE_MAX_RAD if entry["pose_check"] else None,
        "description": entry["description"],
    }


@tool
def g1_list_damp_transitions() -> dict[str, Any]:
    """Return the safe damp-preamble transitions the neon bundle documented.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for the damp-preamble path is called, so a caller can name the
    transitions the neon bundle observed as safe against the real
    robot, and can also compare the driver's live ``fsm_id`` (from
    ``G1Driver.get_status``) and ``avg_knee`` (also from
    ``get_status``) against the per-transition ``expected_fsm_ids``
    and ``avg_knee_max_rad`` to decide whether the damp gate is
    currently open.

    Returns:
        A dict with ``status``; a ``transitions`` list of per-transition
        descriptors sorted by ``name`` ascending, each carrying
        ``name`` (the neon-observed transition label), ``expected_fsm_ids``
        (the FSM set the transition is safe from), ``pose_check``
        (whether the knee-angle refusal is enforced), ``avg_knee_max_rad``
        (the knee threshold in radians when ``pose_check`` is enabled,
        else ``None``), and ``description`` (the neon bundle's one-line
        semantic); a ``walk_ready_fsm_ids`` list quoting
        :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, the set
        the driver's motion gate admits locomotion-shaped writes on;
        and a ``refusals`` list carrying the ``7404`` gate-refusal
        code and its decoded text, the one a future write verb would
        surface. Every field is a snapshot of an observed neon
        precondition or a driver constant; no dynamic decode runs here.
    """
    return {
        "status": "success",
        "transitions": [_describe(name) for name in sorted(_DAMP_TRANSITIONS)],
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
        "refusals": [
            {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
        ],
    }


@tool
def g1_damp_transition_admits(name: str) -> dict[str, Any]:
    """Decide whether a transition ``name`` sits inside the admitted set.

    Read-only. Compares ``name`` against the neon-observed
    :data:`_DAMP_TRANSITIONS` and reports the admitted descriptor on
    match, or the ``7404`` gate-refusal code the driver would quote
    on miss. No driver instance, no DDS, no SDK: the decision reads
    only module-level constants and the argument itself.

    A transition inside the admitted set is *not* the same as an
    admitted write: the driver's motion gate
    (``_check_motion_gates``) also refuses on ``_fsm_id`` outside the
    transition's ``expected_fsm_ids`` and (when ``pose_check`` is
    enabled) on ``avg_knee`` above :data:`_AVG_KNEE_MAX_RAD`, both of
    which this verb does not read (they are live driver-instance
    queries answered by
    :mod:`~strands_robots.tools.g1.g1_state` and
    :mod:`~strands_robots.tools.g1.g1_motion_gates`). The returned
    payload names ``walk_ready_fsm_ids`` so a caller comparing an
    intended write against the general locomotion gate has the FSM
    set on hand too.

    Args:
        name: The transition label to check. Case-sensitive against
            the snapshot in :data:`_DAMP_TRANSITIONS` (``squat_to_stand``,
            ``lie_to_stand``, or ``stand_to_squat`` today). A
            mis-cased or unknown label is refused with the ``7404``
            code. A non-string argument is refused with the same
            code because the lookup is by string key.

    Returns:
        A dict with ``status``; on admit, a ``transition`` descriptor
        with ``name``, ``expected_fsm_ids``, ``pose_check``,
        ``avg_knee_max_rad``, and ``description`` (the same shape
        :func:`g1_list_damp_transitions` returns), plus
        ``walk_ready_fsm_ids`` for the follow-on gate decision. On
        refuse, ``refusal_code`` and ``refusal_text`` name the ``7404``
        code and its decoded text, plus a ``reason`` string that names
        why the argument was refused (unknown name or non-string
        argument).
    """
    if not isinstance(name, str):
        return {
            "status": "error",
            "refusal_code": _INVALID_TRANSITION_CODE,
            "refusal_text": ERR_CODES[_INVALID_TRANSITION_CODE],
            "reason": (f"name={name!r} is not a str; pass one of {sorted(_DAMP_TRANSITIONS)}"),
        }
    if name not in _DAMP_TRANSITIONS:
        return {
            "status": "error",
            "refusal_code": _INVALID_TRANSITION_CODE,
            "refusal_text": ERR_CODES[_INVALID_TRANSITION_CODE],
            "reason": (f"name={name!r} is not in the admitted set {sorted(_DAMP_TRANSITIONS)}"),
        }
    return {
        "status": "success",
        "transition": _describe(name),
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
    }
