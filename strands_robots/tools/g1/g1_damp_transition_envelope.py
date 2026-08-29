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
  damp-preamble method lands, its refusal will name the same
  locomotion gate this module's refusal text names. No SDK return
  code is quoted on either side, because the driver quotes none.

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

from strands_robots.tools.g1._g1_common import WALK_FSMS

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

# Refusals in this module carry module-local text and no ``code`` field,
# because no SDK return code covers either refusal it makes.
#
# ``ERR_CODES[7404]`` ("Invalid FSM id - need FSM in {500, 501, 801}") is the
# nearest-looking candidate and is wrong on both counts. This repository's
# driver quotes it nowhere -- ``G1Driver._check_motion_gates`` returns
# free-text refusals and never names a return code -- so it is not "the code
# the driver quotes". And the set it names is not ``WALK_FSMS``
# (``{501, 801}``), so a payload carrying that text beside its own
# ``walk_ready_fsm_ids`` would contradict itself about which FSMs are
# admitted.
#
# The two refusals also have different remedies, which is why they are
# separate texts rather than one shared code: an FSM outside the locomotion
# gate is a robot-state problem fixed by a transition, while an unknown
# transition label is an argument problem fixed by passing a different
# string. Quoting the FSM code for a mistyped label would hand an agent
# planner a physical motion-state change as the remedy for a typo.


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


def _walk_gate_refusal() -> str:
    """The refusal text a future driver-side damp-preamble wrapper would surface.

    Built from :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`
    rather than spelled as a literal, so a widen of the locomotion
    gate carries into this message instead of leaving a stale set
    behind. This is the refusal a caller gets for a *robot-state*
    problem, and its remedy is an FSM transition.
    """
    return f"fsm_id outside the locomotion gate - need FSM in {sorted(WALK_FSMS)}"


def _unknown_transition_refusal() -> str:
    """The refusal text for a transition label this module does not carry.

    Built from :data:`_DAMP_TRANSITIONS` so an added transition widens
    the message too. Kept strictly distinct from
    :func:`_walk_gate_refusal`: a mistyped or unknown label is an
    *argument* problem whose remedy is to pass a different string, and
    handing such a caller the FSM-gate remedy would point it at a
    physical motion-state change it never asked for.
    """
    return f"unknown damp-preamble transition - need one of {sorted(_DAMP_TRANSITIONS)}"


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
        and a ``refusals`` list carrying the module-local gate-refusal
        ``text`` a future write verb would surface, built from
        ``WALK_FSMS`` so it cannot drift from ``walk_ready_fsm_ids`` in
        the same payload. No ``code`` field: the driver quotes no SDK
        return code on this gate. Every field is a snapshot of an
        observed neon precondition or a driver constant; no dynamic
        decode runs here.
    """
    return {
        "status": "success",
        "transitions": [_describe(name) for name in sorted(_DAMP_TRANSITIONS)],
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
        "refusals": [
            {"text": _walk_gate_refusal()},
        ],
    }


@tool
def g1_damp_transition_admits(name: str) -> dict[str, Any]:
    """Decide whether a transition ``name`` sits inside the admitted set.

    Read-only. Compares ``name`` against the neon-observed
    :data:`_DAMP_TRANSITIONS` and reports the admitted descriptor on
    match, or a refusal naming the unknown label and the admitted set
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
            mis-cased or unknown label is refused with the
            unknown-transition text. A non-string argument is refused
            with the same text, because the lookup is by string key and
            both are the same argument mistake.

    Returns:
        A dict with ``status``; on admit, a ``transition`` descriptor
        with ``name``, ``expected_fsm_ids``, ``pose_check``,
        ``avg_knee_max_rad``, and ``description`` (the same shape
        :func:`g1_list_damp_transitions` returns), plus
        ``walk_ready_fsm_ids`` for the follow-on gate decision. On
        refuse, ``refusal_text`` names the unknown-transition refusal
        (the admitted labels, never the FSM gate -- the remedy for a bad
        label is a different string, not a motion-state change), plus a
        ``reason`` string that names why the argument was refused
        (unknown name or non-string argument). There is no
        ``refusal_code``: no SDK return code covers this refusal.
    """
    if not isinstance(name, str):
        return {
            "status": "error",
            "refusal_text": _unknown_transition_refusal(),
            "reason": (f"name={name!r} is not a str; pass one of {sorted(_DAMP_TRANSITIONS)}"),
        }
    if name not in _DAMP_TRANSITIONS:
        return {
            "status": "error",
            "refusal_text": _unknown_transition_refusal(),
            "reason": (f"name={name!r} is not in the admitted set {sorted(_DAMP_TRANSITIONS)}"),
        }
    return {
        "status": "success",
        "transition": _describe(name),
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
    }
