"""The damp-transition envelope names what the neon damp-preamble verbs admit.

The neon bundle's ``g1_safe_posture.py`` verbs
(``cagataycali/neon-the-g1/tools/g1_safe_posture.py``) each front a
``LocoClient.Damp()`` preamble followed by an FSM transition. The
neon bundle observed against the real robot that each transition is
safe from a small set of controller-managed FSMs, and that a
deep-squat pose (``avg_knee > 1.4 rad``) means the controller has
already released holding torque and no damp-preamble path will
recover it. The :mod:`strands_robots.tools.g1.g1_damp_transition_envelope`
module snapshots those preconditions into a module-level dict and
exposes two agent-facing verbs -
:func:`g1_list_damp_transitions` (name the whole set) and
:func:`g1_damp_transition_admits` (decide one query) - so a caller
can decide the refusal decidably before a future damp-preamble write
path is attempted. The tests here fix that contract without pulling
the SDK: the module is loadable on a host without ``unitree_sdk2py``
(the same SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs strands-labs/robots#358),
and every membership answer is read off the module's own snapshot
rather than restated in the tests, so a widen or narrow to the
observed set surfaces here as a shape change rather than as a
diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The snapshot is the neon
  bundle's observed precondition set, not the SDK's own admissions
  (the SDK admits any FSM id silently; the neon bundle drew the
  refusal boundary from field observation). A driver-side wrapper
  for the damp-preamble path that lands later will re-check the
  preconditions at wire time and its refusal string will quote the
  ``7404`` gate-refusal code the driver's ``_check_motion_gates``
  also quotes.
* Whether the driver's live ``fsm_id`` sits inside a transition's
  ``expected_fsm_ids``. That is a live driver-instance read and
  belongs on :mod:`~strands_robots.tools.g1.g1_state` /
  :mod:`~strands_robots.tools.g1.g1_motion_gates`; the verb surfaces
  the set as a snapshot so a caller comparing an intended write
  against the transition preconditions has the FSM set on hand.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS
from strands_robots.tools.g1.g1_damp_transition_envelope import (
    _AVG_KNEE_MAX_RAD,
    _DAMP_TRANSITIONS,
    _GATE_REFUSAL_CODE,
    _INVALID_TRANSITION_CODE,
    g1_damp_transition_admits,
    g1_list_damp_transitions,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process; this helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent (refs strands-labs/robots#358); a module that
    pulled a submodule at import time would break every headless CI
    runner and Thor before an office bring-up.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_damp_transition_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_damp_transition_envelope imports "
        f"pulled SDK submodules: {leaked}. The rule for this package is "
        f"that the SDK loads only inside function bodies (refs "
        f"strands-labs/robots#358)."
    )


def test_the_snapshot_covers_the_neon_observed_transitions() -> None:
    """The transition map pins the three transitions the neon bundle documented.

    ``squat_to_stand``, ``lie_to_stand``, and ``stand_to_squat`` are
    the three verbs ``cagataycali/neon-the-g1/tools/g1_safe_posture.py``
    exposed against the real robot. A widen or narrow of that set
    surfaces here as a shape change the tests read off the module's
    own snapshot, not as a diverging copy this file would need to
    update.
    """
    assert set(_DAMP_TRANSITIONS) == {
        "squat_to_stand",
        "lie_to_stand",
        "stand_to_squat",
    }, (
        f"Damp-transition snapshot drifted from the neon-observed set. "
        f"Got {sorted(_DAMP_TRANSITIONS)}, expected "
        f"['lie_to_stand', 'squat_to_stand', 'stand_to_squat']. Update "
        f"the snapshot and this test together."
    )


def test_the_squat_to_stand_precondition_matches_the_neon_argument() -> None:
    """The ``squat_to_stand`` entry names the FSMs neon's verb refuses outside.

    The neon bundle's ``g1_safe_squat_to_stand`` verb calls
    ``_assert_safe_for_damp(expected_fsms={3, 4, 706}, pose_check=True)``.
    The snapshot pins that exact set and the pose-check flag so a
    driver-side wrapper's refusal string quotes the same boundary
    the neon bundle observed.
    """
    entry = _DAMP_TRANSITIONS["squat_to_stand"]
    assert entry["expected_fsm_ids"] == frozenset({3, 4, 706}), (
        f"squat_to_stand expected_fsm_ids drifted; got {sorted(entry['expected_fsm_ids'])}, expected [3, 4, 706]"
    )
    assert entry["pose_check"] is True, (
        f"squat_to_stand pose_check drifted; got {entry['pose_check']!r}, "
        f"expected True (the neon bundle's own argument)"
    )


def test_the_lie_to_stand_precondition_matches_the_neon_argument() -> None:
    """The ``lie_to_stand`` entry names the FSMs neon's verb refuses outside.

    The neon bundle's ``g1_safe_lie_to_stand`` verb calls
    ``_assert_safe_for_damp(expected_fsms={1, 702}, pose_check=False)``.
    The snapshot pins that exact set and the pose-check flag: a
    lying robot's knee angle is not the deciding factor, so neon
    skipped the pose check.
    """
    entry = _DAMP_TRANSITIONS["lie_to_stand"]
    assert entry["expected_fsm_ids"] == frozenset({1, 702}), (
        f"lie_to_stand expected_fsm_ids drifted; got {sorted(entry['expected_fsm_ids'])}, expected [1, 702]"
    )
    assert entry["pose_check"] is False


def test_the_stand_to_squat_precondition_matches_the_neon_argument() -> None:
    """The ``stand_to_squat`` entry names the FSMs neon's verb refuses outside.

    The neon bundle's ``g1_safe_stand_to_squat`` verb calls
    ``_assert_safe_for_damp(expected_fsms={500, 501, 801}, pose_check=False)``
    — the upright, actively-balancing FSMs. The snapshot pins that
    exact set and the pose-check flag: a caller asking for a
    stand-to-squat expects a lower knee angle by definition, so neon
    skipped the pose check.
    """
    entry = _DAMP_TRANSITIONS["stand_to_squat"]
    assert entry["expected_fsm_ids"] == frozenset({500, 501, 801}), (
        f"stand_to_squat expected_fsm_ids drifted; got {sorted(entry['expected_fsm_ids'])}, expected [500, 501, 801]"
    )
    assert entry["pose_check"] is False


def test_the_knee_threshold_matches_the_neon_observation() -> None:
    """The knee threshold pins the ``avg_knee > 1.4 rad`` refusal.

    The neon bundle's ``_assert_safe_for_damp`` compared ``avg_knee``
    against the literal ``1.4`` before refusing. Named here so a
    re-observation against the real robot lands as a single constant
    change rather than three verb-side edits.
    """
    assert _AVG_KNEE_MAX_RAD == 1.4


def test_the_gate_refusal_code_matches_the_driver_constant() -> None:
    """The verb's refusal code names the driver's gate refusal.

    The driver's ``_check_motion_gates`` refuses a locomotion-shaped
    write on an FSM outside :data:`WALK_FSMS` with rc=7404, and the
    ``7404`` entry in
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries the
    text a driver-side damp-preamble wrapper would surface. Pinned
    here so a re-wording of that message lands in one place, not one
    in the driver and a diverging copy in this verb.
    """
    assert _GATE_REFUSAL_CODE == 7404
    assert _INVALID_TRANSITION_CODE == 7404
    assert _GATE_REFUSAL_CODE in ERR_CODES, (
        f"verb quotes rc={_GATE_REFUSAL_CODE} but that code is not in "
        f"ERR_CODES. The refusal string would render as 'unknown'."
    )


def test_g1_list_damp_transitions_returns_the_whole_table() -> None:
    """The verb's payload names every transition, the gate set, and the refusal.

    ``transitions`` carries every entry in :data:`_DAMP_TRANSITIONS`
    sorted by ``name``, ``walk_ready_fsm_ids`` quotes
    :data:`WALK_FSMS`, and ``refusals`` names the ``7404``
    gate-refusal code with the decoded text :data:`ERR_CODES` carries.
    """
    result = _call(g1_list_damp_transitions)
    assert result["status"] == "success"
    names = [t["name"] for t in result["transitions"]]
    assert names == sorted(_DAMP_TRANSITIONS), (
        f"g1_list_damp_transitions returned {names}, expected {sorted(_DAMP_TRANSITIONS)} (sorted-ascending)"
    )
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)
    assert result["refusals"] == [
        {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
    ]


def test_g1_list_damp_transitions_descriptor_names_every_field() -> None:
    """Every transition descriptor names every field the module documents.

    The descriptor shape is the contract
    :func:`g1_damp_transition_admits` mirrors on admit; a field added
    in one code path but not the other would drift the payload
    silently. This test reads the field set off the first descriptor
    and pins the full membership.
    """
    result = _call(g1_list_damp_transitions)
    first = result["transitions"][0]
    assert set(first) == {
        "name",
        "expected_fsm_ids",
        "pose_check",
        "avg_knee_max_rad",
        "description",
    }, (
        f"transition descriptor drifted; got fields {sorted(first)}, "
        f"expected ['avg_knee_max_rad', 'description', "
        f"'expected_fsm_ids', 'name', 'pose_check']"
    )


def test_g1_list_damp_transitions_pose_check_field_matches_snapshot() -> None:
    """The ``avg_knee_max_rad`` field is populated exactly when ``pose_check`` is on.

    When ``pose_check`` is ``True`` the threshold is
    :data:`_AVG_KNEE_MAX_RAD`; when ``False`` the field is ``None``
    because there is no refusal boundary to surface. The two fields
    move together; this test pins that pair contract so a future
    widen (e.g. per-transition thresholds) surfaces here.
    """
    result = _call(g1_list_damp_transitions)
    for transition in result["transitions"]:
        if transition["pose_check"]:
            assert transition["avg_knee_max_rad"] == _AVG_KNEE_MAX_RAD, (
                f"{transition['name']}: pose_check True but "
                f"avg_knee_max_rad={transition['avg_knee_max_rad']!r}, "
                f"expected {_AVG_KNEE_MAX_RAD}"
            )
        else:
            assert transition["avg_knee_max_rad"] is None, (
                f"{transition['name']}: pose_check False but "
                f"avg_knee_max_rad={transition['avg_knee_max_rad']!r}, "
                f"expected None"
            )


def test_g1_damp_transition_admits_returns_the_descriptor_on_a_known_name() -> None:
    """The admit-path returns the same descriptor shape ``g1_list_*`` returns.

    Reads the expected descriptor off :data:`_DAMP_TRANSITIONS` rather
    than restating it, so a widen to the descriptor surfaces here
    once. The ``walk_ready_fsm_ids`` field is named too so a caller
    comparing an intended write against the general locomotion gate
    has the FSM set on hand.
    """
    result = _call(g1_damp_transition_admits, name="squat_to_stand")
    assert result["status"] == "success"
    assert result["transition"]["name"] == "squat_to_stand"
    assert result["transition"]["expected_fsm_ids"] == sorted(_DAMP_TRANSITIONS["squat_to_stand"]["expected_fsm_ids"])
    assert result["transition"]["pose_check"] is True
    assert result["transition"]["avg_knee_max_rad"] == _AVG_KNEE_MAX_RAD
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)


def test_g1_damp_transition_admits_refuses_a_bogus_name() -> None:
    """A name outside the snapshot is refused with the ``7404`` code.

    The refusal string quotes the sorted set of admitted names so
    the caller can decide the next intent without a second query.
    """
    result = _call(g1_damp_transition_admits, name="pirouette")
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_TRANSITION_CODE
    assert result["refusal_text"] == ERR_CODES[_INVALID_TRANSITION_CODE]
    assert "pirouette" in result["reason"]
    for admitted in _DAMP_TRANSITIONS:
        assert admitted in result["reason"], (
            f"refusal reason should list the admitted set so the "
            f"caller can decide next; got {result['reason']!r} which "
            f"omits {admitted!r}"
        )


def test_g1_damp_transition_admits_refuses_a_miscased_name() -> None:
    """The name match is case-sensitive against the snapshot.

    The snapshot keys are lowercase-underscore; a caller passing
    ``"Squat_To_Stand"`` is refused so the returned envelope names
    the exact string a driver-side wrapper would refuse on.
    """
    result = _call(g1_damp_transition_admits, name="Squat_To_Stand")
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_TRANSITION_CODE


def test_g1_damp_transition_admits_refuses_a_non_string_argument() -> None:
    """A non-string argument is refused with the ``7404`` code.

    The lookup is by string key; a caller passing an int would
    otherwise raise on the ``in`` check, and this envelope's
    contract is to refuse with a decidable code, not raise.
    """
    result = _call(g1_damp_transition_admits, name=706)  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_TRANSITION_CODE
    assert "not a str" in result["reason"]


def test_every_transition_carries_a_non_empty_description() -> None:
    """Each transition names the semantic the neon bundle documented.

    The description is what an agent-facing planner reads to decide
    which transition to select; an empty description would leave
    the caller with only the FSM set. Pinned here so a snapshot
    edit that drops the description surfaces immediately.
    """
    for name, entry in _DAMP_TRANSITIONS.items():
        assert entry["description"], (
            f"{name}: description is empty; the neon bundle carried a one-line semantic per verb, port that in too"
        )
