"""The four G1 reference verbs answer the driver and SDK tables, and nothing else.

Each verb fronts one constant table under one rule: no query lists the table, a
query resolves one entry in it. The values are read here off the driver's own
constants rather than restated, so a driver-side rename or a widened gate
surfaces as a shape change rather than as a table this file would have to keep
in step. What the cells do restate is the shape of each returned record, the
domain each refusal has to name, and the SDK-load-hygiene contract every file
under :mod:`strands_robots.tools.g1` carries: the import must pull no
``unitree_sdk2py`` submodule, or every headless runner breaks before a robot is
ever wired.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from strands_robots.drivers.g1 import (
    _G1_JOINT_INDEX,
    _G1_NAMED_JOINTS,
    _SDK_KD,
    _SDK_KP,
)
from strands_robots.drivers.unitree._common import (
    ERR_CODES,
    HANDSHAKE_FSMS,
    WALK_FSMS,
)
from strands_robots.tools.g1.g1_reference import (
    _ARM_ACTION_MAP,
    _ARM_RELEASE_ACTION_ID,
    _FSM_REFUSAL_CODE,
    _GROUP_SLOTS,
    _HOLDING_CODE,
    _INVALID_ACTION_CODE,
    _SCOPE_SETS,
    _TOPIC_BUSY_CODE,
    _UNKNOWN_CODE_TEXT,
    g1_arm_actions,
    g1_error_codes,
    g1_joints,
    g1_motion_gates,
)

# The ``@tool`` wrapper returns the wrapped function's value verbatim when
# called in-process; the cells below call the verbs directly for that reason.
_VERBS: dict[str, Any] = {
    "g1_joints": g1_joints,
    "g1_motion_gates": g1_motion_gates,
    "g1_arm_actions": g1_arm_actions,
    "g1_error_codes": g1_error_codes,
}


def test_the_import_pulls_no_sdk_module() -> None:
    """The module is loadable on a host without ``unitree_sdk2py``.

    The SDK loads only inside function bodies in this package; a module that
    pulled a submodule at import time would break every runner with no robot
    attached (refs strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_reference")
    leaked = {name for name in set(sys.modules) - before if "unitree" in name.lower()}
    assert leaked == set(), f"the import pulled SDK submodules: {leaked}"


@pytest.mark.parametrize("name", sorted(_VERBS))
def test_a_verb_with_no_query_lists_its_whole_table(name: str) -> None:
    """The consolidation's rule: no argument means the catalogue, on every verb."""
    result = _VERBS[name]()
    assert result["status"] == "success"
    assert result["count"] > 0, f"{name} answered an empty table"


class TestTheJointTable:
    """``g1_joints`` names exactly what ``G1Driver.send_action`` accepts."""

    def test_the_listing_is_every_slot_the_driver_names_once(self) -> None:
        result = g1_joints()
        assert result["count"] == _G1_NAMED_JOINTS
        assert sorted(row["name"] for row in result["joints"]) == sorted(_G1_JOINT_INDEX)
        indices = [row["index"] for row in result["joints"]]
        assert indices == list(range(_G1_NAMED_JOINTS))
        assert result["group"] is None
        assert result["groups"] == sorted(_GROUP_SLOTS)

    def test_every_row_carries_the_gain_pair_the_driver_would_use(self) -> None:
        for row in g1_joints()["joints"]:
            slot = row["index"]
            assert row["kp"] == _SDK_KP[slot], f"kp for slot {slot} ({row['name']}) drifted from the driver's table"
            assert row["kd"] == _SDK_KD[slot]

    @pytest.mark.parametrize("group", sorted(_GROUP_SLOTS))
    def test_a_group_name_lists_that_group_verbatim(self, group: str) -> None:
        result = g1_joints(group)
        assert result["status"] == "success"
        assert result["group"] == group
        assert result["count"] == len(_GROUP_SLOTS[group])
        assert [row["index"] for row in result["joints"]] == list(_GROUP_SLOTS[group])

    def test_the_groups_partition_the_slots(self) -> None:
        """A slot in two groups would duplicate a row and make ``group`` order-dependent."""
        covered = [slot for slots in _GROUP_SLOTS.values() for slot in slots]
        assert sorted(covered) == list(range(_G1_NAMED_JOINTS))
        assert len(covered) == len(set(covered))

    def test_a_slot_resolves_to_the_driver_key(self) -> None:
        for name, slot in _G1_JOINT_INDEX.items():
            result = g1_joints(slot)
            assert result["status"] == "success"
            assert (result["name"], result["index"]) == (name, slot)
            assert (result["kp"], result["kd"]) == (_SDK_KP[slot], _SDK_KD[slot])

    def test_a_name_resolves_to_the_driver_slot(self) -> None:
        for name, slot in _G1_JOINT_INDEX.items():
            result = g1_joints(name)
            assert result["status"] == "success", f"snake_case {name} refused: {result}"
            assert (result["index"], result["name"]) == (slot, name)

    @pytest.mark.parametrize("alias", ["LeftKnee", "leftKnee", "  LeftKnee  "])
    def test_a_camel_case_alias_resolves_to_the_canonical_key(self, alias: str) -> None:
        """The alias is one-way: the returned ``name`` is the spelling the wire takes."""
        result = g1_joints(alias)
        assert result["status"] == "success", f"alias {alias!r} refused: {result}"
        assert result["name"] == "left_knee"
        assert result["index"] == _G1_JOINT_INDEX["left_knee"]


class TestTheMotionGates:
    """``g1_motion_gates`` names exactly what ``_check_motion_gates`` admits on."""

    def test_the_scope_map_is_the_driver_admission_sets_themselves(self) -> None:
        """Identity, not equality: a widen in the driver cannot diverge from this map."""
        assert _SCOPE_SETS["arm"] is HANDSHAKE_FSMS
        assert _SCOPE_SETS["loco"] is WALK_FSMS
        assert set(_SCOPE_SETS) == {"arm", "loco"}

    def test_the_loco_gate_is_narrower_than_the_arm_gate(self) -> None:
        """The driver's rule, not this verb's: FSM 500 takes arm gestures, not walking."""
        assert WALK_FSMS <= HANDSHAKE_FSMS
        assert WALK_FSMS, "an empty loco gate would admit no locomotion write at all"

    def test_the_listing_carries_every_scope_with_its_set_and_refusal(self) -> None:
        result = g1_motion_gates()
        assert result["count"] == len(_SCOPE_SETS)
        assert result["scope"] is None
        assert result["scopes"] == sorted(_SCOPE_SETS)
        for row in result["gates"]:
            assert row["fsm_ids"] == sorted(_SCOPE_SETS[row["scope"]])
            assert all(isinstance(fsm, int) and not isinstance(fsm, bool) for fsm in row["fsm_ids"])
            assert row["refusal_code"] == _FSM_REFUSAL_CODE
            assert row["refusal_text"] == ERR_CODES[_FSM_REFUSAL_CODE]

    @pytest.mark.parametrize("scope", sorted(_SCOPE_SETS))
    def test_a_scope_filter_lists_that_scope_alone(self, scope: str) -> None:
        result = g1_motion_gates(scope=scope)
        assert result["count"] == 1
        assert result["scope"] == scope
        [row] = result["gates"]
        assert row["fsm_ids"] == sorted(_SCOPE_SETS[scope])

    def test_an_admitted_id_carries_no_refusal(self) -> None:
        for scope, admitted in _SCOPE_SETS.items():
            for fsm_id in admitted:
                result = g1_motion_gates(fsm_id=fsm_id, scope=scope)
                assert result["status"] == "success"
                assert (result["scope"], result["fsm_id"]) == (scope, fsm_id)
                assert result["admitted"] is True
                assert result["fsm_ids"] == sorted(admitted)
                assert "refusal_code" not in result
                assert "refusal_text" not in result

    @pytest.mark.parametrize("scope", sorted(_SCOPE_SETS))
    def test_an_id_outside_the_gate_carries_the_write_path_refusal(self, scope: str) -> None:
        """``42`` is outside both sets and outside the three-digit motion-switcher range."""
        assert 42 not in HANDSHAKE_FSMS and 42 not in WALK_FSMS
        result = g1_motion_gates(fsm_id=42, scope=scope)
        assert result["admitted"] is False
        assert result["refusal_code"] == _FSM_REFUSAL_CODE
        assert result["refusal_text"] == ERR_CODES[_FSM_REFUSAL_CODE]

    def test_a_membership_query_with_no_scope_answers_for_the_arm_gate(self) -> None:
        """An unnamed scope is the arm-SDK gate - the broader of the two the driver keeps."""
        result = g1_motion_gates(fsm_id=next(iter(sorted(HANDSHAKE_FSMS))))
        assert result["scope"] == "arm"
        assert result["fsm_ids"] == sorted(HANDSHAKE_FSMS)


class TestTheArmActionTable:
    """``g1_arm_actions`` names exactly what ``ExecuteAction`` admits."""

    def test_the_snapshot_is_the_sdk_shipped_set_and_names_the_release_id(self) -> None:
        """16 gestures, and ``99`` drops the arm-action hold on both sides."""
        assert len(_ARM_ACTION_MAP) == 16
        assert _ARM_RELEASE_ACTION_ID == 99
        assert _ARM_ACTION_MAP["release arm"] == _ARM_RELEASE_ACTION_ID

    def test_the_listing_carries_the_map_the_gate_and_the_three_refusals(self) -> None:
        result = g1_arm_actions()
        assert result["count"] == len(_ARM_ACTION_MAP)
        assert result["action_map"] == _ARM_ACTION_MAP
        result["action_map"]["synthetic"] = 999  # a fresh dict, not the constant
        assert "synthetic" not in _ARM_ACTION_MAP
        assert result["action_ids"] == sorted(_ARM_ACTION_MAP.values())
        assert result["release_action_id"] == _ARM_RELEASE_ACTION_ID
        assert result["arm_ready_fsm_ids"] == sorted(HANDSHAKE_FSMS)
        assert {row["code"] for row in result["refusals"]} == {
            _INVALID_ACTION_CODE,
            _HOLDING_CODE,
            _TOPIC_BUSY_CODE,
        }
        for row in result["refusals"]:
            assert row["text"] == ERR_CODES[row["code"]]

    @pytest.mark.parametrize(
        ("query", "echo"),
        [
            ("two-hand kiss", {"action": "two-hand kiss"}),
            (99, {"action_id": 99}),
        ],
        ids=["by-name", "by-id"],
    )
    def test_an_admitted_query_resolves_the_name_and_id_pair(self, query: str | int, echo: dict[str, Any]) -> None:
        """Both directions resolve, so a caller with either half gets the other."""
        result = g1_arm_actions(query)
        assert result["status"] == "success"
        assert result["query"] == echo
        assert result["admitted"] is True
        assert result["action_id"] == _ARM_ACTION_MAP[result["action_name"]]
        assert "refusal_code" not in result

    @pytest.mark.parametrize(
        ("query", "echo"),
        [
            # Title case: the SDK does not lower-case its own lookup.
            ("Two-Hand Kiss", {"action": "Two-Hand Kiss"}),
            (42, {"action_id": 42}),
        ],
        ids=["unknown-name", "unknown-id"],
    )
    def test_a_query_outside_the_set_carries_the_sdk_refusal(self, query: str | int, echo: dict[str, Any]) -> None:
        result = g1_arm_actions(query)
        assert result["status"] == "success"
        assert result["query"] == echo
        assert result["admitted"] is False
        assert result["refusal_code"] == _INVALID_ACTION_CODE
        assert result["refusal_text"] == ERR_CODES[_INVALID_ACTION_CODE]
        assert "action_name" not in result


class TestTheErrorCodeCatalogue:
    """``g1_error_codes`` decodes what the SDK's handlers return."""

    @pytest.mark.parametrize(
        ("side", "codes"),
        [
            ("arm", (0, 7400, 7401, 7402, 7404)),
            ("loco", (7301, 7302, 7303)),
            ("rpc transport", (3102, 3103, 3104)),
        ],
    )
    def test_the_catalogue_names_each_side_of_the_sdk(self, side: str, codes: tuple[int, ...]) -> None:
        """A caller comparing a live ``refusal_code`` needs every rc a handler can return.

        The transport codes matter separately: an RPC that never reached a
        handler is retry-worthy, a handler that refused is not.
        """
        listed = {row["code"] for row in g1_error_codes()["error_codes"]}
        for code in codes:
            assert code in listed, f"rc={code} is a {side}-side code the verbs quote; the catalogue must decode it"

    def test_the_listing_is_the_whole_catalogue_in_fresh_containers(self) -> None:
        result = g1_error_codes()
        assert result["count"] == len(ERR_CODES)
        assert result["codes"] == sorted(ERR_CODES)
        for row in result["error_codes"]:
            assert row["text"] == ERR_CODES[row["code"]]
        result["codes"].append(9999)
        result["error_codes"][0]["synthetic"] = True
        fresh = g1_error_codes()
        assert 9999 not in fresh["codes"]
        assert "synthetic" not in fresh["error_codes"][0]

    @pytest.mark.parametrize("code", [0, 7302])
    def test_a_known_code_decodes_to_the_catalogued_text(self, code: int) -> None:
        result = g1_error_codes(code)
        assert result["status"] == "success"
        assert result["query"] == {"code": code}
        assert result["known"] is True
        assert result["text"] == ERR_CODES[code]

    @pytest.mark.parametrize("code", [9999, -1], ids=["above-the-set", "transport-minus-one"])
    def test_a_code_outside_the_catalogue_is_decidably_unknown(self, code: int) -> None:
        """``-1`` is the convention for an SDK call that raised instead of returning a rc.

        It is admitted rather than refused: the driver-side renderer answers for
        every integer, and a lookup narrower than the renderer whose text it
        quotes would send a caller back to the constant. The envelope always
        names something, so a caller never branches on a missing key.
        """
        result = g1_error_codes(code)
        assert result["status"] == "success"
        assert result["known"] is False
        assert result["text"] == _UNKNOWN_CODE_TEXT


@pytest.mark.parametrize(
    ("verb", "kwargs", "named"),
    [
        # A value outside the table names the value and the domain, so the
        # caller reads what is accepted rather than a hint.
        ("g1_joints", {"query": "left_tentacle"}, ["left_tentacle", "left_arm", "left_knee", "#2765"]),
        ("g1_joints", {"query": "ThirdEye"}, ["ThirdEye", "left_knee", "#2765"]),
        ("g1_joints", {"query": -1}, ["[0, 28]", "#2765"]),
        ("g1_joints", {"query": _G1_NAMED_JOINTS}, ["[0, 28]", "#2765"]),
        ("g1_motion_gates", {"scope": "teleport"}, ["teleport", "arm", "loco", "#358"]),
        ("g1_motion_gates", {"fsm_id": 500, "scope": "teleport"}, ["teleport", "arm", "loco", "#358"]),
        # ``True`` is an ``int``; read as a slot, an FSM id or an action id it
        # would answer a question the caller never asked - and a live write
        # built on it would land at the left hip.
        ("g1_joints", {"query": True}, ["bool", "#2765"]),
        ("g1_motion_gates", {"fsm_id": True}, ["bool", "#358"]),
        ("g1_arm_actions", {"query": True}, ["bool", "#358"]),
        ("g1_error_codes", {"code": True}, ["bool", "#358"]),
        # A wrong type names the type, so a caller with many parallel calls in
        # flight sees which one carried the wrong shape.
        ("g1_joints", {"query": ["left_knee"]}, ["list", "#2765"]),
        ("g1_joints", {"query": 5.0}, ["float", "#2765"]),
        ("g1_motion_gates", {"fsm_id": "500"}, ["str", "#358"]),
        ("g1_motion_gates", {"fsm_id": 500.0}, ["float", "#358"]),
        ("g1_arm_actions", {"query": None}, ["NoneType", "#358"]),
        ("g1_error_codes", {"code": "7302"}, ["str", "#358"]),
    ],
)
def test_a_refusal_names_the_value_the_domain_and_the_reference(
    verb: str, kwargs: dict[str, Any], named: list[str]
) -> None:
    """Every refusal in the family carries the same three anchors."""
    result = _VERBS[verb](**kwargs)
    assert result["status"] == "error", f"{verb}({kwargs}) was admitted: {result}"
    for fragment in named:
        assert fragment in result["message"], f"{verb}({kwargs}) refusal did not name {fragment!r}: {result['message']}"
