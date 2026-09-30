"""An operator's yes to a motor target is bound to the calibration it was given under.

The grant store keyed a human's yes on the tool, the action, the port, the instruction and the
motion fields (``motor_name``, ``position``, ...). ``calibration`` was in the payload ``pose_tool``
hands the gate, with a comment saying the operator approves both together, but not in the
identity computed over it. The calibration decides where a degree target puts the joint (a bus
given none commands the servo's full rotation instead of the arm's measured travel), so one yes
for "move this motor to 30" under ``arm_a.json`` was spendable by the same numbers under
``arm_b.json`` or under no calibration at all, with no second prompt and no trace (f017, CWE-863).
The dashboard's detail line was built from the same roster, so the operator never saw the
calibration either.

Now the grant key carries the calibration identity: the content hash of the file (so two spellings
of one file share a grant), the hash of an inline record, or the explicit word ``none`` when the
call has no calibration. ``calibration`` is on the detail roster too, and a call without one tells
the operator so. A structural guard keeps every field ``pose_tool`` and ``serial_tool`` hand the
gate on the roster or among the fixed key parts, so the next field cannot drift the same way.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from strands_robots import _motion_grants
from strands_robots._motion_grants import (
    DETAIL_FIELDS,
    calibration_identity,
    consume_grant,
    deposit_grant,
    grant_key,
    motion_fields,
)

PORT = "/dev/ttyACM0"
RECORD = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 12, "range_min": 700, "range_max": 3300},
}


def _call(calibration: Any = None, **extra: Any) -> dict[str, Any]:
    call: dict[str, Any] = {
        "action": "move_motor",
        "port": PORT,
        "motor_name": "shoulder_pan",
        "position": 30.0,
        **extra,
    }
    if calibration is not None:
        call["calibration"] = calibration
    return call


@pytest.fixture
def cal_a(tmp_path: Path) -> Path:
    p = tmp_path / "arm_a.json"
    p.write_text(json.dumps(RECORD), encoding="utf-8")
    return p


@pytest.fixture
def cal_b(tmp_path: Path) -> Path:
    p = tmp_path / "arm_b.json"
    other = {"shoulder_pan": {**RECORD["shoulder_pan"], "homing_offset": -900}}
    p.write_text(json.dumps(other), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _clean_store() -> Any:
    with _motion_grants._grants_lock:
        _motion_grants._grants.clear()
    yield
    with _motion_grants._grants_lock:
        _motion_grants._grants.clear()


# --- the identity -----------------------------------------------------------------------------


def test_no_calibration_is_an_explicit_identity_not_an_absent_field() -> None:
    assert calibration_identity(None) == "none"
    assert calibration_identity("") == "none"
    assert "calibration=none" in grant_key("pose_tool", _call())


def test_a_file_is_identified_by_its_content(cal_a: Path, tmp_path: Path) -> None:
    digest = hashlib.sha256(cal_a.read_bytes()).hexdigest()[:16]
    assert calibration_identity(str(cal_a)) == f"sha256:{digest}"
    link = tmp_path / "link.json"
    link.symlink_to(cal_a)
    relative = Path(str(cal_a)).parent / "." / cal_a.name
    assert calibration_identity(str(link)) == calibration_identity(str(cal_a))
    assert calibration_identity(str(relative)) == calibration_identity(str(cal_a))


def test_two_files_with_different_offsets_are_different_identities(cal_a: Path, cal_b: Path) -> None:
    assert calibration_identity(str(cal_a)) != calibration_identity(str(cal_b))


def test_an_unreadable_path_is_its_own_identity(tmp_path: Path) -> None:
    missing = str(tmp_path / "nope.json")
    ident = calibration_identity(missing)
    assert ident.startswith("unreadable:") and ident != "none"


def test_an_inline_record_is_hashed_canonically() -> None:
    a = calibration_identity({"x": {"b": 1, "a": 2}})
    b = calibration_identity({"x": {"a": 2, "b": 1}})
    assert a == b and a.startswith("sha256:")


# --- the grant ----------------------------------------------------------------------------------


def test_a_yes_under_one_calibration_is_not_spendable_under_another(cal_a: Path, cal_b: Path) -> None:
    deposit_grant("pose_tool", _call(str(cal_a)))
    assert consume_grant("pose_tool", _call(str(cal_b))) is False
    assert consume_grant("pose_tool", _call(str(cal_a))) is True


def test_a_yes_under_a_calibration_is_not_spendable_with_none(cal_a: Path) -> None:
    deposit_grant("pose_tool", _call(str(cal_a)))
    assert consume_grant("pose_tool", _call()) is False, "no calibration commands the servo's full rotation"
    assert consume_grant("pose_tool", _call(str(cal_a))) is True
    deposit_grant("pose_tool", _call())
    assert consume_grant("pose_tool", _call(str(cal_a))) is False
    assert consume_grant("pose_tool", _call()) is True


def test_the_same_file_by_another_spelling_spends_the_grant(cal_a: Path, tmp_path: Path) -> None:
    link = tmp_path / "same.json"
    link.symlink_to(cal_a)
    deposit_grant("pose_tool", _call(str(cal_a)))
    assert consume_grant("pose_tool", _call(str(link))) is True


def test_the_operator_line_names_the_calibration(cal_a: Path) -> None:
    assert "calibration" in DETAIL_FIELDS
    fields = motion_fields(_call(str(cal_a)))
    assert f"calibration={cal_a}" in fields
    assert fields.index(f"calibration={cal_a}") < fields.index("motor_name=shoulder_pan"), (
        "the frame of reference comes before the numbers read inside it"
    )


def test_the_dashboard_line_says_when_there_is_no_calibration(cal_a: Path) -> None:
    from strands_robots.dashboard.agent_hitl import _direct_serial_detail

    bare = _direct_serial_detail("pose_tool", "move_motor", _call())
    assert "calibration=none" in bare and "full rotation" in bare
    with_file = _direct_serial_detail("pose_tool", "move_motor", _call(str(cal_a)))
    assert f"calibration={cal_a}" in with_file and "full rotation" not in with_file
    raw = _direct_serial_detail("serial_tool", "send", {"action": "send", "port": PORT, "hex_data": "ff"})
    assert "calibration" not in raw, "serial_tool writes raw bytes; it has no calibration to speak of"


def test_grants_for_tools_without_a_calibration_are_unchanged_in_shape() -> None:
    call = {"action": "task", "target": "arm-1", "instruction": "wave"}
    deposit_grant("fleet", call)
    assert consume_grant("fleet", call) is True


# --- the structural guard -----------------------------------------------------------------------

FIXED_KEY_PARTS = frozenset({"action", "port", "target", "instruction", "message", "calibration"})


def _gate_payload_keys(module: Any) -> set[str]:
    """The literal keys of the ``tool_input = {... for key, value in ((k, v), ...)}`` the tool hands the gate."""
    tree = ast.parse(inspect.getsource(module))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "tool_input" for t in node.targets)
        ):
            continue
        comp = node.value
        if not isinstance(comp, ast.DictComp):
            continue
        source = comp.generators[0].iter
        if not isinstance(source, ast.Tuple):
            continue
        for pair in source.elts:
            if isinstance(pair, ast.Tuple) and pair.elts and isinstance(pair.elts[0], ast.Constant):
                keys.add(str(pair.elts[0].value))
    return keys


@pytest.mark.parametrize("module_name", ["strands_robots.tools.pose_tool", "strands_robots.tools.serial_tool"])
def test_every_field_a_tool_hands_the_gate_is_part_of_the_grant_identity(module_name: str) -> None:
    import importlib

    keys = _gate_payload_keys(importlib.import_module(module_name))
    assert keys, f"{module_name}: the gate payload was not found; the guard needs updating with the tool"
    drift = keys - FIXED_KEY_PARTS - set(DETAIL_FIELDS)
    assert not drift, f"{module_name} hands the gate {sorted(drift)}, which neither the key nor the operator line reads"
