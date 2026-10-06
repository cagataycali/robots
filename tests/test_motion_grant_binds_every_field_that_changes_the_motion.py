"""An operator's yes is spendable only by a call that moves the arm the way they were shown.

A grant is filed under :func:`~strands_robots._motion_grants.grant_key`, and every field missing
from that key is a way for a second call to spend the first call's yes. The key carried the target,
the instruction, the calibration and the joint targets, but not the pose library ``robot_id``
selects, not the speed profile (``steps``, ``step_delay``, ``smooth``), not the bus ``baudrate``, and
not which policy an ``execute`` / ``start`` runs. Each is now keyed and shown to the operator; a
grant left behind by a call that failed before its gate is forgotten when that call returns; and
the calibration is identified from the records the controller is built with, not a second read of
the file. The structural guard reads every gated tool's declared parameters, so a new one that
changes the motion cannot be added without being keyed or explicitly exempted.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots import _motion_grants
from strands_robots._motion_grants import (
    DETAIL_FIELDS,
    UNKEYED_FIELDS,
    consume_grant,
    deposit_grant,
    grant_key,
    pending_grants,
)
from strands_robots.drivers.feetech.bus import load_calibration

PORT = "/dev/ttyACM0"
RECORD = {"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 12, "range_min": 700, "range_max": 3300}}
POSE = {"action": "load_pose", "port": PORT, "pose_name": "rest"}
WRITE = {"action": "feetech_position", "port": PORT, "motor_id": 1, "position": 2048}
RUN = {"action": "execute", "instruction": "pick the cube", "duration": 10}


@pytest.fixture(autouse=True)
def _clean_store() -> Any:
    with _motion_grants._grants_lock:
        _motion_grants._grants.clear()
    yield
    with _motion_grants._grants_lock:
        _motion_grants._grants.clear()


@pytest.mark.parametrize(
    ("tool", "base", "field", "approved", "spent"),
    [
        ("pose_tool", POSE, "robot_id", "lib-a", "lib-b"),
        ("pose_tool", POSE, "steps", 100, 2),
        ("pose_tool", POSE, "step_delay", 0.2, 0.01),
        ("pose_tool", POSE, "smooth", None, False),
        ("serial_tool", WRITE, "baudrate", 1_000_000, 115_200),
        ("so101", RUN, "policy_provider", "lerobot_local", "remote"),
        ("so101", RUN, "policy_host", "localhost", "10.0.0.9"),
        ("so101", RUN, "policy_port", 5555, 6666),
        ("so101", RUN, "pretrained_name_or_path", "lab/act_pick", "lab/act_throw"),
        ("so101", RUN, "policy_type", "act", "pi0"),
        ("so101", RUN, "model_path", "/ckpt/a", "/ckpt/b"),
        ("so101", RUN, "embodiment", "so101", "so100"),
        ("so101", RUN, "walk", False, True),
        ("so101", RUN, "target_velocity", [0.1, 0.0, 0.0], [2.0, 0.0, 0.0]),
        ("so101", RUN, "policy_config", {"pretrained_name_or_path": "a"}, {"pretrained_name_or_path": "b"}),
    ],
)
def test_a_yes_is_not_spendable_by_a_call_that_moves_differently(
    tool: str, base: dict[str, Any], field: str, approved: Any, spent: Any
) -> None:
    def call(value: Any) -> dict[str, Any]:
        return {**base, field: value} if value is not None else dict(base)

    assert grant_key(tool, call(approved)) != grant_key(tool, call(spent))
    deposit_grant(tool, call(approved))
    assert consume_grant(tool, call(spent)) is False, f"a yes for {field}={approved!r} spent by {field}={spent!r}"
    assert consume_grant(tool, call(approved)) is True


def test_the_operator_reads_every_field_the_grant_is_keyed_on() -> None:
    from strands_robots.dashboard.agent_hitl import _direct_serial_detail, motion_intent

    line = _direct_serial_detail("pose_tool", "load_pose", {**POSE, "robot_id": "lib-a", "step_delay": 0.01})
    assert "robot_id=lib-a" in line and "step_delay=0.01" in line
    physical = {"arm": {"presence": {"robot_type": "so101_follower", "connected": True}}}
    run = {**RUN, "policy_provider": "remote", "policy_host": "10.0.0.9", "policy_port": 6666}
    reason = motion_intent("arm", run, physical, extra_actions={"arm": frozenset({"execute"})}, bound_targets={})
    assert reason is not None
    for shown in ("pick the cube", "policy_provider=remote", "policy_host=10.0.0.9", "policy_port=6666"):
        assert shown in reason["instruction"]


def _write(path: Path, record: dict[str, Any]) -> Path:
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_the_grant_is_matched_against_the_calibration_records_that_drive_the_servos(tmp_path: Path) -> None:
    shown = _write(tmp_path / "arm.json", RECORD)
    other = _write(tmp_path / "other.json", {"shoulder_pan": {**RECORD["shoulder_pan"], "homing_offset": -900}})
    call = {
        "action": "move_motor",
        "port": PORT,
        "calibration": str(shown),
        "motor_name": "shoulder_pan",
        "position": 30,
    }
    deposit_grant("pose_tool", call)
    # The file the operator was shown changed after the yes and before the tool loaded it.
    assert consume_grant("pose_tool", call, calibration=load_calibration(other)) is False
    assert consume_grant("pose_tool", call, calibration=load_calibration(shown)) is True


def test_a_yes_for_a_call_that_stopped_before_its_gate_does_not_outlive_it() -> None:
    from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
    from strands.interrupt import InterruptException

    from strands_robots.dashboard.agent_hitl import MotionInterruptHook

    agent: Any = SimpleNamespace(_interrupt_state=SimpleNamespace(interrupts={}))
    tool_use: Any = {"name": "pose_tool", "toolUseId": "t1", "input": {**POSE, "robot_id": "lib-a"}}
    hook = MotionInterruptHook(lambda: {PORT: {"presence": {"robot_type": "so101_follower", "connected": True}}})
    before = BeforeToolCallEvent(agent=agent, selected_tool=None, tool_use=tool_use, invocation_state={})
    with pytest.raises(InterruptException) as exc:
        hook._gate(before)
    agent._interrupt_state.interrupts[exc.value.interrupt.id].response = {"approve": True}
    hook._gate(before)
    assert len(pending_grants()) == 1
    # The tool refused before spending it ("Pose 'rest' not found" in lib-a).
    result: Any = {"toolUseId": "t1", "status": "error", "content": [{"text": "Pose 'rest' not found"}]}
    hook._expire(
        AfterToolCallEvent(agent=agent, selected_tool=None, tool_use=tool_use, invocation_state={}, result=result)
    )
    assert pending_grants() == []


# --- the structural guard -----------------------------------------------------------------------

#: The parts every key carries whatever the tool: ``action``, the target the gate resolves
#: (``port`` / ``target``), the instruction (``instruction`` / ``message``), and the calibration.
FIXED_KEY_PARTS = frozenset({"action", "port", "target", "instruction", "message", "calibration"})


def _declared(tool: Any) -> set[str]:
    return set(tool.tool_spec["inputSchema"]["json"]["properties"])


def _gated_tools() -> dict[str, set[str]]:
    from strands_robots.dashboard.peer_tools import KIND_REAL, peer_tool_spec
    from strands_robots.hardware_robot import Robot
    from strands_robots.tools.pose_tool import pose_tool
    from strands_robots.tools.serial_tool import serial_tool

    robot = SimpleNamespace(robot="so101", tool_name_str="so101")
    proxy = peer_tool_spec("arm", KIND_REAL, "arm")
    assert proxy is not None
    return {
        "pose_tool": _declared(pose_tool),
        "serial_tool": _declared(serial_tool),
        "Robot": set(Robot.tool_spec.fget(robot)["inputSchema"]["json"]["properties"]),  # type: ignore[attr-defined]
        "mesh proxy": set(proxy["inputSchema"]["json"]["properties"]),
    }


@pytest.mark.parametrize("tool", ["pose_tool", "serial_tool", "Robot", "mesh proxy"])
def test_every_parameter_of_a_gated_tool_is_keyed_or_exempt_with_a_reason(tool: str) -> None:
    params = _gated_tools()[tool]
    assert params, f"{tool}: no declared parameters were found; the guard needs updating with the tool"
    drift = params - FIXED_KEY_PARTS - set(DETAIL_FIELDS) - set(UNKEYED_FIELDS)
    assert not drift, f"{tool} declares {sorted(drift)}, which neither the grant key nor UNKEYED_FIELDS names"


def _gate_payload_keys(module: Any) -> set[str]:
    """The literal keys of the ``tool_input = {... for key, value in ((k, v), ...)}`` the tool hands the gate."""
    keys: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if not (
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "tool_input" for t in node.targets)
        ):
            continue
        if isinstance(node.value, ast.DictComp) and isinstance(source := node.value.generators[0].iter, ast.Tuple):
            keys |= {
                str(pair.elts[0].value)
                for pair in source.elts
                if isinstance(pair, ast.Tuple) and pair.elts and isinstance(pair.elts[0], ast.Constant)
            }
    return keys


@pytest.mark.parametrize("tool", ["pose_tool", "serial_tool"])
def test_a_direct_serial_tool_spends_its_grant_with_every_keyed_parameter(tool: str) -> None:
    """The tool rebuilds the call it spends against; a keyed parameter it drops is one its key cannot see."""
    import importlib

    payload = _gate_payload_keys(importlib.import_module(f"strands_robots.tools.{tool}"))
    keyed = _gated_tools()[tool] - set(UNKEYED_FIELDS)
    assert payload, f"{tool}: the gate payload was not found; the guard needs updating with the tool"
    assert keyed <= payload, f"{tool} spends its grant without {sorted(keyed - payload)}"
