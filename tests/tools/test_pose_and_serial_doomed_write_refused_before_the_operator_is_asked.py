"""A pose_tool motion or serial_tool write its own branch would refuse is refused before the operator is asked.

Both tools ask the operator to approve every motion/write, then check what
they were given. ``move_motor`` without a position, ``load_pose`` of a pose
that is not stored, ``send`` with no payload or with ``hex_data="ZZ"``:
each raised the approval interrupt, and the operator who typed "y" was
answered with ``motor_name and position required`` or, for the bad hex, a
``ValueError`` from the write after the port was opened. Those checks now
run first; the branches still run them again.
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock

import pytest

pose_mod = importlib.import_module("strands_robots.tools.pose_tool")
serial_mod = importlib.import_module("strands_robots.tools.serial_tool")

PORT = "/dev/cu.does-not-exist"


def _call(fn: Any, **kwargs: Any) -> str:
    """``asked`` when the gate reached the operator, else ``<status>:<text>``.

    The stand-in context declines every question, so a call that asks comes
    back as the gate's own refusal and ``interrupt`` records that it was asked;
    a call refused before the gate never reaches ``interrupt`` at all.
    """
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = "n"
    result = fn(tool_context=ctx, **kwargs)
    if ctx.interrupt.called:
        return "asked"
    return f"{result['status']}:{result['content'][0]['text']}"


@pytest.fixture(autouse=True)
def _no_allowlist(monkeypatch, tmp_path):
    monkeypatch.delenv(pose_mod.COMMAND_ALLOW_ENV, raising=False)
    monkeypatch.delenv(serial_mod.COMMAND_ALLOW_ENV, raising=False)
    monkeypatch.chdir(tmp_path)  # an empty pose library


POSE_DOOMED = [
    pytest.param(
        {"action": "move_motor", "motor_name": "shoulder_pan"},
        "motor_name and position required",
        id="move_motor-no-position",
    ),
    pytest.param(
        {"action": "move_motor", "motor_name": "elbow", "position": 10},
        "names an unknown motor elbow",
        id="move_motor-unknown-motor",
    ),
    pytest.param({"action": "move_multiple", "positions": {}}, "positions dict required", id="move_multiple-empty"),
    pytest.param(
        {"action": "move_multiple", "positions": {"elbow": 10}},
        "positions['elbow'] names an unknown motor",
        id="move_multiple-unknown-motor",
    ),
    pytest.param(
        {"action": "incremental_move", "motor_name": "shoulder_pan"},
        "motor_name and delta required",
        id="incremental_move-no-delta",
    ),
    pytest.param({"action": "load_pose"}, "pose_name required", id="load_pose-no-name"),
    pytest.param({"action": "load_pose", "pose_name": "nope"}, "Pose 'nope' not found", id="load_pose-not-stored"),
]


@pytest.mark.parametrize("kwargs,expected", POSE_DOOMED)
def test_a_doomed_pose_motion_is_refused_without_asking(kwargs, expected):
    out = _call(pose_mod.pose_tool, port=PORT, **kwargs)
    assert out != "asked"
    assert out.startswith("error:") and expected in out, out


@pytest.mark.parametrize("kwargs,expected", [p for p in POSE_DOOMED if "unknown" not in p.id])
def test_the_pose_refusal_matches_the_branch_with_the_gate_bypassed(monkeypatch, kwargs, expected):
    """The pre-gate check speaks with the branch's voice.

    The unknown-motor rows are left out: the branch only learns that from the
    controller after the port is open, which is the point of checking first.
    """
    monkeypatch.setenv(pose_mod.COMMAND_ALLOW_ENV, "*")
    out = _call(pose_mod.pose_tool, port=PORT, **kwargs)
    assert out.startswith("error:") and expected in out, out


def test_the_unknown_motor_refusal_names_the_table():
    out = _call(pose_mod.pose_tool, action="move_motor", motor_name="elbow", position=10, port=PORT)
    for name in pose_mod._DEFAULT_MOTOR_CONFIGS:
        assert name in out


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action": "move_motor", "motor_name": "shoulder_pan", "position": 10},
        {"action": "move_multiple", "positions": {"shoulder_pan": 10}},
        {"action": "incremental_move", "motor_name": "shoulder_pan", "delta": 5},
        {"action": "reset_to_home"},
    ],
)
def test_a_sound_pose_motion_still_asks(kwargs):
    assert _call(pose_mod.pose_tool, port=PORT, **kwargs) == "asked"


def test_a_stored_pose_still_asks(monkeypatch):
    """The library is consulted pre-gate; a pose that is there reaches the operator."""
    monkeypatch.setenv(pose_mod.COMMAND_ALLOW_ENV, "*")
    # store_pose needs the bus; write the library directly.
    pm = pose_mod.PoseManager("so101_follower")
    pm.poses["rest"] = pose_mod.RobotPose(name="rest", positions={"shoulder_pan": 0.0}, timestamp=0.0)
    pm._save_poses()
    monkeypatch.delenv(pose_mod.COMMAND_ALLOW_ENV, raising=False)
    assert _call(pose_mod.pose_tool, action="load_pose", pose_name="rest", port=PORT) == "asked"


SERIAL_DOOMED = [
    pytest.param({"action": "send"}, "No data or hex_data provided", id="send-no-payload"),
    pytest.param({"action": "send_read"}, "No data to send", id="send_read-no-payload"),
    pytest.param({"action": "send", "hex_data": "ZZ 01"}, "hex_data must be hex byte pairs", id="send-not-hex"),
    pytest.param({"action": "send", "hex_data": "FFF"}, "hex_data must be hex byte pairs", id="send-odd-hex"),
    pytest.param({"action": "send_read", "hex_data": "0G"}, "hex_data must be hex byte pairs", id="send_read-not-hex"),
    pytest.param(
        {"action": "feetech_position", "position": 100}, "motor_id and position required", id="feetech_position-no-id"
    ),
    pytest.param(
        {"action": "feetech_position", "motor_id": 1},
        "motor_id and position required",
        id="feetech_position-no-position",
    ),
    pytest.param(
        {"action": "feetech_velocity", "motor_id": 1},
        "motor_id and velocity required",
        id="feetech_velocity-no-velocity",
    ),
]


@pytest.mark.parametrize("kwargs,expected", SERIAL_DOOMED)
def test_a_doomed_serial_write_is_refused_without_asking(kwargs, expected):
    out = _call(serial_mod.serial_tool, port=PORT, **kwargs)
    assert out != "asked"
    assert out.startswith("error:") and expected in out, out


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action": "send", "hex_data": "FF FF 01 04"},
        {"action": "send", "data": "hello"},
        {"action": "send_read", "data": "hello"},
        {"action": "feetech_position", "motor_id": 1, "position": 100},
        {"action": "feetech_velocity", "motor_id": 1, "velocity": 100},
    ],
)
def test_a_sound_serial_write_still_asks(kwargs):
    assert _call(serial_mod.serial_tool, port=PORT, **kwargs) == "asked"


def test_the_bad_hex_refusal_quotes_the_value():
    out = _call(serial_mod.serial_tool, action="send", port=PORT, hex_data="ZZ 01")
    assert "got ZZ 01." in out
