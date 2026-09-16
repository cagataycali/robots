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
def _gate_environment(monkeypatch, tmp_path):
    """A gate that really asks, a private pose library and a private audit log.

    ``BYPASS_TOOL_CONSENT`` switches the gate off wholesale, and the ambient
    environment of an agent run carries it - so every "still asks" case here
    would pass vacuously without clearing it, which is what the sibling gate
    fixtures do. The audit dir is redirected for the same reason those do:
    a call that reaches the operator records their reply, and the default
    destination is the developer's own ``~/.strands_robots``.
    """
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    monkeypatch.delenv(pose_mod.COMMAND_ALLOW_ENV, raising=False)
    monkeypatch.delenv(serial_mod.COMMAND_ALLOW_ENV, raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
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


@pytest.mark.parametrize("kwargs,expected", [p for p in POSE_DOOMED if "unknown" not in str(p.id)])
def test_the_pose_branch_answers_in_the_same_words(monkeypatch, kwargs, expected):
    """The branch's own check, still there, still says what the pre-gate one says.

    The pre-gate check is bypassed and the gate pre-approved, so the answer
    comes from the action's own branch: two copies of one verdict, and an
    operator must not get different words depending on which ran. Without
    bypassing it the pre-gate check answers first, so a cell that only opens
    the gate never reaches the branch at all.

    The unknown-motor rows are left out: the branch only learns that from the
    controller after the port is open, which is the point of checking first.
    """
    monkeypatch.setattr(pose_mod, "_motion_input_error", lambda *_, **__: None)
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


@pytest.mark.parametrize("kwargs,expected", [p for p in SERIAL_DOOMED if "hex" not in str(p.id)])
def test_the_serial_branch_answers_in_the_same_words(monkeypatch, fake_serial, kwargs, expected):
    """The branch's own payload check, still there, still says the same thing.

    Reached the way the pose one is - pre-gate check bypassed, gate
    pre-approved - which for a write means the port is opened first, so a
    stand-in port is installed.
    """
    monkeypatch.setattr(serial_mod, "_write_payload_error", lambda *_, **__: None)
    monkeypatch.setenv(serial_mod.COMMAND_ALLOW_ENV, "*")
    out = _call(serial_mod.serial_tool, port=PORT, **kwargs)
    assert out.startswith("error:") and expected in out, out


def test_bad_hex_has_no_branch_verdict_and_left_the_port_open(monkeypatch, fake_serial):
    """Why the hex rows have no wording to match: the branch has no verdict.

    ``bytes.fromhex`` raises, so the answer is the ValueError's own text -
    naming neither the parameter nor a usable value - from a port that was
    opened and that the exception path never closes. The pre-gate check is the
    only verdict there is, and it opens no port at all.
    """
    monkeypatch.setenv(serial_mod.COMMAND_ALLOW_ENV, "*")
    assert _call(serial_mod.serial_tool, action="send", port=PORT, hex_data="ZZ 01").startswith("error:")
    assert fake_serial == []

    monkeypatch.setattr(serial_mod, "_write_payload_error", lambda *_, **__: None)
    out = _call(serial_mod.serial_tool, action="send", port=PORT, hex_data="ZZ 01")
    assert "non-hexadecimal" in out, out
    assert [port.is_open for port in fake_serial] == [True]
