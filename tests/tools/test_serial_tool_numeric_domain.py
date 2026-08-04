# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``serial_tool`` agent tool must refuse a numeric option it cannot honor.

Six numeric options reach the raw servo bus, and none of them was checked. Each
one failed differently and none of the failures was reportable:

* The three Feetech register fields are encoded into fixed-width bytes with a
  mask (``value & 0xFF``, ``(value >> 8) & 0xFF``), so an out-of-range value was
  not rejected - it was silently truncated into a different, reachable command.
  ``position=70000`` put 4464 on the wire and ``position=-1`` put 65535, the
  largest value the two-byte field can hold, i.e. a full-scale move for a caller
  who asked for one step below zero. Both returned ``status="success"`` with a
  message quoting the number the caller supplied ("Position 70000 (6153.8 deg)"),
  so the report described a command the servo never received.
* ``motor_id=255`` built a frame whose ID byte is ``0xFF`` - a third copy of the
  two-byte header - and ``motor_id=True`` silently addressed motor 1 while the
  message read "Feetech Motor True". ``motor_id=300`` leaked
  ``bytes must be in range(0, 256)``, naming neither the tool nor the parameter,
  from inside the packet builder and before the port was closed.
* ``read_bytes`` in ``{0, -1, -100, True}`` was not refused by pyserial either:
  its read loop is ``while len(read) < size``, so a non-positive size returns
  immediately with no bytes and the tool reported ``success`` "Read 0 bytes" -
  indistinguishable from a timed-out read on a healthy port with data pending.
  ``2.7`` leaked ``'float' object cannot be interpreted as an integer``.
* ``timeout=nan`` waited no time at all, because every comparison against ``nan``
  is false, and still reported ``success``; ``timeout=inf`` failed with
  ``timestamp out of range for platform time_t``; ``True`` was a silent 1.0 s.
* ``baudrate`` is coerced by pyserial rather than checked, so ``2.7`` opened the
  port at 2 baud and ``True`` at 1 baud - neither can carry a servo frame - while
  ``0`` was accepted outright.

:mod:`~strands_robots.tools.pose_tool` writes the same ``Goal_Position`` register
through the same mask and is unaffected: it clamps to each motor's declared range
in ``degrees_to_position`` before encoding, so its mask can only ever see a value
that fits. That is the contract these tests give the raw-bus tool.

They pin the domain, the per-action scoping (an option an action never reads must
not be refused), the guard's placement before the port is opened, that a value
the tool accepts is the value the wire carries, and the option tables so a new
action or a new numeric option cannot ship unguarded.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
import serial

# The module object itself is needed - for the option tables the drift guard
# scans and for the source it parses - so every name in it, the tool included, is
# reached through this one alias.
import strands_robots.tools.serial_tool as serial_mod

# One value per rejection reason.
BAD_COUNTS = (0, -1, -100, 2.7, True, "8", None, [8])
BAD_TIMEOUTS = (-1, -0.5, float("nan"), float("inf"), True, "1.0", None, [1.0])

# (action, the register field it consumes, a value that fits every other field).
REGISTER_ACTIONS = (
    ("feetech_position", "position", 2048),
    ("feetech_velocity", "velocity", 100),
)


class _FakeSerial:
    """Stand-in for ``serial.Serial`` recording the bytes that reach the wire."""

    def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.writes: list[bytes] = []
        self.closed = False
        self.in_waiting = 0

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))

    def read(self, n: int = 1) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[_FakeSerial]:
    """Record every port this tool opens; empty means the bus was never touched."""
    created: list[_FakeSerial] = []

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> _FakeSerial:
        instance = _FakeSerial(port, baudrate, timeout)
        created.append(instance)
        return instance

    monkeypatch.setattr(serial, "Serial", _ctor)
    return created


def _call(**kwargs: Any) -> dict[str, Any]:
    """Invoke the tool with a fake port, splatted so off-type values reach it."""
    return serial_mod.serial_tool(port="/dev/fake0", **kwargs)


def _text(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _goal_from(packet: bytes) -> int:
    """Decode the two-byte little-endian register value out of a written packet."""
    return packet[6] | (packet[7] << 8)


class TestARegisterFieldIsNeverSilentlyTruncated:
    """A value the field cannot hold is refused, not masked into another command."""

    @pytest.mark.parametrize("action,field,other", REGISTER_ACTIONS)
    @pytest.mark.parametrize("value", (-1, -100, 70000, 2.7, True, "100", [100]))
    def test_an_unusable_register_value_is_refused_before_the_port_opens(
        self, action: str, field: str, other: int, value: Any, opened: list[_FakeSerial]
    ) -> None:
        result = _call(action=action, motor_id=1, **{field: value})
        assert result["status"] == "error"
        assert field in _text(result)
        assert action in _text(result)
        assert opened == []

    def test_a_negative_position_is_refused_rather_than_encoded_as_the_field_maximum(
        self, opened: list[_FakeSerial]
    ) -> None:
        result = _call(action="feetech_position", motor_id=1, position=-1)

        assert result["status"] == "error"
        # -1 & 0xFF and (-1 >> 8) & 0xFF are both 0xFF, so the pre-fix encoding
        # was 65535 - the largest the field holds - for a caller asking for one
        # step below zero.
        assert "-1" in _text(result)
        assert opened == []

    @pytest.mark.parametrize("position", (0, 1, 2048, 4095))
    def test_the_reported_position_is_the_position_on_the_wire(self, position: int, opened: list[_FakeSerial]) -> None:
        result = _call(action="feetech_position", motor_id=1, position=position)

        assert result["status"] == "success"
        assert _goal_from(opened[0].writes[0]) == position
        # The message quotes the caller's value, which the bound makes truthful.
        assert f"Position {position} " in _text(result)
        assert f"{position / 4095 * 360:.1f} deg" in _text(result)

    @pytest.mark.parametrize("velocity", (0, 100, 65535))
    def test_the_reported_velocity_is_the_velocity_on_the_wire(self, velocity: int, opened: list[_FakeSerial]) -> None:
        result = _call(action="feetech_velocity", motor_id=1, velocity=velocity)

        assert result["status"] == "success"
        assert _goal_from(opened[0].writes[0]) == velocity

    @pytest.mark.parametrize("motor_id", (0, 255, 256, 300, -1, True, 2.5, "1"))
    def test_a_motor_id_outside_the_frames_id_byte_is_refused(self, motor_id: Any, opened: list[_FakeSerial]) -> None:
        result = _call(action="feetech_ping", motor_id=motor_id)

        assert result["status"] == "error"
        assert "motor_id" in _text(result)
        assert opened == []

    @pytest.mark.parametrize("motor_id", (1, 2, 6, 254))
    def test_an_addressable_motor_id_still_reaches_the_wire(self, motor_id: int, opened: list[_FakeSerial]) -> None:
        result = _call(action="feetech_position", motor_id=motor_id, position=100)

        assert result["status"] == "success"
        assert opened[0].writes[0][2] == motor_id


class TestTheReadBudgetAndLengthMustBeHonorable:
    """A length that reads nothing, or a budget that waits no time, is refused."""

    @pytest.mark.parametrize("value", BAD_COUNTS)
    def test_read_refuses_a_length_it_cannot_read(self, value: Any, opened: list[_FakeSerial]) -> None:
        result = _call(action="read", read_bytes=value)
        assert result["status"] == "error"
        assert "read_bytes" in _text(result)
        assert opened == []

    @pytest.mark.parametrize("value", BAD_TIMEOUTS)
    def test_read_refuses_a_budget_it_cannot_wait(self, value: Any, opened: list[_FakeSerial]) -> None:
        result = _call(action="read", timeout=value)
        assert result["status"] == "error"
        assert "timeout" in _text(result)
        assert opened == []

    def test_a_non_blocking_poll_is_still_accepted(self, opened: list[_FakeSerial]) -> None:
        """pyserial documents ``timeout=0`` as non-blocking: return what is buffered."""
        result = _call(action="read", timeout=0, read_bytes=8)

        assert result["status"] == "success"
        assert opened[0].timeout == 0

    @pytest.mark.parametrize("value", (0.05, 1.0, 30))
    def test_a_usable_budget_may_be_fractional(self, value: float, opened: list[_FakeSerial]) -> None:
        assert _call(action="read", timeout=value)["status"] == "success"
        assert opened[0].timeout == value

    @pytest.mark.parametrize("value", (0, -1, 2.7, True, "9600", None))
    def test_a_line_speed_no_frame_can_be_carried_at_is_refused(self, value: Any, opened: list[_FakeSerial]) -> None:
        result = _call(action="read", baudrate=value)
        assert result["status"] == "error"
        assert "baudrate" in _text(result)
        assert opened == []

    @pytest.mark.parametrize("value", (9600, 115200, 1000000))
    def test_a_real_line_speed_still_opens_the_port(self, value: int, opened: list[_FakeSerial]) -> None:
        assert _call(action="read", baudrate=value)["status"] == "success"
        assert opened[0].baudrate == value


class TestOnlyTheOptionsAnActionReadsAreChecked:
    """A value the requested action never looks at must not refuse a working call."""

    def test_port_discovery_reads_no_option_at_all(self, opened: list[_FakeSerial], monkeypatch) -> None:
        monkeypatch.setattr(serial.tools.list_ports, "comports", lambda: [])

        result = serial_mod.serial_tool(
            action="list_ports", baudrate=0, timeout=float("nan"), read_bytes=-1, motor_id=999
        )

        assert result["status"] == "success"
        assert opened == []

    def test_send_ignores_the_read_length(self, opened: list[_FakeSerial]) -> None:
        assert _call(action="send", data="PING", read_bytes=0)["status"] == "success"

    def test_read_ignores_the_servo_registers(self, opened: list[_FakeSerial]) -> None:
        assert _call(action="read", motor_id=999, position=-5, velocity=-5)["status"] == "success"

    def test_a_ping_ignores_the_position_and_velocity_registers(self, opened: list[_FakeSerial], monkeypatch) -> None:
        monkeypatch.setattr(serial_mod.time, "sleep", lambda *_: None)

        result = _call(action="feetech_ping", motor_id=1, position=-5, velocity=70000)

        assert "position" not in _text(result)
        assert "velocity" not in _text(result)

    def test_a_position_write_ignores_the_velocity_register(self, opened: list[_FakeSerial]) -> None:
        assert _call(action="feetech_position", motor_id=1, position=100, velocity=-5)["status"] == "success"


class TestTheRequiredFieldCheckStillOwnsAnAbsentRegister:
    """An unset register is reported by the action's own check, not as a bad value."""

    def test_an_absent_position_still_reports_the_whole_missing_pair(self, opened: list[_FakeSerial]) -> None:
        result = _call(action="feetech_position", motor_id=1)

        assert result["status"] == "error"
        assert "motor_id and position required" in _text(result)
        assert opened[0].closed

    def test_an_absent_motor_id_still_reports_the_whole_missing_pair(self, opened: list[_FakeSerial]) -> None:
        result = _call(action="feetech_velocity", velocity=100)

        assert result["status"] == "error"
        assert "motor_id and velocity required" in _text(result)


class TestTheOptionTablesCannotDriftApart:
    """A new action or a new numeric option cannot ship without a decision."""

    def _tool_parameters(self) -> dict[str, str]:
        """Map parameter name -> annotation, read from the tool's own source."""
        source = Path(inspect.getfile(serial_mod)).read_text(encoding="utf-8")
        (function,) = [
            node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "serial_tool"
        ]
        args = function.args.args + function.args.kwonlyargs
        return {a.arg: ast.unparse(a.annotation) for a in args if a.annotation is not None}

    def test_every_documented_action_has_a_scoping_decision(self) -> None:
        documented = {
            line.split('"')[1]
            for line in (serial_mod.serial_tool.tool_spec["description"] or "").splitlines()
            if line.strip().startswith('- "')
        }
        # ``list_ports`` returns before the port is opened, so it reads none of
        # the options and is deliberately absent from the table.
        assert documented == set(serial_mod._OPTIONS_BY_ACTION) | {"list_ports"}

    def test_every_numeric_parameter_is_covered_by_a_domain(self) -> None:
        numeric = {
            name
            for name, annotation in self._tool_parameters().items()
            if annotation.replace(" ", "").removesuffix("|None") in {"int", "float"}
        }
        assert numeric == {name for name, _ in serial_mod._OPTION_DOMAINS}

    def test_every_option_an_action_reads_has_a_declared_domain(self) -> None:
        declared = {name for name, _ in serial_mod._OPTION_DOMAINS}
        consumed = {option for options in serial_mod._OPTIONS_BY_ACTION.values() for option in options}
        assert consumed == declared

    def test_every_register_field_is_an_option_some_action_reads(self) -> None:
        consumed = {option for options in serial_mod._OPTIONS_BY_ACTION.values() for option in options}
        assert set(serial_mod._REGISTER_FIELDS) <= consumed

    @pytest.mark.parametrize("field", ("motor_id", "position", "velocity"))
    def test_every_register_bound_is_a_value_the_field_can_hold(self, field: str) -> None:
        low, high, why = serial_mod._REGISTER_FIELDS[field]
        assert 0 <= low <= high
        # ``position`` / ``velocity`` are written as two bytes and ``motor_id``
        # as one, so no bound may exceed what its field encodes.
        assert high <= (0xFF if field == "motor_id" else 0xFFFF)
        assert why
