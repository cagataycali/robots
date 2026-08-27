# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``serial_tool`` must not address the whole bus when it is about to read one reply.

The Feetech Protocol 1 ID byte carries every unicast address and one address that
is no servo: ``0xfe`` is the broadcast. ``motor_id`` was bounded at ``254`` for a
reason about the field's width -- "the packet carries the ID in one byte, and 255
is the frame header" -- which is true of the byte and says nothing about which of
its values a servo can hold. So ``action="feetech_ping"`` accepted the broadcast:
it wrote ``FF FF FE 02 01 FE``, slept, read ten bytes and reported
``Feetech Motor 254 responded: <hex>``. Every servo on the bus receives that
frame and, because ``PING`` is answered, every one of them replies at once; on a
half-duplex bus those replies collide, so the bytes read back belong to no single
servo and the ID quoted beside them belongs to none either.

The address space is not this module's to invent.
:mod:`~strands_robots.drivers.feetech.protocol` declares ``BROADCAST_ID`` and
``MAX_UNICAST_ID`` ("Highest ID a specific servo may hold. 0xFE is the
broadcast") and refuses the same intent from ``build_packet`` unless the caller
passes ``allow_broadcast=True`` "for ``SYNC_WRITE`` and other reply-less
instructions that legitimately target the broadcast ID". The tool now applies
that same rule one layer up, before the port is opened, so the two cannot
disagree about which address is a servo and which is the whole bus.

A reply-less write is deliberately still allowed to address the broadcast:
``feetech_position`` and ``feetech_velocity`` never read a status packet, so
moving every servo with one frame means exactly what it says. That asymmetry is
the point, and it is pinned in both directions here.

Why nothing caught this: the numeric-domain suite's refusal sweep is
``(0, 255, 256, 300, -1, True, 2.5, "1")`` -- the broadcast is inside the ID byte
and so is absent from a sweep of values outside it -- while its acceptance sweep
listed ``254`` under the name
``test_an_addressable_motor_id_still_reaches_the_wire``. The broadcast is the one
value in range that is not addressable, and that name recorded the opposite.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest
import serial

import strands_robots.tools.serial_tool as serial_mod
from strands_robots.drivers.feetech import protocol as feetech_protocol

# Stated here rather than read off the module under test, so these cells grade
# the Protocol 1 address space rather than agreeing with whatever the module
# currently believes. One cell below asserts the module's constants are these.
BROADCAST_ID = 0xFE
MAX_UNICAST_ID = 0xFD

# The action that writes and then reads a status packet back, and the two that
# write and never read. Named rather than derived so the behavioural cells state
# the split; the derived guard below is what keeps the module's own set honest.
REPLY_EXPECTING_ACTION = "feetech_ping"
REPLY_LESS_WRITES = (("feetech_position", {"position": 2048}), ("feetech_velocity", {"velocity": 100}))

# A well-formed six-byte status packet, so an accepted ping reports success and
# the control cells grade acceptance rather than a short read.
_PING_REPLY = bytes([0xFF, 0xFF, 0x01, 0x02, 0x00, (~(0x01 + 0x02 + 0x00)) & 0xFF])


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
        return _PING_REPLY[:n]

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


def _actions_that_read_a_reply() -> set[str]:
    """Actions whose own branch reads the port back and which consume ``motor_id``.

    Derived from the tool's body rather than listed, so an action added later
    that reads a status packet for a caller-supplied ID is held to the same rule
    the hour it lands. An action that reads without consuming ``motor_id``
    (``read``, ``send_read``, ``monitor``) addresses nothing, and a write that
    consumes ``motor_id`` without reading (``feetech_position``,
    ``feetech_velocity``) needs no reply, so neither is in scope.

    Returns:
        The action names that both read a reply and address a servo.
    """
    (body,) = [
        node
        for node in ast.parse(inspect.getsource(serial_mod)).body
        if isinstance(node, ast.FunctionDef) and node.name == "serial_tool"
    ]
    reads_by_action: dict[str, bool] = {}
    for node in ast.walk(body):
        if not isinstance(node, ast.If) or "action" not in ast.unparse(node.test):
            continue
        reads = any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "read"
            for statement in node.body
            for call in ast.walk(statement)
        )
        for literal in ast.walk(node.test):
            if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                reads_by_action.setdefault(literal.value, reads)
    return {
        action
        for action, reads in reads_by_action.items()
        if reads and "motor_id" in serial_mod._OPTIONS_BY_ACTION.get(action, ())
    }


class TestAReplyExpectingActionRefusesTheBroadcast:
    """The one address no servo holds is refused where a servo must answer."""

    def test_ping_refuses_the_broadcast(self, opened: list[_FakeSerial]) -> None:
        result = _call(action=REPLY_EXPECTING_ACTION, motor_id=BROADCAST_ID)

        assert result["status"] == "error"

    def test_ping_refuses_the_broadcast_before_the_port_is_opened(self, opened: list[_FakeSerial]) -> None:
        _call(action=REPLY_EXPECTING_ACTION, motor_id=BROADCAST_ID)

        assert opened == []

    def test_no_broadcast_frame_reaches_the_wire(self, opened: list[_FakeSerial]) -> None:
        _call(action=REPLY_EXPECTING_ACTION, motor_id=BROADCAST_ID)

        assert [packet for port in opened for packet in port.writes] == []


class TestTheRefusalNamesWhatWentWrong:
    """A caller is told which address it used, why it cannot answer, and what can."""

    @pytest.fixture
    def refusal(self, opened: list[_FakeSerial]) -> str:
        return _text(_call(action=REPLY_EXPECTING_ACTION, motor_id=BROADCAST_ID))

    def test_the_refusal_names_the_action(self, refusal: str) -> None:
        assert REPLY_EXPECTING_ACTION in refusal

    def test_the_refusal_names_the_parameter(self, refusal: str) -> None:
        assert "motor_id" in refusal

    def test_the_refusal_names_the_broadcast_address(self, refusal: str) -> None:
        assert f"{BROADCAST_ID:#x}" in refusal
        assert "broadcast" in refusal

    def test_the_refusal_says_why_no_reply_can_be_read(self, refusal: str) -> None:
        assert "every servo" in refusal
        assert "reply" in refusal

    def test_the_refusal_names_the_range_that_can_answer(self, refusal: str) -> None:
        assert f"[1, {MAX_UNICAST_ID}]" in refusal


class TestAReplyLessWriteStillAddressesTheWholeBus:
    """Moving every servo with one frame is not the defect and stays available."""

    @pytest.mark.parametrize("action,field", REPLY_LESS_WRITES)
    def test_a_write_accepts_the_broadcast(self, action: str, field: dict[str, int], opened: list[_FakeSerial]) -> None:
        result = _call(action=action, motor_id=BROADCAST_ID, **field)

        assert result["status"] == "success"

    @pytest.mark.parametrize("action,field", REPLY_LESS_WRITES)
    def test_the_broadcast_id_is_the_frames_id_byte(
        self, action: str, field: dict[str, int], opened: list[_FakeSerial]
    ) -> None:
        _call(action=action, motor_id=BROADCAST_ID, **field)

        assert opened[0].writes[0][2] == BROADCAST_ID

    @pytest.mark.parametrize("motor_id", (1, 6, MAX_UNICAST_ID))
    def test_ping_still_accepts_every_address_a_servo_can_hold(self, motor_id: int, opened: list[_FakeSerial]) -> None:
        result = _call(action=REPLY_EXPECTING_ACTION, motor_id=motor_id)

        assert result["status"] == "success"
        assert opened[0].writes[0][2] == motor_id


class TestTheAddressSpaceIsTheCodecs:
    """The tool and the Protocol 1 codec answer the same question the same way."""

    def test_the_codec_refuses_a_reply_expecting_broadcast(self) -> None:
        with pytest.raises(ValueError, match="broadcast"):
            feetech_protocol.ping_packet(BROADCAST_ID)

    def test_the_codec_accepts_the_highest_single_servo(self) -> None:
        assert feetech_protocol.ping_packet(MAX_UNICAST_ID)[2] == MAX_UNICAST_ID

    def test_the_codec_allows_a_reply_less_broadcast(self) -> None:
        frame = feetech_protocol.build_packet(BROADCAST_ID, 0x03, b"\x2a\x00\x08", allow_broadcast=True)

        assert frame[2] == BROADCAST_ID

    def test_the_module_reads_the_codecs_constants(self) -> None:
        assert serial_mod.BROADCAST_ID == feetech_protocol.BROADCAST_ID == BROADCAST_ID
        assert serial_mod.MAX_UNICAST_ID == feetech_protocol.MAX_UNICAST_ID == MAX_UNICAST_ID

    def test_the_constants_are_imported_rather_than_restated(self) -> None:
        """A second copy of the same numbers is what a drift looks like before it drifts.

        Equal values are exactly what the previous state looked like, so the
        agreement above cannot tell one owner from two. This reads the import.
        """
        module = ast.parse(inspect.getsource(serial_mod))
        imported = {
            alias.name
            for node in ast.walk(module)
            if isinstance(node, ast.ImportFrom) and node.module == feetech_protocol.__name__
            for alias in node.names
        }

        assert {"BROADCAST_ID", "MAX_UNICAST_ID"} <= imported

    def test_the_field_bound_alone_cannot_refuse_the_broadcast(self) -> None:
        _, ceiling, _ = serial_mod._REGISTER_FIELDS["motor_id"]

        assert ceiling == BROADCAST_ID

    def test_the_field_reason_names_the_address_space(self) -> None:
        _, _, why = serial_mod._REGISTER_FIELDS["motor_id"]

        assert "broadcast" in why
        assert f"{MAX_UNICAST_ID:#x}" in why


class TestEveryReplyExpectingActionIsHeldToASingleServo:
    """A new action that reads a reply for a caller's ID cannot ship unheld."""

    def test_the_module_holds_every_action_that_reads_a_reply(self) -> None:
        assert _actions_that_read_a_reply() == set(serial_mod._REPLY_EXPECTING_ACTIONS)

    def test_the_derivation_separates_reading_from_addressing(self) -> None:
        derived = _actions_that_read_a_reply()

        assert REPLY_EXPECTING_ACTION in derived
        assert {action for action, _ in REPLY_LESS_WRITES}.isdisjoint(derived)
        assert {"read", "send_read"}.isdisjoint(derived)
