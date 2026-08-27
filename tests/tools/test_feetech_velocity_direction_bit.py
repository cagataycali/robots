# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ``Goal_Velocity`` magnitude that would set the direction bit is refused.

The STS/SMS series encodes ``Goal_Velocity`` (register 0x2E) as sign-magnitude,
not as a plain unsigned 16-bit value: bit 15 carries the direction and bits 0-14
the magnitude. :mod:`~strands_robots.tools.serial_tool` bounded the field to the
two-byte maximum instead, so every magnitude from 32768 up was accepted and put
its own two bytes on the wire - where the servo read them as a command in the
opposite direction:

===============  ==================  ==========================================
``velocity=``    bytes on the wire   what the servo executes
===============  ==================  ==========================================
32767            ``2E FF 7F``        +32767, forward (the largest true magnitude)
32768            ``2E 00 80``        magnitude 0 - it *stops*
40000            ``2E 40 9C``        -7232, reverse
65535            ``2E FF FF``        -32767, full speed reverse
===============  ==================  ==========================================

Each returned ``status="success"`` quoting the number supplied, so the report
described a command the servo never received - and for 65535 the arm ran at full
speed the wrong way. That is the same silent-reinterpretation the sibling fields
are bounded to prevent, which is why the module's own docstring says bounding the
field is what makes the success message true.

``Goal_Position`` (0x2A) is sign-magnitude on bit 15 too, and needed no change:
its 12-bit ceiling of 4095 sits far below the direction bit, so a position the
tool accepts can never reach it. Its bound was already keyed to the register
rather than to the byte width, which is exactly what ``velocity``'s was not.

The vendor semantics is not this suite's invention. lerobot's Feetech tables
declare ``"Goal_Velocity": 15`` in ``STS_SMS_SERIES_ENCODINGS_TABLE`` and encode
through ``encode_sign_magnitude``, which raises for a magnitude above 32767;
:class:`TestTheVendorAgreesWhereTheMagnitudeEnds` pins that agreement whenever
lerobot is installed.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
import serial

import strands_robots.tools.serial_tool as serial_mod

#: Bit index the vendor declares for ``Goal_Velocity`` on the STS/SMS series -
#: lerobot spells it ``STS_SMS_SERIES_ENCODINGS_TABLE["Goal_Velocity"] = 15`` and
#: :class:`TestTheVendorAgreesWhereTheMagnitudeEnds` checks that claim against
#: lerobot itself. Stated here rather than read off the module under test so the
#: premises and the over-reach controls grade the same register semantics whether
#: or not the module has been fixed.
DIRECTION_BIT = 15

#: Largest magnitude leaving :data:`DIRECTION_BIT` clear.
MAX_MAGNITUDE = (1 << DIRECTION_BIT) - 1

#: Magnitudes that overflow into the direction bit, and what the servo reads
#: each of them as once bit 15 is stripped off as the sign.
REVERSING_MAGNITUDES = (
    (32768, 0),
    (40000, -7232),
    (50000, -17232),
    (65535, -32767),
)

#: Magnitudes the field carries with the direction bit still clear.
TRUE_MAGNITUDES = (0, 1, 100, 4095, 32767)


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


def _register_value(packet: bytes) -> int:
    """Decode the two-byte little-endian register value out of a written packet."""
    return packet[6] | (packet[7] << 8)


def _as_sign_magnitude(encoded: int, sign_bit: int) -> int:
    """Read ``encoded`` the way the servo does: a sign bit and a magnitude.

    Spelled out rather than imported so this file grades the register semantics
    on an install with no lerobot at all.
    """
    magnitude = encoded & ((1 << sign_bit) - 1)
    return -magnitude if (encoded >> sign_bit) & 1 else magnitude


def _velocity_ceiling_expression() -> str:
    """The source expression ``_REGISTER_FIELDS["velocity"]`` uses as its ceiling.

    A literal 32767 there would behave identically today and drift the moment the
    direction bit is reconsidered, so what is pinned is that the bound is *derived
    from* :data:`~strands_robots.tools.serial_tool._MAX_MAGNITUDE` rather than
    equal to it by coincidence.
    """
    source = Path(inspect.getfile(serial_mod)).read_text(encoding="utf-8")
    (assignment,) = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_REGISTER_FIELDS"
    ]
    fields = assignment.value
    assert isinstance(fields, ast.Dict)
    (entry,) = [
        value
        for key, value in zip(fields.keys, fields.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "velocity"
    ]
    assert isinstance(entry, ast.Tuple)
    return ast.unparse(entry.elts[1])


class TestTheRegisterSemanticsThisBoundEncodes:
    """Premises: what bit 15 does, and that it is reachable inside two bytes."""

    def test_the_direction_bit_is_inside_the_two_byte_field(self) -> None:
        # If the direction bit sat outside the field, the byte-width bound would
        # have been correct and there would be nothing to refuse.
        assert (1 << DIRECTION_BIT) <= 0xFFFF
        assert MAX_MAGNITUDE < 0xFFFF

    @pytest.mark.parametrize("magnitude,executed", REVERSING_MAGNITUDES)
    def test_a_reversing_magnitude_really_does_change_the_command(self, magnitude: int, executed: int) -> None:
        # The premise the refusal exists for: these are not truncations of the
        # caller's number, they are different commands.
        assert _as_sign_magnitude(magnitude, DIRECTION_BIT) == executed
        assert executed != magnitude

    @pytest.mark.parametrize("magnitude", TRUE_MAGNITUDES)
    def test_an_accepted_magnitude_reads_back_as_itself(self, magnitude: int) -> None:
        assert _as_sign_magnitude(magnitude, DIRECTION_BIT) == magnitude


class TestAVelocityThatWouldReverseTheCommandIsRefused:
    """The regression: the bus is never given a magnitude that means something else."""

    @pytest.mark.parametrize("magnitude,_executed", REVERSING_MAGNITUDES)
    def test_it_is_refused_before_the_port_opens(
        self, magnitude: int, _executed: int, opened: list[_FakeSerial]
    ) -> None:
        result = _call(action="feetech_velocity", motor_id=1, velocity=magnitude)

        assert result["status"] == "error"
        assert "velocity" in _text(result)
        assert "feetech_velocity" in _text(result)
        assert opened == []

    def test_the_refusal_explains_the_direction_bit_rather_than_the_byte_width(self) -> None:
        result = _call(action="feetech_velocity", motor_id=1, velocity=65535)
        text = _text(result)

        # A caller told only "must be at most 32767" would reasonably read it as
        # an arbitrary cap; the reason is what makes the number make sense.
        assert "32767" in text
        assert "sign-magnitude" in text
        assert "direction" in text
        assert "65535" in text

    def test_the_ceiling_is_the_largest_magnitude_the_direction_bit_leaves_free(self) -> None:
        _low, high, _why = serial_mod._REGISTER_FIELDS["velocity"]

        assert high == MAX_MAGNITUDE == 32767
        # One above the ceiling is exactly the direction bit, alone.
        assert high + 1 == 1 << DIRECTION_BIT
        # And the module derives that ceiling from the bit rather than restating
        # a number, so the two cannot drift apart.
        assert serial_mod._DIRECTION_BIT == DIRECTION_BIT
        assert serial_mod._MAX_MAGNITUDE == MAX_MAGNITUDE
        assert _velocity_ceiling_expression() == "_MAX_MAGNITUDE"


class TestEveryMagnitudeTheToolAcceptsStillReachesTheWire:
    """The over-reach guard: bounding the field must not narrow a true command."""

    @pytest.mark.parametrize("magnitude", TRUE_MAGNITUDES)
    def test_an_accepted_velocity_is_the_velocity_on_the_wire(self, magnitude: int, opened: list[_FakeSerial]) -> None:
        result = _call(action="feetech_velocity", motor_id=1, velocity=magnitude)

        assert result["status"] == "success"
        assert _register_value(opened[0].writes[0]) == magnitude
        # And it means what it says, which is the whole point of the bound.
        assert _as_sign_magnitude(_register_value(opened[0].writes[0]), DIRECTION_BIT) == magnitude

    def test_the_register_address_is_unchanged(self, opened: list[_FakeSerial]) -> None:
        _call(action="feetech_velocity", motor_id=1, velocity=100)

        # Goal_Velocity, address 46 - the bound must not move the write.
        assert opened[0].writes[0][5] == 0x2E


class TestTheSiblingPositionRegisterNeededNoChange:
    """``Goal_Position`` shares the encoding and was already bounded below it."""

    def test_its_ceiling_already_left_the_direction_bit_clear(self) -> None:
        _low, high, _why = serial_mod._REGISTER_FIELDS["position"]

        assert high == 4095
        assert high <= MAX_MAGNITUDE
        assert not (high >> DIRECTION_BIT) & 1

    @pytest.mark.parametrize("position", (4096, 32768, 40000, 65535))
    def test_a_position_that_could_reach_the_direction_bit_was_always_refused(
        self, position: int, opened: list[_FakeSerial]
    ) -> None:
        result = _call(action="feetech_position", motor_id=1, position=position)

        assert result["status"] == "error"
        assert "position" in _text(result)
        assert opened == []


class TestTheVendorAgreesWhereTheMagnitudeEnds:
    """lerobot encodes the same register, and refuses the same magnitudes."""

    @pytest.mark.parametrize("magnitude,_executed", REVERSING_MAGNITUDES)
    def test_lerobot_refuses_every_magnitude_this_tool_refuses(self, magnitude: int, _executed: int) -> None:
        encoding = pytest.importorskip("lerobot.motors.encoding_utils")

        with pytest.raises(ValueError):
            encoding.encode_sign_magnitude(magnitude, DIRECTION_BIT)

    @pytest.mark.parametrize("magnitude", TRUE_MAGNITUDES)
    def test_lerobot_accepts_every_magnitude_this_tool_accepts(self, magnitude: int) -> None:
        encoding = pytest.importorskip("lerobot.motors.encoding_utils")

        assert encoding.encode_sign_magnitude(magnitude, DIRECTION_BIT) == magnitude

    def test_the_sign_bit_index_is_the_one_the_vendor_table_declares(self) -> None:
        tables = pytest.importorskip("lerobot.motors.feetech.tables")

        assert tables.STS_SMS_SERIES_ENCODINGS_TABLE["Goal_Velocity"] == DIRECTION_BIT
        assert tables.STS_SMS_SERIES_ENCODINGS_TABLE["Goal_Position"] == DIRECTION_BIT
        # And that the address this tool writes is the one that table describes.
        assert tables.STS_SMS_SERIES_CONTROL_TABLE["Goal_Velocity"] == (0x2E, 2)
