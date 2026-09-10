# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Present_Position`` carries a sign, and the bus reads it as one.

The STS/SMS series encodes ``Present_Position`` (0x38) as **sign-magnitude**:
bit 15 is the direction and bits 0-14 the magnitude. A servo carrying a homing
offset reports a joint just past its zero with that bit set, which is a routine
reading on a calibrated arm rather than an edge case.

``READABLE_REGISTERS`` spelled the sign per entry and gave ``Present_Position``
none, while giving its two neighbours theirs. The bit was then read as part of
the magnitude, and the degrees that came out were off by the whole top half of
the field - reported as a measurement, with no error:

===============  ==========  ===================  ==================
joint            wire word   reported before      reported after
===============  ==========  ===================  ==================
shoulder_pan     ``0x0800``  0.04 deg             0.04 deg
shoulder_lift    ``0x8064``  **1354.75 deg**      -94.40 deg
elbow_flex       ``0x8320``  **2309.19 deg**      -208.61 deg
wrist_flex       ``0x0064``  -85.60 deg           -85.60 deg
wrist_roll       ``0x8001``  **2700.79 deg**      -180.09 deg
gripper          ``0x0800``  50.01 percent        50.01 percent
===============  ==========  ===================  ==================

``shoulder_lift`` spans -90..90 degrees. 1354.75 is not a value that joint can
hold, and nothing on the read path said so.

The sign is not this suite's invention, and not the bus's to declare: lerobot's
``STS_SMS_SERIES_ENCODINGS_TABLE`` - the table an SO-arm is calibrated and read
by - gives ``Present_Position`` bit 15 exactly as it gives ``Present_Velocity``
bit 15 and ``Present_Load`` bit 10.
:data:`~strands_robots.drivers.feetech.protocol.SIGN_BIT` is now that table, and
:class:`TestTheSignIsDeclaredOnceForThePackage` pins that no reader keeps a
second copy of it.

The two neighbours are the control: both were already signed, both are unchanged.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest

import strands_robots.tools.serial_tool as serial_mod
from strands_robots.drivers.feetech.bus import (
    READABLE_REGISTERS,
    SO_ARM_MOTORS,
    FeetechBus,
)
from strands_robots.drivers.feetech.protocol import (
    SIGN_BIT,
    WORD_LENGTH,
    Register,
    decode_sign_magnitude,
    max_magnitude,
    read_packet,
)
from tests.drivers.conftest import FakeServoPort

#: The vendor SDK is an incidental dependency here - it arrives transitively
#: through ``lerobot[feetech]``. Scoped to the one class that needs it rather
#: than a module-level skip, which would silently deselect every cell above.
_scservo_sdk_available = importlib.util.find_spec("scservo_sdk") is not None

#: Wire word -> the counts the servo means by it. Bit 15 clear is a plain
#: magnitude; bit 15 set is that magnitude in the negative direction. The
#: magnitudes are bounded by the 12-bit position field, so these are words a
#: real STS3215 emits rather than the whole two-byte space.
SIGNED_REGISTERS: tuple[Register, ...] = tuple(sorted(SIGN_BIT, key=lambda register: register.name))

SIGNED_WORDS: tuple[tuple[int, int], ...] = (
    (0x0000, 0),
    (0x0064, 100),
    (0x0800, 2048),
    (0x0FFF, 4095),
    (0x8001, -1),
    (0x8064, -100),
    (0x8320, -800),
    (0x8FFF, -4095),
)


@pytest.fixture(scope="module")
def sdk():  # type: ignore[no-untyped-def]
    """Feetech's vendor SDK, or a skip where it is not installed.

    Imported inside the fixture so this module still imports on a box without
    it - the class ``skipif`` guards collection, and the 35 cells above grade the
    codec with no SDK at all.
    """
    return pytest.importorskip("scservo_sdk", reason="scservo_sdk not installed on this box")


def _assigned_expression(module: object, name: str) -> str:
    """The source expression ``name`` is assigned in ``module``.

    Read off the shipped file rather than the imported value, because the value
    is what cannot tell a lookup from a coincidence.
    """
    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")  # type: ignore[arg-type]
    (assignment,) = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    return ast.unparse(assignment.value)


def _open_bus(port: FakeServoPort) -> FeetechBus:
    """A bus already holding ``port``, skipping the real ``connect()``."""
    bus = FeetechBus(port="/dev/fake")
    bus._conn = port
    return bus


def _read_one(register: str, word: int, motor: str = "shoulder_lift") -> float:
    """What the bus reports for ``register`` when ``motor``'s servo replies ``word``."""
    motor_id = SO_ARM_MOTORS[motor].motor_id
    bus = _open_bus(FakeServoPort({motor_id: word}))
    return bus.sync_read(register)[motor]


class TestTheVendorDeclaresTheSign:
    """The premise: bit 15 is Feetech's, not ours, and lerobot reads it too."""

    def test_present_position_is_in_the_table_at_all(self) -> None:
        # The defect in one line: the register was absent, and absent means
        # unsigned.
        assert SIGN_BIT[Register.PRESENT_POSITION] == 15

    @pytest.mark.parametrize("register", SIGNED_REGISTERS)
    def test_every_entry_is_the_bit_lerobot_declares(self, register: Register) -> None:
        tables = pytest.importorskip("lerobot.motors.feetech.tables")
        vendor = tables.STS_SMS_SERIES_ENCODINGS_TABLE

        # ``PRESENT_POSITION`` -> ``Present_Position``: the two spellings of one
        # register, checked rather than assumed.
        name = register.name.title()
        assert vendor[name] == SIGN_BIT[register], f"{name}: lerobot says {vendor[name]}"

    @pytest.mark.parametrize(("word", "counts"), SIGNED_WORDS)
    def test_lerobot_decodes_the_same_word_to_the_same_counts(self, word: int, counts: int) -> None:
        encoding = pytest.importorskip("lerobot.motors.encoding_utils")

        assert encoding.decode_sign_magnitude(word, 15) == counts
        assert decode_sign_magnitude(word, 15) == counts


class TestAJointPastItsZeroIsReportedWhereItIs:
    """The regression: a set direction bit is a direction, not more magnitude."""

    @pytest.mark.parametrize(("word", "counts"), SIGNED_WORDS)
    def test_the_reported_counts_are_the_ones_the_servo_sent(self, word: int, counts: int) -> None:
        expected = FeetechBus(port="/dev/fake").to_value("shoulder_lift", counts)

        assert _read_one("Present_Position", word) == pytest.approx(expected)

    def test_a_negative_reading_is_below_the_joints_own_range(self) -> None:
        """The consequence in the caller's unit, stated without a magic float.

        Before, the same word read as a value *above* the top of the servo's
        rotation by more than a whole turn - a position the joint cannot hold,
        reported as one it does.
        """
        bus = FeetechBus(port="/dev/fake")

        reported = _read_one("Present_Position", 0x8064)

        assert reported < bus.to_value("shoulder_lift", 0)
        assert reported == pytest.approx(-188.79, abs=0.01)

    def test_an_unsigned_reading_is_untouched(self) -> None:
        """The other half of the field must not move: bit 15 clear reads as before."""
        assert _read_one("Present_Position", 0x0800) == pytest.approx(0.0, abs=0.1)


class TestEveryReadableRegisterTakesItsSignFromTheVendor:
    """The anti-recurrence pin: a register added here cannot pick unsigned by silence."""

    @pytest.mark.parametrize("register", sorted(READABLE_REGISTERS))
    def test_a_top_bit_reply_is_negative_exactly_when_lerobot_says_it_is(self, register: str) -> None:
        tables = pytest.importorskip("lerobot.motors.feetech.tables")
        vendor_bit = tables.STS_SMS_SERIES_ENCODINGS_TABLE.get(register)
        # A word with the vendor's direction bit set over a magnitude of 50.
        word = 50 if vendor_bit is None else (1 << vendor_bit) | 50

        reported = _read_one(register, word)

        assert (reported < 0) is (vendor_bit is not None), (
            f"{register}: lerobot bit {vendor_bit}, this bus reported {reported}"
        )


class TestTheSignIsDeclaredOnceForThePackage:
    """One table. A second copy is how the two neighbours came to disagree."""

    def test_the_readable_table_carries_registers_and_no_sign_of_its_own(self) -> None:
        assert all(isinstance(value, Register) for value in READABLE_REGISTERS.values())

    def test_the_serial_tools_write_ceiling_reads_the_same_table(self) -> None:
        # The tool bounds ``Goal_Velocity`` by the bit; the bus decodes
        # ``Present_Velocity`` by the bit. One table, so they cannot drift.
        assert serial_mod._DIRECTION_BIT == SIGN_BIT[Register.GOAL_VELOCITY]
        assert serial_mod._MAX_MAGNITUDE == max_magnitude(SIGN_BIT[Register.GOAL_VELOCITY])

    def test_the_tool_reads_that_bit_rather_than_restating_it(self) -> None:
        """A literal 15 there behaves identically today and drifts tomorrow.

        Equal-by-coincidence is what the two neighbouring registers already were,
        so what is pinned is that the value is *looked up* - the same shape
        ``test_feetech_velocity_direction_bit`` pins for the ceiling it feeds.
        """
        assert _assigned_expression(serial_mod, "_DIRECTION_BIT") == "SIGN_BIT[Register.GOAL_VELOCITY]"


class TestAMaskThatIsNotTheMagnitudeIsRefused:
    """A sign bit outside the word returns the unsigned value - the very defect.

    The off-type refusals are graded by the package-wide refusal table in
    ``test_feetech_refusals_are_documented_and_catchable``; what is pinned here
    is the range, which is the half that carries the defect's own shape.
    """

    @pytest.mark.parametrize("sign_bit", (0, 8 * WORD_LENGTH, 32))
    def test_a_bit_that_leaves_no_magnitude_is_refused(self, sign_bit: int) -> None:
        with pytest.raises(ValueError, match="sign_bit"):
            max_magnitude(sign_bit)
        with pytest.raises(ValueError, match="sign_bit"):
            decode_sign_magnitude(0x8064, sign_bit)

    @pytest.mark.parametrize("value", (-1, 0x10000))
    def test_a_value_that_is_not_a_word_is_refused(self, value: int) -> None:
        with pytest.raises(ValueError, match="value"):
            decode_sign_magnitude(value, 15)


@pytest.mark.skipif(not _scservo_sdk_available, reason="scservo_sdk (feetech-servo-sdk) not installed")
class TestTheVendorSdkAgreesOnTheBytes:
    """Graded against Feetech's own SDK, not against this package's idea of it.

    The sign lives *above* the SDK - it hands back the unsigned word, exactly as
    it does for lerobot, which applies the same table on top. So the two facts
    worth grading here are that the request frame is byte-identical and that the
    reply frame the read tests are built on is one the SDK itself accepts and
    decodes to the same word.
    """

    def test_the_request_frame_is_byte_identical(self, sdk) -> None:  # type: ignore[no-untyped-def]
        port = _SdkPort()
        sdk.PacketHandler(0).readTxRx(port, 1, Register.PRESENT_POSITION, WORD_LENGTH)

        assert read_packet(1, Register.PRESENT_POSITION, WORD_LENGTH) == port.written[0]

    @pytest.mark.parametrize(("word", "counts"), SIGNED_WORDS)
    def test_the_sdk_reads_the_same_reply_frame_as_the_same_unsigned_word(  # type: ignore[no-untyped-def]
        self, sdk, word: int, counts: int
    ) -> None:
        frame = FakeServoPort._status_frame(1, word)
        value, comm, _error = sdk.PacketHandler(0).read2ByteTxRx(_SdkPort(frame), 1, Register.PRESENT_POSITION)

        assert comm == sdk.COMM_SUCCESS
        assert value == word
        # And the sign the SDK leaves to its caller is the one applied here.
        assert decode_sign_magnitude(value, SIGN_BIT[Register.PRESENT_POSITION]) == counts


class _SdkPort:
    """The ``PortHandler`` surface ``scservo_sdk`` drives, over a byte buffer.

    Enough of it to capture what the SDK would write and to feed it a reply,
    with no serial device: the SDK's own framing is what is being read off.
    """

    is_using = False

    def __init__(self, reply: bytes = b"") -> None:
        self.written: list[bytes] = []
        self._buffer = bytearray(reply)

    def clearPort(self) -> None:  # noqa: N802 - the SDK's own spelling
        pass

    def writePort(self, data: list[int]) -> int:  # noqa: N802
        self.written.append(bytes(bytearray(data)))
        return len(data)

    def readPort(self, length: int) -> list[int]:  # noqa: N802
        chunk, self._buffer = self._buffer[:length], self._buffer[length:]
        return list(chunk)

    def setPacketTimeout(self, length: int) -> None:  # noqa: N802
        pass

    def isPacketTimeout(self) -> bool:  # noqa: N802
        return not self._buffer
