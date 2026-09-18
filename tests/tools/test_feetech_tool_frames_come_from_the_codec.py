# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The servo tools put the codec's frames on the wire, and read its sign.

``strands_robots.drivers.feetech.protocol`` owns this wire format. The two tools
that drive a Feetech bus each carried a ``build_feetech_packet`` of their own -
six lines assembling ``FF FF ID LEN INST <params> CHK`` and spelling the
register address as a hex literal at each call site - so the package held three
copies of the format and three chances to disagree with it.

They did disagree, in the half that is not the frame. ``Present_Position``
(0x38) is sign-magnitude on the STS/SMS series: bit 15 is the direction. The
driver's bus reads it that way; the pose tool decoded the whole field as a
magnitude, so a servo reporting a joint just past its homing zero was quoted as
a position more than a full turn away:

===========  ==========  ==================  =================
wire word    counts      reported before     reported after
===========  ==========  ==================  =================
``0x0800``   2048        0.04 deg            0.04 deg
``0x8001``   -1          **2700.79 deg**     -180.09 deg
``0x8064``   -100        **2709.49 deg**     -188.79 deg
``0x8320``   -800        **2771.03 deg**     -250.33 deg
===========  ==========  ==================  =================

``shoulder_pan`` spans -180..180 degrees, so every emboldened number is one the
joint cannot hold, reported as a measurement. The bit is not this suite's
invention: ``tests/drivers/test_feetech_position_carries_a_sign.py`` grades
:data:`~strands_robots.drivers.feetech.protocol.SIGN_BIT` against lerobot's
``STS_SMS_SERIES_ENCODINGS_TABLE`` and the vendor SDK, and one cell here pins
the entry this tool now reads.

The frames are the control: byte for byte the same as before, which is what
makes the sign the only behaviour that moved.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
import serial

import strands_robots.tools.pose_tool as pose_mod
import strands_robots.tools.serial_tool as serial_mod
from strands_robots.drivers.feetech.protocol import (
    SIGN_BIT,
    WORD_LENGTH,
    Register,
    build_packet,
    encode_word,
    read_packet,
    write_packet,
)
from strands_robots.tools.pose_tool import MotorController

#: The joint every cell drives, and its servo ID.
MOTOR = "shoulder_pan"
MOTOR_ID = 1

#: A target far from the middle of the travel, so a frame carrying the wrong
#: count cannot pass by landing near mid-scale.
MOVE_DEGREES = 90.0

#: Wire word -> the counts a servo means by it, bit 15 being the direction.
#: Magnitudes stay inside the 12-bit position field, so these are words a real
#: STS3215 emits.
SIGNED_WORDS: tuple[tuple[int, int], ...] = (
    (0x0800, 2048),
    (0x0FFF, 4095),
    (0x8001, -1),
    (0x8064, -100),
    (0x8320, -800),
)


def _status_frame(motor_id: int, word: int) -> bytes:
    """One status packet carrying ``word``, framed by the codec itself.

    A reply has the layout of a request with the error byte where the
    instruction goes, so the builder that frames what the host sends frames what
    the servo answers with - and no second framing lives in this file.
    """
    return build_packet(motor_id, 0x00, encode_word(word))


@pytest.fixture
def controller(monkeypatch: pytest.MonkeyPatch) -> tuple[MotorController, Any]:
    """A connected :class:`MotorController` over a recording fake port."""

    class Fake:
        def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
            self.is_open = True
            self.writes: list[bytes] = []
            self.replies: list[bytes] = []

        def write(self, data: bytes) -> None:
            self.writes.append(bytes(data))

        def read(self, size: int = 1) -> bytes:
            return self.replies.pop(0) if self.replies else b""

        def close(self) -> None:
            self.is_open = False

    made: list[Fake] = []

    def ctor(port: str, baudrate: int, timeout: float = 1.0) -> Fake:
        made.append(Fake(port, baudrate, timeout))
        return made[-1]

    monkeypatch.setattr(serial, "Serial", ctor)
    built = MotorController("/dev/ttyTEST")
    assert built.connect()[0] is True
    return built, made[0]


def _drive_read(built: MotorController) -> None:
    built.read_motor_position(MOTOR)


def _drive_move(built: MotorController) -> None:
    built.move_motor(MOTOR, MOVE_DEGREES)


def _drive_release(built: MotorController) -> None:
    built.disable_torque()


def _read_frame(built: MotorController) -> bytes:
    return read_packet(MOTOR_ID, Register.PRESENT_POSITION, WORD_LENGTH)


def _move_frame(built: MotorController) -> bytes:
    return write_packet(MOTOR_ID, Register.GOAL_POSITION, encode_word(built.degrees_to_position(MOTOR, MOVE_DEGREES)))


def _release_frame(built: MotorController) -> bytes:
    return write_packet(MOTOR_ID, Register.TORQUE_ENABLE, b"\x00")


class TestEveryFrameTheToolWritesIsTheCodecs:
    """Each verb's frame, compared against the codec's for the same register."""

    @pytest.mark.parametrize(
        ("drive", "expected"),
        [(_drive_read, _read_frame), (_drive_move, _move_frame), (_drive_release, _release_frame)],
        ids=["present_position_read", "goal_position_write", "torque_release"],
    )
    def test_the_first_frame_on_the_wire_is_byte_identical(
        self,
        controller: tuple[MotorController, Any],
        drive: Any,
        expected: Any,
    ) -> None:
        built, port = controller

        drive(built)

        assert port.writes[0] == expected(built)


class TestThePositionReadTakesItsSignFromTheCodec:
    """The regression: a set direction bit is a direction, not more magnitude."""

    def test_the_bit_is_the_one_the_vendor_declares(self) -> None:
        tables = pytest.importorskip("lerobot.motors.feetech.tables")

        assert SIGN_BIT[Register.PRESENT_POSITION] == tables.STS_SMS_SERIES_ENCODINGS_TABLE["Present_Position"]

    @pytest.mark.parametrize(("word", "counts"), SIGNED_WORDS)
    def test_the_angle_reported_is_the_one_those_counts_mean(
        self,
        controller: tuple[MotorController, Any],
        word: int,
        counts: int,
    ) -> None:
        built, port = controller
        port.replies.append(_status_frame(MOTOR_ID, word))

        reported = built.read_motor_position(MOTOR)

        assert reported == pytest.approx(built.position_to_degrees(MOTOR, counts))

    def test_a_joint_past_its_zero_reads_below_its_own_range(self, controller: tuple[MotorController, Any]) -> None:
        """The consequence in the caller's unit, with no magic float.

        Before, the same word read as a position above the top of the joint's
        travel by more than a whole turn - which is why the number was quoted
        without an error: nothing on this path bounds a reading.
        """
        built, port = controller
        port.replies.append(_status_frame(MOTOR_ID, 0x8064))
        floor, _ceiling = built.motor_configs[MOTOR]["range"]

        reported = built.read_motor_position(MOTOR)

        assert reported is not None
        assert reported < floor


class TestNeitherToolAssemblesAFrameByHand:
    """The anti-recurrence pin: the header belongs to the codec.

    A hand-rolled builder is how the register address came to be a hex literal
    at each call site, which is how the sign came to be nobody's to look up.
    """

    @pytest.mark.parametrize("module", [pose_mod, serial_mod], ids=["pose_tool", "serial_tool"])
    def test_no_sequence_literal_starts_with_the_frame_header(self, module: Any) -> None:
        source = Path(inspect.getfile(module)).read_text(encoding="utf-8")

        hand_rolled = [
            ast.unparse(node)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.List, ast.Tuple))
            and len(node.elts) >= 2
            and all(isinstance(element, ast.Constant) and element.value == 0xFF for element in node.elts[:2])
        ]

        assert hand_rolled == []
