# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``FeetechBus.sync_read`` asks the whole arm once, in the frame it is named for.

``SYNC_READ`` (0x82) is the one Feetech instruction where a broadcast expects
replies: one frame names the register and the IDs, and every addressed servo
answers with a status packet, back to back. Reading a six-servo arm with six
unicast READs instead costs six round trips and six reply settles, and - because
each reply is 8 bytes while the read asked for 10 - each of those round trips
waits out the port's entire read window for two bytes no servo sends.

Three things are graded here, in the order a caller meets them:

1. the frame, against ``scservo_sdk.GroupSyncRead`` driven for real (the vendor
   SDK builds the bytes, this codec must match them) and against
   datasheet-shaped literals so the pin holds on a box without the SDK;
2. the arm being asked *once* - one frame, one settle, no window waited out;
3. what a servo that does not answer costs, which is itself and not the arm:
   it is absent from the reading, its neighbours are present, and a retry
   re-asks only the servos still missing.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

import strands_robots.drivers.feetech.bus as bus_module
from strands_robots.drivers.feetech.bus import MotorSpec
from strands_robots.drivers.feetech.protocol import (
    Instruction,
    sync_read_packet,
    sync_read_reply_size,
)
from tests.drivers.conftest import MIDPOINT_COUNTS, FakeServoPort, open_bus

_scservo_sdk_available = importlib.util.find_spec("scservo_sdk") is not None

#: The SO-arm's six IDs, and three narrower reads: one servo, a wider register,
#: and a register other than position. ``(motor_ids, address, width, frame)``.
#:
#: Each ``frame`` is the byte string ``scservo_sdk``'s ``GroupSyncRead.txPacket``
#: puts on the wire for that row, captured from the SDK and written out here so
#: the codec is graded against the vendor's bytes on any box - the same division
#: :mod:`tests.drivers.test_feetech_protocol` draws between its datasheet cells
#: and its SDK-gated ones.
SYNC_READ_FRAMES = [
    pytest.param(
        [1, 2, 3, 4, 5, 6],
        0x38,
        2,
        bytes.fromhex("ffff fe 0a 82 38 02 010203040506 26"),
        48,
        id="six-servo-arm-present-position",
    ),
    pytest.param([1], 0x38, 2, bytes.fromhex("ffff fe 05 82 38 02 01 3f"), 8, id="one-servo"),
    pytest.param([3, 9], 0x2A, 4, bytes.fromhex("ffff fe 06 82 2a 04 0309 3f"), 20, id="four-byte-register"),
    pytest.param([6], 0x3C, 2, bytes.fromhex("ffff fe 05 82 3c 02 06 36"), 8, id="present-load"),
]


class TestTheFrameIsTheVendorsFrame:
    """The bytes ``sync_read_packet`` builds are the bytes the SDK builds."""

    @pytest.mark.parametrize("motor_ids,address,width,frame,reply_bytes", SYNC_READ_FRAMES)
    def test_the_frame_matches_the_vendor_bytes(
        self, motor_ids: list[int], address: int, width: int, frame: bytes, reply_bytes: int
    ) -> None:
        assert sync_read_packet(address, width, motor_ids) == frame

    @pytest.mark.parametrize("motor_ids,address,width,frame,reply_bytes", SYNC_READ_FRAMES)
    def test_the_frame_is_a_broadcast_carrying_the_sync_read_instruction(
        self, motor_ids: list[int], address: int, width: int, frame: bytes, reply_bytes: int
    ) -> None:
        """A broadcast, so every servo reads it; 0x82, so every servo answers."""
        assert frame[2] == 0xFE
        assert frame[4] == Instruction.SYNC_READ == 0x82
        assert list(frame[7:-1]) == motor_ids

    @pytest.mark.parametrize("motor_ids,address,width,frame,reply_bytes", SYNC_READ_FRAMES)
    def test_the_reply_size_is_one_status_frame_per_servo(
        self, motor_ids: list[int], address: int, width: int, frame: bytes, reply_bytes: int
    ) -> None:
        """The byte budget the SDK computes for the same read, captured from it."""
        assert sync_read_reply_size(len(motor_ids), width) == reply_bytes

    @pytest.mark.skipif(not _scservo_sdk_available, reason="scservo_sdk not installed on this box")
    @pytest.mark.parametrize("motor_ids,address,width,frame,reply_bytes", SYNC_READ_FRAMES)
    def test_the_vendor_sdk_builds_the_same_frame(
        self, motor_ids: list[int], address: int, width: int, frame: bytes, reply_bytes: int
    ) -> None:
        """Drive ``GroupSyncRead`` itself, so the SDK's builder is the grader.

        The SDK writes through a ``PortHandler``; the capture below is the two
        methods ``txPacket`` reaches for, so no port is opened. Its computed
        packet timeout is the reply budget, which grades
        :func:`~strands_robots.drivers.feetech.protocol.sync_read_reply_size`
        against the vendor's own arithmetic rather than against a restatement.
        """
        sdk = pytest.importorskip("scservo_sdk", reason="scservo_sdk not installed on this box")

        class CapturePort:
            def __init__(self) -> None:
                self.frames: list[bytes] = []
                self.reply_budget: int | None = None
                self.is_using = False

            def clearPort(self) -> None:  # noqa: N802 - the SDK's spelling
                pass

            def writePort(self, packet: Any) -> int:  # noqa: N802 - the SDK's spelling
                self.frames.append(bytes(packet))
                return len(packet)

            def setPacketTimeout(self, budget: int) -> None:  # noqa: N802 - the SDK's spelling
                self.reply_budget = budget

        port = CapturePort()
        reader = sdk.GroupSyncRead(port, sdk.PacketHandler(0), address, width)
        for motor_id in motor_ids:
            reader.addParam(motor_id)
        assert reader.txPacket() == sdk.COMM_SUCCESS

        assert port.frames == [frame]
        assert sync_read_packet(address, width, motor_ids) == port.frames[0]
        assert sync_read_reply_size(len(motor_ids), width) == port.reply_budget

    @pytest.mark.parametrize(
        "motor_ids,match",
        [
            ([], "no motors"),
            ([1, 1], "twice"),
            ([1, 0xFE], "broadcast"),
        ],
    )
    def test_a_frame_no_servo_could_answer_is_refused(self, motor_ids: list[int], match: str) -> None:
        """One servo named twice sends two replies nothing can tell apart."""
        with pytest.raises(ValueError, match=match):
            sync_read_packet(0x38, 2, motor_ids)


class TestTheArmIsAskedOnce:
    """One frame, one settle, and no read window waited out."""

    @pytest.fixture
    def sleeps(self, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        """Every settle the read pays, so the count is graded and not the clock."""
        recorded: list[float] = []
        monkeypatch.setattr(bus_module.time, "sleep", recorded.append)
        return recorded

    def test_six_joints_come_back_from_one_frame(self, servo_port: FakeServoPort, sleeps: list[float]) -> None:
        bus = open_bus(servo_port)

        reading = bus.sync_read("Present_Position")

        assert sorted(reading) == ["elbow_flex", "gripper", "shoulder_lift", "shoulder_pan", "wrist_flex", "wrist_roll"]
        assert len(servo_port.writes) == 1, "one frame asks the whole arm"
        assert servo_port.writes[0][4] == Instruction.SYNC_READ
        assert len(sleeps) == 1, "one settle for the arm, not one per servo"

    def test_no_read_window_is_waited_out_on_a_healthy_arm(self, servo_port: FakeServoPort) -> None:
        """The read asks for the bytes the arm sends, so nothing is waited for.

        A read that asks for more than the reply carries returns only when the
        window expires, so a healthy arm reports at the timeout rather than at
        the wire - and the timeout is a second by default.
        """
        bus = open_bus(servo_port, timeout=servo_port.timeout)

        assert len(bus.sync_read("Present_Position")) == 6
        assert servo_port.blocked_s == 0.0

    def test_every_joint_reads_the_value_its_servo_reported(self, sleeps: list[float]) -> None:
        """Position is per servo, so a stream read in the wrong order is caught."""
        port = FakeServoPort({1: 0, 2: 1024, 3: 2048, 4: 3072, 5: 4095, 6: 2048})
        bus = open_bus(port)

        reading = bus.sync_read("Present_Position")

        assert reading["shoulder_pan"] == pytest.approx(-180.0, abs=0.1)  # id 1, count 0
        assert reading["shoulder_lift"] == pytest.approx(-45.0, abs=0.1)  # id 2, count 1024
        assert reading["elbow_flex"] == pytest.approx(0.0, abs=0.1)  # id 3, count 2048
        assert reading["wrist_flex"] == pytest.approx(45.0, abs=0.1)  # id 4, count 3072
        assert reading["wrist_roll"] == pytest.approx(180.0, abs=0.1)  # id 5, count 4095
        assert reading["gripper"] == pytest.approx(50.0, abs=0.1)  # id 6, count 2048

    def test_an_echoed_frame_does_not_cost_the_last_servo(self, sleeps: list[float]) -> None:
        """A half-duplex bus can echo the host's own frame in front of the replies.

        The echo pushes the last servo's frame past the byte count the read asked
        for, so the buffered remainder is drained rather than dropped - and the
        echo itself is addressed to the broadcast ID, which no servo holds, so it
        is skipped rather than read as a position.
        """
        echo = sync_read_packet(0x38, 2, [1, 2, 3, 4, 5, 6])
        port = FakeServoPort(MIDPOINT_COUNTS, leading_noise=echo)
        bus = open_bus(port)

        assert len(bus.sync_read("Present_Position")) == 6
        assert len(port.writes) == 1


class TestAServoThatDoesNotAnswerCostsOnlyItself:
    """A gap in the stream is one absent joint, not a failed read."""

    def test_a_mute_servo_is_absent_and_its_neighbours_are_not(self) -> None:
        port = FakeServoPort({motor_id: 2048 for motor_id in (1, 2, 3, 5, 6)})  # id 4 is mute
        bus = open_bus(port)

        reading = bus.sync_read("Present_Position")

        assert "wrist_flex" not in reading
        assert sorted(reading) == ["elbow_flex", "gripper", "shoulder_lift", "shoulder_pan", "wrist_roll"]

    def test_a_reply_that_fails_its_checksum_is_not_a_position(self) -> None:
        port = FakeServoPort(MIDPOINT_COUNTS)
        bus = open_bus(port)
        original = port.read

        def corrupt_the_third_frame(size: int) -> bytes:
            raw = bytearray(original(size))
            if len(raw) >= 24:
                raw[16 + 7] ^= 0xFF  # id 3's checksum byte
            return bytes(raw)

        port.read = corrupt_the_third_frame  # type: ignore[method-assign]
        reading = bus.sync_read("Present_Position")

        assert "elbow_flex" not in reading
        assert len(reading) == 5

    def test_a_retry_re_asks_only_the_servos_still_missing(self) -> None:
        """The arm that already answered is not asked again, so a retry is cheap."""
        port = FakeServoPort({motor_id: 2048 for motor_id in (1, 2, 3, 5, 6)})  # id 4 is mute
        bus = open_bus(port)

        bus.sync_read("Present_Position", num_retry=2)

        assert len(port.writes) == 3, "the first frame plus the two retries it was given"
        assert list(port.writes[0][7:-1]) == [1, 2, 3, 4, 5, 6]
        assert list(port.writes[1][7:-1]) == [4]
        assert list(port.writes[2][7:-1]) == [4]

    def test_a_servo_that_answers_on_the_retry_is_read(self) -> None:
        port = FakeServoPort({motor_id: 2048 for motor_id in (1, 2, 3, 5, 6)})
        bus = open_bus(port)
        original = port.write

        def wake_id_four(data: bytes) -> int:
            port.counts[4] = 4095
            return original(data)

        port.write = wake_id_four  # type: ignore[method-assign]
        reading = bus.sync_read("Present_Position", num_retry=1)

        assert reading["wrist_flex"] == pytest.approx(90.0, abs=0.1)
        assert len(reading) == 6

    def test_an_arm_that_says_nothing_reads_as_no_joints(self) -> None:
        """No frames back is an empty reading, not a guess and not an exception."""
        port = FakeServoPort({})
        bus = open_bus(port)

        assert bus.sync_read("Present_Position") == {}


class TestABusNamingOneServoTwiceAsksForItOnce:
    """Two names for one ID is one servo, and the frame it answers must be legal."""

    def test_both_names_report_the_one_servo_the_frame_asked_for(self) -> None:
        port = FakeServoPort({1: 2048})
        bus = open_bus(port, motors={"pan": MotorSpec(1, -180, 180), "pan_alias": MotorSpec(1, -180, 180)})

        reading = bus.sync_read("Present_Position")

        assert reading == {"pan": pytest.approx(0.0, abs=0.1), "pan_alias": pytest.approx(0.0, abs=0.1)}
        assert list(port.writes[0][7:-1]) == [1], "one ID, listed once - a servo cannot answer twice"
