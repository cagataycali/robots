"""A Feetech servo's reply is located and verified, not read at fixed offsets.

The servo bus is half-duplex and every motor on the arm shares it, so the bytes
that come back from a read may carry a leading byte the host's own transmission
echoed, or the late answer to a read that already timed out, sent by a different
motor. Indexing straight into that buffer produces a joint angle that is wrong
rather than absent, and the tool quotes it as a measurement.

The frames these tests feed are graded against ``scservo_sdk`` - the vendor SDK
that owns this wire format - in :class:`TestTheFramesHereAreWhatAServoSends`, so
the well-formed cases are known-good rather than assumed-good and the malformed
ones are known-bad. Everything else needs no dependency at all: the framing is
pure byte handling, so the regression cells run on an install with no servo SDK
and no arm attached.
"""

from __future__ import annotations

from typing import Any

import pytest
import serial

import strands_robots.tools.pose_tool as pose_mod
from strands_robots.tools.pose_tool import MotorController, _parse_status_packet, pose_tool

#: The ID of ``shoulder_pan``, and the position its reply carries throughout.
#: 1024 is a quarter turn from the 4095-count full scale, which puts the joint at
#: -90 degrees on its -180..180 range - far enough from both the value a shifted
#: read reports (0 counts, -180 degrees) and the mid-scale default that a wrong
#: answer cannot be mistaken for the right one.
MOTOR = "shoulder_pan"
MOTOR_ID = 1
TRUE_COUNTS = 1024
TRUE_DEGREES = -89.97802197802199

#: The other motor on the bus, whose stale reply must not answer for MOTOR.
OTHER_ID = 2


def _status(motor_id: int, params: list[int], error: int = 0, checksum: int | None = None) -> bytes:
    """Build one status packet: ``FF FF ID LEN ERR <params> CHK``.

    Args:
        motor_id: The responding servo's ID.
        params: The reply's parameter bytes.
        error: The error byte the servo reports.
        checksum: Override the checksum, to build a corrupt frame.

    Returns:
        The frame's bytes.
    """
    body = [motor_id, len(params) + 2, error, *params]
    check = (~sum(body)) & 0xFF if checksum is None else checksum
    return bytes([0xFF, 0xFF, *body, check])


def _position_reply(motor_id: int = MOTOR_ID, counts: int = TRUE_COUNTS, **kwargs: Any) -> bytes:
    """A well-formed ``Present_Position`` reply carrying ``counts``."""
    return _status(motor_id, [counts & 0xFF, (counts >> 8) & 0xFF], **kwargs)


@pytest.fixture
def bus(monkeypatch):
    """A connected MotorController over a recording fake bus.

    Returns:
        A ``(controller, fake)`` pair. ``fake.queue_read`` adds one reply,
        ``fake.serve`` replaces the whole queue, and ``fake.writes`` records
        every packet written.
    """

    class Fake:
        def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
            self.port = port
            self.writes: list[bytes] = []
            self._queue: list[bytes] = []
            self.is_open = True

        def queue_read(self, data: bytes) -> None:
            self._queue.append(data)

        def serve(self, data: bytes, times: int) -> None:
            """Answer the next ``times`` reads with ``data``, discarding any queue.

            Replacing rather than appending is what keeps a two-phase test
            honest: leftovers from an earlier phase would answer the later one
            and the second phase would silently re-measure the first.
            """
            self._queue = [data] * times

        def write(self, data: bytes) -> None:
            self.writes.append(bytes(data))

        def read(self, n: int = 1) -> bytes:
            return self._queue.pop(0) if self._queue else b""

        def close(self) -> None:
            self.is_open = False

    made: list[Fake] = []

    def ctor(port: str, baudrate: int, timeout: float = 1.0) -> Fake:
        fake = Fake(port, baudrate, timeout)
        made.append(fake)
        return fake

    monkeypatch.setattr(serial, "Serial", ctor)
    controller = MotorController("/dev/ttyTEST")
    assert controller.connect()[0] is True
    return controller, made[0]


def _goal_positions(writes: list[bytes]) -> list[int]:
    """Every Goal_Position value written, in order.

    Args:
        writes: Raw packets recorded from the bus.

    Returns:
        The little-endian position each ``INST_WRITE`` to register 0x2A carried.
    """
    return [w[6] | (w[7] << 8) for w in writes if len(w) >= 8 and w[4] == 0x03 and w[5] == 0x2A]


# --------------------------------------------------------------------------- #
# Premise: the frames below are the frames a servo really sends
# --------------------------------------------------------------------------- #
class TestTheFramesHereAreWhatAServoSends:
    """Grade this file's fixtures against the SDK that owns the wire format.

    Without this the suite could pass by agreeing with itself: a builder and a
    parser written together share any misunderstanding of the format. The SDK
    arrives with ``lerobot[feetech]``, which the ``[lerobot]`` extra declares, so
    it is present wherever the rest of the servo path is - but it is skipped
    rather than required, because nothing else in this file needs it.
    """

    @staticmethod
    def _verdict(payload: bytes) -> int:
        """The SDK's communication result for ``payload``."""
        handler = pytest.importorskip("scservo_sdk.protocol_packet_handler").protocol_packet_handler()

        class Port:
            def __init__(self, data: bytes) -> None:
                self.buf = bytearray(data)
                self.dry = False

            def readPort(self, length: int) -> list[int]:  # noqa: N802 - SDK's spelling
                out = self.buf[:length]
                del self.buf[:length]
                if not out:
                    self.dry = True
                return list(out)

            def setPacketTimeout(self, n: int) -> None:  # noqa: N802 - SDK's spelling
                pass

            def isPacketTimeout(self) -> bool:  # noqa: N802 - SDK's spelling
                return self.dry

        _packet, result = handler.rxPacket(Port(payload))
        return int(result)

    @pytest.mark.parametrize(
        "label,payload",
        [
            ("clean", _position_reply()),
            ("one-leading-byte", b"\x00" + _position_reply()),
            ("two-leading-bytes", b"\x00\x00" + _position_reply()),
            ("another-motor", _position_reply(motor_id=OTHER_ID)),
        ],
    )
    def test_the_well_formed_frames_are_accepted_by_the_sdk(self, label: str, payload: bytes) -> None:
        # Reached through importorskip rather than an import statement: the SDK
        # ships no py.typed, so a static import would need a mypy override for a
        # module only this file reads.
        codes = pytest.importorskip("scservo_sdk.scservo_def")
        assert self._verdict(payload) == codes.COMM_SUCCESS, f"{label} should be a frame the SDK accepts"

    @pytest.mark.parametrize(
        "label,payload",
        [
            ("bad-checksum", _position_reply(checksum=0x00)),
            ("truncated", _position_reply()[:-1]),
        ],
    )
    def test_the_malformed_frames_are_refused_by_the_sdk(self, label: str, payload: bytes) -> None:
        codes = pytest.importorskip("scservo_sdk.scservo_def")
        assert self._verdict(payload) == codes.COMM_RX_CORRUPT, f"{label} should be a frame the SDK refuses"

    def test_the_sdk_recovers_the_position_from_a_shifted_frame(self) -> None:
        """The SDK reads 1024 from a frame behind a leading byte.

        This is what makes recovery - rather than refusal - the right answer for
        leading noise: the vendor implementation gets the true position from
        exactly these bytes.
        """
        handler = pytest.importorskip("scservo_sdk.protocol_packet_handler").protocol_packet_handler()

        class Port:
            def __init__(self, data: bytes) -> None:
                self.buf = bytearray(data)
                self.dry = False

            def readPort(self, length: int) -> list[int]:  # noqa: N802 - SDK's spelling
                out = self.buf[:length]
                del self.buf[:length]
                if not out:
                    self.dry = True
                return list(out)

            def setPacketTimeout(self, n: int) -> None:  # noqa: N802 - SDK's spelling
                pass

            def isPacketTimeout(self) -> bool:  # noqa: N802 - SDK's spelling
                return self.dry

        packet, _result = handler.rxPacket(Port(b"\x00" + _position_reply()))
        assert packet[5] | (packet[6] << 8) == TRUE_COUNTS


# --------------------------------------------------------------------------- #
# Regression: a reply behind leading bytes is still the reading
# --------------------------------------------------------------------------- #
class TestAReplyBehindLeadingBytesIsStillTheReading:
    """Leading bytes offset a reply; they do not make it a different position."""

    @pytest.mark.parametrize("leading", [0, 1, 2, 3, 7])
    def test_the_true_position_is_reported_however_far_the_frame_is_offset(self, bus, leading: int) -> None:
        controller, fake = bus
        fake.queue_read(bytes(leading) + _position_reply())
        assert controller.read_motor_position(MOTOR) == pytest.approx(TRUE_DEGREES)

    def test_the_parse_returns_the_parameter_bytes(self) -> None:
        assert _parse_status_packet(b"\x00" + _position_reply(), MOTOR_ID, 2) == (0x00, 0x04)

    def test_a_coincidental_header_does_not_end_the_search(self) -> None:
        """A frame that fails verification is skipped, not treated as the answer.

        The corrupt frame in front carries a real header, so a parse that stopped
        at the first header it found would report no reading at all while the
        motor's answer sat in the same buffer.
        """
        raw = _position_reply(checksum=0x00) + _position_reply()
        assert _parse_status_packet(raw, MOTOR_ID, 2) == (0x00, 0x04)

    def test_another_motors_reply_does_not_hide_this_motors(self, bus) -> None:
        controller, fake = bus
        fake.queue_read(_position_reply(motor_id=OTHER_ID) + _position_reply())
        assert controller.read_motor_position(MOTOR) == pytest.approx(TRUE_DEGREES)


# --------------------------------------------------------------------------- #
# Regression: an unverified reply is not a reading
# --------------------------------------------------------------------------- #
class TestAnUnverifiedReplyIsNotAReading:
    """A reply that cannot be verified reports nothing rather than a number."""

    @pytest.mark.parametrize(
        "label,payload",
        [
            ("another-motor", _position_reply(motor_id=OTHER_ID)),
            ("broadcast-id", _position_reply(motor_id=0xFE)),
            ("bad-checksum", _position_reply(checksum=0x00)),
            ("truncated-by-one", _position_reply()[:-1]),
            ("header-only", b"\xff\xff"),
            ("error-byte-not-an-error-byte", _position_reply(error=0x80)),
            ("wrong-parameter-count", _status(MOTOR_ID, [0x00])),
            ("no-bytes", b""),
        ],
    )
    def test_the_read_reports_no_position(self, bus, label: str, payload: bytes) -> None:
        controller, fake = bus
        fake.queue_read(payload)
        assert controller.read_motor_position(MOTOR) is None, f"{label} must not read as a position"

    @pytest.mark.parametrize(
        "label,payload",
        [
            ("another-motor", _position_reply(motor_id=OTHER_ID)),
            ("bad-checksum", _position_reply(checksum=0x00)),
            ("truncated-by-one", _position_reply()[:-1]),
        ],
    )
    def test_the_discarded_bytes_are_reported(self, bus, caplog, label: str, payload: bytes) -> None:
        """The bytes that were thrown away are named, so a bus fault is diagnosable."""
        controller, fake = bus
        fake.queue_read(payload)
        with caplog.at_level("WARNING", logger=pose_mod.__name__):
            assert controller.read_motor_position(MOTOR) is None
        assert any(MOTOR in record.getMessage() for record in caplog.records), label


class TestTheParseRefusesAFrameThatIsNotTheAnswerToThisRead:
    """Each verification carries a case the others cannot catch.

    Every frame here is well formed enough to pass the checks around the one
    under test, so a check that were removed would not simply be covered by a
    neighbour. The read itself would answer ``None`` for some of these anyway --
    by raising inside the broad handler rather than by refusing -- so they are
    graded on the parse, where the difference is visible.
    """

    def test_a_truncated_reply_is_refused_even_when_its_checksum_happens_to_agree(self) -> None:
        """Length is checked, not inferred from a checksum that might still add up.

        A frame reporting 64000 counts is ``FF FF 01 04 00 00 FA 00``. Drop the
        checksum byte and the byte now at the end, 0xFA, is exactly the checksum
        of everything before it -- so verifying the sum alone accepts a frame one
        byte short and reads a single parameter as though it were two.
        """
        frame = _position_reply(counts=64000)
        truncated = frame[:-1]
        assert (~sum(truncated[2:-1])) & 0xFF == truncated[-1], "the coincidence this case rests on"
        assert _parse_status_packet(truncated, MOTOR_ID, 2) is None

    def test_a_reply_carrying_the_wrong_number_of_parameters_is_not_this_read(self) -> None:
        """A well-formed one-byte reply is a different read's answer, not ours."""
        frame = _status(MOTOR_ID, [0x00])
        assert (~sum(frame[2:-1])) & 0xFF == frame[-1], "well formed - just not two parameters"
        assert _parse_status_packet(frame, MOTOR_ID, 2) is None

    def test_the_broadcast_address_never_answers_even_when_asked(self) -> None:
        """0xFE addresses every motor at once, so no reply can come from it.

        The vendor SDK refuses the same thing one layer up: ``readTxRx`` returns
        ``COMM_NOT_AVAILABLE`` without reading when asked for an ID at or above
        the broadcast address.
        """
        assert _parse_status_packet(_position_reply(motor_id=0xFE), 0xFE, 2) is None


# --------------------------------------------------------------------------- #
# Controls: what a verified reply must keep doing
# --------------------------------------------------------------------------- #
class TestAVerifiedReplyStillReads:
    """Every expectation here also held before the frame was verified."""

    def test_a_clean_reply_decodes_to_its_angle(self, bus) -> None:
        controller, fake = bus
        fake.queue_read(_position_reply())
        assert controller.read_motor_position(MOTOR) == pytest.approx(TRUE_DEGREES)

    @pytest.mark.parametrize("counts,degrees", [(0, -180.0), (2048, 0.043956043956), (4095, 180.0)])
    def test_the_whole_register_range_reads(self, bus, counts: int, degrees: float) -> None:
        controller, fake = bus
        fake.queue_read(_position_reply(counts=counts))
        assert controller.read_motor_position(MOTOR) == pytest.approx(degrees)

    def test_a_closed_bus_still_reports_nothing(self) -> None:
        assert MotorController("/dev/ttyTEST").read_motor_position(MOTOR) is None

    def test_the_read_still_asks_for_present_position(self, bus) -> None:
        """The request is unchanged: INST_READ of register 0x38, two bytes."""
        controller, fake = bus
        fake.queue_read(_position_reply())
        controller.read_motor_position(MOTOR)
        request = fake.writes[-1]
        # FF FF | id | LEN=4 | INST_READ=0x02 | Present_Position=0x38 | 2 bytes
        assert request[:7] == bytes([0xFF, 0xFF, MOTOR_ID, 0x04, 0x02, 0x38, 0x02])
        assert request[7] == (~sum(request[2:7])) & 0xFF

    def test_reading_every_motor_reads_each_one(self, bus) -> None:
        controller, fake = bus
        for name, config in controller.motor_configs.items():
            fake.queue_read(_position_reply(motor_id=config["id"]))
            assert name  # every configured motor is answered by its own ID
        assert len(controller.read_all_positions()) == len(controller.motor_configs)


# --------------------------------------------------------------------------- #
# Over-reach: verification must not refuse a frame the format allows
# --------------------------------------------------------------------------- #
class TestVerificationRefusesNothingTheFormatAllows:
    """A stricter parse must not turn working reads into failures."""

    def test_a_payload_that_looks_like_a_header_still_reads(self, bus) -> None:
        """Position 0xFFFF puts the header bytes inside the parameters.

        The frame is well formed, so it reads. Bounding the *value* to the
        register's 12-bit range is a separate question from framing, and this
        pins that the parse does not quietly answer it.
        """
        controller, fake = bus
        fake.queue_read(_position_reply(counts=0xFFFF))
        assert controller.read_motor_position(MOTOR) is not None

    @pytest.mark.parametrize("error", [0x00, 0x01, 0x04, 0x20, 0x7F])
    def test_a_servo_reporting_a_fault_still_reports_its_position(self, bus, error: int) -> None:
        """An error byte within range is a valid frame, as the SDK also holds.

        Acting on the fault bits - refusing to move an overheating servo - is a
        capability this parse deliberately does not decide; it would change what
        a caller is told about a healthy reading.
        """
        controller, fake = bus
        fake.queue_read(_position_reply(error=error))
        assert controller.read_motor_position(MOTOR) == pytest.approx(TRUE_DEGREES)


# --------------------------------------------------------------------------- #
# The consequences a caller sees
# --------------------------------------------------------------------------- #
class TestTheToolReportsAnErrorRatherThanAWrongNumber:
    """``pose_tool`` must not quote an unverified reply as a measurement."""

    @pytest.mark.parametrize("action", ["read_position", "read_all"])
    def test_an_unverifiable_reply_is_an_error_envelope(self, monkeypatch, action: str) -> None:
        made: list[Any] = []

        class Fake:
            def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
                self.is_open = True
                made.append(self)

            def write(self, data: bytes) -> None:
                pass

            def read(self, n: int = 1) -> bytes:
                return _position_reply(checksum=0x00)

            def close(self) -> None:
                self.is_open = False

        monkeypatch.setattr(serial, "Serial", Fake)
        result = pose_tool(action=action, port="/dev/ttyTEST", motor_name=MOTOR)
        assert result["status"] == "error"
        assert made, "the tool opened the bus"

    def test_a_shifted_reply_is_reported_as_the_true_angle(self, monkeypatch) -> None:
        class Fake:
            def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
                self.is_open = True

            def write(self, data: bytes) -> None:
                pass

            def read(self, n: int = 1) -> bytes:
                return b"\x00" + _position_reply()

            def close(self) -> None:
                self.is_open = False

        monkeypatch.setattr(serial, "Serial", Fake)
        result = pose_tool(action="read_position", port="/dev/ttyTEST", motor_name=MOTOR)
        assert result["status"] == "success"
        assert result["content"][1]["json"]["position"] == pytest.approx(TRUE_DEGREES)


class TestAnInterpolatedMoveStartsFromTheVerifiedPosition:
    """A trajectory is built from where the arm is, not from a shifted read.

    ``_smooth_move`` reads the current position and divides the travel to the
    target into increments. A reply read at the wrong offset reports 0 counts for
    a joint at 1024, so the first packet of a *smooth* move commands the joint
    ninety degrees away from its target instead of toward it.
    """

    def test_the_first_commanded_step_is_the_arms_own_position(self, bus) -> None:
        controller, fake = bus
        fake.serve(b"\x00" + _position_reply(), 64)
        controller._smooth_move({MOTOR: 180.0}, steps=4, step_delay=0.0)
        assert _goal_positions(fake.writes)[0] == TRUE_COUNTS

    def test_the_trajectory_matches_the_one_a_clean_bus_produces(self, bus) -> None:
        """Leading noise changes nothing about the path the arm is given."""
        controller, fake = bus

        fake.serve(_position_reply(), 64)
        controller._smooth_move({MOTOR: 180.0}, steps=4, step_delay=0.0)
        clean = _goal_positions(fake.writes)
        assert clean, "the clean bus produced a trajectory to compare against"

        fake.writes.clear()
        fake.serve(b"\x00\x00" + _position_reply(), 64)
        controller._smooth_move({MOTOR: 180.0}, steps=4, step_delay=0.0)
        assert _goal_positions(fake.writes) == clean

    def test_an_incremental_move_refuses_an_unverifiable_current_position(self, bus) -> None:
        controller, fake = bus
        fake.queue_read(_position_reply(checksum=0x00))
        assert controller.incremental_move(MOTOR, 5.0) is False
