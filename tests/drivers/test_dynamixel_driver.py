"""Tests for :mod:`strands_robots.drivers.dynamixel`.

The codec:

* :class:`TestProtocol` grades the wire format against expected bytes. Every
  case works from a known-good frame (either hand-computed from the manual or
  a fixture recorded from the Robotis SDK), so a passing test says the codec
  round-trips a real packet, not that it round-trips itself.
* :class:`TestByteStuffing` grades the escape the protocol requires around the
  reserved ``FF FF FD`` run, against the Robotis SDK's own framing.
* :class:`TestSixteenBitFrameFields` grades the boundary of each two-byte
  frame field, where a value the field cannot hold would otherwise be
  truncated into one it can.

* :class:`TestTheBusOnTheWire` grades the bus against frames ``dynamixel_sdk``
  sent, then drives koch end to end - read, move, release, a policy rollout -
  over a port with six servos behind it. The other Dynamixel arms are refused.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from strands_robots.drivers import get_native_driver_class
from strands_robots.drivers.dynamixel import (
    CONTROL_TABLE,
    KOCH_MOTORS,
    DynamixelBus,
    DynamixelDriver,
    Instruction,
    build_packet,
    checksum,
    decode_model_number,
    parse_status_packet,
    parse_status_stream,
    sync_write_packet,
    write_packet,
)
from strands_robots.drivers.dynamixel.protocol import (
    BROADCAST_ID,
    HEADER,
    MAX_UNICAST_ID,
    RESERVED_RUN,
)

# ============================================================================
# Codec.
# ============================================================================


class TestProtocol:
    """Wire format tests.

    Values throughout this class come from the Robotis Protocol 2.0 e-manual
    example packets and from a small set of frames captured against a running
    ``dynamixel_sdk`` on a live bus. Where the manual's example has been
    reproduced, the reference is inline.
    """

    # ---------------------------------- CRC ----------------------------------
    #
    # The expected CRC values below were captured against a fresh install of
    # ``dynamixel_sdk`` (Robotis' Python binding) as an independent oracle.
    # ``dynamixel_sdk`` is on PyPI and the check reproduces trivially::
    #
    #     >>> from dynamixel_sdk.protocol2_packet_handler import (
    #     ...     Protocol2PacketHandler,
    #     ... )
    #     >>> h = Protocol2PacketHandler()
    #     >>> frame = bytearray([0xFF, 0xFF, 0xFD, 0x00, 0x01, 6, 0, 3,
    #     ...                    0x41, 0x00, 0x01, 0, 0])
    #     >>> h.updateCRC(0, frame, len(frame) - 2)
    #     59084  # 0xE6CC
    #
    # We do not require ``dynamixel_sdk`` at test time - the point of vendoring
    # the codec is to remove that dependency from the driver's test surface -
    # so the expected values are baked into the tests as literals.

    def test_crc_matches_the_dynamixel_sdk_write_led_example(self) -> None:
        """WRITE to the LED register of ID 1, value 1.

        Frame preamble (all bytes before the CRC): FF FF FD 00 01 06 00 03 41 00 01
        Expected CRC (from dynamixel_sdk): 0xE6CC (bytes CC E6 on the wire).
        """
        frame_without_crc = bytes.fromhex("fffffd0001060003410001")
        crc = checksum(frame_without_crc)
        assert crc == 0xE6CC, f"CRC {crc:#06x} != expected 0xE6CC (matches dynamixel_sdk)"

    def test_crc_matches_the_dynamixel_sdk_ping_example(self) -> None:
        """PING to ID 1.

        Frame preamble: FF FF FD 00 01 03 00 01
        Expected CRC (from dynamixel_sdk): 0x4E19 (bytes 19 4E on the wire).
        """
        frame_without_crc = bytes.fromhex("fffffd0001030001")
        crc = checksum(frame_without_crc)
        assert crc == 0x4E19, f"CRC {crc:#06x} != expected 0x4E19"

    def test_crc_covers_the_whole_frame_not_only_the_body(self) -> None:
        """Truncating the header must change the CRC.

        A common regression: computing the CRC over the body only. The manual
        is explicit that the CRC covers everything from HEADER1 onwards.
        """
        frame = bytes.fromhex("fffffd0001060003410001")
        assert checksum(frame) != checksum(frame[4:])

    # -------------------------------- build ---------------------------------

    def test_build_packet_write_led_matches_dynamixel_sdk_output(self) -> None:
        """The full framed packet reproduces dynamixel_sdk byte-for-byte.

        Expected (from dynamixel_sdk): fffffd0001060003410001cce6
        """
        packet = build_packet(1, Instruction.WRITE, bytes([0x41, 0x00, 0x01]))
        assert packet.hex() == "fffffd0001060003410001cce6"

    def test_build_packet_ping_matches_dynamixel_sdk_output(self) -> None:
        """Expected (from dynamixel_sdk): fffffd0001030001194e"""
        packet = build_packet(1, Instruction.PING)
        assert packet.hex() == "fffffd0001030001194e"

    def test_build_packet_refuses_reserved_and_broadcast_ids(self) -> None:
        for bad in (0xFD, 0xFE, 0xFF, 256):
            with pytest.raises(ValueError, match="servo_id must be"):
                build_packet(bad, Instruction.PING)

    def test_build_packet_refuses_negative_id(self) -> None:
        with pytest.raises(ValueError, match="servo_id must be"):
            build_packet(-1, Instruction.PING)

    def test_build_packet_write_carries_params(self) -> None:
        """A WRITE of one byte at register 65 (LED) to servo 1 is the
        manual's example: params are ``41 00 01`` (address LE + value)."""
        packet = build_packet(1, Instruction.WRITE, bytes.fromhex("4100 01".replace(" ", "")))
        # LEN counts INST + params (3) + CRC (2) = 6.
        assert packet[5] == 0x06
        assert packet[6] == 0x00
        assert packet[7] == int(Instruction.WRITE)

    # -------------------------------- sync ---------------------------------

    def test_sync_write_targets_the_broadcast_id(self) -> None:
        packet = sync_write_packet(
            register_address=116,  # GOAL_POSITION
            data_length=4,
            entries=[(1, b"\x00\x00\x00\x00"), (2, b"\xff\x03\x00\x00")],
        )
        # Body starts at byte 4.
        assert packet[4] == BROADCAST_ID

    def test_sync_write_matches_dynamixel_sdk_output(self) -> None:
        """SYNC_WRITE of GOAL_POSITION (register 116, 4 bytes) for IDs 1 and 2.

        Expected (from dynamixel_sdk): fffffd00fe11008374000400010000000002ff030000ef40
        """
        packet = sync_write_packet(
            register_address=116,
            data_length=4,
            entries=[(1, b"\x00\x00\x00\x00"), (2, b"\xff\x03\x00\x00")],
        )
        assert packet.hex() == "fffffd00fe11008374000400010000000002ff030000ef40"

    def test_sync_write_refuses_a_data_of_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="expected 4"):
            sync_write_packet(
                register_address=116,
                data_length=4,
                entries=[(1, b"\x00\x00\x00")],  # 3 bytes for a 4-byte write
            )

    def test_sync_write_refuses_a_broadcast_entry(self) -> None:
        with pytest.raises(ValueError, match="entry id must be"):
            sync_write_packet(
                register_address=116,
                data_length=4,
                entries=[(BROADCAST_ID, b"\x00\x00\x00\x00")],
            )

    def test_sync_write_refuses_a_zero_data_length(self) -> None:
        with pytest.raises(ValueError, match="data_length"):
            sync_write_packet(register_address=116, data_length=0, entries=[])

    # ------------------- sync width against the register --------------------
    #
    # A servo answers a SYNC_WRITE with nothing, so a data_length that does not
    # fit the register it addresses cannot come back as an error - it comes back
    # as a joint somewhere nobody asked for. The widths below are read from
    # CONTROL_TABLE rather than retyped, so a table edit moves these cases with
    # it instead of leaving them pinning a stale width.

    _MISMATCHED_WIDTHS = [
        # A 4-byte current command runs its top half into GOAL_VELOCITY, so
        # asking for 500 mA also commands a velocity of 0 on a moving joint.
        pytest.param("GOAL_CURRENT", 4, "runs the extra 2 byte(s) on into GOAL_VELOCITY", id="current-into-velocity"),
        pytest.param("TORQUE_ENABLE", 4, "runs the extra 3 byte(s) on into LED", id="torque-into-led"),
        # The register above GOAL_VELOCITY is one the table does not name, so the
        # refusal says where the bytes go without inventing a name for it.
        pytest.param("GOAL_VELOCITY", 8, "on into the register above it", id="velocity-into-unlisted"),
        pytest.param("GOAL_POSITION", 2, "leaves the remaining 2 byte(s) of it unwritten", id="position-short"),
        pytest.param("GOAL_CURRENT", 1, "leaves the remaining 1 byte(s) of it unwritten", id="current-short"),
    ]

    @pytest.mark.parametrize(("register", "data_length", "consequence"), _MISMATCHED_WIDTHS)
    def test_sync_write_refuses_a_data_length_the_register_is_not(
        self, register: str, data_length: int, consequence: str
    ) -> None:
        """The refusal names the register, its width, and where the bytes land.

        Naming the consequence is the point: a caller who picked the wrong width
        picked it from a register map, and "GOAL_CURRENT is 2 bytes wide" sends
        them back to the map, while "runs on into GOAL_VELOCITY" tells them what
        the servo would have done with the packet.
        """
        address, width, _ = CONTROL_TABLE[register]
        assert data_length != width
        with pytest.raises(ValueError) as excinfo:
            sync_write_packet(address, data_length, [(1, bytes(data_length))])
        message = str(excinfo.value)
        assert f"{register} at register_address={address} is {width} bytes wide" in message
        assert consequence in message

    def test_sync_write_accepts_every_listed_register_at_its_own_width(self) -> None:
        """The gate narrows nothing a correct caller does.

        Every register the table names, framed at the width the table declares
        for it, still produces a broadcast packet - so the refusal above cannot
        be passing merely because the width check refuses everything.
        """
        for name, (address, width, _) in CONTROL_TABLE.items():
            packet = sync_write_packet(address, width, [(1, bytes(width))])
            assert packet[4] == BROADCAST_ID, name

    def test_sync_write_leaves_an_address_the_table_does_not_name_ungraded(self) -> None:
        """CONTROL_TABLE is a curated subset of the servo's registers, not an
        allowlist. PROFILE_VELOCITY (112) is a real 4-byte register it omits;
        grading unlisted addresses would refuse writes to every register the
        table has not got round to naming.
        """
        assert 112 not in {address for address, _, _ in CONTROL_TABLE.values()}
        packet = sync_write_packet(112, 4, [(1, bytes(4))])
        assert packet[4] == BROADCAST_ID

    def test_a_zero_data_length_is_still_reported_as_a_count(self) -> None:
        """The width gate reads a data_length that is already a positive count,
        so the domain check keeps its place in front of it: a 0 is diagnosed as
        a 0 rather than as a width GOAL_POSITION happens not to be.
        """
        with pytest.raises(ValueError, match=r"data_length must be > 0"):
            sync_write_packet(register_address=116, data_length=0, entries=[])

    def test_the_register_width_is_reported_before_the_entry_length(self) -> None:
        """A caller who picks the wrong width sizes their entries to it, so both
        checks have something to say. The register is the fault that explains the
        other one, and a caller told only "expected 2" would re-send entries at a
        width GOAL_POSITION still will not take.
        """
        with pytest.raises(ValueError) as excinfo:
            sync_write_packet(register_address=116, data_length=2, entries=[(1, b"\x00")])
        message = str(excinfo.value)
        assert "GOAL_POSITION at register_address=116 is 4 bytes wide" in message
        assert "expected 2" not in message

    # -------------------------------- parse --------------------------------

    def _make_status(self, servo_id: int, err: int, params: bytes) -> bytes:
        length = len(params) + 4  # inst + err + params + crc(2)
        body = bytes([servo_id, length & 0xFF, (length >> 8) & 0xFF, 0x55, err]) + params
        frame = HEADER + body
        crc = checksum(frame)
        return frame + bytes([crc & 0xFF, (crc >> 8) & 0xFF])

    def test_parse_status_round_trips_a_good_frame(self) -> None:
        frame = self._make_status(servo_id=1, err=0, params=b"\x24\x04")  # MODEL_NUMBER=1060
        result = parse_status_packet(frame)
        assert result == {"servo_id": 1, "err": 0, "params": b"\x24\x04", "crc_ok": True}

    def test_parse_status_reports_a_bad_crc_without_raising(self) -> None:
        """A caller who wants to retry an unreliable line needs to see the shape,
        not an exception."""
        frame = bytearray(self._make_status(servo_id=1, err=0, params=b"\x24\x04"))
        frame[-1] ^= 0xFF  # flip the high CRC byte
        result = parse_status_packet(bytes(frame))
        assert result["crc_ok"] is False
        # The rest of the shape survives.
        assert result["servo_id"] == 1
        assert result["params"] == b"\x24\x04"

    def test_parse_status_refuses_a_short_frame(self) -> None:
        with pytest.raises(ValueError, match="frame too short"):
            parse_status_packet(b"\xff\xff\xfd\x00\x01\x03\x00")  # 7 bytes

    def test_parse_status_refuses_a_bad_header(self) -> None:
        frame = bytearray(self._make_status(servo_id=1, err=0, params=b""))
        frame[0] = 0x00
        with pytest.raises(ValueError, match="header mismatch"):
            parse_status_packet(bytes(frame))

    def test_parse_status_refuses_a_non_status_instruction_byte(self) -> None:
        frame = bytearray(self._make_status(servo_id=1, err=0, params=b""))
        frame[7] = 0x03  # WRITE instead of the 0x55 status marker
        # CRC changes, but the shape check runs first.
        with pytest.raises(ValueError, match="0x55 status marker"):
            parse_status_packet(bytes(frame))

    def test_parse_status_refuses_a_length_field_mismatch(self) -> None:
        """The length field is authoritative; a truncated frame must not parse."""
        frame = self._make_status(servo_id=1, err=0, params=b"\x24\x04")
        with pytest.raises(ValueError, match="does not match length field"):
            parse_status_packet(frame[:-1])  # drop the last CRC byte

    # ------------------------------- model ---------------------------------

    @pytest.mark.parametrize(
        "params,expected_number",
        [
            (b"\x00\x00", 0x0000),
            (b"\x24\x04", 0x0424),
            (b"\xa6\x04", 0x04A6),
            (b"\x60\x04", 0x0460),
            (b"\x37\x01", 0x0137),
            (b"\xff\xff", 0xFFFF),
        ],
    )
    def test_decode_model_number_is_little_endian(self, params: bytes, expected_number: int) -> None:
        """Register 0 is little-endian: the low byte arrives first. Pinned at
        both ends of the range because a byte-swap is invisible on a payload
        whose two bytes happen to be equal, and every model number a real
        servo reports has a non-zero high byte."""
        assert decode_model_number(params) == expected_number

    def test_decode_model_number_refuses_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="expected 2 bytes"):
            decode_model_number(b"\x24")

    # ---------------------------- control table ----------------------------

    def test_goal_position_and_present_position_widths_match_the_manual(self) -> None:
        """A read of PRESENT_POSITION returns 4 bytes; a write of GOAL_POSITION
        takes 4 bytes. This is the pair the Aloha bimanual sync-writes at
        100Hz, so getting the width wrong is on the acceptance path."""
        assert CONTROL_TABLE["GOAL_POSITION"][:2] == (116, 4)
        assert CONTROL_TABLE["PRESENT_POSITION"][:2] == (132, 4)

    def test_torque_enable_width_is_one_byte(self) -> None:
        assert CONTROL_TABLE["TORQUE_ENABLE"][:2] == (64, 1)

    def test_no_two_registers_share_an_address(self) -> None:
        """A sync-write's width is looked up by address, so two names at one
        address would silently grade one of them by the other's width."""
        addresses = [address for address, _, _ in CONTROL_TABLE.values()]
        assert len(addresses) == len(set(addresses))

    def test_max_unicast_id_below_the_broadcast(self) -> None:
        """A codec-level invariant. The values are the manual's; a change here
        should trip the tests, not slide into a release."""
        assert BROADCAST_ID == 0xFE
        assert MAX_UNICAST_ID == BROADCAST_ID - 2  # 0xFD is reserved


# ============================================================================
# The bus and the driver, over a port with six servos behind it.
# ============================================================================

#: Frames ``dynamixel_sdk`` 4.1.0 put on the wire (``GroupSyncRead`` of
#: Present_Position for ids 1-6, ``GroupSyncWrite`` of Goal_Position
#: {1: 2048, 2: 1024, 6: 3000}, ``write1ByteTxOnly(3, TORQUE_ENABLE, 0)``) and
#: the status packet it accepts from id 2 at 2048 counts. Captured once against
#: the SDK so the transcript below grades bytes, not this codec against itself.
SDK_SYNC_READ = bytes.fromhex("fffffd00fe0d008284000400010203040506b29b")
SDK_SYNC_WRITE = bytes.fromhex("fffffd00fe160083740004000100080000020004000006b80b0000c2ea")
SDK_TORQUE_OFF = bytes.fromhex("fffffd0003060003400000fd64")
SDK_STATUS_ID2_2048 = bytes.fromhex("fffffd00020800550000080000bc32")


def _status(servo_id: int, params: bytes = b"", err: int = 0) -> bytes:
    """The status packet a servo answers with (``INST`` 0x55, then ``ERR``)."""
    length = len(params) + 4
    frame = HEADER + bytes([servo_id, length & 0xFF, length >> 8, 0x55, err]) + params
    crc = checksum(frame)
    return frame + bytes([crc & 0xFF, crc >> 8])


class FakeDynamixelPort:
    """Six X-series servos on a half-duplex line, decoded by offset, not by the codec.

    A servo moves to a ``Goal_Position`` only while torqued, as the hardware
    does. ``mute`` servos answer nothing; ``err`` sets a servo's ``ERR`` byte.
    """

    def __init__(self, counts: dict[int, int], *, mute: tuple[int, ...] = (), err: dict[int, int] | None = None):
        self.counts, self.mute, self.err = dict(counts), set(mute), dict(err or {})
        self.torque = dict.fromkeys(counts, 0)
        self.writes: list[bytes] = []
        self.is_open, self._pending = True, b""

    @property
    def in_waiting(self) -> int:
        return len(self._pending)

    def read(self, size: int) -> bytes:
        out, self._pending = self._pending[:size], self._pending[size:]
        return out

    def close(self) -> None:
        self.is_open = False

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        servo_id, inst, params = data[4], data[7], data[8:-2]
        address = params[0] | params[1] << 8
        if inst == Instruction.SYNC_READ:
            for i in params[4:]:
                if i not in self.mute:
                    self._pending += _status(i, self.counts[i].to_bytes(4, "little", signed=True), self.err.get(i, 0))
        elif inst == Instruction.SYNC_WRITE and address == CONTROL_TABLE["GOAL_POSITION"][0]:
            for k in range(4, len(params), 5):
                if self.torque.get(params[k]):
                    self.counts[params[k]] = int.from_bytes(params[k + 1 : k + 5], "little", signed=True)
        elif inst == Instruction.WRITE and servo_id not in self.mute:
            if address == CONTROL_TABLE["TORQUE_ENABLE"][0]:
                self.torque[servo_id] = params[2]
            self._pending += _status(servo_id, err=self.err.get(servo_id, 0))
        return len(data)


def _koch(monkeypatch: pytest.MonkeyPatch, port: FakeDynamixelPort) -> Any:
    """``Robot("koch", mode="real", driver="strands")`` with ``port`` behind ``serial.Serial``."""
    import serial

    from strands_robots import Robot

    monkeypatch.setattr(serial, "Serial", lambda *args, **kwargs: port)
    return Robot("koch", mode="real", driver="strands", port="/dev/ttyUSB0")


class TestTheBusOnTheWire:
    """The frames the bus sends are the SDK's; the arm reads, moves, releases and runs a policy."""

    def test_every_frame_is_the_sdk_frame(self, monkeypatch: pytest.MonkeyPatch) -> None:
        port = FakeDynamixelPort(dict.fromkeys(range(1, 7), 2048))
        bus = DynamixelBus(port="/dev/fake")
        bus._conn = port
        bus.sync_read()
        bus.set_torque(True)
        bus.write_goal_positions({name: 0.0 for name in ("shoulder_pan", "shoulder_lift")})
        assert port.writes[0] == SDK_SYNC_READ
        assert write_packet(3, CONTROL_TABLE["TORQUE_ENABLE"][0], b"\x00") == SDK_TORQUE_OFF
        pairs = [(1, 2048), (2, 1024), (6, 3000)]
        assert sync_write_packet(116, 4, [(i, c.to_bytes(4, "little")) for i, c in pairs]) == SDK_SYNC_WRITE
        assert parse_status_stream(b"\x00" + SDK_STATUS_ID2_2048, [2], 4) == {2: (2048).to_bytes(4, "little")}

    @pytest.mark.parametrize(
        ("err", "verified"),
        [(0x00, True), (0x80, True), (0x04, False), (0x84, False)],
        ids=["clean", "hardware-alert-still-executed", "data-range-error", "alert-and-error"],
    )
    def test_only_an_error_number_refuses_a_reply(self, err: int, verified: bool) -> None:
        stream = SDK_SYNC_READ + _status(4, (100).to_bytes(4, "little"), err)
        assert (4 in parse_status_stream(stream, [4], 4)) is verified

    def test_koch_reads_moves_and_releases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        port = FakeDynamixelPort(dict.fromkeys(range(1, 7), 2048))
        robot = _koch(monkeypatch, port)
        assert type(robot) is DynamixelDriver
        assert robot._set_torque_envelope(True)["status"] == "success"
        assert robot.send_action({"shoulder_pan": 90.0, "gripper.pos": 100.0})["status"] == "success"
        assert port.counts[1] == 3071 and port.counts[6] == 4095
        joints = robot._read_joints_envelope()["content"][0]["json"]["joint_state"]
        assert set(joints) == set(KOCH_MOTORS)
        assert joints["shoulder_pan"] == pytest.approx(90.0, abs=360 / 4095)
        asyncio.run(robot.stop())
        assert set(port.torque.values()) == {0}

    def test_a_mute_servo_is_named_as_possibly_still_driven(self, monkeypatch: pytest.MonkeyPatch) -> None:
        robot = _koch(monkeypatch, FakeDynamixelPort(dict.fromkeys(range(1, 7), 2048), mute=(6,)))
        refusal = robot._set_torque_envelope(False)
        assert refusal["status"] == "error" and "['gripper']" in refusal["content"][0]["text"]
        assert "gripper" not in robot._read_joints_envelope()["content"][0]["json"]["joint_state"]

    def test_a_policy_runs_at_thirty_hertz_and_stops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        port = FakeDynamixelPort(dict.fromkeys(range(1, 7), 2048))
        robot = _koch(monkeypatch, port)
        robot._set_torque_envelope(True)
        seen: list[dict[str, Any]] = []

        def policy(observation: dict[str, Any]) -> dict[str, Any]:
            seen.append(observation)
            return {"shoulder_pan.pos": float(len(seen))}

        assert robot.run_policy(policy, n_steps=15, control_frequency=30.0)["status"] == "success"
        robot._rollout.join()
        status = robot.get_task_status()["content"][0]["json"]
        assert (status["steps"], status["running"]) == (15, False)
        assert set(seen[0]) == {f"{name}.pos" for name in KOCH_MOTORS}
        assert robot._read_joints_envelope()["content"][0]["json"]["joint_state"]["shoulder_pan"] == pytest.approx(
            15.0, abs=360 / 4095
        )
        assert robot.stop_task()["status"] == "success"


#: The Dynamixel arms with no verified motor map: ``driver="strands"`` refuses
#: each rather than commanding joints it cannot name.
UNMAPPED_DYNAMIXEL_ROBOTS = ("aloha", "vx300s", "wx250s", "trossen_wxai", "dynamixel_2r")


@pytest.mark.parametrize("canonical", UNMAPPED_DYNAMIXEL_ROBOTS)
def test_a_dynamixel_robot_without_a_motor_map_is_refused(canonical: str) -> None:
    from strands_robots import Robot

    assert get_native_driver_class(canonical) is None
    with pytest.raises(ValueError, match="lerobot has no robot type for it either"):
        Robot(canonical, mode="real", driver="strands", port="/dev/null")


# ============================================================================
# Byte stuffing.
# ============================================================================


class TestByteStuffing:
    """Protocol 2.0 forbids the run ``FF FF FD`` inside a payload.

    A servo watching the bus reads those three bytes as the start of the next
    packet, so the protocol escapes the run with an extra ``0xFD`` and counts
    the inserted byte in ``LEN``. The escape is applied BEFORE the CRC, so the
    CRC covers the stuffed frame -- get that order wrong and every frame
    carrying a run is rejected even though the bytes look plausible.

    Every expected value in this class was produced by Robotis'
    ``dynamixel_sdk`` 4.0.5 -- specifically ``Protocol2PacketHandler``'s
    ``addStuffing`` followed by ``updateCRC``, which is the exact pair its
    ``txPacket`` uses to put bytes on the wire. No serial port is involved:
    the SDK's framing is pure, so it can be used as an oracle on any host.
    The codec here is expected to be byte-identical to it, which is what the
    module docstring claims and what these cases hold it to.

    ``dynamixel_sdk`` is deliberately NOT a test dependency -- it is not
    declared in ``pyproject.toml`` and the point of owning the codec is that
    the wire format is gradeable without it. The vectors are therefore frozen
    here rather than recomputed, matching how :class:`TestProtocol` already
    records its own expected frames.
    """

    def test_the_reserved_run_is_the_packet_header(self) -> None:
        """Why the run must be escaped at all.

        ``FF FF FD`` is not an arbitrary forbidden sequence: it is the first
        three bytes of :data:`HEADER`. This is the premise the whole class
        rests on, so it is asserted rather than assumed.
        """
        from strands_robots.drivers.dynamixel.protocol import RESERVED_RUN

        assert RESERVED_RUN == HEADER[:3]
        assert RESERVED_RUN == b"\xff\xff\xfd"

    def test_build_packet_escapes_a_single_run(self) -> None:
        """A ``WRITE`` whose parameters are exactly the reserved run.

        Expected (dynamixel_sdk): ``fffffd0001070003fffffdfd7cd1`` -- note the
        payload reads ``fffffdfd`` (the escape) and ``LEN`` is 7, not the 6 an
        unescaped three-byte parameter block would give.
        """
        packet = build_packet(1, Instruction.WRITE, b"\xff\xff\xfd")
        assert packet.hex() == "fffffd0001070003fffffdfd7cd1"
        assert packet[5] | (packet[6] << 8) == 7

    def test_build_packet_escapes_only_the_run_not_every_fd(self) -> None:
        """``FF FF FD FD`` gains one escape, not two.

        Only the run gets an escape. The second ``0xFD`` is not itself
        preceded by ``FF FF``, so it is left alone and the payload becomes
        ``FF FF FD FD FD``: three ``0xFD`` bytes, which looks like one too
        many until they are counted.

        Expected (dynamixel_sdk): ``fffffd0001080003fffffdfdfdc90c``
        """
        packet = build_packet(1, Instruction.WRITE, b"\xff\xff\xfd\xfd")
        assert packet.hex() == "fffffd0001080003fffffdfdfdc90c"

    def test_build_packet_escapes_every_run_in_the_payload(self) -> None:
        """Two runs cost two escape bytes and ``LEN`` counts both.

        Expected (dynamixel_sdk): ``fffffd00010b0003fffffdfdfffffdfd3121``
        """
        packet = build_packet(1, Instruction.WRITE, b"\xff\xff\xfd\xff\xff\xfd")
        assert packet.hex() == "fffffd00010b0003fffffdfdfffffdfd3121"
        assert packet[5] | (packet[6] << 8) == 11

    def test_a_goal_position_that_needs_escaping_is_a_reachable_command(self) -> None:
        """The one legal goal position whose encoding contains the run.

        ``GOAL_POSITION`` is a signed 32-bit little-endian value. In
        extended-position (multi-turn) mode the legal range is
        -1048575..1048575, and exactly one value in it encodes to bytes
        containing ``FF FF FD``: -131073, which is -32.0 turns at 4096 counts
        per revolution. That is an ordinary place to drive a multi-turn joint,
        which is what makes an unescaped write here worth a test rather than a
        note -- it is data-dependent and would appear once in two million
        commands.
        """
        import struct

        assert struct.pack("<i", -131073) == b"\xff\xff\xfd\xff"
        needing_escape = [value for value in range(-1048575, 1048576) if b"\xff\xff\xfd" in struct.pack("<i", value)]
        assert needing_escape == [-131073]
        assert -131073 / 4096 == pytest.approx(-32.0, abs=0.001)

    def test_unicast_write_of_that_goal_position_is_escaped(self) -> None:
        """Expected (dynamixel_sdk): ``fffffd00010a00037400fffffdfdff23e5``"""
        import struct

        address, width, _ = CONTROL_TABLE["GOAL_POSITION"]
        params = bytes([address & 0xFF, (address >> 8) & 0xFF]) + struct.pack("<i", -131073)
        packet = build_packet(1, Instruction.WRITE, params)
        assert packet.hex() == "fffffd00010a00037400fffffdfdff23e5"

    def test_sync_write_of_that_goal_position_is_escaped(self) -> None:
        """The same value through the broadcast path the driver will use.

        Expected (dynamixel_sdk): ``fffffd00fe0d00837400040001fffffdfdff6084``
        """
        import struct

        address, width, _ = CONTROL_TABLE["GOAL_POSITION"]
        packet = sync_write_packet(address, width, [(1, struct.pack("<i", -131073))])
        assert packet.hex() == "fffffd00fe0d00837400040001fffffdfdff6084"
        assert packet[5] | (packet[6] << 8) == 13

    def test_parse_status_unescapes_the_payload(self) -> None:
        """A servo escapes its reply too, so the parser must reverse it.

        The frame below is what ``dynamixel_sdk`` produces for a status packet
        from id 7 with ``err=0`` and parameters ``FF FF FD 2A``:
        ``fffffd000709005500fffffdfd2a5adc``. The parameters read back must be
        the four bytes the servo measured, not the five that travelled.
        """
        frame = bytes.fromhex("fffffd000709005500fffffdfd2a5adc")
        parsed = parse_status_packet(frame)
        assert parsed["servo_id"] == 7
        assert parsed["err"] == 0
        assert parsed["params"] == b"\xff\xff\xfd\x2a"
        assert parsed["crc_ok"] is True

    def test_stuffing_round_trips_every_adversarial_payload(self) -> None:
        """Build then parse recovers the payload, for payloads built from the
        bytes that make runs likely.

        The corpus is enumerated rather than random so a failure names a
        reproducible payload. It covers every 4-byte string over
        ``{00, 2A, FD, FE, FF}``, which includes every arrangement of the run
        and of the ``FD FD`` sequence that the look-back treats specially.
        """
        import itertools

        alphabet = (0x00, 0x2A, 0xFD, 0xFE, 0xFF)
        checked = 0
        carried_an_escape = 0
        for combo in itertools.product(alphabet, repeat=4):
            params = bytes(combo)
            frame = build_packet(3, Instruction.WRITE, params)
            # Re-frame it as the status packet a servo would send back, so the
            # parse path sees the same escaping the build path produced.
            status = build_packet(3, Instruction.WRITE, b"\x00" + params)
            status = status[:7] + b"\x55" + status[8:]
            rebuilt = status[:-2]
            crc = checksum(rebuilt)
            parsed = parse_status_packet(rebuilt + bytes([crc & 0xFF, (crc >> 8) & 0xFF]))
            assert parsed["params"] == params, f"payload {params.hex()} did not round-trip"
            if len(frame) != 10 + len(params):
                carried_an_escape += 1
            checked += 1
        assert checked == len(alphabet) ** 4
        assert carried_an_escape > 0, "corpus never exercised an escape - it grades nothing"

    def test_a_payload_without_the_run_is_untouched(self) -> None:
        """Over-reach control.

        The overwhelmingly common case must be byte-identical to what the
        codec produced before stuffing existed, and ``LEN`` must not move.
        This is the same vector :class:`TestProtocol` already pins, repeated
        here so a stuffing change that corrupts ordinary traffic fails in the
        class that caused it.
        """
        packet = build_packet(1, Instruction.WRITE, bytes([0x41, 0x00, 0x01]))
        assert packet.hex() == "fffffd0001060003410001cce6"
        assert packet[5] | (packet[6] << 8) == 6

    def test_an_ordinary_sync_write_is_untouched(self) -> None:
        """Over-reach control for the broadcast path."""
        packet = sync_write_packet(116, 4, [(1, bytes(4)), (2, bytes([0xFF, 0x03, 0x00, 0x00]))])
        assert packet.hex() == "fffffd00fe11008374000400010000000002ff030000ef40"


# ============================================================================
# The frame's 16-bit fields.
# ============================================================================


#: The largest value a Protocol 2.0 ``LEN`` or register address can hold. Both
#: are two little-endian bytes, so a value above this is not "large" on the
#: wire - it is a different value, and the codec writes it as one.
_FIELD_MAX = 0xFFFF

#: The longest ``params`` :func:`build_packet` accepts: ``LEN`` counts ``INST``
#: plus the parameters plus the two CRC bytes, so this lands ``LEN`` on
#: :data:`_FIELD_MAX` exactly.
_LONGEST_PARAMS = _FIELD_MAX - 3

#: Eight reserved runs, padded so escaping them lands ``LEN`` on
#: :data:`_FIELD_MAX` exactly. Stuffing happens after ``LEN`` is computed, so
#: this is the largest payload the escape can still fit - and one byte more is
#: a frame :func:`build_packet` accepts and :func:`_stuff` cannot.
_ESCAPES_TO_THE_LIMIT = RESERVED_RUN * 8 + bytes(_LONGEST_PARAMS - 8 - 3 * 8)

#: A ``data_length`` whose single-entry parameter block (address 2 + length 2 +
#: id 1 + data) is the longest :func:`sync_write_packet` can frame.
_LONGEST_SYNC_DATA_LENGTH = _LONGEST_PARAMS - 5

#: An address :data:`CONTROL_TABLE` does not name, so the register-width gate
#: has nothing to say about it and the length checks are what answer.
_UNLISTED_ADDRESS = 0x0100


class TestSixteenBitFrameFields:
    """A value too wide for a frame field is refused, not truncated into it.

    ``LEN`` and the sync-write register address are two little-endian bytes
    each, and the codec writes them with ``& 0xFF`` / ``>> 8``. Truncation is
    therefore the default behaviour of every one of these writes, and it is
    silent in the way that matters most on this bus: a servo answers a
    ``SYNC_WRITE`` with nothing at all, and a unicast reply that never arrives
    is indistinguishable from a servo that is powered off.

    Three separate functions compute a ``LEN`` and each refuses its own
    overflow - :func:`build_packet` from the parameter count,
    :func:`sync_write_packet` from the parameter block, and :func:`_stuff` from
    the bytes the escape inserted, which is the one door a caller cannot
    anticipate because the frame it is handed was already legal. The cells
    below pin all three at the boundary, from both sides, plus the address
    field's own domain.
    """

    # ------------------------------- LEN ------------------------------------

    def test_the_longest_frame_the_field_can_declare_is_accepted(self) -> None:
        """Over-reach control for the three refusals below.

        ``LEN`` is INST-inclusive, so the largest frame the field can describe
        carries ``_FIELD_MAX - 3`` parameters. Pinning that it is accepted, and
        that the field really does read back full, is what makes the refusals
        a boundary rather than a cap somewhere below it.
        """
        packet = build_packet(1, Instruction.WRITE, bytes(_LONGEST_PARAMS))
        assert packet[5] | (packet[6] << 8) == _FIELD_MAX
        assert len(packet) == 7 + _FIELD_MAX

    def test_one_parameter_byte_past_the_field_is_refused(self) -> None:
        """The refusal counts the parameters, which is the number the caller
        controls; ``LEN`` is derived from it and reporting the derived value
        would send them looking for a field they never set.
        """
        with pytest.raises(ValueError, match=r"params too long \(65533 bytes\)"):
            build_packet(1, Instruction.WRITE, bytes(_LONGEST_PARAMS + 1))

    def test_a_payload_the_escape_fills_the_field_with_is_accepted(self) -> None:
        """Stuffing rewrites ``LEN`` upward, and reaching the maximum exactly is
        still a frame the servo can read.
        """
        packet = build_packet(1, Instruction.WRITE, _ESCAPES_TO_THE_LIMIT)
        assert packet.count(RESERVED_RUN + b"\xfd") == 8, "the payload must really be escaped"
        assert packet[5] | (packet[6] << 8) == _FIELD_MAX
        assert len(packet) == 7 + _FIELD_MAX

    def test_an_escape_that_pushes_the_field_over_is_refused(self) -> None:
        """The one overflow a caller cannot see coming.

        These parameters are shorter than :func:`build_packet` refuses, so its
        own length check passes and the frame handed to :func:`_stuff` is legal.
        The escape bytes are what overflow the field, and the count of escaped
        runs is in the message because that is the part of the frame the caller
        did not put there.
        """
        with pytest.raises(ValueError, match=r"_stuff: escaping 8 reserved run\(s\) overflows"):
            build_packet(1, Instruction.WRITE, _ESCAPES_TO_THE_LIMIT + b"\x00")

    def test_the_longest_sync_write_the_field_can_declare_is_accepted(self) -> None:
        """Over-reach control for the broadcast path's own length check."""
        packet = sync_write_packet(
            _UNLISTED_ADDRESS,
            _LONGEST_SYNC_DATA_LENGTH,
            [(1, bytes(_LONGEST_SYNC_DATA_LENGTH))],
        )
        assert packet[4] == BROADCAST_ID
        assert packet[5] | (packet[6] << 8) == _FIELD_MAX

    def test_a_sync_write_parameter_block_past_the_field_is_refused(self) -> None:
        """The broadcast's parameter block grows with both ``data_length`` and
        the entry count, so the refusal reports the block it built rather than
        either input: a caller writing 300 servos and a caller writing one wide
        register arrive at the same limit from different directions.
        """
        with pytest.raises(ValueError, match=r"parameter block too long \(65533 bytes\)"):
            sync_write_packet(
                _UNLISTED_ADDRESS,
                _LONGEST_SYNC_DATA_LENGTH + 1,
                [(1, bytes(_LONGEST_SYNC_DATA_LENGTH + 1))],
            )

    # --------------------------- register address ---------------------------

    @pytest.mark.parametrize(
        "register_address",
        [
            pytest.param(_FIELD_MAX + 1, id="one-past-the-field"),
            pytest.param(0x10074, id="aliases-onto-goal-position"),
            pytest.param(-1, id="negative"),
        ],
    )
    def test_a_register_address_outside_the_field_is_refused(self, register_address: int) -> None:
        """An address the field cannot hold is refused before the packet is
        framed, because the two bytes it would be written as address a real
        register and a broadcast draws no reply to contradict them.
        """
        with pytest.raises(ValueError, match=r"register_address must be 0\.\.0xFFFF"):
            sync_write_packet(register_address, 2, [(1, bytes(2))])

    def test_the_widest_address_the_field_holds_is_accepted(self) -> None:
        """Over-reach control: the refusal is about the field's width, not about
        high addresses, and the top of the range is framed as itself.
        """
        packet = sync_write_packet(_FIELD_MAX, 2, [(1, bytes(2))])
        assert packet[8] | (packet[9] << 8) == _FIELD_MAX

    def test_the_refused_address_aliases_onto_a_register_the_table_names(self) -> None:
        """Why truncation is worse here than a wrong number usually is.

        ``0x10074`` is not a register at all, but the low two bytes of it are
        ``GOAL_POSITION``. Truncation would frame a broadcast command to a
        motion register on every servo on the bus, and it would do so at a
        ``data_length`` the register-width gate never saw: that gate looks the
        caller's address up in :data:`CONTROL_TABLE` unmasked, so a value above
        the field misses the table, is treated as an unlisted register, and
        carries whatever width the caller asked for onto a register the table
        does declare a width for.
        """
        assert 0x10074 & _FIELD_MAX == CONTROL_TABLE["GOAL_POSITION"][0]
        assert CONTROL_TABLE["GOAL_POSITION"][1] == 4
        assert 0x10074 not in {address for address, _, _ in CONTROL_TABLE.values()}
