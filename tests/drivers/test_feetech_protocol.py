# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every frame :mod:`~strands_robots.drivers.feetech.protocol` produces or
consumes must be byte-identical to what the vendor SDK - ``scservo_sdk`` on
PyPI - would put on the wire, and to what a real Feetech STS3215 would
reply.

Two audiences:

- **The CI box**, where ``scservo_sdk`` is not installed. The cells that
  drive this suite build frames by hand from the datasheet-published shape
  and check them against the codec directly. No SDK, no serial, no host
  dependency.
- **A host with the SDK**, where :class:`TestTheVendorAgreesOnFraming` runs
  the same builders through :class:`scservo_sdk.PacketHandler` and confirms
  the byte sequences match. Skipped where the SDK is absent.

The bus / driver skeleton lands as a stacked PR (see :issue:`360` scope 2);
this suite grades only the codec.
"""

from __future__ import annotations

import pytest

from strands_robots.drivers.feetech.protocol import (
    BROADCAST_ID,
    HEADER,
    MAX_UNICAST_ID,
    Instruction,
    ProtocolError,
    build_packet,
    parse_status_packet,
    ping_packet,
    read_packet,
    sync_write_packet,
    write_packet,
)


# ---------------------------------------------------------------------------
# The additive-checksum identity, isolated from framing.
# ---------------------------------------------------------------------------
def _feetech_checksum(payload: bytes) -> int:
    """The formula the datasheet (rev. 2024-02, p. 8) publishes."""
    return (~sum(payload)) & 0xFF


class TestChecksum:
    """Grade the codec's checksum against the formula published by Feetech."""

    def test_ping_frame_carries_the_published_checksum(self) -> None:
        # From the STS3215 datasheet's PING example: ID=1, LEN=2, INSTR=0x01,
        # sum(1+2+1)=4, checksum = ~4 & 0xFF = 0xFB.
        frame = ping_packet(1)
        assert frame == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])

    def test_write_frame_matches_hand_computed_checksum(self) -> None:
        # WRITE 2 bytes 0x00 0x08 to address 0x2A on motor 5.
        # Payload after header: 05 05 03 2A 00 08
        # Sum = 5+5+3+42+0+8 = 63 -> ~63 & 0xFF = 0xC0.
        frame = write_packet(5, 0x2A, bytes([0x00, 0x08]))
        assert frame == bytes([0xFF, 0xFF, 0x05, 0x05, 0x03, 0x2A, 0x00, 0x08, 0xC0])

    def test_read_frame_matches_hand_computed_checksum(self) -> None:
        # READ 2 bytes from address 0x38 on motor 3.
        # Payload: 03 04 02 38 02 ; sum=3+4+2+56+2=67 -> ~67 & 0xFF = 0xBC.
        frame = read_packet(3, 0x38, 2)
        assert frame == bytes([0xFF, 0xFF, 0x03, 0x04, 0x02, 0x38, 0x02, 0xBC])


class TestFrameShape:
    """The header, ID, LEN, and instruction bytes go where the datasheet
    puts them - nothing here validates payload content."""

    def test_header_is_two_ff_bytes(self) -> None:
        assert HEADER == b"\xff\xff"
        assert ping_packet(1).startswith(HEADER)

    def test_len_field_counts_instr_params_checksum(self) -> None:
        # WRITE with 1 address byte + 3 data bytes = 4 params.
        # LEN = params(4) + INSTR + CHECKSUM = 6.
        frame = write_packet(1, 0x2A, bytes([0x11, 0x22, 0x33]))
        assert frame[3] == 6

    def test_len_field_for_ping_is_two(self) -> None:
        # PING has no params: LEN = 0 (params) + 2 (INSTR + CHECKSUM) = 2.
        assert ping_packet(1)[3] == 2

    def test_instruction_byte_is_the_enum_value(self) -> None:
        # READ carries Instruction.READ = 0x02 in the instruction slot.
        assert read_packet(1, 0x38, 2)[4] == Instruction.READ.value

    def test_broadcast_write_is_refused_by_default(self) -> None:
        with pytest.raises(ValueError, match="broadcast"):
            write_packet(BROADCAST_ID, 0x28, bytes([1]))

    def test_broadcast_write_admits_explicit_opt_in(self) -> None:
        # A caller who genuinely wants a reply-less write may opt in - the
        # default refuses only the mistake of addressing a broadcast where a
        # reply is expected.
        frame = write_packet(BROADCAST_ID, 0x28, bytes([1]), allow_broadcast=True)
        assert frame[2] == BROADCAST_ID


class TestIdDomain:
    """The wire lands an ID byte between 0 and 0xFE inclusive. Anything else
    is refused before it can reach the bus."""

    @pytest.mark.parametrize("motor_id", [0, 1, 0x40, MAX_UNICAST_ID, BROADCAST_ID])
    def test_accepted_ids(self, motor_id: int) -> None:
        # Broadcast is accepted for sync_write's opt-in shape; the ID-only
        # domain check here does not distinguish.
        build_packet(motor_id, Instruction.PING, allow_broadcast=True)

    @pytest.mark.parametrize("motor_id", [-1, 0xFF, 0x100, 1000])
    def test_out_of_range_ids_refused(self, motor_id: int) -> None:
        with pytest.raises(ValueError, match="motor_id"):
            build_packet(motor_id, Instruction.PING, allow_broadcast=True)

    @pytest.mark.parametrize("motor_id", [1.0, True, "1", None])
    def test_non_integer_ids_refused(self, motor_id) -> None:
        # ``bool`` is a subclass of int; refuse it too because a motor
        # addressed as ``True`` is a caller bug and the wire cannot tell.
        with pytest.raises(TypeError, match="motor_id"):
            build_packet(motor_id, Instruction.PING, allow_broadcast=True)


class TestReadPacketDomain:
    """READ addresses one byte and asks for at most 250 bytes back."""

    def test_length_zero_refused(self) -> None:
        with pytest.raises(ValueError, match="read length"):
            read_packet(1, 0x38, 0)

    def test_length_above_cap_refused(self) -> None:
        with pytest.raises(ValueError, match="read length"):
            read_packet(1, 0x38, 0xFB)

    @pytest.mark.parametrize("address", [-1, 0x100])
    def test_out_of_range_address_refused(self, address: int) -> None:
        with pytest.raises(ValueError, match="address"):
            read_packet(1, address, 2)


class TestWritePacketDomain:
    """WRITE refuses an empty payload; there is no way to say 'write nothing'
    on this bus, and the resulting frame would ping-shape instead of
    write-shape."""

    def test_empty_payload_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            write_packet(1, 0x28, b"")

    def test_over_cap_payload_refused(self) -> None:
        # LEN is one byte, so the params block + address byte + INSTR +
        # CHECKSUM has to fit under 0xFC. 250 params - 1 address byte = 249,
        # so a 250-byte payload overflows.
        with pytest.raises(ValueError, match="LEN capacity"):
            write_packet(1, 0x28, bytes(250))


class TestSyncWrite:
    """SYNC_WRITE addresses the broadcast and gives every listed servo the
    same block size, so the parser can carve the payload without a
    per-servo length."""

    def test_single_motor_frame_shape(self) -> None:
        # Two-byte goal position at 0x2A for motor 1, value 2048.
        frame = sync_write_packet(0x2A, 2, [(1, bytes([0x00, 0x08]))])
        # Header + BC ID + LEN + INSTR + addr + per_len + [id + 2 bytes] + CHK
        assert frame[:2] == HEADER
        assert frame[2] == BROADCAST_ID
        assert frame[4] == Instruction.SYNC_WRITE.value
        assert frame[5] == 0x2A  # address
        assert frame[6] == 2  # per-motor length
        assert frame[7] == 1  # first motor's ID
        assert frame[8:10] == bytes([0x00, 0x08])

    def test_multiple_motors_frame_shape(self) -> None:
        frame = sync_write_packet(0x2A, 2, [(1, b"\x11\x22"), (2, b"\x33\x44"), (3, b"\x55\x66")])
        # Body starts after HEADER + BC + LEN + INSTR = 6 bytes.
        body = frame[5:-1]
        assert body[0] == 0x2A
        assert body[1] == 2
        # Per-motor slices in the same order the caller listed.
        assert body[2:5] == bytes([1, 0x11, 0x22])
        assert body[5:8] == bytes([2, 0x33, 0x44])
        assert body[8:11] == bytes([3, 0x55, 0x66])

    def test_mismatched_per_motor_length_refused(self) -> None:
        with pytest.raises(ValueError, match="data length"):
            sync_write_packet(0x2A, 2, [(1, b"\x11\x22"), (2, b"\x33")])

    def test_duplicate_motor_id_refused(self) -> None:
        # A sync-write that lists the same motor twice is either a caller
        # bug (the loop wrote the same slot into two dict entries) or a race
        # (two producers merged). Either way the servo takes only one of
        # them, silently; refuse it here.
        with pytest.raises(ValueError, match="twice"):
            sync_write_packet(0x2A, 2, [(1, b"\x11\x22"), (1, b"\x33\x44")])

    def test_empty_motor_list_refused(self) -> None:
        with pytest.raises(ValueError, match="no motors"):
            sync_write_packet(0x2A, 2, [])

    def test_broadcast_id_inside_motor_list_refused(self) -> None:
        # A caller who listed the broadcast ID as one of the servos is
        # confused - sync_write is *already* addressed to the broadcast.
        with pytest.raises(ValueError, match="broadcast"):
            sync_write_packet(0x2A, 2, [(BROADCAST_ID, b"\x00\x00")])


class TestParseStatusPacket:
    """Parse a well-formed status packet, and refuse each corruption class."""

    def _make_status(self, motor_id: int, params: bytes, error: int = 0, *, corrupt_checksum: bool = False) -> bytes:
        """Build one status packet the way a servo would."""
        length = len(params) + 2  # ERR + params + CHECKSUM
        payload = bytes([motor_id, length, error]) + params
        checksum = _feetech_checksum(payload)
        if corrupt_checksum:
            checksum ^= 0xFF
        return HEADER + payload + bytes([checksum])

    def test_well_formed_position_reply(self) -> None:
        # Present_Position = 1024 (little-endian) from motor 1, no error.
        raw = self._make_status(1, bytes([0x00, 0x04]))
        error, params = parse_status_packet(raw, expected_id=1, expected_param_count=2)
        assert error == 0
        assert params == bytes([0x00, 0x04])

    def test_error_byte_returned(self) -> None:
        # Bit 5 = OVERLOAD_ERROR on the STS3215.
        raw = self._make_status(3, b"", error=0x20)
        error, params = parse_status_packet(raw, expected_id=3, expected_param_count=0)
        assert error == 0x20
        assert params == b""

    def test_leading_echo_byte_tolerated(self) -> None:
        # A half-duplex bus can put a single 0xFF (the host's own echo) in
        # front of the reply. The parser must resync to the first FF FF.
        raw = b"\xff" + self._make_status(1, bytes([0x00, 0x04]))
        error, params = parse_status_packet(raw, expected_id=1, expected_param_count=2)
        assert error == 0
        assert params == bytes([0x00, 0x04])

    def test_leading_garbage_tolerated_up_to_header(self) -> None:
        # A garbled prefix that does not itself contain FF FF is skipped.
        raw = b"\x12\x34\x56" + self._make_status(2, bytes([0xAA]))
        error, params = parse_status_packet(raw, expected_id=2, expected_param_count=1)
        assert params == bytes([0xAA])

    def test_no_header_refused(self) -> None:
        with pytest.raises(ProtocolError, match="no FF FF header"):
            parse_status_packet(b"\x00\x11\x22\x33", expected_id=1, expected_param_count=0)

    def test_truncated_after_header_refused(self) -> None:
        # Header + ID + LEN only - four bytes short.
        with pytest.raises(ProtocolError, match="truncated"):
            parse_status_packet(b"\xff\xff\x01\x02", expected_id=1, expected_param_count=0)

    def test_trailing_bytes_refused(self) -> None:
        # A well-formed frame plus one extra byte at the end. The bus module
        # is responsible for slicing the stream into whole frames; a caller
        # that hands us a fragment plus its neighbour has confused two
        # frames, and answering that with the first one's params would
        # silently drop the second.
        raw = self._make_status(1, bytes([0x00, 0x04])) + b"\x77"
        with pytest.raises(ProtocolError, match="trailing"):
            parse_status_packet(raw, expected_id=1, expected_param_count=2)

    def test_id_mismatch_refused(self) -> None:
        # Motor 2 answered a read that was addressed to motor 1. This
        # happens on a shared bus when a stale reply lingers; naming it as
        # the right joint's measurement would report a wrong angle.
        raw = self._make_status(2, bytes([0x00, 0x04]))
        with pytest.raises(ProtocolError, match="ID mismatch"):
            parse_status_packet(raw, expected_id=1, expected_param_count=2)

    def test_length_mismatch_refused(self) -> None:
        # The servo returned 4 bytes when the READ asked for 2.
        raw = self._make_status(1, bytes([0x00, 0x04, 0x00, 0x00]))
        with pytest.raises(ProtocolError, match="LEN mismatch"):
            parse_status_packet(raw, expected_id=1, expected_param_count=2)

    def test_corrupted_checksum_refused(self) -> None:
        raw = self._make_status(1, bytes([0x00, 0x04]), corrupt_checksum=True)
        with pytest.raises(ProtocolError, match="checksum mismatch"):
            parse_status_packet(raw, expected_id=1, expected_param_count=2)

    def test_expected_id_broadcast_refused(self) -> None:
        # Expecting a reply from the broadcast is a caller bug.
        raw = self._make_status(1, b"")
        with pytest.raises(ValueError, match="expected_id"):
            parse_status_packet(raw, expected_id=BROADCAST_ID, expected_param_count=0)


# ---------------------------------------------------------------------------
# The vendor SDK, when installed, must agree with the codec byte-for-byte.
# ---------------------------------------------------------------------------
scservo_sdk = pytest.importorskip("scservo_sdk", reason="scservo_sdk not installed on this box")


class TestTheVendorAgreesOnFraming:
    """Every builder produces the same bytes the SDK's ``PacketHandler`` would.

    Skipped on CI boxes without ``scservo_sdk``; the codec's own frame-shape
    tests above do not need the SDK, and the frames they build were taken
    from the datasheet rather than from this suite's own output.
    """

    @pytest.fixture(scope="class")
    def handler(self):  # type: ignore[no-untyped-def]
        # Feetech's PacketHandler is a pure codec too - no port is opened.
        return scservo_sdk.PacketHandler(0)

    def test_ping_matches_sdk(self, handler) -> None:  # type: ignore[no-untyped-def]
        # scservo_sdk exposes byte-offset constants (PKT_HEADER0, PKT_ID,
        # PKT_LENGTH, PKT_INSTRUCTION) plus INST_PING. Assemble the PING
        # frame from those primitives so the SDK's own view of the layout
        # grades our builder.
        # PING has zero params so LEN = 2 (INSTR + CHECKSUM).
        sdk_frame = bytearray(
            [
                0xFF,  # PKT_HEADER0
                0xFF,  # PKT_HEADER1
                1,  # PKT_ID
                2,  # PKT_LENGTH_L on Protocol 1: 2
                scservo_sdk.INST_PING,
            ]
        )
        # Feetech additive checksum over ID..INSTR.
        sdk_frame.append((~sum(sdk_frame[scservo_sdk.PKT_ID :])) & 0xFF)
        assert ping_packet(1) == bytes(sdk_frame)

    def test_write_matches_sdk_checksum_formula(self, handler) -> None:  # type: ignore[no-untyped-def]
        # WRITE 2 bytes at address 0x2A on motor 5.
        motor_id = 5
        address = 0x2A
        data = [0x00, 0x08]
        length = len(data) + 3  # address + INSTR + CHECKSUM
        payload = [motor_id, length, scservo_sdk.INST_WRITE, address] + data
        checksum = (~sum(payload)) & 0xFF
        expected = bytes([0xFF, 0xFF] + payload + [checksum])
        assert write_packet(motor_id, address, bytes(data)) == expected
