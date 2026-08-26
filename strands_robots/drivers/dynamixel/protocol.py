"""Dynamixel Protocol 2.0 wire format. Pure, no I/O.

Every function here is verifiable against Robotis' ``dynamixel_sdk`` byte-for
-byte. The point of extracting the codec from the SDK is not to avoid the
dependency - ``dynamixel_sdk`` is on PyPI and small - but to make the wire
format part of the driver's own test surface. The SDK's parser lives inside
``PacketHandler.readTxRx`` and cannot be exercised without opening a port,
which makes a mocked bus impossible to grade against a real one.

Protocol 2.0 packet shape (from the Robotis e-manual, `Protocol 2.0`_):

.. code-block:: text

    HEADER1  HEADER2  HEADER3  RESERVED  ID    LEN_L  LEN_H  INST    P1..PN    CRC_L  CRC_H
    0xFF     0xFF     0xFD     0x00      <id>  <n+3>          <n params>

- ``ID`` is 0..252 for a specific servo, ``0xFE`` for a broadcast that expects
  no reply.
- ``LEN`` is the count of the bytes that follow it up to and including the CRC
  (i.e. ``INST`` + params + CRC = ``N + 3`` where ``N`` is the parameter
  count).
- The CRC is CRC-16/BUYPASS (polynomial ``0x8005``, no reflection, init
  ``0x0000``, xor-out ``0x0000``) computed over the full frame from
  ``HEADER1`` through the last parameter byte.
- A payload may not contain the byte run ``FF FF FD``: a servo reading the bus
  would take it for the start of the next packet. Protocol 2.0 therefore
  *stuffs* an extra ``0xFD`` after any such run, and the ``LEN`` field counts
  the stuffed bytes. Stuffing happens BEFORE the CRC is taken, so the CRC
  covers the stuffed frame -- the order matters and is the one Robotis'
  ``txPacket`` uses. :func:`_stuff` and :func:`_unstuff` are the two halves;
  every builder here goes through them, and :func:`parse_status_packet`
  reverses them.

.. _Protocol 2.0: https://emanual.robotis.com/docs/en/dxl/protocol2/

The status packet Robotis returns is the same shape with an ERR byte in front
of the parameters, so :func:`parse_status_packet` shares the frame check with
:func:`build_packet` and only differs in what it accepts after ``INST``.

Nothing here opens a serial port. The one function that would want the port -
``read_register`` say - is deliberately in the bus module (see :issue:`359`
scope 1) so this module can be imported on any host in CI.
"""

from __future__ import annotations

import enum
from typing import Final

# ---------------------------------------------------------------------------
# Framing constants. Named because two of them look confusingly interchangeable
# (0xFF vs 0xFD), and because the manual talks about them by name rather than
# by value.
# ---------------------------------------------------------------------------
HEADER: Final[bytes] = b"\xff\xff\xfd\x00"
BROADCAST_ID: Final[int] = 0xFE
"""ID a controller writes to when it wants every servo to receive the packet
and none to reply. Used by :func:`sync_write_packet`; a broadcast that expects
a reply (``BULK_READ``) needs a distinct sync-read primitive that reads N
status packets back."""

MAX_UNICAST_ID: Final[int] = 0xFC
"""Highest ID a specific servo may hold. 0xFD is reserved and 0xFE is the
broadcast."""


class Instruction(enum.IntEnum):
    """The instruction bytes Protocol 2.0 defines.

    Only the members :class:`DynamixelDriver` actually issues appear here; the
    remainder (``FACTORY_RESET`` etc.) will land as they are wired.
    """

    PING = 0x01
    READ = 0x02
    WRITE = 0x03
    REG_WRITE = 0x04
    ACTION = 0x05
    FACTORY_RESET = 0x06
    REBOOT = 0x08
    SYNC_READ = 0x82
    SYNC_WRITE = 0x83
    BULK_READ = 0x92
    BULK_WRITE = 0x93


# ---------------------------------------------------------------------------
# Control table. Register addresses are the same across the XL/XM/XH lines the
# supported robots use; the ranges differ but the addresses do not, which is
# why this table is a single dict rather than one per model.
#
# The full control table is ~50 registers wide. This subset is what the
# driver's read and write paths touch - adding a register here is meant to be
# the shortest legal change, so the entries carry only address + width + a
# short description. A wider surface is the bus module's job (Step 2), not
# the codec's.
# ---------------------------------------------------------------------------
CONTROL_TABLE: Final[dict[str, tuple[int, int, str]]] = {
    # name: (address, width_bytes, description)
    "MODEL_NUMBER": (0, 2, "Read-only. Decoded by :func:`decode_model_number`."),
    "MODEL_INFORMATION": (2, 4, "Read-only, opaque."),
    "FIRMWARE_VERSION": (6, 1, "Read-only."),
    "ID": (7, 1, "Read-write when torque is off. Servo's bus address."),
    "BAUD_RATE": (8, 1, "Enum: 0=9600, 1=57600, 2=115200, 3=1000000, 4=2000000, 5=3000000, 6=4000000, 7=4500000."),
    "RETURN_DELAY_TIME": (9, 1, "Microseconds*2 before status packet is sent."),
    "DRIVE_MODE": (10, 1, "Bitfield: reverse, profile config, torque-on-by-goal-update."),
    "OPERATING_MODE": (
        11,
        1,
        "Enum: 0=current, 1=velocity, 3=position, 4=extended_position, 5=current_position, 16=pwm.",
    ),
    "TORQUE_ENABLE": (64, 1, "0 or 1. Register must be off to change most rw fields above."),
    "LED": (65, 1, "0 or 1."),
    "GOAL_CURRENT": (102, 2, "Signed 16-bit, mA. Range depends on model."),
    "GOAL_VELOCITY": (104, 4, "Signed 32-bit. In rev/min * 0.229 for XL/XM. NOT sign-magnitude - two's complement."),
    "GOAL_POSITION": (116, 4, "0..4095 for XL330/XM430; wider range in extended-position mode."),
    "PRESENT_CURRENT": (126, 2, "Signed 16-bit, mA."),
    "PRESENT_VELOCITY": (128, 4, "Signed 32-bit."),
    "PRESENT_POSITION": (132, 4, "Signed 32-bit; single-turn 0..4095, wraps in extended-position mode."),
    "PRESENT_TEMPERATURE": (146, 1, "Degrees C."),
    "PRESENT_INPUT_VOLTAGE": (144, 2, "0.1V units."),
}


def checksum(frame: bytes) -> int:
    """Return the Protocol 2.0 CRC over ``frame``.

    The CRC covers the whole frame from ``HEADER1`` up to and including the
    last parameter byte - i.e. everything :func:`build_packet` produces except
    the two CRC bytes it appends. Robotis publishes this exact routine in
    ``dynamixel_sdk/protocol2_packet_handler.py``; the table there is the
    reference and this implementation should be byte-identical.

    Args:
        frame: The bytes to CRC. May be any length; the polynomial does not
            require alignment.

    Returns:
        The 16-bit CRC as an int in ``0..0xFFFF``.
    """
    crc = 0
    for byte in frame:
        idx = ((crc >> 8) ^ byte) & 0xFF
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[idx]
    return crc


#: Index of the ``INST`` byte in a framed packet: ``HEADER`` (4) + ``ID`` (1) +
#: ``LEN_L`` + ``LEN_H``. Stuffing scans from here to the last parameter, and
#: the two-byte look-back at this position deliberately reads the ``LEN`` bytes
#: -- that is what Robotis' ``addStuffing`` does, so a servo expects it.
_INST_INDEX: Final[int] = 7

#: The run a payload may not contain: a servo would read it as the next header.
RESERVED_RUN: Final[bytes] = b"\xff\xff\xfd"


def _stuff(frame: bytes) -> bytes:
    """Insert the Protocol 2.0 escape byte after every reserved run.

    Protocol 2.0 forbids ``FF FF FD`` inside a payload, because a servo
    watching the bus would take it for the start of the next packet. The
    escape is an extra ``0xFD`` immediately after the run; the ``LEN`` field
    then counts the inserted bytes, so it is rewritten here.

    Only the run itself is escaped, so a payload of ``FF FF FD FD`` becomes
    ``FF FF FD FD FD`` -- three ``0xFD`` bytes, which reads as one too many
    until they are counted. The second ``0xFD`` is not itself preceded by
    ``FF FF``, so it needs no escape of its own.

    The look-back below reads the original frame rather than the bytes emitted
    so far, mirroring how Robotis' ``addStuffing`` is written. The two
    formulations are equivalent -- an escape is always ``0xFD``, so after
    escaping at some position the preceding pair is ``FD FF`` in the original
    and ``FD FD`` in the output, and neither is ``FF FF`` -- so this is a
    readability choice, not a correctness one. It is spelled the SDK's way
    because that is the implementation a reader is most likely to compare
    against.

    Args:
        frame: An unstuffed frame -- ``HEADER`` through the last parameter,
            with no CRC appended yet. Its ``LEN`` field must already hold the
            unstuffed count.

    Returns:
        The stuffed frame with ``LEN`` rewritten. Returned unchanged (a new
        object either way) when the payload contains no reserved run, which
        is the overwhelmingly common case.
    """
    out = bytearray(frame[:_INST_INDEX])
    inserted = 0
    for position in range(_INST_INDEX, len(frame)):
        out.append(frame[position])
        if frame[position] == 0xFD and frame[position - 1] == 0xFF and frame[position - 2] == 0xFF:
            out.append(0xFD)
            inserted += 1
    if inserted:
        length = (frame[5] | (frame[6] << 8)) + inserted
        if length > 0xFFFF:
            raise ValueError(
                f"_stuff: escaping {inserted} reserved run(s) overflows the 16-bit length field",
            )
        out[5] = length & 0xFF
        out[6] = (length >> 8) & 0xFF
    return bytes(out)


def _unstuff(payload: bytes) -> bytes:
    """Remove Protocol 2.0 escape bytes from a received payload.

    The inverse of :func:`_stuff`, and the mirror of Robotis'
    ``removeStuffing``: a ``0xFD`` that follows a ``FF FF FD`` run is an
    escape the servo inserted and is not part of the data.

    Args:
        payload: The bytes from ``INST`` through the last parameter of a
            received frame, with the CRC already stripped.

    Returns:
        The payload with escapes removed. A payload that carries none is
        returned byte-for-byte.
    """
    out = bytearray()
    skip_next = False
    for position, byte in enumerate(payload):
        if skip_next:
            skip_next = False
            continue
        out.append(byte)
        # An escape follows the run, so look at what we have just emitted.
        if byte == 0xFD and len(out) >= 3 and out[-3] == 0xFF and out[-2] == 0xFF:
            if position + 1 < len(payload) and payload[position + 1] == 0xFD:
                skip_next = True
    return bytes(out)


def build_packet(servo_id: int, instruction: Instruction, params: bytes = b"") -> bytes:
    """Return a Protocol 2.0 packet for a unicast instruction.

    The shape is documented on the module. This does not build broadcast or
    sync packets; those have their own primitive (:func:`sync_write_packet`)
    because their length calculation is different in a way worth naming.

    Args:
        servo_id: The target servo's ID, in ``0..0xFC``. ``0xFD`` is reserved
            and ``0xFE`` is the broadcast - a caller who wants to broadcast
            calls the sync-write primitive.
        instruction: The instruction byte.
        params: The instruction's parameters, as raw bytes. The caller
            encodes register addresses and integer widths.

    Returns:
        The full framed packet, ready to write to the port.

    Raises:
        ValueError: If ``servo_id`` is outside ``0..0xFC``, or if ``params``
            would produce a packet longer than the 16-bit length field can
            represent (``params`` longer than ~65530 bytes).
    """
    if not 0 <= servo_id <= MAX_UNICAST_ID:
        raise ValueError(
            f"build_packet: servo_id must be 0..{MAX_UNICAST_ID} (0xFD is reserved, 0xFE is the broadcast); got {servo_id}",
        )
    length = len(params) + 3  # INST + params + CRC(2). LEN is INST-inclusive.
    if length > 0xFFFF:
        raise ValueError(f"build_packet: params too long ({len(params)} bytes) for the 16-bit length field")
    body = (
        bytes(
            [
                servo_id,
                length & 0xFF,
                (length >> 8) & 0xFF,
                int(instruction),
            ]
        )
        + params
    )
    frame = HEADER + body
    # Escape any reserved run BEFORE the CRC: the CRC covers the stuffed frame,
    # and _stuff rewrites LEN to count the inserted bytes.
    frame = _stuff(frame)
    crc = checksum(frame)
    return frame + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def sync_write_packet(register_address: int, data_length: int, entries: list[tuple[int, bytes]]) -> bytes:
    """Return a ``SYNC_WRITE`` packet writing ``entries`` to ``register_address``.

    Broadcast: the packet targets :data:`BROADCAST_ID` and no servo replies.
    The length field is different from a unicast write because the packet
    carries N (id, data) tuples and the servos self-select on ID.

    Args:
        register_address: The register to write.
        data_length: Bytes per servo. Must match the register's width in
            :data:`CONTROL_TABLE` - a 4-byte write to a 2-byte register is
            accepted by the servo but writes into the next register.
        entries: List of ``(servo_id, data)`` pairs. Each ``data`` must be
            exactly ``data_length`` bytes; a data of the wrong length is
            :class:`ValueError`, not a silent truncation, because the servo
            has no way to tell the driver the sync-write's shape was wrong.

    Returns:
        The full framed packet, targeting :data:`BROADCAST_ID`.

    Raises:
        ValueError: If any entry's data is not ``data_length`` bytes, or if
            an entry's ID is outside ``0..0xFC``, or if the parameter block
            would overflow the 16-bit length field.
    """
    if data_length <= 0 or data_length > 0xFFFF:
        raise ValueError(f"sync_write_packet: data_length must be > 0 and <= 0xFFFF; got {data_length}")
    if not 0 <= register_address <= 0xFFFF:
        raise ValueError(f"sync_write_packet: register_address must be 0..0xFFFF; got {register_address}")
    params = bytearray(
        [
            register_address & 0xFF,
            (register_address >> 8) & 0xFF,
            data_length & 0xFF,
            (data_length >> 8) & 0xFF,
        ]
    )
    for servo_id, data in entries:
        if not 0 <= servo_id <= MAX_UNICAST_ID:
            raise ValueError(
                f"sync_write_packet: entry id must be 0..{MAX_UNICAST_ID}; got {servo_id}",
            )
        if len(data) != data_length:
            raise ValueError(
                f"sync_write_packet: entry for id={servo_id} has {len(data)} bytes, expected {data_length}",
            )
        params.append(servo_id)
        params.extend(data)
    length = len(params) + 3  # INST + params + CRC(2)
    if length > 0xFFFF:
        raise ValueError(f"sync_write_packet: parameter block too long ({len(params)} bytes)")
    body = bytes(
        [
            BROADCAST_ID,
            length & 0xFF,
            (length >> 8) & 0xFF,
            int(Instruction.SYNC_WRITE),
        ]
    ) + bytes(params)
    frame = HEADER + body
    # Escape any reserved run BEFORE the CRC: the CRC covers the stuffed frame,
    # and _stuff rewrites LEN to count the inserted bytes.
    frame = _stuff(frame)
    crc = checksum(frame)
    return frame + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def parse_status_packet(frame: bytes) -> dict[str, object]:
    """Parse a Protocol 2.0 status packet.

    A status packet has the same framing as an instruction packet but the
    ``INST`` byte is always ``0x55`` (the status marker) and is followed by
    an ``ERR`` byte before the parameters.

    Args:
        frame: The bytes returned on the port. Must be at least a full frame
            (11 bytes minimum: header 4 + id 1 + len 2 + inst 1 + err 1 + crc 2).

    Returns:
        Dict with:

        * ``servo_id`` (int) - who replied,
        * ``err`` (int) - the ``ERR`` byte, 0 for a successful read,
        * ``params`` (bytes) - the register data, empty for a write's ack,
        * ``crc_ok`` (bool) - whether the appended CRC matches the frame.

        The parse does not raise on a bad CRC because a caller who wants to
        retry an unreliable line needs to distinguish "arrived but corrupt"
        from "did not arrive". A malformed shape (short frame, wrong length
        field, missing header) is :class:`ValueError` because there is
        nothing a retry can recover.

    Raises:
        ValueError: The frame is too short to be a status packet, or the
            header / length field are wrong.
    """
    if len(frame) < 11:
        raise ValueError(f"parse_status_packet: frame too short ({len(frame)} bytes, minimum 11)")
    if frame[:4] != HEADER:
        raise ValueError(f"parse_status_packet: header mismatch, got {frame[:4].hex()}")
    servo_id = frame[4]
    length = frame[5] | (frame[6] << 8)
    # length counts INST + ERR + params + CRC, all as stuffed on the wire.
    if len(frame) != 7 + length:
        raise ValueError(
            f"parse_status_packet: frame length {len(frame)} does not match length field {length} (expected {7 + length})",
        )
    inst = frame[7]
    if inst != 0x55:
        raise ValueError(f"parse_status_packet: expected 0x55 status marker, got {inst:#04x}")
    expected_crc = checksum(frame[:-2])
    actual_crc = frame[-2] | (frame[-1] << 8)
    # The bytes arrived stuffed and the servo's CRC covers them as received, so
    # the CRC is taken first and the payload unescaped afterwards. ``length``
    # counts the stuffed bytes, which is why the slice uses it before
    # unstuffing rather than after.
    payload = _unstuff(bytes(frame[_INST_INDEX : _INST_INDEX + length - 2]))
    err = payload[1]
    params = payload[2:]
    return {
        "servo_id": servo_id,
        "err": err,
        "params": params,
        "crc_ok": expected_crc == actual_crc,
    }


def decode_model_number(params: bytes) -> int:
    """Decode a ``MODEL_NUMBER`` read's params.

    ``MODEL_NUMBER`` is a two-byte little-endian unsigned integer at register
    0, and every Protocol 2.0 servo reports it in its ``PING`` reply. That
    decode is wire format, so it belongs here.

    Resolving the number to a model name (``1060`` -> ``XL430-W250``) is
    deliberately *not* here. A name is per-model hardware metadata, checkable
    against a servo rather than against a byte string, so a table of names in
    this module can only be graded on its shape - upper-case, hyphenated,
    unique - and never on whether each name belongs to the number it sits
    against. That gap is not academic: this function has no error path for a
    *wrong* name, so one misassigned row reports an XL430-W250 as an
    XL330-M077 - a servo with roughly seven times less stall torque and a
    different voltage envelope - and the probe still looks like it succeeded.
    Silent hardware misidentification is a worse failure than no name at all,
    so name resolution lands with the bus that can read register 0 off real
    hardware and check itself (:issue:`359` scope 1).

    Args:
        params: The status packet's parameter bytes for a read of two bytes
            at register 0. Must be exactly two bytes.

    Returns:
        The model number, in ``0..0xFFFF``. An unrecognised number is not an
        error here: the caller still knows which servo answered and can refuse
        it by ID.

    Raises:
        ValueError: If ``params`` is not exactly two bytes.
    """
    if len(params) != 2:
        raise ValueError(f"decode_model_number: expected 2 bytes, got {len(params)}")
    return int.from_bytes(params, "little")


# ---------------------------------------------------------------------------
# CRC-16/BUYPASS lookup table. Precomputed at import time; the same values
# ship in dynamixel_sdk. Kept as a private module attribute rather than a
# constant because it is neither user-facing nor useful outside :func:`checksum`.
# ---------------------------------------------------------------------------
def _build_crc_table() -> tuple[int, ...]:
    poly = 0x8005
    table = []
    for value in range(256):
        crc = value << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) & 0xFFFF) ^ poly
            else:
                crc = (crc << 1) & 0xFFFF
        table.append(crc)
    return tuple(table)


_CRC_TABLE: Final[tuple[int, ...]] = _build_crc_table()
