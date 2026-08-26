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
    "MODEL_NUMBER": (0, 2, "Read-only. Used by :func:`decode_model_number` for auto-detection."),
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


# ---------------------------------------------------------------------------
# Model number -> canonical model name. Read from register 0 during a probe.
# The identifiers here are the ones the Robotis e-manual and dynamixel_sdk
# use; a downstream translator maps them to lerobot's names if needed.
#
# Not every Dynamixel model is here - only the ones present on the robots
# :issue:`359` lists. Adding one is a one-line change and is deliberately
# not gated behind a class hierarchy.
# ---------------------------------------------------------------------------
MODEL_NUMBERS: Final[dict[int, str]] = {
    1060: "XL330-M077",
    1190: "XL330-M288",
    1030: "XL430-W250",
    1050: "XM430-W210",
    1120: "XM430-W350",
    1130: "XM540-W150",
    1140: "XM540-W270",
    1150: "XM540-W270-R",
    1180: "XM540-W150-R",
    1020: "XH430-W210",
    311: "MX-64",
    321: "MX-106",
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
    # length counts INST + ERR + params + CRC.
    if len(frame) != 7 + length:
        raise ValueError(
            f"parse_status_packet: frame length {len(frame)} does not match length field {length} (expected {7 + length})",
        )
    inst = frame[7]
    if inst != 0x55:
        raise ValueError(f"parse_status_packet: expected 0x55 status marker, got {inst:#04x}")
    err = frame[8]
    params = bytes(frame[9 : 9 + length - 4])
    expected_crc = checksum(frame[:-2])
    actual_crc = frame[-2] | (frame[-1] << 8)
    return {
        "servo_id": servo_id,
        "err": err,
        "params": params,
        "crc_ok": expected_crc == actual_crc,
    }


def decode_model_number(params: bytes) -> tuple[int, str | None]:
    """Decode a ``MODEL_NUMBER`` read's params.

    ``MODEL_NUMBER`` is a two-byte little-endian read at register 0. The
    driver probes this to auto-detect XL330 vs XM430 vs XM540 without user
    config, as :issue:`359` acceptance-criterion 5 requires.

    Args:
        params: The status packet's parameter bytes for a read of two bytes
            at register 0. Must be exactly two bytes.

    Returns:
        ``(model_number, canonical_name)`` where the name is ``None`` for a
        model not in :data:`MODEL_NUMBERS`. Returning the number rather than
        raising is deliberate: an unknown model is a "add a row here"
        situation, not a hardware error, and a caller can still make
        progress with the raw number.

    Raises:
        ValueError: If ``params`` is not exactly two bytes.
    """
    if len(params) != 2:
        raise ValueError(f"decode_model_number: expected 2 bytes, got {len(params)}")
    number = params[0] | (params[1] << 8)
    return number, MODEL_NUMBERS.get(number)


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
