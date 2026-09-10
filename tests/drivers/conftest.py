# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A hardware-shaped Feetech servo bus, shared by the bus and driver tests.

:class:`FakeServoPort` stands in for ``serial.Serial``. It is deliberately
*shaped like the hardware* rather than a bare mock: it answers a READ with a
real status frame it builds itself (header, length, error byte, little-endian
parameters, checksum), so a driver reading through it exercises the same codec
path a physical arm does. A mock returning a canned dict would pass while the
framing was wrong, which is the one thing worth grading here.

The frame it builds is the one
:func:`strands_robots.drivers.feetech.protocol.parse_status_packet` verifies -
they are written from the same datasheet layout, and if they ever disagree the
read tests fail, which is the point.
"""

from __future__ import annotations

import pytest

from strands_robots.drivers.feetech.bus import FeetechBus


class FakeServoPort:
    """A serial port with servos behind it.

    A ``SYNC_READ`` is answered the way the hardware answers it: one status
    frame per addressed servo, in the order the frame listed them, concatenated
    into a single reply stream. A servo absent from ``counts`` contributes no
    frame, so the stream a reader has to survive is exactly the one a mute servo
    produces - the frames of its neighbours, back to back, with a gap.

    A unicast ``WRITE`` is answered too, with the empty status frame the servo
    sends to acknowledge it - the vendor SDK's ``txRxPacket`` sets a six-byte
    packet timeout after every one of them. A fake that answered only reads
    would let a write-and-forget path pass while the acks it left behind piled
    up in front of the next reader's frame.

    Args:
        counts: Motor ID -> the raw register value that motor answers with.
            A motor absent from this map answers nothing, which is how a
            missing or mute servo is simulated.
        leading_noise: Bytes to put in front of the next reply, simulating the
            host's own echo on a half-duplex bus. The parser must skip these.
        timeout: The read window, in seconds, that ``serial.Serial`` was opened
            with. A read asking for more bytes than are pending blocks for the
            whole window and then returns what arrived, which is what pyserial
            does - and the cost is recorded in :attr:`blocked_s` so a test can
            grade it. The bytes returned are the same either way.
    """

    def __init__(
        self,
        counts: dict[int, int] | None = None,
        leading_noise: bytes = b"",
        timeout: float = 1.0,
    ) -> None:
        self.counts = dict(counts or {})
        self.leading_noise = leading_noise
        self.timeout = timeout
        self.is_open = True
        #: Every frame written, in order, for a test to decode.
        self.writes: list[bytes] = []
        #: Seconds of read window waited out for bytes that never came.
        self.blocked_s = 0.0
        self._pending = b""

    # -- the serial.Serial surface the bus uses ---------------------------- #

    @property
    def in_waiting(self) -> int:
        """Bytes already buffered, as ``serial.Serial`` reports them."""
        return len(self._pending)

    def write(self, data: bytes) -> int:
        """Record the frame and queue whatever the servos answer it with."""
        self.writes.append(bytes(data))
        instruction = data[4]
        if instruction == 0x02:  # READ: the addressed servo answers
            motor_id = data[2]
            if motor_id in self.counts:
                self._pending += self.leading_noise + self._status_frame(motor_id, self.counts[motor_id])
        elif instruction == 0x03:  # WRITE: the addressed servo acks with an empty frame
            motor_id = data[2]
            if motor_id in self.counts:
                self._pending += self.leading_noise + self._frame(motor_id, b"")
        elif instruction == 0x82:  # SYNC_READ: every addressed servo answers, in order
            replies = b"".join(
                self._status_frame(motor_id, self.counts[motor_id])
                for motor_id in data[7:-1]
                if motor_id in self.counts
            )
            if replies:
                self._pending += self.leading_noise + replies
        return len(data)

    def read(self, size: int) -> bytes:
        """Return up to ``size`` buffered bytes, waiting out the window for the rest."""
        if size > len(self._pending):
            self.blocked_s += self.timeout
        taken, self._pending = self._pending[:size], self._pending[size:]
        return taken

    def close(self) -> None:
        self.is_open = False

    # -- frame construction ------------------------------------------------ #

    @staticmethod
    def _frame(motor_id: int, params: bytes) -> bytes:
        """Build one status frame from ``motor_id`` carrying ``params``.

        Empty params is the acknowledgement of a ``WRITE``; two bytes is the
        answer to a register read. One builder for both, so the ack and the
        reading cannot drift into two spellings of the same layout.
        """
        body = bytes([motor_id, len(params) + 2, 0x00]) + params
        return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])

    @classmethod
    def _status_frame(cls, motor_id: int, value: int) -> bytes:
        """Build the reply a servo sends for a two-byte register read."""
        return cls._frame(motor_id, bytes([value & 0xFF, (value >> 8) & 0xFF]))


def open_bus(port: FakeServoPort, **kwargs: object) -> FeetechBus:
    """A bus already holding ``port``, skipping the real ``connect()``.

    Reaching for ``_conn`` is how a test drives the bus without a serial stack,
    and it is spelled here once so the suites that do it do not each spell it -
    and so the day the bus grows a seam for this, one line changes.
    """
    bus = FeetechBus(port="/dev/fake", **kwargs)  # type: ignore[arg-type]
    bus._conn = port
    return bus


#: Raw counts putting every SO-arm joint at a value a test can name.
#: 0 is the low end of a joint's range, 4095 the high end, 2048 the middle.
MIDPOINT_COUNTS: dict[int, int] = dict.fromkeys((1, 2, 3, 4, 5, 6), 2048)


@pytest.fixture
def servo_port() -> FakeServoPort:
    """A six-servo SO-arm sitting at the middle of every joint's range."""
    return FakeServoPort(MIDPOINT_COUNTS)
