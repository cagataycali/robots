# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared serial-layer doubles for the ``pose_tool`` test modules.

These live in a conftest rather than in one test module because several
modules now need them, and importing a fixture across test modules re-binds
the name in the importing module (ruff F811) as well as making the owning
module an implicit dependency of the other.

``pose_tool``'s ``port`` defaults to ``/dev/ttyACM0`` and several actions --
``emergency_stop`` above all -- write to the bus, so every test that reaches
the motor path must both take ``fake_serial`` and pass an explicit fake
``port``. Otherwise the suite drives whatever arm is plugged into the machine
running it.
"""

from __future__ import annotations

import pytest
import serial


class FakeSerial:
    """Minimal stand-in for ``serial.Serial`` recording writes and serving reads."""

    def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.writes: list[bytes] = []
        self._read_queue: list[bytes] = []
        self.is_open = True

    def queue_read(self, data: bytes) -> None:
        self._read_queue.append(data)

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))

    def read(self, n: int = 1) -> bytes:
        if self._read_queue:
            return self._read_queue.pop(0)
        return b""

    def close(self) -> None:
        self.is_open = False


def position_packet(raw: int = 0x0800, motor_id: int = 0x01) -> bytes:
    """A Feetech status packet reporting ``raw`` counts for ``motor_id``.

    ``FF FF ID LEN ERR <lo> <hi> CHK``, with the checksum a servo would send so
    the frame passes the verification ``read_motor_position`` performs. Framing
    itself is graded in ``test_feetech_status_packet_framing``.
    """
    body = [motor_id, 0x04, 0x00, raw & 0xFF, (raw >> 8) & 0xFF]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


class ReadingSerial(FakeSerial):
    """A ``FakeSerial`` that always answers a read with a decodable position.

    ``_smooth_move`` reads the current pose before interpolating and builds a
    trajectory only for the motors it could read, so a source that never
    answers makes the interpolation vacuous -- and, until the interpolating path
    reported the joints it had dropped, made a move that wrote nothing look
    like a success.
    """

    def read(self, n: int = 1) -> bytes:
        # Answer as the motor the outgoing packet addressed; a servo bus does,
        # and a fake that always answered as motor 1 would let a read attribute
        # one motor's position to another with no test able to see it.
        asked = self.writes[-1][2] if self.writes else 0x01
        return position_packet(motor_id=asked)


@pytest.fixture
def reading_serial(monkeypatch):
    """Patch ``serial.Serial`` with an always-answering position source."""
    instances: list[ReadingSerial] = []

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> ReadingSerial:
        fs = ReadingSerial(port, baudrate, timeout)
        instances.append(fs)
        return fs

    monkeypatch.setattr(serial, "Serial", _ctor)
    return instances


@pytest.fixture
def fake_serial(monkeypatch):
    """Patch ``serial.Serial`` to return a single shared FakeSerial instance."""
    instances: list[FakeSerial] = []

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> FakeSerial:
        fs = FakeSerial(port, baudrate, timeout)
        instances.append(fs)
        return fs

    monkeypatch.setattr(serial, "Serial", _ctor)
    return instances


@pytest.fixture
def cwd_tmp(tmp_path, monkeypatch):
    """Run with cwd in a temp dir so PoseManager persists under tmp."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
