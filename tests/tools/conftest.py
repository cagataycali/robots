# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared serial-layer doubles for the ``pose_tool`` test modules.

These live in a conftest rather than in one test module because two modules
now need them, and importing a fixture across test modules re-binds the name
in the importing module (ruff F811) as well as making the owning module an
implicit dependency of the other.

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
