# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""``pose_tool(action="emergency_stop")`` must actually de-energize the arm.

The handler used to be:

    if action == "emergency_stop":
        # This would require torque disable in real implementation
        return {"status": "success",
                "content": [{"text": "Emergency stop executed (torque disabled)"}]}

i.e. it executed NO code while reporting that torque had been disabled. That is a
fabricated safety confirmation, the worst failure mode a safety path can have: an
operator - or an agent reading the tool result - would believe a moving arm had
been released when nothing was sent to it.

It now writes ``Torque_Enable = 0`` (register address 40 / 0x28, 1 byte on the
Feetech STS3215 - authority: ``lerobot.motors.feetech.tables``) to every
configured motor, and reports an error listing any motor it could not release.

No serial port is opened: a fake connection captures the bytes so the wire
protocol itself is asserted.
"""

from __future__ import annotations

from strands_robots.tools.pose_tool import MotorController, pose_tool

_TORQUE_ENABLE_ADDR = 0x28  # 40, per lerobot.motors.feetech.tables
_INST_WRITE = 0x03


class _FakeSerial:
    """Captures written packets instead of touching a serial device."""

    def __init__(self, *, fail_on_write: bool = False) -> None:
        self.is_open = True
        self.written: list[bytes] = []
        self._fail_on_write = fail_on_write

    def write(self, data: bytes) -> int:
        if self._fail_on_write:
            raise OSError("simulated bus failure")
        self.written.append(bytes(data))
        return len(data)

    def read(self, n: int = 1) -> bytes:
        return b""

    def close(self) -> None:
        self.is_open = False


def _controller(fake: _FakeSerial) -> MotorController:
    controller = MotorController(port="/dev/null")
    controller.serial_conn = fake
    return controller


def _decode(packet: bytes) -> dict:
    """Unpack a Feetech packet: FF FF id len inst params... checksum."""
    return {
        "motor_id": packet[2],
        "instruction": packet[4],
        "address": packet[5],
        "value": packet[6],
    }


class TestDisableTorqueWritesTheRightBytes:
    def test_every_motor_receives_torque_enable_zero(self):
        fake = _FakeSerial()
        controller = _controller(fake)

        failed = controller.disable_torque()

        assert failed == []
        assert len(fake.written) == len(controller.motor_configs)
        expected_ids = {c["id"] for c in controller.motor_configs.values()}
        seen_ids = set()
        for packet in fake.written:
            decoded = _decode(packet)
            assert decoded["instruction"] == _INST_WRITE
            assert decoded["address"] == _TORQUE_ENABLE_ADDR
            assert decoded["value"] == 0
            seen_ids.add(decoded["motor_id"])
        assert seen_ids == expected_ids

    def test_packet_checksum_is_valid(self):
        """A malformed packet would be ignored by the servo - a silent non-stop."""
        fake = _FakeSerial()

        _controller(fake).disable_torque()

        for packet in fake.written:
            assert packet[0:2] == b"\xff\xff"
            body = packet[2:-1]
            assert packet[-1] == (~sum(body) & 0xFF)

    def test_reports_all_motors_when_not_connected(self):
        controller = MotorController(port="/dev/null")  # serial_conn stays None

        assert controller.disable_torque() == list(controller.motor_configs)

    def test_reports_every_motor_that_failed(self):
        fake = _FakeSerial(fail_on_write=True)
        controller = _controller(fake)

        failed = controller.disable_torque()

        # Every motor is attempted even after the first failure: giving up early
        # would leave joints energized while reporting only one name.
        assert sorted(failed) == sorted(controller.motor_configs)


class TestEmergencyStopActionIsHonest:
    def test_success_only_when_the_writes_happened(self, monkeypatch):
        fake = _FakeSerial()

        def _fake_connect(self):
            self.serial_conn = fake
            return True, ""

        monkeypatch.setattr(MotorController, "connect", _fake_connect)

        result = pose_tool(action="emergency_stop", port="/dev/null")

        assert result["status"] == "success", result
        # The bytes really went out.
        assert len(fake.written) == 6
        payload = next(b["json"] for b in result["content"] if "json" in b)
        assert payload["torque_disabled"] is True
        text = result["content"][0]["text"]
        # The consequence must be stated: a de-energized arm drops its payload.
        assert "de-energized" in text
        assert "NOT holding position" in text

    def test_connect_failure_reports_error_not_success(self, monkeypatch):
        """Pre-fix this returned success even with no arm attached at all."""
        monkeypatch.setattr(MotorController, "connect", lambda self: (False, "no such port"))

        result = pose_tool(action="emergency_stop", port="/dev/null")

        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "NOT de-energized" in text
        assert "hardware power cutoff" in text

    def test_partial_failure_reports_error_and_names_the_motors(self, monkeypatch):
        fake = _FakeSerial(fail_on_write=True)

        def _fake_connect(self):
            self.serial_conn = fake
            return True, ""

        monkeypatch.setattr(MotorController, "connect", _fake_connect)

        result = pose_tool(action="emergency_stop", port="/dev/null")

        assert result["status"] == "error"
        payload = next(b["json"] for b in result["content"] if "json" in b)
        assert payload["torque_disabled"] is False
        assert payload["failed_motors"], "the un-released motors must be named"
        assert "still enabled" in result["content"][0]["text"]

    def test_result_text_is_plain_ascii(self, monkeypatch):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        fake = _FakeSerial()

        def _fake_connect(self):
            self.serial_conn = fake
            return True, ""

        monkeypatch.setattr(MotorController, "connect", _fake_connect)

        result = pose_tool(action="emergency_stop", port="/dev/null")

        for block in result["content"]:
            if "text" in block:
                assert block["text"].isascii(), block["text"]

    def test_serial_port_is_released_afterwards(self, monkeypatch):
        """A held port would block the next connect - including a retry of the stop."""
        fake = _FakeSerial()

        def _fake_connect(self):
            self.serial_conn = fake
            return True, ""

        monkeypatch.setattr(MotorController, "connect", _fake_connect)

        pose_tool(action="emergency_stop", port="/dev/null")

        assert fake.is_open is False
