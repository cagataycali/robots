"""A failed real-robot connect names the device that failed and the remedy for that device.

Measured with an agent: an unplugged arm returned ``Could not connect on port
'/dev/cu.usbmodem…'. … Try running lerobot-find-port. Ensure robot is
calibrated and accessible on the specified port`` and the agent advised
"Recalibrate if needed"; a camera at an index that does not exist returned
``Failed to open OpenCVCamera(99).Run lerobot-find-cameras opencv…. Ensure
robot is calibrated and accessible on the specified port`` - lerobot's object
name, not the ``cameras=`` key the operator wrote, plus a calibration hint for
a camera fault.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from strands_robots import Robot


@pytest.fixture
def arm():
    robot = Robot(
        "so101",
        mode="real",
        port=os.devnull,
        cameras={"wrist": {"type": "opencv", "index_or_path": 99, "fps": 30, "width": 640, "height": 480}},
    )
    yield robot
    robot.cleanup()


def _connect_with(arm, exc: BaseException) -> str:
    def failing_connect(calibrate=True):
        raise exc

    arm.robot.connect = failing_connect
    ok, message = asyncio.run(arm._connect_robot())
    assert ok is False
    return message


class TestCamera:
    def test_names_the_cameras_key_and_drops_the_calibration_hint(self, arm) -> None:
        message = _connect_with(
            arm,
            ConnectionError(
                "Failed to open OpenCVCamera(99).Run `lerobot-find-cameras opencv` to find available cameras."
            ),
        )
        assert message.startswith(
            "Robot connection failed: camera 'wrist' did not open - Failed to open OpenCVCamera(99)."
        )
        assert "lerobot-find-cameras opencv" in message
        assert "Fix or remove that entry in cameras=" in message
        assert "calibrated" not in message

    def test_is_reported_in_the_task_envelope(self, arm, monkeypatch) -> None:
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        def failing_connect(calibrate=True):
            raise ConnectionError("Failed to open OpenCVCamera(99).Run `lerobot-find-cameras opencv`.")

        arm.robot.connect = failing_connect
        result = arm._execute_task_sync("wave", policy_provider="mock", duration=1)
        text = " ".join(c.get("text", "") for c in result["content"] if isinstance(c, dict))
        assert result["status"] == "error"
        assert "camera 'wrist' did not open" in text


class TestBus:
    def test_port_failure_names_the_cable_and_the_port_finders(self, arm) -> None:
        message = _connect_with(
            arm,
            ConnectionError(
                "\nCould not connect on port '/dev/cu.usbmodem5AB01818061'. Make sure you are using the correct port.\n"
                "Try running `lerobot-find-port`\n"
            ),
        )
        assert message.startswith("Robot connection failed: Could not connect on port '/dev/cu.usbmodem5AB01818061'.")
        assert "check the USB cable and power" in message
        assert "lerobot-find-port or scan_serial_devices" in message
        assert "calibrated" not in message
        assert "\n" not in message

    def test_anything_else_keeps_a_general_remedy(self, arm) -> None:
        message = _connect_with(arm, RuntimeError("motor 3 did not answer"))
        assert message == (
            "Robot connection failed: motor 3 did not answer. "
            "Ensure the robot is powered, on the right port and calibrated."
        )
