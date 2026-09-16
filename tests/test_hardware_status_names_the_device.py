"""The real robot's ``status`` names the device it would drive, not only the task.

``Robot("so101", mode="real", port=...)`` connects lazily, on the first task.
Until then the tool's ``status`` action read ``Robot Status: IDLE`` and nothing
else - byte-identical for an arm whose port does not exist on this host and a
connected arm at rest - and the Python-side ``get_status()`` probe raised
through lerobot's ``is_calibrated`` (a bus read that refuses before
``connect()``), degrading every idle arm to the ``{"error": ..., "task_status":
"error"}`` shape.

The stand-ins here mirror lerobot's contract: ``is_calibrated`` raises while
disconnected, the config carries ``port`` and ``cameras``. A port is not always
a device path, so the shipped ``Reachy2RobotConfig`` is used as-is for the
network case rather than imitated.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from lerobot.robots.reachy2 import Reachy2RobotConfig

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState
from tests._daemon_executor import DaemonThreadExecutor


class _NotConnected(Exception):
    pass


class _Camera:
    def __init__(self, connected: bool) -> None:
        self.is_connected = connected


class _Arm:
    """A lerobot-shaped arm: ``is_calibrated`` refuses until connected."""

    def __init__(self, *, port: str | None, connected: bool = False, cameras: dict[str, bool] | None = None) -> None:
        self.name = "so101"
        self.robot_type = "so_follower"
        self._connected = connected
        self.config = type("Cfg", (), {"port": port, "cameras": dict.fromkeys(cameras or {}, object())})()
        self.cameras = {n: _Camera(c) for n, c in (cameras or {}).items()}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        if not self._connected:
            raise _NotConnected("FeetechMotorsBus is not connected. Run `.connect()` first.")
        return True


class _Device:
    """An arm built around a config as given - including a non-path port."""

    def __init__(self, config: Any) -> None:
        self.name = "device"
        self.robot_type = "device"
        self.is_connected = False
        self.config = config
        self.cameras: dict[str, _Camera] = {}


def _hw(arm: _Arm | _Device) -> HwRobot:
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "so101"
    hw.data_config = None
    hw._task_state = RobotTaskState()
    hw._executor = DaemonThreadExecutor(max_workers=1, thread_name_prefix="t")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = arm
    return hw


def _text(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


class TestTheToolStatus:
    def test_an_absent_port_is_named_under_the_task_state(self, tmp_path):
        hw = _hw(_Arm(port=str(tmp_path / "cu.usbmodem-gone"), cameras={"top": False}))
        result = hw.get_task_status()
        text = _text(result)
        assert result["status"] == "success"
        assert text.startswith("Robot Status: IDLE\n")  # the first line is what it always was
        assert "Device: not connected (the bus is opened by the first task)" in text
        assert f"Port: {tmp_path / 'cu.usbmodem-gone'} is not present on this host" in text
        assert "the first task will fail to connect" in text
        assert "Cameras: top (not connected)" in text
        facts = result["content"][1]["json"]
        assert facts["port_present"] is False
        assert facts["is_connected"] is False
        assert facts["is_calibrated"] is None  # not readable before connect, and not a raise

    def test_a_present_port_reads_as_present(self, tmp_path):
        port = tmp_path / "cu.usbmodem-here"
        port.write_bytes(b"")
        text = _text(_hw(_Arm(port=str(port))).get_task_status())
        assert f"Port: {port} is present on this host" in text
        assert "Cameras: none configured" in text

    def test_a_connected_arm_reports_the_port_and_calibration(self, tmp_path):
        port = tmp_path / "cu.usbmodem-here"
        port.write_bytes(b"")
        text = _text(_hw(_Arm(port=str(port), connected=True, cameras={"top": True})).get_task_status())
        assert f"Device: connected on {port} (calibrated)" in text
        assert "Cameras: top (connected)" in text

    def test_a_driver_without_a_port_has_no_port_line(self):
        result = _hw(_Arm(port=None)).get_task_status()
        assert "Port:" not in _text(result)
        assert result["content"][1]["json"]["port"] is None
        assert result["content"][1]["json"]["port_present"] is None

    def test_the_task_state_still_leads_when_the_device_cannot_answer(self):
        hw = _hw(_Arm(port=None))
        hw.robot = None  # nothing to read facts from
        result = hw.get_task_status()
        assert _text(result).startswith("Robot Status: IDLE\n")
        assert result["status"] == "success"


class TestAPortThatIsNotAPath:
    """``port_present`` is three-state, and the text may not flatten it.

    lerobot's network drivers carry a TCP port, not a device path: nothing on
    this host is stat-ed for one, so ``port_present`` stays ``None``. Reporting
    it as present would be the very defect ``status`` exists to fix - a
    sentence an agent cannot tell apart from a reading.
    """

    def test_a_network_port_is_not_read_as_present_on_this_host(self):
        config = Reachy2RobotConfig()  # the shipped config: port is a TCP port, reached at ip_address
        assert isinstance(config.port, int)  # the shape is lerobot's, not this test's
        result = _hw(_Device(config)).get_task_status()
        text, facts = _text(result), result["content"][1]["json"]
        assert facts["port"] == config.port
        assert facts["port_present"] is None  # nothing on this host was stat-ed
        assert "present on this host" not in text  # so presence is claimed neither way
        assert f"Port: {config.port} is a network port, reached at {config.ip_address}, not a device path" in text
        assert "this host has no such path to check" in text

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("ip_address", "reachy.local", "reachy.local"),  # Reachy 2
            ("remote_ip", "192.168.1.9", "192.168.1.9"),  # LeKiwi client
            ("robot_ip", "192.168.123.161", "192.168.123.161"),  # Unitree G1
            ("host", "kiwi.local", "kiwi.local"),
            ("host", object(), None),  # a host that is not an address names nothing
            ("ip_address", "", None),
        ],
    )
    def test_the_address_a_network_port_is_reached_at_is_named(self, field, value, expected):
        config = type("Cfg", (), {"port": 50065, "cameras": {}, field: value})()
        facts = _hw(_Device(config))._device_facts()
        assert facts["address"] == expected
        text = HwRobot._device_lines(facts)
        assert (f"reached at {expected}" in text) is (expected is not None)


class TestThePythonProbe:
    def test_an_idle_arm_is_not_an_error(self, tmp_path):
        hw = _hw(_Arm(port=str(tmp_path / "gone"), cameras={"top": False}))
        status = asyncio.run(hw.get_status())
        assert "error" not in status, status
        assert status["task_status"] == "idle"
        assert status["is_connected"] is False
        assert status["is_calibrated"] is None
        assert status["port_present"] is False
        assert status["cameras"] == ["top"]
        assert status["cameras_connected"] == {"top": False}

    def test_a_connected_arm_reads_calibration(self, tmp_path):
        status = asyncio.run(_hw(_Arm(port=str(tmp_path), connected=True)).get_status())
        assert status["is_calibrated"] is True
        assert status["port_present"] is True


@pytest.mark.parametrize(
    "attr",
    ["port", "port_present", "address", "is_connected", "is_calibrated", "cameras", "cameras_connected"],
)
def test_the_tool_and_the_probe_carry_the_same_facts(attr, tmp_path):
    hw = _hw(_Arm(port=str(tmp_path / "gone"), cameras={"top": False}))
    assert hw.get_task_status()["content"][1]["json"][attr] == asyncio.run(hw.get_status())[attr]
