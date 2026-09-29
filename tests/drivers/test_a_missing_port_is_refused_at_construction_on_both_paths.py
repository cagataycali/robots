"""A serial arm with no usable ``port=`` is refused when ``Robot()`` is called, on both paths.

Before: the lerobot path refused an omitted ``port`` at construction (naming this
host's serial devices) but accepted ``port=""`` and called the empty string a
network port at the first bus action (#4168); the native path returned a
``FeetechDriver(port=None)`` that refused only at the first bus action, in other
words (#4152). Now both paths refuse ``None``, a blank string and a non-string
port with the same sentence, at the same moment.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from strands_robots import Robot
from strands_robots.drivers.feetech import FeetechDriver
from strands_robots.hardware_robot import _is_blank_port

_MISSING = r"missing required parameter\(s\) \['port'\]"


class TestTheBlankPortRule:
    @pytest.mark.parametrize("value", ["", "   ", None, 3, b"/dev/ttyACM0"])
    def test_a_value_no_bus_can_open_is_missing(self, value: object) -> None:
        assert _is_blank_port("port", value) is True
        assert _is_blank_port("serial_port", value) is True

    def test_a_device_path_is_supplied(self) -> None:
        assert _is_blank_port("port", "/dev/ttyACM0") is False

    def test_only_port_shaped_fields_are_graded(self) -> None:
        assert _is_blank_port("remote_ip", "") is False


@pytest.mark.parametrize("kwargs", [{}, {"port": ""}, {"port": "  "}])
def test_the_native_driver_refuses_at_construction(kwargs: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("strands_robots.robot.scan_serial_devices", lambda: [])
    with pytest.raises(ValueError, match=_MISSING) as info:
        Robot("so101", mode="real", driver="strands", mesh=False, **kwargs)
    assert "driver='strands', port=..." in str(info.value)
    assert "usb id" in str(info.value)


def test_the_twin_transport_needs_no_port() -> None:
    pytest.importorskip("mujoco")
    arm = Robot("so101", mode="real", driver="strands", transport="twin", mesh=False)
    try:
        assert isinstance(arm, FeetechDriver)
        assert arm.transport == "twin"
    finally:
        asyncio.run(arm.stop())


def test_the_lerobot_path_refuses_a_blank_port_like_an_omitted_one(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("lerobot")
    monkeypatch.setattr("strands_robots.hardware_robot.scan_serial_devices", lambda: [])
    messages = []
    cases: tuple[dict[str, Any], ...] = ({}, {"port": ""})
    for kwargs in cases:
        with pytest.raises(ValueError, match=_MISSING) as info:
            Robot("so101", mode="real", driver="lerobot", mesh=False, **kwargs)
        messages.append(str(info.value).split(" Config:")[0])
    assert messages[0] == messages[1], "the blank port must get the omitted port's sentence"
