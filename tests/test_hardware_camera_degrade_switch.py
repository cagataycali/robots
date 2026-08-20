"""The camera-degrade divergence is switchable, and BOTH contracts are pinned.

``_degrade_to_available_cameras`` makes a camera that will not open cost the
camera instead of the whole arm. That is deliberate for this fleet: on a Mac
where macOS never granted camera access, the strict upstream behaviour means no
arm can be brought up at all. But it answers the same failure that upstream's
rollback contract answers (tests/test_hardware_camera_rollback.py), so exactly
one of them can be in force per process - and the choice has to be visible.

These tests pin the switch itself: default degrades, ``STRANDS_ROBOT_CAMERA_DEGRADE=0``
restores the strict path, and the flag is read at CALL time so a test (or an
operator debugging one arm) can change it without reimporting the module.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.hardware_robot import _degrade_to_available_cameras


class _Cam:
    def __init__(self, name: str, *, fails: bool = False) -> None:
        self.name = name
        self.fails = fails
        self.is_connected = False

    def connect(self) -> None:
        if self.fails:
            raise RuntimeError(f"{self.name}: could not open")
        self.is_connected = True


class _Bus:
    is_connected = True


class _Config:
    def __init__(self, cameras: dict[str, Any]) -> None:
        self.cameras = cameras


class _Robot:
    """Mirrors the two properties the degrade path actually consults.

    ``is_connected`` is lerobot's own definition - bus plus every REMAINING
    camera - which is the reason the degrade can succeed at all: dropping the
    blind camera is what lets that expression describe the motors.
    """

    def __init__(self) -> None:
        self.bus = _Bus()
        self.cameras: dict[str, Any] = {"wrist": _Cam("wrist_cam"), "top": _Cam("top_cam", fails=True)}
        self.config = _Config(dict(self.cameras))

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(c.is_connected for c in self.cameras.values())


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " 0 "])
def test_the_strict_contract_is_reachable_by_env(monkeypatch, value: str) -> None:
    """With the switch off nothing is dropped, so the caller's error stands.

    Returning ``{}`` is what makes ``_connect_robot`` re-raise and roll every
    device back - the behaviour the upstream rollback tests describe.
    """
    monkeypatch.setenv("STRANDS_ROBOT_CAMERA_DEGRADE", value)
    robot = _Robot()

    assert _degrade_to_available_cameras(robot) == {}
    assert set(robot.cameras) == {"wrist", "top"}, "a refused degrade must not mutate the camera set"


def test_the_default_degrades(monkeypatch) -> None:
    """Unset means degrade: this fleet's arms must survive a blind camera."""
    monkeypatch.delenv("STRANDS_ROBOT_CAMERA_DEGRADE", raising=False)
    robot = _Robot()

    dropped = _degrade_to_available_cameras(robot)

    assert set(dropped) == {"top"}
    assert "could not open" in dropped["top"], "the reason must name the real fault"
    assert set(robot.cameras) == {"wrist"}
    assert set(robot.config.cameras) == {"wrist"}, "config drives status/dataset listings"


def test_the_flag_is_read_at_call_time(monkeypatch) -> None:
    """Import-time reads would make the switch untestable and unusable live."""
    monkeypatch.setenv("STRANDS_ROBOT_CAMERA_DEGRADE", "0")
    assert _degrade_to_available_cameras(_Robot()) == {}
    monkeypatch.setenv("STRANDS_ROBOT_CAMERA_DEGRADE", "1")
    assert set(_degrade_to_available_cameras(_Robot())) == {"top"}
