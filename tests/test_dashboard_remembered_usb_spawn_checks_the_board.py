"""Spawning a remembered USB profile checks the board it remembered, and never rebinds it.

``POST /devices/spawn-remembered`` found the profile by the board's self-reported serial alone and
spawned it as that real robot, then re-saved the profile with whatever chip had just confirmed, so a
board reporting a cloned serial on a different chip came up under the remembered peer id and
calibration after one click and became the board on file. Now the chip (``vid:pid``) and the USB bus
location the profile recorded must match the live scan, a mismatch is a 409 naming both, a spawn
never rewrites them, and the watcher holds such a board (allowlisted or not) with both side by side.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import device_manager, routes_devices
from strands_robots.dashboard.device_manager import AUTOSPAWN_REAL_ALLOWLIST_ENV, AutoSpawnWatcher, DeviceManager

SERIAL = "S1"
GENUINE: dict[str, Any] = {
    "device": "/dev/ttyACM9",
    "serial_number": SERIAL,
    "vid": "1a86",
    "pid": "7523",
    "location": "1-2.3",
}
PROFILE: dict[str, Any] = {
    "peer_id": "arm-a",
    "robot_name": "so101",
    "mode": "real",
    "robot_id": "arm_a_follower",
    "port": "/dev/ttyACM9",
    "usb": "1a86:7523",
    "location": "1-2.3",
}
OTHER_CHIP: dict[str, Any] = {**GENUINE, "vid": "0403", "pid": "6001"}
OTHER_SOCKET: dict[str, Any] = {**GENUINE, "location": "1-2.4"}
NO_LOCATION: dict[str, Any] = {**GENUINE, "location": None}


class _Manager(DeviceManager):
    """A DeviceManager whose spawn records (and remembers, like the real one) instead of forking."""

    def __init__(self, tmp_path: Any) -> None:
        super().__init__(profiles_path=str(tmp_path / "profiles.json"))
        self.spawns: list[dict[str, Any]] = []

    def spawn(
        self, robot_name: str, mode: str = "sim", peer_id: str | None = None, *args: Any, **kw: Any
    ) -> dict[str, Any]:  # type: ignore[override]
        defaults: list[Any] = [None, None, None, True]  # port, cameras, robot_id, remember
        port, _cameras, _robot_id, remember = [*args, *defaults[len(args) :]]
        self.spawns.append({"peer_id": peer_id, "port": port, "remember": remember})
        if remember:
            self.remember_profile({"robot_name": robot_name, "mode": mode, "peer_id": peer_id, "port": port})
        return {"peer_id": peer_id, "pid": 1, "mode": mode}


class _Bridge:
    peers: dict[str, Any] = {}

    def __init__(self) -> None:
        self.trail: list[dict[str, Any]] = []

    def record_activity(self, source: str, action: str, **kw: Any) -> None:
        self.trail.append({"action": action, **kw})


@pytest.fixture
def setup(tmp_path: Any, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(AUTOSPAWN_REAL_ALLOWLIST_ENV, raising=False)
    monkeypatch.setenv("STRANDS_DASHBOARD_AUTOSPAWN", "1")
    monkeypatch.setattr(routes_devices.env_install, "spawn_preflight", lambda *a, **k: None)
    dm = _Manager(tmp_path)
    dm.profiles.save(SERIAL, dict(PROFILE))
    bridge = _Bridge()
    request: Any = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(devices=dm, bridge=bridge)))

    def call(board: dict[str, Any], **body: Any) -> dict[str, Any]:
        monkeypatch.setattr(device_manager, "scan_serial_ports", lambda: [dict(board)])
        return asyncio.run(routes_devices.spawn_remembered(request, {"port": board["device"], **body}, {}))

    return dm, bridge, call


@pytest.mark.parametrize(
    ("board", "named"),
    [
        (OTHER_CHIP, ["1a86:7523", "0403:6001"]),
        (OTHER_SOCKET, ["1-2.3", "1-2.4"]),
        (NO_LOCATION, ["1-2.3", "cannot read"]),
    ],
    ids=["different-chip", "different-socket", "location-unreadable"],
)
def test_a_board_that_differs_from_the_remembered_one_is_refused(setup, board, named) -> None:
    dm, bridge, call = setup
    with pytest.raises(HTTPException) as refused:
        call(board)
    assert refused.value.status_code == 409
    detail: Any = refused.value.detail
    assert all(word in detail["error"] for word in named)
    assert detail["remembered"] == {"usb": "1a86:7523", "location": "1-2.3"}
    assert dm.spawns == [], "a real arm was started on a board that is not the one remembered"
    assert bridge.trail[-1]["ok"] is False
    assert dm.profiles.get(SERIAL)["usb"] == "1a86:7523"


def test_the_genuine_board_spawns_without_rewriting_its_profile(setup) -> None:
    dm, _, call = setup
    out = call(GENUINE)
    assert out["peer_id"] == "arm-a" and "error" not in out
    assert dm.spawns[0]["remember"] is False
    assert "board_recorded" not in out


def test_a_spawn_never_rebinds_the_remembered_chip(setup, monkeypatch: pytest.MonkeyPatch) -> None:
    """Any spawn that re-saves the profile (the plain form, a camera change) keeps the board on file."""
    dm, _, _ = setup
    monkeypatch.setattr(device_manager, "scan_serial_ports", lambda: [dict(OTHER_CHIP)])
    dm.remember_profile({**PROFILE, "usb": None, "location": None})
    saved = dm.profiles.get(SERIAL)
    assert (saved["usb"], saved["location"]) == ("1a86:7523", "1-2.3")


def test_a_profile_without_a_recorded_board_records_it_once_and_says_so(setup) -> None:
    dm, _, call = setup
    dm.profiles._data[SERIAL] = {k: v for k, v in PROFILE.items() if k not in ("usb", "location")}
    out = call(GENUINE)
    assert out["board_recorded"] == {"usb": "1a86:7523", "location": "1-2.3"}
    assert dm.spawns[0]["remember"] is True
    with pytest.raises(HTTPException):
        call(OTHER_CHIP)


def test_accepting_a_different_board_is_audited_and_rebinds_the_profile(setup) -> None:
    dm, bridge, call = setup
    out = call(OTHER_SOCKET, accept_different_board=True)
    assert out["board_rebound"] == {"usb": "1a86:7523", "location": "1-2.4"}
    assert dm.profiles.get(SERIAL)["location"] == "1-2.4"
    accepted = [e for e in bridge.trail if e["action"] == "spawn_accept_different_board"]
    assert accepted and "1-2.4" in accepted[0]["detail"]


@pytest.mark.parametrize("allowlisted", [False, True], ids=["proposal-path", "allowlist-path"])
def test_the_watcher_holds_a_different_board_and_shows_both(setup, monkeypatch, allowlisted) -> None:
    dm, _, _ = setup
    if allowlisted:
        monkeypatch.setenv(AUTOSPAWN_REAL_ALLOWLIST_ENV, SERIAL)
    did = AutoSpawnWatcher(dm, list_ports=lambda: [dict(OTHER_CHIP)], peer_ids=lambda: ()).poll()
    assert dm.spawns == [] and did["proposed"] == []
    (held,) = did["held"]
    assert (held["usb"], held["remembered_usb"]) == ("0403:6001", "1a86:7523")
    assert (held["location"], held["remembered_location"]) == ("1-2.3", "1-2.3")
    assert "different board" in held["reason"] and "0403:6001" in held["reason"]


def test_the_scan_reports_the_usb_location(monkeypatch: pytest.MonkeyPatch) -> None:
    import serial.tools.list_ports

    port = SimpleNamespace(
        device="/dev/ttyACM9", description="USB Serial", vid=0x1A86, pid=0x7523, serial_number=SERIAL, location="1-2.3"
    )
    monkeypatch.setattr(serial.tools.list_ports, "comports", lambda: [port])
    monkeypatch.setattr("strands_robots._serial_discovery.matches_servo_bus", lambda p: True)
    (entry,) = device_manager.scan_serial_ports()
    assert entry["location"] == "1-2.3"
