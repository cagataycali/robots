"""A USB board that says a remembered serial is proposed to the operator, not adopted as that robot.

The auto-spawn watcher matched a plugged-in board to a saved profile on the board's self-reported
USB serial number alone, then started a ``mode=real`` child under the remembered peer id with the
remembered calibration, with nobody asked; a board that reported no serial was keyed on its ``/dev``
path instead. A serial is a string a device chooses, so a device an attacker controls could be
adopted as a specific real robot, hold its name on the mesh, and make the genuine arm look absent
when it arrived afterwards (f014, CWE-290).

Now a real-mode match is a proposal: it lands in the poll result and the activity trail with what
was seen (serial, vid:pid, path) and what it would become, and the spawn waits for the operator's
``spawn-remembered`` click, or for the serial to be on the allowlist the operator wrote in
``STRANDS_DASHBOARD_AUTOSPAWN_REAL_SERIALS`` with a vid:pid that matches what was remembered.
Sim profiles keep coming up on their own. No serial, or two boards with one serial, means no
spawn and an entry that says so.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.dashboard import device_manager
from strands_robots.dashboard.device_manager import (
    AUTOSPAWN_REAL_ALLOWLIST_ENV,
    AutoSpawnWatcher,
    DeviceManager,
    profile_key,
)

SERIAL = "5A7F00E1"
BOARD = {
    "device": "/dev/ttyACM0",
    "serial_number": SERIAL,
    "vid": "1a86",
    "pid": "55d3",
    "description": "USB Single Serial",
}
REAL_PROFILE = {
    "peer_id": "arm-left",
    "robot_name": "so101",
    "mode": "real",
    "robot_id": "left_follower",
    "port": "/dev/ttyACM0",
}
SIM_PROFILE = {"peer_id": "bench-sim", "robot_name": "so101", "mode": "sim"}


def test_a_board_without_a_serial_has_no_profile_identity() -> None:
    assert profile_key({"device": "/dev/ttyACM0"}) == ""
    assert profile_key({"device": "/dev/ttyACM0", "serial_number": ""}) == ""
    assert profile_key(BOARD) == SERIAL


class _Manager(DeviceManager):
    """A DeviceManager whose spawn records instead of forking."""

    def __init__(self, tmp_path: Any) -> None:
        super().__init__(profiles_path=str(tmp_path / "profiles.json"))
        self.spawns: list[dict[str, Any]] = []

    def spawn(self, robot_name: str, mode: str = "sim", peer_id: str | None = None, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        self.spawns.append({"robot_name": robot_name, "mode": mode, "peer_id": peer_id, **kwargs})
        return {"peer_id": peer_id, "pid": 1, "mode": mode}


@pytest.fixture
def manager(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> _Manager:
    monkeypatch.setenv("STRANDS_DASHBOARD_AUTOSPAWN", "1")
    monkeypatch.delenv(AUTOSPAWN_REAL_ALLOWLIST_ENV, raising=False)
    return _Manager(tmp_path)


def _watcher(manager: _Manager, *ports: dict[str, Any], peers: tuple[str, ...] = ()) -> AutoSpawnWatcher:
    return AutoSpawnWatcher(manager, list_ports=lambda: [dict(p) for p in ports], peer_ids=lambda: peers)


def test_a_real_profile_match_is_proposed_not_spawned(manager: _Manager) -> None:
    manager.profiles.save(SERIAL, REAL_PROFILE)
    did = _watcher(manager, BOARD).poll()
    assert manager.spawns == [], "real hardware must not start from a hotplug event alone"
    assert did["spawned"] == []
    (proposal,) = did["proposed"]
    assert proposal["serial"] == SERIAL
    assert proposal["device"] == "/dev/ttyACM0"
    assert proposal["usb"] == "1a86:55d3"
    assert proposal["peer_id"] == "arm-left"
    assert "confirm" in proposal["reason"] and AUTOSPAWN_REAL_ALLOWLIST_ENV in proposal["reason"]


def test_a_proposal_is_announced_once_and_kept_while_the_board_stays(manager: _Manager) -> None:
    manager.profiles.save(SERIAL, REAL_PROFILE)
    w = _watcher(manager, BOARD)
    assert len(w.poll()["proposed"]) == 1
    assert w.poll()["proposed"] == [], "the trail must not repeat the proposal every two seconds"
    assert SERIAL in w.pending
    gone = _watcher(manager)
    gone.pending = dict(w.pending)
    gone.poll()
    assert gone.pending == {}, "an unplugged proposal is withdrawn"


def test_a_sim_profile_still_comes_up_on_its_own(manager: _Manager) -> None:
    manager.profiles.save(SERIAL, SIM_PROFILE)
    did = _watcher(manager, BOARD).poll()
    assert did["spawned"] == ["bench-sim"]
    assert did["proposed"] == []
    assert manager.spawns[0]["mode"] == "sim"


def test_an_allowlisted_serial_with_the_remembered_usb_identity_spawns(
    manager: _Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager.profiles.save(SERIAL, {**REAL_PROFILE, "usb": "1a86:55d3"})
    monkeypatch.setenv(AUTOSPAWN_REAL_ALLOWLIST_ENV, f"OTHER, {SERIAL}")
    did = _watcher(manager, BOARD).poll()
    assert did["spawned"] == ["arm-left"]
    assert manager.spawns[0]["mode"] == "real"
    assert manager.spawns[0]["remember"] is False


def test_an_allowlisted_serial_on_a_different_chip_is_held(manager: _Manager, monkeypatch: pytest.MonkeyPatch) -> None:
    manager.profiles.save(SERIAL, {**REAL_PROFILE, "usb": "0403:6001"})
    monkeypatch.setenv(AUTOSPAWN_REAL_ALLOWLIST_ENV, SERIAL)
    did = _watcher(manager, BOARD).poll()
    assert manager.spawns == []
    assert did["spawned"] == []
    (held,) = did["held"]
    assert held["serial"] == SERIAL
    assert "0403:6001" in held["reason"] and "1a86:55d3" in held["reason"]


def test_an_allowlisted_serial_with_no_remembered_usb_identity_is_proposed(
    manager: _Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old profile that never recorded its chip cannot be matched on it; the operator confirms once."""
    manager.profiles.save(SERIAL, REAL_PROFILE)
    monkeypatch.setenv(AUTOSPAWN_REAL_ALLOWLIST_ENV, SERIAL)
    did = _watcher(manager, BOARD).poll()
    assert manager.spawns == []
    assert len(did["proposed"]) == 1
    assert "usb" in did["proposed"][0]["reason"]


def test_two_boards_with_one_serial_spawn_nothing(manager: _Manager) -> None:
    manager.profiles.save(SERIAL, SIM_PROFILE)
    twin = {**BOARD, "device": "/dev/ttyACM1"}
    did = _watcher(manager, BOARD, twin).poll()
    assert manager.spawns == []
    (held,) = did["held"]
    assert held["serial"] == SERIAL
    assert "/dev/ttyACM0" in held["reason"] and "/dev/ttyACM1" in held["reason"]


def test_a_board_without_a_serial_is_reported_not_matched(manager: _Manager) -> None:
    manager.profiles.save("/dev/ttyACM0", SIM_PROFILE)  # a key an older store may still hold
    did = _watcher(manager, {"device": "/dev/ttyACM0", "vid": "1a86", "pid": "55d3"}).poll()
    assert manager.spawns == []
    assert did["unidentified"] == ["/dev/ttyACM0"]


def test_a_board_without_a_serial_is_reported_once_while_it_stays_plugged_in(manager: _Manager) -> None:
    """A no-serial board is a steady state (CH34x adapters), not a two-second drumbeat.

    The trail is a bounded deque; one entry per poll would rotate the proposal
    and hold evidence out of it. The path is announced when it appears, kept on
    the watcher while it stays, and announced again only after it left.
    """
    no_serial = {"device": "/dev/ttyACM0", "vid": "1a86", "pid": "55d3"}
    ports: list[dict[str, Any]] = [no_serial]
    w = AutoSpawnWatcher(manager, list_ports=lambda: list(ports), peer_ids=lambda: ())
    assert w.poll()["unidentified"] == ["/dev/ttyACM0"]
    assert w.poll()["unidentified"] == [], "the same plugged-in board was announced a second time"
    assert w.poll()["unidentified"] == []
    assert w.unidentified == {"/dev/ttyACM0"}, "the devices screen still needs to know the board is there"
    ports.clear()
    assert w.poll()["unidentified"] == []
    assert w.unidentified == set(), "a board that left the scan is still remembered as present"
    ports.append(no_serial)
    assert w.poll()["unidentified"] == ["/dev/ttyACM0"], "re-plugging the board is a new event the trail should hear"


def test_a_claimed_peer_is_held_out_loud(manager: _Manager) -> None:
    manager.profiles.save(SERIAL, SIM_PROFILE)
    w = _watcher(manager, BOARD, peers=("bench-sim",))
    did = w.poll()
    assert manager.spawns == []
    (held,) = did["held"]
    assert held["peer_id"] == "bench-sim" and "already present on the mesh" in held["reason"]
    assert w.poll()["held"] == [], "a hold is announced once, not every poll"


def test_remembering_a_board_records_its_usb_identity(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    monkeypatch.setattr(device_manager, "scan_serial_ports", lambda: [dict(BOARD)])
    dm.remember_profile({"peer_id": "arm-left", "robot_name": "so101", "mode": "real", "port": "/dev/ttyACM0"})
    saved = dm.profiles.get(SERIAL)
    assert saved is not None and saved["usb"] == "1a86:55d3"


def test_the_trail_says_what_was_seen() -> None:
    from strands_robots.dashboard.routes_devices import _audit_autospawn

    entries: list[dict[str, Any]] = []

    class _Bridge:
        def record_activity(self, source: str, action: str, **kw: Any) -> None:
            entries.append({"source": source, "action": action, **kw})

    _audit_autospawn(
        _Bridge(),
        {
            "spawned": ["bench-sim"],
            "despawned": [],
            "detected_unknown": [],
            "spawned_from": {"bench-sim": {"serial": SERIAL, "usb": "1a86:55d3", "device": "/dev/ttyACM0"}},
            "proposed": [
                {
                    "serial": SERIAL,
                    "usb": "1a86:55d3",
                    "device": "/dev/ttyACM0",
                    "peer_id": "arm-left",
                    "reason": "confirm it",
                }
            ],
            "held": [{"serial": "DUP", "usb": None, "device": None, "peer_id": "x", "reason": "two boards"}],
            "unidentified": ["/dev/ttyACM2"],
        },
    )
    by_action = {e["action"]: e for e in entries}
    assert SERIAL in by_action["spawn"]["detail"] and "1a86:55d3" in by_action["spawn"]["detail"]
    assert by_action["autospawn_proposed"]["target"] == "arm-left"
    assert by_action["autospawn_proposed"]["ok"] is None
    assert SERIAL in by_action["autospawn_proposed"]["detail"]
    assert by_action["autospawn_held"]["ok"] is False and "two boards" in by_action["autospawn_held"]["detail"]
    assert by_action["autospawn_unidentified"]["target"] == "/dev/ttyACM2"
