"""USB auto-spawn: plug a known board -> its robot comes up; unplug -> it goes away.

Covers the AutoSpawnWatcher + ProfileStore pair in
strands_robots.dashboard.device_manager with a fake serial bus and a fake
manager — no real hardware, no real subprocesses.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from strands_robots.dashboard.device_manager import (
    AutoSpawnWatcher,
    ProfileStore,
    profile_key,
)

ARM1 = {"device": "/dev/cu.usbmodem101", "serial_number": "SER-ARM-1"}
ARM_NO_SERIAL = {"device": "/dev/cu.usbserial-2", "serial_number": None}


class FakeManaged:
    def __init__(self, peer_id: str, port: str | None, alive: bool = True) -> None:
        self.peer_id = peer_id
        self.port = port
        self._alive = alive

    def alive(self) -> bool:
        return self._alive


class FakeManager:
    """Just enough DeviceManager for the watcher: robots, profiles, spawn, despawn."""

    def __init__(self, profiles: ProfileStore) -> None:
        self.profiles = profiles
        self.robots: dict[str, FakeManaged] = {}
        self.spawn_calls: list[dict[str, Any]] = []
        self.despawn_calls: list[str] = []
        self.spawn_error: str | None = None

    def spawn(self, **kwargs: Any) -> dict[str, Any]:
        self.spawn_calls.append(kwargs)
        if self.spawn_error:
            return {"error": self.spawn_error}
        peer_id = kwargs.get("peer_id") or "anon"
        self.robots[peer_id] = FakeManaged(peer_id, kwargs.get("port"))
        return {"peer_id": peer_id, "pid": 4242, "mode": kwargs.get("mode")}

    def despawn(self, peer_id: str) -> dict[str, Any]:
        self.despawn_calls.append(peer_id)
        self.robots.pop(peer_id, None)
        return {"stopped": peer_id}


@pytest.fixture()
def store(tmp_path):
    return ProfileStore(path=str(tmp_path / "profiles.json"))


def make_watcher(manager, ports, peers=(), missing_polls=2):
    bus = {"ports": list(ports)}
    w = AutoSpawnWatcher(
        manager,
        list_ports=lambda: list(bus["ports"]),
        peer_ids=lambda: list(peers),
        missing_polls=missing_polls,
    )
    return w, bus


# ---------------------------------------------------------------- profile_key

def test_profile_key_prefers_serial_and_falls_back_to_path():
    assert profile_key(ARM1) == "SER-ARM-1"
    assert profile_key(ARM_NO_SERIAL) == "/dev/cu.usbserial-2"
    assert profile_key({}) == ""


# ---------------------------------------------------------------- ProfileStore

def test_profile_store_round_trip(store, tmp_path):
    saved = store.save("SER-ARM-1", {"peer_id": "so101-arm-1", "port": "/dev/cu.usbmodem101"})
    assert saved["serial_number"] == "SER-ARM-1"
    # A fresh store instance reads the same file back.
    again = ProfileStore(path=store.path)
    assert again.get("SER-ARM-1")["peer_id"] == "so101-arm-1"
    assert "SER-ARM-1" in again.all()
    assert again.get("nope") is None


def test_profile_store_corrupt_file_starts_empty(tmp_path):
    p = tmp_path / "profiles.json"
    p.write_text("{this is not json")
    assert ProfileStore(path=str(p)).all() == {}
    p.write_text(json.dumps(["a", "list"]))
    assert ProfileStore(path=str(p)).all() == {}


# ------------------------------------------------------------- appear -> spawn

def test_known_board_appears_and_is_spawned(store):
    store.save("SER-ARM-1", {"peer_id": "so101-arm-1", "robot_name": "so101",
                             "mode": "real", "port": "/dev/cu.OLD-PATH"})
    mgr = FakeManager(store)
    w, _ = make_watcher(mgr, [ARM1])
    res = w.poll()
    assert res["spawned"] == ["so101-arm-1"]
    call = mgr.spawn_calls[0]
    # The live enumerated path wins over the remembered one, and the watcher
    # must not re-save the profile it is replaying.
    assert call["port"] == "/dev/cu.usbmodem101"
    assert call["remember"] is False
    # Second poll: adopted, not spawned twice.
    assert w.poll()["spawned"] == []
    assert len(mgr.spawn_calls) == 1


def test_unknown_board_is_only_reported(store):
    mgr = FakeManager(store)
    w, _ = make_watcher(mgr, [ARM1])
    res = w.poll()
    assert res["spawned"] == []
    assert "SER-ARM-1" in res["detected_unknown"]
    assert mgr.spawn_calls == []


# --------------------------------------------------------------------- dedupe

def test_port_claimed_by_managed_robot_is_skipped(store):
    store.save("SER-ARM-1", {"peer_id": "so101-arm-1", "port": ARM1["device"]})
    mgr = FakeManager(store)
    mgr.robots["so101-arm-1"] = FakeManaged("so101-arm-1", ARM1["device"])
    w, _ = make_watcher(mgr, [ARM1])
    assert w.poll()["spawned"] == []
    assert mgr.spawn_calls == []


def test_peer_already_on_mesh_is_skipped(store):
    store.save("SER-ARM-1", {"peer_id": "so101-arm-1", "port": "/dev/cu.OLD"})
    mgr = FakeManager(store)
    w, _ = make_watcher(mgr, [ARM1], peers=["so101-arm-1"])
    assert w.poll()["spawned"] == []
    assert mgr.spawn_calls == []


def test_mesh_lookup_failure_refuses_to_spawn(store):
    store.save("SER-ARM-1", {"peer_id": "so101-arm-1", "port": "/dev/cu.OLD"})
    mgr = FakeManager(store)

    def boom():
        raise RuntimeError("zenoh down")

    w = AutoSpawnWatcher(mgr, list_ports=lambda: [ARM1], peer_ids=boom)
    assert w.poll()["spawned"] == []
    assert mgr.spawn_calls == []


# --------------------------------------------------------- disappear -> despawn

def test_unplug_despawns_after_consecutive_misses(store):
    store.save("SER-ARM-1", {"peer_id": "so101-arm-1", "port": "/dev/cu.OLD"})
    mgr = FakeManager(store)
    w, bus = make_watcher(mgr, [ARM1], missing_polls=2)
    assert w.poll()["spawned"] == ["so101-arm-1"]

    bus["ports"] = []  # unplugged
    assert w.poll()["despawned"] == []          # miss 1 of 2 — debounce
    assert w.poll()["despawned"] == ["so101-arm-1"]  # miss 2 — stopped
    assert mgr.despawn_calls == ["so101-arm-1"]
    assert w.adopted == {}


def test_flapping_port_is_not_despawned(store):
    store.save("SER-ARM-1", {"peer_id": "so101-arm-1", "port": "/dev/cu.OLD"})
    mgr = FakeManager(store)
    w, bus = make_watcher(mgr, [ARM1], missing_polls=2)
    w.poll()
    bus["ports"] = []       # one missed poll…
    w.poll()
    bus["ports"] = [ARM1]   # …but it re-enumerates (USB flap)
    w.poll()
    assert mgr.despawn_calls == []
    assert w.adopted  # still adopted


def test_operator_spawned_robots_are_never_despawned(store):
    mgr = FakeManager(store)
    mgr.robots["hand-started"] = FakeManaged("hand-started", "/dev/cu.X")
    w, _ = make_watcher(mgr, [], missing_polls=1)
    w.poll()
    assert mgr.despawn_calls == []


def test_failed_spawn_is_not_adopted(store):
    store.save("SER-ARM-1", {"peer_id": "so101-arm-1", "port": "/dev/cu.OLD"})
    mgr = FakeManager(store)
    mgr.spawn_error = "servo bus dead"
    w, _ = make_watcher(mgr, [ARM1])
    assert w.poll()["spawned"] == []
    assert w.adopted == {}


# ---------------------------------------------------------------- kill switch

def test_kill_switch_disables_polling(store, monkeypatch):
    store.save("SER-ARM-1", {"peer_id": "so101-arm-1", "port": "/dev/cu.OLD"})
    mgr = FakeManager(store)
    w, _ = make_watcher(mgr, [ARM1])
    monkeypatch.setenv("STRANDS_DASHBOARD_AUTOSPAWN", "0")
    assert w.poll() == {"skipped": "autospawn disabled"}
    assert mgr.spawn_calls == []
    monkeypatch.setenv("STRANDS_DASHBOARD_AUTOSPAWN", "off")
    assert AutoSpawnWatcher.enabled() is False
    monkeypatch.delenv("STRANDS_DASHBOARD_AUTOSPAWN")
    assert AutoSpawnWatcher.enabled() is True
