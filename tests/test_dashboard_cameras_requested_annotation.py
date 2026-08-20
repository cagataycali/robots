"""The dashboard tells the fleet view which cameras a spawn ASKED for.

An empty `cameras` in the snapshot means "no frames have arrived" and presence means "the robot announces
these". Neither can distinguish a deliberately joints-only robot from cameras that failed to open — and
hardware_robot DROPS a camera it cannot open at connect, so a blocked camera erases its own evidence
(BUGS.md Q25: macOS refused capture on this Mac and both arms silently became joints-only). The dashboard
is the only party that remembers what it requested, so that memory belongs in the snapshot.
"""

from __future__ import annotations

from strands_robots.dashboard.device_manager import (
    DeviceManager,
    ManagedRobot,
    requested_camera_names,
)


def test_names_only_sorted() -> None:
    assert requested_camera_names({"wrist": {"index_or_path": 1}, "top": {"index_or_path": 0}}) == ["top", "wrist"]


def test_a_spawn_with_no_cameras_asked_for_none() -> None:
    assert requested_camera_names({}) == []
    assert requested_camera_names(None) == []
    assert requested_camera_names("top") == []  # not a config; nothing to claim


def test_indices_and_paths_are_not_broadcast() -> None:
    # Only names go to every websocket client; the config's indices/paths stay server-side.
    names = requested_camera_names({"top": {"index_or_path": "/dev/video0", "fps": 30}})
    assert names == ["top"]


def _manager(robots: dict[str, ManagedRobot]) -> DeviceManager:
    dm = DeviceManager.__new__(DeviceManager)
    dm.robots = robots  # type: ignore[attr-defined]
    return dm


def test_annotation_carries_requested_names_for_a_managed_peer(monkeypatch) -> None:
    dm = _manager({"so101-arm-2": ManagedRobot("so101-arm-2", "so101", "real", cameras={"top": 0, "wrist": 1})})
    monkeypatch.setattr(DeviceManager, "roles_by_peer", lambda self: {})

    assert dm.annotations_by_peer() == {"so101-arm-2": {"cameras_requested": ["top", "wrist"]}}


def test_it_does_not_shadow_the_measured_role(monkeypatch) -> None:
    """U2's role and this must coexist: one hook, one story per peer."""
    dm = _manager({"a": ManagedRobot("a", "so101", "real", cameras={"top": 0})})
    monkeypatch.setattr(DeviceManager, "roles_by_peer", lambda self: {"a": {"role": "follower", "role_volts": 12.6}})

    ann = dm.annotations_by_peer()["a"]
    assert ann["role"] == "follower" and ann["role_volts"] == 12.6
    assert ann["cameras_requested"] == ["top"]


def test_a_peer_that_asked_for_nothing_contributes_no_key(monkeypatch) -> None:
    """Absence must stay absence: a joints-only robot must not read as `cameras_requested: []`,
    which a consumer would show as a positive claim about zero cameras."""
    dm = _manager({"sim-a": ManagedRobot("sim-a", "so101", "sim", cameras={})})
    monkeypatch.setattr(DeviceManager, "roles_by_peer", lambda self: {})

    assert dm.annotations_by_peer() == {}


def test_role_only_peers_survive_untouched(monkeypatch) -> None:
    dm = _manager({})
    monkeypatch.setattr(DeviceManager, "roles_by_peer", lambda self: {"x": {"role": "leader"}})

    assert dm.annotations_by_peer() == {"x": {"role": "leader"}}


def test_the_annotation_hook_the_server_installs_is_this_one() -> None:
    """A test at the wiring, because the U2 bug was exactly a route and a websocket disagreeing:
    the hook must be the merged annotation, not roles alone."""
    import inspect

    from strands_robots.dashboard import server

    src = inspect.getsource(server.create_app)
    assert "peer_annotations = app.state.devices.annotations_by_peer" in src
