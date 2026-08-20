"""A board the dashboard already knows how to spawn says so (Q41).

`managed` is in-memory: after a restart — or after a recording session parked both arms and the
dashboard died (Q40) — the devices screen shows "Managed robots (0)" and two boards that read as
never-configured hardware, even though profiles.json holds the exact payload that brought them up.
"""

from __future__ import annotations

from strands_robots.dashboard.device_manager import remembered_spawn


def test_nothing_remembered_renders_as_nothing() -> None:
    # A board nobody has configured is a normal state, not a gap to fill with guesses.
    assert remembered_spawn(None) == {}
    assert remembered_spawn({}) == {}


def test_a_payload_that_cannot_be_respawned_is_not_offered() -> None:
    # No peer_id = not a spawn recipe. Showing it would offer a button that cannot work.
    assert remembered_spawn({"robot_name": "so101", "port": "/dev/cu.x"}) == {}
    assert remembered_spawn({"peer_id": "   ", "robot_name": "so101"}) == {}


def test_it_reports_how_the_board_comes_up() -> None:
    r = remembered_spawn({
        "peer_id": "so101-arm-1", "name": "so101-arm-1", "robot_name": "so101", "mode": "real",
        "port": "/dev/cu.usbmodem5AB01818061", "saved_at": 1787115801.0,
        "cameras": {
            "top": {"type": "opencv", "index_or_path": 2, "fps": 30, "width": 1920, "height": 1080},
            "wrist": {"type": "opencv", "index_or_path": 1, "fps": 30, "width": 1920, "height": 1080},
        },
    })
    assert r["peer_id"] == "so101-arm-1"
    assert r["robot_name"] == "so101" and r["mode"] == "real"
    # NAMES, not indices: the operator recognises "top, wrist", and the indices behind them are
    # exactly what may have moved since (macOS renumbers cameras between reboots).
    assert r["cameras"] == ["top", "wrist"]
    assert "2" not in str(r["cameras"]) and "index_or_path" not in str(r["cameras"])
    assert r["saved_at"] == 1787115801.0


def test_the_calibration_id_is_shown_when_remembered_and_never_invented() -> None:
    # A wrong lerobot id moves a real arm with another arm's zero points, so it is worth showing -
    # and worth never guessing from the peer name.
    with_id = remembered_spawn({"peer_id": "so101-arm-2", "robot_id": "leader_arm"})
    assert with_id["robot_id"] == "leader_arm"
    assert "robot_id" not in remembered_spawn({"peer_id": "so101-arm-2", "robot_id": None})
    assert "robot_id" not in remembered_spawn({"peer_id": "so101-arm-2"})


def test_no_cameras_is_an_empty_list_not_a_missing_key() -> None:
    # A sim or a joints-only arm: the screen should be able to say "no cameras" from this alone.
    assert remembered_spawn({"peer_id": "sim-a", "cameras": None})["cameras"] == []
    assert remembered_spawn({"peer_id": "sim-a", "cameras": "top"})["cameras"] == []


def test_it_carries_no_secrets_or_paths_it_was_not_asked_for() -> None:
    r = remembered_spawn({
        "peer_id": "so101-arm-1", "port": "/dev/cu.x", "env": {"TOKEN": "s3cret"},
        "role": "follower", "role_volts": 12.6,
    })
    # The port already sits on the row this rides on, and the measured role has its own fields -
    # duplicating either invites the two copies to disagree.
    assert set(r) <= {"peer_id", "robot_name", "mode", "cameras", "saved_at", "robot_id"}
    assert "s3cret" not in str(r)


def test_devices_payload_carries_it_per_board(tmp_path) -> None:
    from strands_robots.dashboard import device_manager as dm

    mgr = dm.DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    mgr.profiles.save("5AB0181806", {
        "peer_id": "so101-arm-1", "robot_name": "so101", "mode": "real",
        "port": "/dev/cu.usbmodem5AB01818061", "cameras": {"top": {"index_or_path": 2}},
    })
    ports = [
        {"device": "/dev/cu.usbmodem5AB01818061", "serial_number": "5AB0181806"},
        {"device": "/dev/cu.unknown", "serial_number": "NEVERSEEN"},
    ]
    mgr._cameras = lambda **_: {}  # type: ignore[assignment]
    orig = dm.scan_serial_ports
    dm.scan_serial_ports = lambda: ports  # type: ignore[assignment]
    try:
        doc = mgr.devices()
    finally:
        dm.scan_serial_ports = orig  # type: ignore[assignment]

    rows = {r["device"]: r for r in doc["serial_ports"]}
    assert rows["/dev/cu.usbmodem5AB01818061"]["remembered"]["peer_id"] == "so101-arm-1"
    # The unknown board must carry NO key at all, so a screen never has to special-case an empty
    # object that means the same as absence.
    assert "remembered" not in rows["/dev/cu.unknown"]
