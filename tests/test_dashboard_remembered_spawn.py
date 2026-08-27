"""A board the dashboard already knows how to spawn says so (Q41).

`managed` is in-memory: after a restart -- or after a recording session parked both arms and the
dashboard died (Q40) -- the devices screen shows "Managed robots (0)" and two boards that read as
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
    r = remembered_spawn(
        {
            "peer_id": "so101-arm-1",
            "name": "so101-arm-1",
            "robot_name": "so101",
            "mode": "real",
            "port": "/dev/cu.usbmodem5AB01818061",
            "saved_at": 1787115801.0,
            "cameras": {
                "top": {"type": "opencv", "index_or_path": 2, "fps": 30, "width": 1920, "height": 1080},
                "wrist": {"type": "opencv", "index_or_path": 1, "fps": 30, "width": 1920, "height": 1080},
            },
        }
    )
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
    r = remembered_spawn(
        {
            "peer_id": "so101-arm-1",
            "port": "/dev/cu.x",
            "env": {"TOKEN": "s3cret"},
            "role": "follower",
            "role_volts": 12.6,
        }
    )
    # The port already sits on the row this rides on, and the measured role has its own fields -
    # duplicating either invites the two copies to disagree.
    assert set(r) <= {"peer_id", "robot_name", "mode", "cameras", "saved_at", "robot_id"}
    assert "s3cret" not in str(r)


def test_devices_payload_carries_it_per_board(tmp_path) -> None:
    from strands_robots.dashboard import device_manager as dm

    mgr = dm.DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    mgr.profiles.save(
        "5AB0181806",
        {
            "peer_id": "so101-arm-1",
            "robot_name": "so101",
            "mode": "real",
            "port": "/dev/cu.usbmodem5AB01818061",
            "cameras": {"top": {"index_or_path": 2}},
        },
    )
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
    # Q43 rides along: the saved index 2 is nowhere in this (empty) camera scan, so the row says so
    # BEFORE the operator clicks spawn and reads it out of a child's log.
    health = rows["/dev/cu.usbmodem5AB01818061"]["remembered"]["camera_health"]
    assert "no camera at index 2" in health["text"] and health["ok"] is False
    # The unknown board must carry NO key at all, so a screen never has to special-case an empty
    # object that means the same as absence.
    assert "remembered" not in rows["/dev/cu.unknown"]


# ---------------------------------------------------------------------------
# Bringing it back up (Q41 part 2)
# ---------------------------------------------------------------------------


def test_nothing_remembered_is_a_sentence_not_a_spawn() -> None:
    from strands_robots.dashboard.device_manager import respawn_payload

    r = respawn_payload(None, "/dev/cu.x")
    assert "error" in r and "nothing to bring back" in r["error"]
    # It says how the memory gets made, because "no profile" is a state the operator can leave.
    assert "spawn it once" in r["error"]
    assert "robot_name" not in r, "half a memory must never become a process"


def test_an_incomplete_profile_names_what_is_missing() -> None:
    from strands_robots.dashboard.device_manager import respawn_payload

    assert "peer name" in respawn_payload({"robot_name": "so101"}, "/dev/cu.x")["error"]
    assert "robot family" in respawn_payload({"peer_id": "arm-1"}, "/dev/cu.x")["error"]


def test_the_payload_is_the_remembered_one() -> None:
    from strands_robots.dashboard.device_manager import respawn_payload

    cams = {"top": {"index_or_path": 2}, "wrist": {"index_or_path": 1}}
    r = respawn_payload(
        {
            "peer_id": "so101-arm-1",
            "robot_name": "so101",
            "mode": "real",
            "port": "/dev/cu.usbmodem5AB01818061",
            "cameras": cams,
            "robot_id": "arm_1",
        },
        "/dev/cu.usbmodem5AB01818061",
    )
    assert r == {
        "robot_name": "so101",
        "mode": "real",
        "peer_id": "so101-arm-1",
        "port": "/dev/cu.usbmodem5AB01818061",
        "cameras": cams,
        "robot_id": "arm_1",
    }
    # A two-camera config is exactly what a client could not re-type, which is why the payload is
    # assembled server-side.
    assert len(r["cameras"]) == 2


def test_the_port_is_where_the_board_is_now_not_where_it_was() -> None:
    from strands_robots.dashboard.device_manager import respawn_payload

    # THE POINT OF THE WHOLE FEATURE: profiles are keyed by USB serial because /dev names move.
    # Re-using the remembered path would either find nothing, or open a DIFFERENT board with this
    # arm's calibration id.
    r = respawn_payload(
        {
            "peer_id": "so101-arm-1",
            "robot_name": "so101",
            "port": "/dev/cu.usbmodem5AB01818061",
            "robot_id": "arm_1",
        },
        "/dev/cu.usbmodem5AB01818062",
    )
    assert r["port"] == "/dev/cu.usbmodem5AB01818062"
    # ...and the move is stated, so the operator can see the board they are looking at is the one
    # that came up: same serial, new path.
    assert r["port_moved"] == {"was": "/dev/cu.usbmodem5AB01818061", "now": "/dev/cu.usbmodem5AB01818062"}


def test_a_profile_without_a_saved_port_says_nothing_about_moving() -> None:
    from strands_robots.dashboard.device_manager import respawn_payload

    r = respawn_payload({"peer_id": "arm-1", "robot_name": "so101"}, "/dev/cu.x")
    assert "port_moved" not in r
    # An absent mode means the board is real hardware on a real port - the sim default of the form
    # would be wrong here, and a sim spawn on a remembered arm is a silently different robot.
    assert r["mode"] == "real"
    assert r["cameras"] is None and r["robot_id"] is None


def test_a_non_mapping_camera_memory_is_dropped_not_forwarded() -> None:
    from strands_robots.dashboard.device_manager import respawn_payload

    # hardware_robot raises "Camera 'main' config must be a mapping ... got int: 3" on this shape -
    # a live failure once. A junk memory must not be handed to Popen.
    assert respawn_payload({"peer_id": "a", "robot_name": "so101", "cameras": 3}, "/dev/x")["cameras"] is None


# ---------------------------------------------------------------------------
# Are the remembered cameras usable right now? (Q43)
# ---------------------------------------------------------------------------

READY = {"index": 2, "state": "ready", "reason": "opened and delivered a frame just now"}
BLOCKED = {
    "index": 1,
    "state": "blocked",
    "reason": "macOS has not granted camera access to this process",
    "remedy": "start the dashboard from a terminal and allow access",
}


def _health(cameras, rows, peer_id=""):
    from strands_robots.dashboard.device_manager import remembered_camera_health

    return remembered_camera_health(cameras, rows, peer_id)


def test_all_ready_says_nothing() -> None:
    # Silence is the common case: a notice that is always on is not a notice.
    assert _health({"top": {"index_or_path": 2}}, [READY]) == {}
    assert _health(None, [READY]) == {}
    assert _health({}, [READY]) == {}


def test_a_blocked_camera_is_named_with_its_consequence() -> None:
    h = _health({"top": {"index_or_path": 2}, "wrist": {"index_or_path": 1}}, [READY, BLOCKED])
    assert h["ok"] is False
    assert "wrist (index 1)" in h["text"]
    assert "top" not in h["text"], "a ready camera must not be listed as trouble"
    # THE POINT: spawning still WORKS, which is why this needs saying at all - the arm drops the
    # camera it cannot open and records episodes with no pictures in them.
    assert "no pictures" in h["text"]
    assert h["cameras"][1]["remedy"].startswith("start the dashboard")


def test_an_index_that_no_longer_exists_is_the_normal_macos_failure() -> None:
    # The saved index is the least stable thing in the payload: macOS renumbers between reboots.
    h = _health({"top": {"index_or_path": 7}}, [READY])
    assert h["cameras"][0]["state"] == "absent"
    assert "no camera at index 7" in h["text"]


def test_our_own_stream_is_not_a_problem() -> None:
    # The peer we are about to respawn held that camera a moment ago - reporting it as taken would
    # warn about the arm's own picture.
    rows = [{"index": 2, "state": "in_use", "reason": "streaming for so101-arm-1", "claimed_by": "so101-arm-1"}]
    assert _health({"top": {"index_or_path": 2}}, rows, "so101-arm-1") == {}
    # ...but ANOTHER robot holding it is exactly what the operator needs to know.
    h = _health({"top": {"index_or_path": 2}}, rows, "so101-arm-2")
    assert "streaming for so101-arm-1" in h["text"]


def test_a_path_configured_camera_is_admitted_as_unchecked_not_declared_fine() -> None:
    h = _health({"top": {"index_or_path": "/dev/video3"}}, [READY])
    # Nothing to warn about, so no banner - but the entry says plainly it was not tested rather than
    # claiming a path is ready.
    assert h == {}
    full = _health({"top": {"index_or_path": "/dev/video3"}, "wrist": {"index_or_path": 1}}, [READY, BLOCKED])
    assert full["cameras"][0]["state"] == "unchecked"
    assert "not checked" in full["cameras"][0]["reason"]


def test_it_opens_nothing() -> None:
    # The judgment is made from rows /api/devices already computed. Probing here would steal a
    # device from a running robot - so the function must not touch cv2 at all.
    import inspect

    from strands_robots.dashboard import device_manager as dm

    src = inspect.getsource(dm.remembered_camera_health)
    for verb in ("VideoCapture", "cv2", "scan_cameras", "preview_frame", "subprocess"):
        assert verb not in src, f"remembered_camera_health must not {verb}"
