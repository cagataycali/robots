"""The stamp reaches a real spawn — and the CHILD never sees it.

``stamp_device_names`` was pure and dormant (35488a21). Wiring it is where two things can go wrong
in ways no unit test of the rule would catch:

* the note is the dashboard's own, and ``_build_camera_config`` refuses ANY key
  ``OpenCVCameraConfig`` does not declare — so a stamp forwarded to the child does not degrade one
  camera, it kills every camera on the arm. The Popen payload is asserted here, byte for byte.
* the stamp is written into the remembered profile, and a profile is handed straight BACK to
  ``spawn`` by autospawn and by reconfigure. If ``validate_cameras`` treated the note as an unknown
  option, a stamped arm would refuse to come back — the memory would break the fleet it protects.
  That round trip is asserted, not assumed.

And the stamp must never cost hardware: reading the roster here must not trigger a scan (ffmpeg,
10s timeout, and an enumeration can OPEN the very index the arm is about to take).
"""

from __future__ import annotations

import json
import pathlib

import strands_robots.dashboard.device_manager as dm
from strands_robots.dashboard.device_manager import DeviceManager, validate_cameras

_ROSTER = [
    {"listing_index": 0, "name": "USB2.0_CAM1"},
    {"listing_index": 2, "name": "Logi 4K Pro"},
]


class FakeProc:
    _next_pid = 8200
    payloads: list[dict] = []

    def __init__(self, argv, *a, **kw):
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self.stdout = None
        FakeProc.payloads.append(json.loads(argv[-1]))

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        return None

    def kill(self):
        return None


def _manager(tmp_path, monkeypatch, *, roster=_ROSTER, roster_age=0.0):
    FakeProc.payloads = []
    monkeypatch.setattr(dm.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(dm.threading, "Thread", lambda *a, **kw: type("T", (), {"start": lambda self: None})())
    mgr = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    mgr._camera_names_cache = list(roster or [])
    mgr._camera_names_cache_t = dm.time.time() - roster_age if roster else 0.0
    # A scan from inside spawn would be the bug, not the fixture: make it loud.
    monkeypatch.setattr(
        dm,
        "scan_camera_names",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("spawn must never trigger a camera scan")),
    )
    monkeypatch.setattr(
        dm,
        "scan_serial_ports",
        lambda *a, **kw: [{"device": "/dev/tty.fake", "serial_number": "5AB0181806"}],
    )
    monkeypatch.setattr(
        dm,
        "bus_claim",
        type(
            "B",
            (),
            {
                "bus_holders": staticmethod(lambda port: []),
                "bus_conflict": staticmethod(lambda *a, **kw: None),
            },
        )(),
    )
    return mgr


def _spawn(mgr, cameras):
    return mgr.spawn(
        robot_name="so101",
        mode="real",
        port="/dev/tty.fake",
        peer_id="stamp-test",
        cameras=cameras,
    )


def test_the_spawned_arm_remembers_which_device_its_index_was(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)
    out = _spawn(mgr, {"top": {"index_or_path": 2, "fps": 30}})
    assert "error" not in out
    assert mgr.robots["stamp-test"].cameras["top"]["device_name"] == "Logi 4K Pro"
    saved = json.loads(pathlib.Path(mgr.profiles.path).read_text())
    profile = next(p for p in saved.values() if (p or {}).get("peer_id") == "stamp-test")
    assert profile["cameras"]["top"]["device_name"] == "Logi 4K Pro"


def test_the_child_is_handed_the_config_without_the_dashboard_s_note(tmp_path, monkeypatch):
    """The whole reason the strip exists: an unknown key kills EVERY camera on the arm."""
    mgr = _manager(tmp_path, monkeypatch)
    _spawn(mgr, {"top": {"index_or_path": 2, "fps": 30}})
    assert FakeProc.payloads[-1]["cameras"] == {"top": {"index_or_path": 2, "fps": 30}}


def test_a_stamped_profile_can_spawn_again(tmp_path, monkeypatch):
    """Autospawn and reconfigure feed a remembered profile straight back in."""
    mgr = _manager(tmp_path, monkeypatch)
    _spawn(mgr, {"top": {"index_or_path": 2}})
    saved = json.loads(pathlib.Path(mgr.profiles.path).read_text())
    remembered = next(p for p in saved.values() if (p or {}).get("peer_id") == "stamp-test")["cameras"]
    assert validate_cameras(remembered) is None, "the note must not read as an unknown option"
    mgr.despawn("stamp-test")
    again = _spawn(mgr, remembered)
    assert "error" not in again
    assert FakeProc.payloads[-1]["cameras"] == {"top": {"index_or_path": 2}}


def test_a_note_that_is_not_a_name_is_refused(tmp_path, monkeypatch):
    assert validate_cameras({"top": {"index_or_path": 0, "device_name": 3}})["error"].startswith(
        "camera 'top': device_name is the dashboard's note"
    )


def test_a_stale_roster_stamps_nothing_rather_than_a_wrong_name(tmp_path, monkeypatch):
    """Nothing is worse here than a confident wrong memory: it refuses a healthy rig later."""
    mgr = _manager(tmp_path, monkeypatch, roster_age=DeviceManager.ROSTER_MAX_AGE_S + 1)
    _spawn(mgr, {"top": {"index_or_path": 2}})
    assert "device_name" not in mgr.robots["stamp-test"].cameras["top"]


def test_no_roster_at_all_still_spawns(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch, roster=[])
    out = _spawn(mgr, {"top": {"index_or_path": 2}})
    assert "error" not in out and out["pid"]
    assert mgr.robots["stamp-test"].cameras == {"top": {"index_or_path": 2}}


def test_a_camera_less_spawn_is_untouched(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)
    out = _spawn(mgr, None)
    assert "error" not in out
    assert FakeProc.payloads[-1]["cameras"] is None
