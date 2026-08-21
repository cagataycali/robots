"""U19 backend verify: does a camera setting the operator changed actually reach the CHILD? (2026-08-21)

The reconfigure path was already covered - but the test that covers it monkeypatches ``spawn``, so it
proves reconfigure ASKS for the new cameras and stops exactly where the interesting part starts.
Between that ask and a camera actually running at 60fps there are three more steps that have each
been a real bug in this repo: ``stamp_device_names`` ADDS a key, ``without_annotations`` REMOVES
keys, and the profile store rewrites the config for the next autospawn. A strip that took one field
too many, or a profile that remembered only the index, would leave the UI reporting success while the
camera streamed at its default resolution for ever - the silent-degradation class AGENTS.md #86 is
about, and the one no counter on the record screen can see.

So this file drives the real ``reconfigure_cameras`` with ``spawn`` UNMOCKED and reads the child's
argv, which is the only place the truth exists. No process is started (Popen is a fake) and no
camera is opened.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from strands_robots.dashboard import device_manager as dm
from strands_robots.dashboard.device_manager import DeviceManager

_ROSTER = ["USB2.0_CAM1", "USB2.0_CAM1 #2", "Logi 4K Pro"]


class FakeProc:
    _next_pid = 9100
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


@pytest.fixture()
def mgr(tmp_path, monkeypatch):
    FakeProc.payloads = []
    monkeypatch.setattr(dm.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(
        dm.threading, "Thread",
        lambda *a, **kw: type("T", (), {"start": lambda self: None})(),
    )
    m = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    m._camera_names_cache = list(_ROSTER)
    m._camera_names_cache_t = dm.time.time()
    monkeypatch.setattr(dm, "scan_serial_ports", lambda *a, **kw: [
        {"device": "/dev/tty.fake", "serial_number": "5AB0181806"},
    ])
    monkeypatch.setattr(dm, "bus_claim", type("B", (), {
        "bus_holders": staticmethod(lambda port: []),
        "bus_conflict": staticmethod(lambda *a, **kw: None),
    })())
    return m


def _spawn(m, cameras):
    return m.spawn(
        robot_name="so101", mode="real", port="/dev/tty.fake",
        peer_id="u19-arm", cameras=cameras,
    )


def test_the_new_fps_and_resolution_reach_the_child_process(mgr):
    _spawn(mgr, {"wrist": {"index_or_path": 1, "fps": 30}})
    out = mgr.reconfigure_cameras(
        "u19-arm", {"wrist": {"index_or_path": 1, "fps": 60, "width": 1280, "height": 720}}
    )
    assert out.get("reconfigured") is True
    child = FakeProc.payloads[-1]
    assert child["cameras"] == {
        "wrist": {"index_or_path": 1, "fps": 60, "width": 1280, "height": 720}
    }, "the exact settings the operator asked for, with the dashboard's own note stripped"
    assert "device_name" not in child["cameras"]["wrist"], (
        "an undeclared key does not degrade one camera, it kills every camera on the arm"
    )


def test_the_respawn_keeps_the_arm_s_identity_not_just_its_cameras(mgr):
    _spawn(mgr, {"wrist": {"index_or_path": 1, "fps": 30}})
    mgr.reconfigure_cameras("u19-arm", {"wrist": {"index_or_path": 1, "fps": 60}})
    child = FakeProc.payloads[-1]
    assert child["peer_id"] == "u19-arm" and child["port"] == "/dev/tty.fake"
    assert child["robot_name"] == "so101" and child["mode"] == "real"


def test_a_detach_reaches_the_child_as_no_cameras_at_all(mgr):
    _spawn(mgr, {"wrist": {"index_or_path": 1, "fps": 30}})
    mgr.reconfigure_cameras("u19-arm", None)
    assert not FakeProc.payloads[-1]["cameras"], (
        "detach must arrive as empty/None, never as the old config: the operator unplugged a camera"
    )


def test_the_change_survives_a_replug_because_the_profile_remembers_it(mgr):
    """reconfigure passes remember=True - this is what that has to MEAN."""
    _spawn(mgr, {"wrist": {"index_or_path": 1, "fps": 30}})
    mgr.reconfigure_cameras(
        "u19-arm", {"wrist": {"index_or_path": 1, "fps": 60, "width": 1280, "height": 720}}
    )
    saved = json.loads(pathlib.Path(mgr.profiles.path).read_text())
    profile = next(p for p in saved.values() if (p or {}).get("peer_id") == "u19-arm")
    cams = profile["cameras"]["wrist"]
    assert cams["fps"] == 60 and cams["width"] == 1280 and cams["height"] == 720, (
        "an autospawn after a replug feeds this profile straight back to spawn, so a profile that "
        "remembered only the index would quietly undo the operator's change on the next unplug"
    )


def test_the_old_settings_are_replaced_not_merged(mgr):
    """A merge would make a setting impossible to REMOVE - back to the driver default."""
    _spawn(mgr, {"wrist": {"index_or_path": 1, "fps": 60, "width": 1280, "height": 720}})
    mgr.reconfigure_cameras("u19-arm", {"wrist": {"index_or_path": 1, "fps": 30}})
    assert FakeProc.payloads[-1]["cameras"] == {"wrist": {"index_or_path": 1, "fps": 30}}
