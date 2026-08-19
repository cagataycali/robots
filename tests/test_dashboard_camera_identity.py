"""Camera identity for the devices screen (Feature 3: Cameras).

An OpenCV index is a position, never an identity: names are listed by the
OS in a DIFFERENT order than OpenCV enumerates (Continuity cameras renumber,
a spawned robot claiming index 0 shifts the probe list). So the surface is:

* ``camera_names`` in /api/devices - a best-effort roster parsed from the
  platform listing, explicitly NOT index-aligned;
* ``GET /api/devices/camera/{index}/preview`` - one JPEG frame, the
  authoritative answer to "which camera is index N": a picture cannot lie.

The preview must refuse a claimed index (opening a device that is streaming
for a running robot steals its frames mid-episode) and turn camera faults
into HTTP statuses the UI can branch on, never tracebacks.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard import device_manager as dm
from strands_robots.dashboard import server as srv

FFMPEG_STDERR = """\
[AVFoundation indev @ 0x158e04510] AVFoundation video devices:
[AVFoundation indev @ 0x158e04510] [0] USB2.0_CAM1
[AVFoundation indev @ 0x158e04510] [1] USB2.0_CAM1
[AVFoundation indev @ 0x158e04510] [2] Logi 4K Pro
[AVFoundation indev @ 0x158e04510] [3] Capture screen 0
[AVFoundation indev @ 0x158e04510] AVFoundation audio devices:
[AVFoundation indev @ 0x158e04510] [0] MacBook Microphone
: Input/output error
"""


# ---------------------------------------------------------------- name scan


def test_scan_camera_names_parses_avfoundation_listing(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(
        dm.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stderr=FFMPEG_STDERR),
    )
    monkeypatch.setattr("shutil.which", lambda _: "/opt/homebrew/bin/ffmpeg")
    names = dm.scan_camera_names()
    assert [n["name"] for n in names] == [
        "USB2.0_CAM1", "USB2.0_CAM1", "Logi 4K Pro", "Capture screen 0",
    ]
    assert [n["listing_index"] for n in names] == [0, 1, 2, 3]
    # audio devices never leak into the roster
    assert all("Microphone" not in n["name"] for n in names)


def test_scan_camera_names_survives_missing_ffmpeg(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("os.path.exists", lambda _: False)
    assert dm.scan_camera_names() == []


def test_scan_camera_names_survives_ffmpeg_blowup(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("shutil.which", lambda _: "/opt/homebrew/bin/ffmpeg")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("ffmpeg", 10)

    monkeypatch.setattr(dm.subprocess, "run", boom)
    assert dm.scan_camera_names() == []


def test_devices_payload_carries_camera_names(monkeypatch):
    mgr = dm.DeviceManager(profiles_path="/tmp/does-not-matter-profiles.json")
    monkeypatch.setattr(dm, "scan_serial_ports", lambda: [])
    monkeypatch.setattr(dm, "scan_cameras", lambda **k: [])
    monkeypatch.setattr(
        dm, "scan_camera_names", lambda: [{"listing_index": 0, "name": "Logi 4K Pro"}]
    )
    payload = mgr.devices(refresh=True)
    assert payload["camera_names"] == [{"listing_index": 0, "name": "Logi 4K Pro"}]


# ---------------------------------------------------------------- preview


class _FakeCap:
    """Stands in for cv2.VideoCapture."""

    def __init__(self, opened=True, frames=True):
        self._opened = opened
        self._frames = frames
        self.released = False

    def isOpened(self):  # noqa: N802 - cv2 casing
        return self._opened

    def read(self):
        import numpy as np

        if not self._frames:
            return False, None
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def _manager(monkeypatch, cap: _FakeCap, claimed: dict[int, str] | None = None):
    mgr = dm.DeviceManager(profiles_path="/tmp/does-not-matter-profiles.json")
    monkeypatch.setattr(mgr, "_claimed_camera_indices", lambda: claimed or {})
    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", lambda i: cap)
    return mgr


def test_preview_refuses_claimed_index(monkeypatch):
    cap = _FakeCap()
    mgr = _manager(monkeypatch, cap, claimed={0: "so101-arm-1"})
    with pytest.raises(PermissionError, match="so101-arm-1"):
        mgr.preview_frame(0)
    # and it never opened the device
    assert cap.released is False


def test_preview_returns_jpeg_and_releases(monkeypatch):
    cap = _FakeCap()
    mgr = _manager(monkeypatch, cap)
    jpeg = mgr.preview_frame(1)
    assert jpeg[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert cap.released is True


def test_preview_faults_are_runtime_errors(monkeypatch):
    cap = _FakeCap(frames=False)
    mgr = _manager(monkeypatch, cap)
    with pytest.raises(RuntimeError, match="no frame"):
        mgr.preview_frame(1)
    assert cap.released is True

    cap2 = _FakeCap(opened=False)
    mgr2 = _manager(monkeypatch, cap2)
    with pytest.raises(RuntimeError, match="would not open"):
        mgr2.preview_frame(1)


# ---------------------------------------------------------------- HTTP


class _StubBridge:
    peers: dict = {}

    def snapshot(self):
        return {"peers": {}}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}
    return TestClient(srv.create_app(bridge=_StubBridge()))


def test_preview_http_status_mapping(client, monkeypatch):
    devices = client.app.state.devices

    monkeypatch.setattr(
        devices, "preview_frame",
        lambda i: (_ for _ in ()).throw(PermissionError("index 0 is streaming for so101-arm-1")),
    )
    r = client.get("/api/devices/camera/0/preview")
    assert r.status_code == 409
    assert "so101-arm-1" in r.json()["detail"]

    monkeypatch.setattr(
        devices, "preview_frame",
        lambda i: (_ for _ in ()).throw(RuntimeError("camera index 3 would not open")),
    )
    r = client.get("/api/devices/camera/3/preview")
    assert r.status_code == 503

    monkeypatch.setattr(devices, "preview_frame", lambda i: b"\xff\xd8fakejpeg")
    r = client.get("/api/devices/camera/1/preview")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["cache-control"] == "no-store"
    assert r.content.startswith(b"\xff\xd8")
