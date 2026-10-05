"""A camera name sits next to the index OpenCV really opens, not the OS listing position.

OpenCV's AVFoundation backend sorts devices by ``uniqueID`` before indexing them; the
OS (and ffmpeg's ``-list_devices``) lists the built-in camera first. Pairing the two by
position labelled a Mac's USB wrist camera "FaceTime HD Camera".
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from strands_robots.dashboard import device_manager

FACETIME = {"_name": "FaceTime HD Camera", "spcamera_unique-id": "47B4B64B70674B9CAD2BAE273A71F4B5"}
USB_CAM = {"_name": "USB2.0_CAM1", "spcamera_unique-id": "0x1100000a16f3b00"}


@pytest.mark.parametrize(
    ("listing", "expected"),
    [
        # The OS lists FaceTime first; OpenCV opens the USB camera at index 0.
        ([FACETIME, USB_CAM], [(0, "USB2.0_CAM1"), (1, "FaceTime HD Camera")]),
        ([USB_CAM, FACETIME], [(0, "USB2.0_CAM1"), (1, "FaceTime HD Camera")]),
        # One camera without an id makes the whole order unknowable: no names beats wrong ones.
        ([FACETIME, {"_name": "USB2.0_CAM1"}], []),
        ([], []),
    ],
)
def test_names_are_numbered_in_opencv_order(listing, expected):
    got = device_manager.opencv_ordered_camera_names(listing)
    assert [(r["listing_index"], r["name"]) for r in got] == expected


def test_macos_scan_reads_system_profiler_and_returns_opencv_indices(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=json.dumps({"SPCameraDataType": [FACETIME, USB_CAM]}), stderr="")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert device_manager.scan_camera_names() == [
        {"listing_index": 0, "name": "USB2.0_CAM1"},
        {"listing_index": 1, "name": "FaceTime HD Camera"},
    ]
    assert calls == [["/usr/sbin/system_profiler", "SPCameraDataType", "-json"]]
