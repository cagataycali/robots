"""U19: the reconfigure sheet's fps/resolution selects offer REAL modes only.

A cv2 driver accepts any set() without complaint and then delivers whatever it
wants; the only truth is the read-back. modes_from_readbacks is the pure
distillation: keep a candidate only when the camera agreed to it, always keep
the native mode, dedupe, sort for a <select>. probe_modes carries the same
streaming guard as preview_frame - probing an index steals the device on macOS.
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard.device_manager import (
    CAMERA_FPS_CANDIDATES,
    CAMERA_MODE_CANDIDATES,
    DeviceManager,
    modes_from_readbacks,
)


def _rb(req_w, req_h, req_fps, got_w, got_h, got_fps):
    return {
        "requested": {"width": req_w, "height": req_h, "fps": req_fps},
        "got": {"width": got_w, "height": got_h, "fps": got_fps},
    }


NATIVE = {"width": 1280, "height": 720, "fps": 30.0}


class TestModesFromReadbacks:
    def test_agreed_mode_is_kept(self) -> None:
        modes = modes_from_readbacks(NATIVE, [_rb(640, 480, 30, 640, 480, 30.0)])
        assert {"width": 640, "height": 480, "fps": 30} in modes

    def test_ignored_set_contributes_nothing(self) -> None:
        # Driver answers its native mode for every request: only native survives.
        rbs = [_rb(w, h, fps, 1280, 720, 30.0)
               for w, h in CAMERA_MODE_CANDIDATES for fps in CAMERA_FPS_CANDIDATES
               if (w, h) != (1280, 720)]
        modes = modes_from_readbacks(NATIVE, rbs)
        assert modes == [{"width": 1280, "height": 720, "fps": 30}]

    def test_native_mode_always_included_even_with_zero_readbacks(self) -> None:
        assert modes_from_readbacks(NATIVE, []) == [{"width": 1280, "height": 720, "fps": 30}]

    def test_fps_within_one_counts_as_agreement(self) -> None:
        # NTSC drivers report 29.97 for 30 - that IS the 30fps mode.
        modes = modes_from_readbacks(NATIVE, [_rb(640, 480, 30, 640, 480, 29.97)])
        assert {"width": 640, "height": 480, "fps": 30} in modes

    def test_fps_off_by_more_than_one_is_refused(self) -> None:
        modes = modes_from_readbacks(NATIVE, [_rb(640, 480, 60, 640, 480, 30.0)])
        assert {"width": 640, "height": 480, "fps": 60} not in modes

    def test_deduped_and_sorted_by_area_then_fps(self) -> None:
        modes = modes_from_readbacks(NATIVE, [
            _rb(1920, 1080, 30, 1920, 1080, 30),
            _rb(640, 480, 60, 640, 480, 60),
            _rb(640, 480, 15, 640, 480, 15),
            _rb(640, 480, 15, 640, 480, 15),  # duplicate probe
        ])
        keys = [(m["width"], m["height"], m["fps"]) for m in modes]
        assert keys == sorted(set(keys), key=lambda k: (k[0] * k[1], k[2]))
        assert len(keys) == len(set(keys))

    def test_garbage_native_yields_no_phantom_mode(self) -> None:
        # A broken driver reporting 0x0@0 must not put an unusable row in the select.
        modes = modes_from_readbacks({"width": 0, "height": 0, "fps": 0.0},
                                     [_rb(640, 480, 30, 640, 480, 30)])
        assert modes == [{"width": 640, "height": 480, "fps": 30}]

    def test_non_numeric_readback_is_skipped_not_fatal(self) -> None:
        modes = modes_from_readbacks(NATIVE, [
            {"requested": {"width": "x"}, "got": None},
            _rb(640, 480, 30, 640, 480, 30),
        ])
        assert {"width": 640, "height": 480, "fps": 30} in modes


class TestProbeModesGuard:
    def test_streaming_index_is_refused_before_any_open(self, tmp_path) -> None:
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
        dm._claimed_camera_indices = lambda: {0: "so101-arm-1"}  # type: ignore[method-assign]
        dm._streaming_indices = lambda live: {0}  # type: ignore[method-assign]
        with pytest.raises(PermissionError, match="so101-arm-1"):
            dm.probe_modes(0, {"so101-arm-1": ["top"]})
