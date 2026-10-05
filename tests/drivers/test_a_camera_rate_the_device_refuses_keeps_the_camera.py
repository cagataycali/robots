"""A camera that accepts a rate it cannot deliver keeps working, and is never offered that rate.

Measured on macOS (AVFoundation, OpenCV 4.14.0, a Sonix UVC camera) before the
change: the camera reports ``fps=5.0`` when idle, the dashboard's mode probe
offered ``640x480@5`` as a native mode, and ``CAP_PROP_FPS=5`` made the next
``read()`` return ``False`` (10/15/24/30 all delivered). So spawning an arm with
the advertised mode dropped the camera with "another process may hold the
device" while nothing held it.

The fake below is that camera: it reports 640x480@5 when nobody asks, honours
any size, and delivers frames at every rate except 5. One pin per surface:
``OpenCVCamera.open`` falls back and names the refused mode, and
``DeviceManager.probe_modes`` only offers a mode a frame arrived at.
"""

from __future__ import annotations

from typing import Any

import pytest

import strands_robots.drivers.cameras as cameras_mod
from strands_robots.dashboard.device_manager import DeviceManager
from strands_robots.drivers.cameras import CameraSpec, OpenCVCamera

W, H, FPS = 3, 4, 5  # the cv2.CAP_PROP_* ids


class _SonixLikeCv2:
    """Just enough of ``cv2`` for one camera whose idle rate is a rate it refuses."""

    CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS = W, H, FPS

    def __init__(self, refused_rates: set[float], delivers: bool = True) -> None:
        self.refused_rates = refused_rates
        self.delivers = delivers

    def VideoCapture(self, _source: Any) -> Any:  # noqa: N802 - cv2's name
        cv2 = self

        class _Capture:
            props = {W: 640.0, H: 480.0, FPS: 5.0}
            fps_set: float | None = None

            def isOpened(self) -> bool:  # noqa: N802
                return True

            def set(self, prop: int, value: float) -> bool:
                self.props = {**self.props, prop: float(value)}
                if prop == FPS:
                    self.fps_set = float(value)
                return True

            def get(self, prop: int) -> float:
                return self.props[prop]

            def read(self) -> tuple[bool, Any]:
                ok = cv2.delivers and self.fps_set not in cv2.refused_rates
                return ok, (object() if ok else None)

            def release(self) -> None:
                pass

        return _Capture()


@pytest.mark.parametrize(
    ("fps", "delivers", "refused_mode", "error"),
    [
        pytest.param(10, True, None, None, id="a-rate-it-delivers-opens-as-asked"),
        pytest.param(5, True, "640x480@5", None, id="a-refused-rate-falls-back-and-is-named"),
        pytest.param(5, False, None, "no frame arrived at 640x480@5", id="nothing-delivers-names-the-mode"),
    ],
)
def test_open_keeps_the_camera_when_only_the_rate_is_refused(
    monkeypatch: pytest.MonkeyPatch, fps: int, delivers: bool, refused_mode: str | None, error: str | None
) -> None:
    fake = _SonixLikeCv2(refused_rates={5.0}, delivers=delivers)
    monkeypatch.setattr(cameras_mod, "require_optional", lambda *a, **k: fake)
    camera = OpenCVCamera(CameraSpec(name="wrist", index_or_path=0, width=640, height=480, fps=fps))
    if error:
        with pytest.raises(OSError, match=error):
            camera.open()
        assert not camera.is_open
        return
    camera.open()
    assert camera.is_open
    assert camera.describe()["refused_mode"] == refused_mode


def test_probe_offers_only_modes_a_frame_arrived_at(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    fake = _SonixLikeCv2(refused_rates={5.0, 60.0})
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake)
    dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    probed = dm.probe_modes(0)
    offered = {(m["width"], m["height"], m["fps"]) for m in probed["modes"]}
    assert probed["native"]["fps"] == 5.0  # still reported, as the device's idle rate
    assert (640, 480, 5) not in offered and not any(fps == 60 for _, _, fps in offered)
    assert (640, 480, 15) in offered and (640, 480, 30) in offered
