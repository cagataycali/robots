"""A camera that opens but sends no frame says which mode it was asked for and got.

``OpenCVCamera.open`` used to end this case with "another process may hold the
device", which is wrong as often as it is right: a UVC camera asked for a rate
it does not offer opens, accepts the request, and sends nothing. The message is
what the dashboard shows next to the dropped camera, so it must carry the mode
that was tried and the mode the device settled on. ``cv2`` is replaced with a
device stand-in that answers ``set``/``get`` the way a real backend does.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from strands_robots.drivers.cameras import CameraSpec, OpenCVCamera


class _Capture:
    def __init__(self, device: dict[int, float], refuses: set[int]) -> None:
        self.device, self.refuses, self.released = dict(device), refuses, False

    def isOpened(self) -> bool:  # noqa: N802 - the cv2 name
        return not self.released

    def set(self, prop: int, value: float) -> bool:
        return prop not in self.refuses

    def get(self, prop: int) -> float:
        return self.device.get(prop, 0.0)

    def read(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        self.released = True


def _cv2(captures: list[_Capture], refuses: set[int]) -> types.ModuleType:
    """Every ``VideoCapture`` call opens a fresh handle on the same device, as a real backend does."""

    def video_capture(_source: Any) -> _Capture:
        captures.append(_Capture({3: 1280.0, 4: 720.0, 5: 30.0}, refuses))
        return captures[-1]

    cv2 = types.ModuleType("cv2")
    cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS = 3, 4, 5  # type: ignore[attr-defined]
    cv2.VideoCapture = video_capture  # type: ignore[attr-defined]
    return cv2


@pytest.mark.parametrize(
    ("spec", "refuses", "says", "never"),
    [
        # The reported case: fps 5 asked, the device keeps 30 and sends nothing.
        (
            {"width": 640, "height": 480, "fps": 5.0},
            set(),
            ["no frame arrived at 640x480@5", "the device answers 1280x720@30", "pick a mode the device lists"],
            ["refused"],
        ),
        # A backend that rejects the value outright is named for the field it rejected.
        ({"fps": 5.0}, {5}, ["the device refused fps;"], []),
        # Nothing asked: the device's own mode is what failed, and the message says so.
        (
            {},
            set(),
            ["no frame arrived at the device's own size at the device's own rate;", "answers 1280x720@30"],
            ["refused", "nor without"],
        ),
    ],
)
def test_the_message_names_the_mode_tried_and_the_mode_the_device_answers(
    monkeypatch: pytest.MonkeyPatch, spec: dict[str, Any], refuses: set[int], says: list[str], never: list[str]
) -> None:
    captures: list[_Capture] = []
    monkeypatch.setitem(sys.modules, "cv2", _cv2(captures, refuses))
    camera = OpenCVCamera(CameraSpec(name="wrist", index_or_path=0, **spec))

    with pytest.raises(OSError) as raised:
        camera.open()

    message = str(raised.value)
    assert message.startswith("camera 'wrist': opened 0 ")
    for fragment in says:
        assert fragment in message
    for fragment in never:
        assert fragment not in message
    assert captures and all(c.released for c in captures) and not camera.is_open
