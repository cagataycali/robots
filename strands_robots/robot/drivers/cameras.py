"""Hardware cameras: ``cameras={"front": 0, "wrist": {"index": 1, "width": 640, "height": 480, "fps": 30}}``.

A bare index/path takes the device's default mode. A spec sets width/height/fps (LeRobot's ``OpenCVCameraConfig``
fields, same names) and is CHECKED: a camera that delivers another size than asked is refused at the first frame,
naming both sizes (GR00T and every LeRobot checkpoint are trained at one size; a silent resize would feed them
something else). The spec is what the Contract records for the camera.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    index: int | str
    width: int | None = None
    height: int | None = None
    fps: float | None = None

    @classmethod
    def of(cls, name: str, v: Any) -> CameraSpec:
        if isinstance(v, CameraSpec):
            return v
        if isinstance(v, int | str):
            return cls(v)
        if isinstance(v, Mapping):
            unknown = set(v) - {"index", "index_or_path", "width", "height", "fps"}
            if unknown:
                raise TypeError(f"camera {name!r}: unknown field(s) {sorted(unknown)}; use index, width, height, fps")
            idx = v.get("index", v.get("index_or_path"))
            if idx is None:
                raise TypeError(f"camera {name!r}: needs index= (a device number or path)")
            return cls(idx, v.get("width"), v.get("height"), v.get("fps"))
        raise TypeError(f"camera {name!r}: give an index, a path or {{index, width, height, fps}}, got {v!r}")


class Cameras:
    """The cameras of one hardware body, opened lazily with OpenCV, read as HxWx3 uint8 RGB."""

    def __init__(self, cfg: Mapping[str, Any] | None) -> None:
        self.specs = {n: CameraSpec.of(n, v) for n, v in (cfg or {}).items()}
        self._caps: dict[str, Any] = {}

    def _open(self, name: str) -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("cameras need opencv: pip install opencv-python") from exc
        s = self.specs[name]
        cap = cv2.VideoCapture(s.index)
        for prop, v in ((cv2.CAP_PROP_FRAME_WIDTH, s.width), (cv2.CAP_PROP_FRAME_HEIGHT, s.height),
                        (cv2.CAP_PROP_FPS, s.fps)):  # fmt: skip
            if v is not None:
                cap.set(prop, v)
        self._caps[name] = cap
        return cap

    def read(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, s in self.specs.items():
            cap = self._caps.get(name) or self._open(name)
            ok, bgr = cap.read()
            if not ok:
                continue
            h, w = bgr.shape[:2]
            if (s.width is not None and w != s.width) or (s.height is not None and h != s.height):
                raise ValueError(f"camera {name!r} delivers {w}x{h}, asked for {s.width}x{s.height}")
            out[name] = np.ascontiguousarray(bgr[:, :, ::-1])
        return out

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
