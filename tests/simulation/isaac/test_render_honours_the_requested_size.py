"""``render(width=, height=)`` answers at the size asked for, as MuJoCo does.

The RTX render product is created at the camera's ``add_camera`` resolution, and
``render``'s ``width``/``height`` were documented as "ignored on the RTX path":
measured on 6.1, ``render(width=320, height=240)`` returned a 640x480 PNG, where
MuJoCo returns 320x240. The public frame is now resampled to the request, the
json reports ``resolution`` (what was returned) beside ``native_resolution``, and
the internal ``_render_frame`` consumers keep the native frame.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .test_render_refuses_a_camera_the_scene_does_not_carry import CAM, NATIVE_H, NATIVE_W, _engine


def _png(result: dict) -> np.ndarray:
    block = next(b for b in result["content"] if "image" in b)
    return np.asarray(Image.open(io.BytesIO(block["image"]["source"]["bytes"])))


def _json(result: dict) -> dict:
    return next(b["json"] for b in result["content"] if "json" in b)


def test_a_requested_size_is_the_size_returned() -> None:
    result = _engine().render(camera_name=CAM, width=NATIVE_W // 2, height=NATIVE_H // 2)

    assert result["status"] == "success"
    assert _png(result).shape[:2] == (NATIVE_H // 2, NATIVE_W // 2)
    payload = _json(result)
    assert payload["resolution"] == [NATIVE_W // 2, NATIVE_H // 2]
    assert payload["native_resolution"] == [NATIVE_W, NATIVE_H]


def test_one_axis_keeps_the_native_other() -> None:
    result = _engine().render(camera_name=CAM, width=NATIVE_W * 2)
    assert _png(result).shape[:2] == (NATIVE_H, NATIVE_W * 2)


def test_no_request_is_the_native_frame_unchanged() -> None:
    result = _engine().render(camera_name=CAM)
    assert _png(result).shape[:2] == (NATIVE_H, NATIVE_W)
    assert "native_resolution" not in _json(result)


def test_the_internal_frame_stays_native() -> None:
    rgb, _depth, _meta = _engine()._render_frame(CAM, NATIVE_W // 2, NATIVE_H // 2)
    assert rgb is not None
    assert rgb.shape[:2] == (NATIVE_H, NATIVE_W)
