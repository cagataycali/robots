"""An Isaac camera's frames have the size it was added with.

``add_camera(width=224, height=224)`` renders at 640x640 (the DLSS ghosting
floor, ``_MIN_RENDER_PX``), and every consumer used to get that 640x640 frame:
``get_observation`` and recordings handed pi0.5 640x640 views and wrote dataset
features ``[640, 640, 3]`` where MuJoCo writes ``[224, 224, 3]``. The camera now
keeps both sizes and every read-back is resampled to the requested one.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.simulation.isaac.simulation import _CameraState, _frame_at_camera_size


def _cam() -> _CameraState:
    return _CameraState(
        name="wrist", prim_path="/World/Cameras/wrist", width=224, height=224, render_width=640, render_height=640
    )


def test_the_camera_remembers_both_sizes() -> None:
    cam = _cam()
    assert (cam.width, cam.height, cam.render_width, cam.render_height) == (224, 224, 640, 640)
    plain = _CameraState(name="f", prim_path="/p", width=800, height=600)
    assert (plain.render_width, plain.render_height) == (800, 600)


def test_a_colour_frame_is_downsampled_to_the_request() -> None:
    pytest.importorskip("cv2")
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    frame[:, 320:] = 200  # right half bright
    out = _frame_at_camera_size(_cam(), frame)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert out[:, :100].max() == 0 and out[:, 130:].min() == 200


def test_depth_is_resampled_without_blending_two_surfaces() -> None:
    pytest.importorskip("cv2")
    depth = np.full((640, 640), 1.0, dtype=np.float32)
    depth[:, 320:] = 5.0
    out = _frame_at_camera_size(_cam(), depth, nearest=True)
    assert out.shape == (224, 224) and set(np.unique(out).tolist()) == {1.0, 5.0}


def test_a_frame_already_at_the_request_is_returned_as_is() -> None:
    frame = np.ones((224, 224, 3), dtype=np.uint8)
    assert _frame_at_camera_size(_cam(), frame) is frame
