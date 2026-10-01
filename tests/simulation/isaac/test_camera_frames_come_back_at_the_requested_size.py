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


def test_camera_params_describe_the_frame_get_frame_returns() -> None:
    """``get_camera_params`` and ``get_frame`` are consumed as a pair (the compositor aligns a
    background off ``K`` and reads a frame at ``width x height``), so the intrinsics must be in the
    requested size's pixels: a 224x224 camera rendered at 640x640 reports a principal point at 112,
    not 320, and a focal length scaled by 224/640."""
    import types

    pytest.importorskip("strands_robots.simulation.isaac")
    from tests.simulation._isaac_engine import isaac_engine

    engine = isaac_engine()
    engine._world_created = True
    cam = _cam()
    render_k = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 320.0], [0.0, 0.0, 1.0]])
    cam.handle = types.SimpleNamespace(
        get_intrinsics_matrix=lambda: render_k,
        get_world_pose=lambda: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
    )
    engine._cameras = {"wrist": cam}

    params = engine.get_camera_params("wrist")

    assert (params.width, params.height) == (224, 224)
    assert params.K[0, 2] == pytest.approx(112.0) and params.K[1, 2] == pytest.approx(112.0)
    assert params.K[0, 0] == pytest.approx(400.0 * 224 / 640) and params.K[1, 1] == pytest.approx(400.0 * 224 / 640)
    assert params.K[2, 2] == 1.0
    with pytest.raises(ValueError, match=r"640x640.*224x224"):
        engine.get_camera_params("wrist", width=640, height=640)
