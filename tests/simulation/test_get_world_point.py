# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic contract of ``SimEngine.get_world_point``.

``get_world_point`` is a concrete facade over ``get_frame`` /
``get_camera_params``, so its unprojection math and its error contract are
identical on every backend. These tests drive it through a stub engine with a
synthetic pinhole camera and a hand-authored depth buffer: the expected world
points are computed by hand from the documented conventions (OpenGL optical
frame - +X right, +Y up, -Z forward - meeting image coordinates where ``v``
grows downward), so a flipped sign or a transposed pose fails here rather than
being absorbed by a rendered scene.
"""

from typing import Any

import numpy as np
import pytest

from strands_robots.rendering import CameraParams
from strands_robots.simulation.base import SimEngine

# Synthetic camera: 100x80 image, fx = fy = 50 px, principal point at the center.
WIDTH, HEIGHT = 100, 80
FOCAL = 50.0
CX, CY = 50.0, 40.0
HALF_FOCAL = 25.0
ZFAR = 10.0
CAM_POS = np.array([1.0, 2.0, 3.0])


class _StubEngine(SimEngine):
    """Stub engine exposing one synthetic camera with a caller-authored depth map."""

    def __init__(self, depth: np.ndarray | None, rotation: np.ndarray | None = None) -> None:
        self._depth = depth
        self._rotation = np.eye(3) if rotation is None else rotation
        self.raise_on_frame: Exception | None = None

    # ----- the two APIs get_world_point is built on ----- #

    def get_frame(self, camera_name="default", width=None, height=None):  # type: ignore[no-untyped-def]
        if self.raise_on_frame is not None:
            raise self.raise_on_frame
        rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        return rgb, self._depth

    def get_camera_params(self, camera_name="default", width=None, height=None) -> CameraParams:
        pose = np.eye(4)
        pose[:3, :3] = self._rotation
        pose[:3, 3] = CAM_POS
        return CameraParams(
            K=np.array([[FOCAL, 0.0, CX], [0.0, FOCAL, CY], [0.0, 0.0, 1.0]]),
            T_world_cam=pose,
            width=WIDTH,
            height=HEIGHT,
            znear=0.01,
            zfar=ZFAR,
        )

    # ----- unused abstract surface ----- #

    def create_world(self, timestep=None, gravity=None, ground_plane=True):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def destroy(self):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def reset(self):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def step(self, n_steps: int = 1):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def get_state(self):  # type: ignore[no-untyped-def]
        return {"sim_time": 0.0, "step_count": 0}

    def add_robot(self, name, **kw):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def remove_robot(self, name):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return []

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return []

    def add_object(self, name, **kw):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def remove_object(self, name):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):  # type: ignore[no-untyped-def]
        return {}

    def send_action(self, action, robot_name=None, n_substeps=1):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def render(self, camera_name="default", width=None, height=None):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}


def _depth_map(fill: float = ZFAR) -> np.ndarray:
    return np.full((HEIGHT, WIDTH), fill, dtype=np.float32)


def _json_block(result: dict[str, Any]) -> dict[str, Any]:
    return next(block["json"] for block in result["content"] if "json" in block)


def test_unprojects_pixels_through_the_documented_convention() -> None:
    """Three hand-computable pixels, identity camera rotation.

    With ``T_world_cam`` rotation = I the camera axes ARE the world axes, so for
    depth ``d``:

    * principal point ``(cx, cy)``            -> ``CAM_POS + [0, 0, -d]``
    * half a focal length right ``(cx+f/2)``  -> ``CAM_POS + [d/2, 0, -d]``
    * half a focal length down ``(cy+f/2)``   -> ``CAM_POS + [0, -d/2, -d]``
      (``v`` grows downward while camera +Y points up)
    """
    depth = _depth_map()
    depth[int(CY), int(CX)] = 2.0
    depth[int(CY), int(CX + HALF_FOCAL)] = 3.0
    depth[int(CY + HALF_FOCAL), int(CX)] = 4.0
    engine = _StubEngine(depth)

    result = engine.get_world_point("synthetic", pixels=[[CX, CY], [CX + HALF_FOCAL, CY], [CX, CY + HALF_FOCAL]])

    assert result["status"] == "success"
    block = _json_block(result)
    assert block["n_valid"] == 3
    assert block["n_requested"] == 3
    assert block["dropped"] == []
    assert block["depths"] == [2.0, 3.0, 4.0]
    expected = [
        [1.0, 2.0, 1.0],  # CAM_POS + [0, 0, -2]
        [2.5, 2.0, 0.0],  # CAM_POS + [1.5, 0, -3]
        [1.0, 0.0, -1.0],  # CAM_POS + [0, -2, -4]
    ]
    assert np.allclose(block["points"], expected)
    # Per-axis median of the three points, not the centroid.
    assert np.allclose(block["point"], [1.0, 2.0, 0.0])
    assert block["width"] == WIDTH and block["height"] == HEIGHT


def test_camera_rotation_is_applied_world_from_camera() -> None:
    """A 90-degree yaw must rotate the camera-frame point, not its transpose.

    ``Rz(90)`` maps camera +X to world +Y, so a pixel half a focal length right
    of center at depth 2 (camera-frame ``[1, 0, -2]``) lands at
    ``CAM_POS + [0, +1, -2]``. Applying ``T_world_cam`` transposed would instead
    give ``CAM_POS + [0, -1, -2]``.
    """
    depth = _depth_map()
    depth[int(CY), int(CX + HALF_FOCAL)] = 2.0
    yaw90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    engine = _StubEngine(depth, rotation=yaw90)

    block = _json_block(engine.get_world_point("synthetic", pixels=[[CX + HALF_FOCAL, CY]]))

    assert np.allclose(block["point"], CAM_POS + np.array([0.0, 1.0, -2.0]))


def test_background_pixels_are_dropped_not_fabricated() -> None:
    """Far-clip pixels contribute nothing and are reported, never counted in."""
    depth = _depth_map()
    depth[int(CY), int(CX)] = 2.0
    engine = _StubEngine(depth)

    result = engine.get_world_point("synthetic", pixels=[[CX, CY], [0, 0], [WIDTH - 1, HEIGHT - 1]])

    assert result["status"] == "success"
    block = _json_block(result)
    assert block["n_valid"] == 1
    assert block["dropped"] == [1, 2]
    assert block["points"][1] is None and block["points"][2] is None
    assert block["depths"] == [2.0, None, None]
    # The median is taken over the surviving sample only.
    assert np.allclose(block["point"], [1.0, 2.0, 1.0])
    assert "dropped pixels [1, 2]" in result["content"][0]["text"]


def test_every_pixel_on_background_is_an_error() -> None:
    """No usable depth means no world point - never a far-plane coordinate."""
    engine = _StubEngine(_depth_map())

    result = engine.get_world_point("synthetic", pixels=[[10, 10], [20, 20]])

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "none of the 2 requested pixels carried usable depth" in text
    assert "zfar=10.000m" in text


def test_backend_without_depth_reports_a_structured_error() -> None:
    """A depth-less backend (Newton returns ``(rgb, None)``) errors, not raises."""
    engine = _StubEngine(None)

    result = engine.get_world_point("synthetic", pixels=[[10, 10]])

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "renders no depth on this backend" in text
    assert "get_body_state" in text


def test_render_failure_surfaces_as_an_error_naming_the_camera() -> None:
    """An unknown camera comes back through the envelope, unquoted."""
    engine = _StubEngine(_depth_map())
    engine.raise_on_frame = KeyError("Camera 'nope' not found. Available: ['synthetic']")

    result = engine.get_world_point("nope", pixels=[[10, 10]])

    assert result["status"] == "error"
    assert result["content"][0]["text"] == ("get_world_point: Camera 'nope' not found. Available: ['synthetic']")


@pytest.mark.parametrize(
    ("pixels", "fragment"),
    [
        (None, "'pixels' is required"),
        ([], "'pixels' is empty"),
        ("10,20", "must be a sequence of [u, v] pairs, got str"),
        ({"u": 10, "v": 20}, "must be a sequence of [u, v] pairs, got dict"),
        (7, "must be a sequence of [u, v] pairs, got int"),
        ([[10]], "'pixels'[0] must be a [u, v] pair of 2 numbers, got 1"),
        ([[10, 20, 30]], "'pixels'[0] must be a [u, v] pair of 2 numbers, got 3"),
        ([10, 20], "'pixels'[0] must be a [u, v] pair, got int"),
        ([[10, 20], [10.5, 20]], "'pixels'[1] u must be a whole image coordinate, got 10.5"),
        ([[10, float("nan")]], "'pixels'[0] v must be a whole image coordinate"),
        ([[True, 20]], "'pixels'[0] u must be an image coordinate, got bool"),
        ([[10, "20"]], "'pixels'[0] v must be an image coordinate, got str"),
    ],
)
def test_malformed_pixels_are_rejected(pixels: Any, fragment: str) -> None:
    """Every shape/type a caller can get wrong names the pixel and the fix."""
    engine = _StubEngine(_depth_map())

    result = engine.get_world_point("synthetic", pixels=pixels)

    assert result["status"] == "error"
    assert fragment in result["content"][0]["text"]


def test_pixel_outside_the_frame_is_rejected_with_the_bounds() -> None:
    """Out-of-frame indices are refused, not wrapped around by NumPy."""
    engine = _StubEngine(_depth_map())

    result = engine.get_world_point("synthetic", pixels=[[10, 10], [WIDTH, 10], [-1, 0]])

    assert result["status"] == "error"
    assert result["content"][0]["text"] == (
        f"get_world_point: 'pixels'[1] = [{WIDTH}, 10] is outside the "
        f"{WIDTH}x{HEIGHT} frame of camera 'synthetic' "
        f"(u in [0, {WIDTH - 1}], v in [0, {HEIGHT - 1}])."
    )


def test_numpy_pixel_arrays_are_accepted() -> None:
    """Pixels picked off a rendered frame arrive as NumPy integers."""
    depth = _depth_map()
    depth[int(CY), int(CX)] = 2.0
    engine = _StubEngine(depth)

    block = _json_block(
        engine.get_world_point("synthetic", pixels=np.array([[CX, CY]], dtype=np.int64))  # type: ignore[arg-type]
    )

    assert np.allclose(block["point"], [1.0, 2.0, 1.0])
