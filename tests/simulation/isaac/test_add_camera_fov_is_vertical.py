# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``add_camera(fov=)`` is the vertical FOV (fovy) on the Isaac backend.

``add_camera(fov=)`` is documented as one shared surface across backends, but
the Isaac backend used to map it onto the *horizontal* axis while MuJoCo and
Newton pass it to MuJoCo's ``fovy`` and ``get_camera_params`` falls back to
``fovy``. The same call framed a different scene on Isaac than on the other
two.

These pins do not need a running Isaac Sim: ``_create_camera_prim`` lazy-imports
the ``Camera`` sensor, so a fake sensor injected into ``sys.modules`` drives the
real focal-length / aperture code. The intrinsics the RTX handle would report
are then reconstructed with USD's own formula (``fx = width*f/h_ap``,
``fy = height*f/v_ap``) and compared, for the issue call, against the intrinsics
the MuJoCo backend reports for the same ``add_camera`` -- so the two backends
are pinned to agree at unit level.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.simulation import (
    _vertical_fov_lens_mm,
)
from tests.simulation._isaac_engine import isaac_engine

# The issue scene: a 640x480 camera at the default fov. A deliberately
# non-24 mm horizontal aperture proves the result is independent of the
# aperture's absolute value (only the aspect ratio and fov set the intrinsics).
WIDTH, HEIGHT, FOV = 640, 480, 60.0
H_APERTURE_MM = 20.955


class _FakeCamera:
    """Records the lens calls ``_create_camera_prim`` makes.

    Implements only the handful of methods the focal-length block touches.
    Its ``get_intrinsics_matrix`` applies USD's documented pinhole formula to
    the aperture + focal length actually set, so the matrix is what a real RTX
    handle would report for this configuration.
    """

    def __init__(self, **_: Any) -> None:
        self._h_ap = H_APERTURE_MM
        self._v_ap = H_APERTURE_MM
        self._focal = 0.0
        self.set_vertical_aperture_called = False

    def initialize(self) -> None:
        pass

    def get_horizontal_aperture(self) -> float:
        return self._h_ap

    def set_vertical_aperture(self, value: float) -> None:
        self.set_vertical_aperture_called = True
        self._v_ap = float(value)

    def set_focal_length(self, value: float) -> None:
        self._focal = float(value)

    def add_distance_to_image_plane_to_frame(self) -> None:
        pass

    def get_intrinsics_matrix(self) -> np.ndarray:
        fx = WIDTH * self._focal / self._h_ap
        fy = HEIGHT * self._focal / self._v_ap
        return np.array(
            [[fx, 0.0, WIDTH / 2.0], [0.0, fy, HEIGHT / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def _make_camera(monkeypatch: pytest.MonkeyPatch) -> _FakeCamera:
    """Run the real ``_create_camera_prim`` against a fake ``Camera`` sensor.

    Injects a fake ``isaacsim.sensors.camera`` module so the method's lazy
    import resolves without Isaac Sim installed and without importing the real
    (heavyweight) ``isaacsim`` package.
    """
    made: list[_FakeCamera] = []

    def _factory(**kwargs: Any) -> _FakeCamera:
        cam = _FakeCamera(**kwargs)
        made.append(cam)
        return cam

    for name in ("isaacsim", "isaacsim.sensors", "isaacsim.sensors.camera"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["isaacsim.sensors.camera"].Camera = _factory  # type: ignore[attr-defined]

    engine = isaac_engine()
    handle, _focal = engine._create_camera_prim(
        name="cam",
        prim_path="/World/cameras/cam",
        position=[0.0, 0.0, 1.5],
        target=None,  # no look-at -> no set_camera_view import
        width=WIDTH,
        height=HEIGHT,
        fov_deg=FOV,
    )
    assert made and handle is made[0]
    return handle


def _mujoco_intrinsics() -> tuple[float, float]:
    """(fx, fy) the MuJoCo backend reports for the same add_camera call."""
    pytest.importorskip("mujoco")
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    try:
        sim.create_world()
        sim.add_camera(
            "front",
            position=[0.0, 0.0, 1.5],
            target=[1.0, 0.0, 1.5],
            fov=FOV,
            width=WIDTH,
            height=HEIGHT,
        )
        cam = sim.get_camera_params("front")
        return float(cam.K[0, 0]), float(cam.K[1, 1])
    finally:
        sim.destroy()


def test_isaac_maps_fov_to_the_vertical_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """fov=60 sets fy from the HEIGHT (vertical), not the width (horizontal)."""
    cam = _make_camera(monkeypatch)
    K = cam.get_intrinsics_matrix()
    fx, fy = K[0, 0], K[1, 1]

    fy_vertical = 0.5 * HEIGHT / np.tan(np.deg2rad(FOV) / 2.0)
    fx_horizontal = 0.5 * WIDTH / np.tan(np.deg2rad(FOV) / 2.0)

    # The vertical aperture was pinned from the aspect ratio ...
    assert cam.set_vertical_aperture_called
    # ... and both axes read the vertical-FOV value (square pixels).
    assert fy == pytest.approx(fy_vertical, rel=1e-6)
    assert fx == pytest.approx(fy_vertical, rel=1e-6)
    # The old horizontal mapping (fx from width) is now clearly wrong.
    assert fx != pytest.approx(fx_horizontal, rel=1e-3)


def test_isaac_and_mujoco_intrinsics_agree_for_the_same_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The (fx, fy) the two backends report for one add_camera call agree."""
    cam = _make_camera(monkeypatch)
    K = cam.get_intrinsics_matrix()
    isaac_fx, isaac_fy = K[0, 0], K[1, 1]

    mj_fx, mj_fy = _mujoco_intrinsics()

    assert isaac_fx == pytest.approx(mj_fx, rel=1e-6)
    assert isaac_fy == pytest.approx(mj_fy, rel=1e-6)
    assert isaac_fx == pytest.approx(isaac_fy, rel=1e-6)


def test_helper_predicts_the_issue_scene_pixel_offsets() -> None:
    """A 25 deg-horizontal / 20 deg-vertical marker 3 m out lands at 194 / 151 px.

    The GPU render of the issue scene must match these analytic offsets.
    """
    v_ap, focal = _vertical_fov_lens_mm(FOV, WIDTH, HEIGHT, H_APERTURE_MM)
    fx = WIDTH * focal / H_APERTURE_MM
    fy = HEIGHT * focal / v_ap
    assert fx == pytest.approx(fy, rel=1e-9)

    red_horizontal_px = fx * np.tan(np.deg2rad(25.0))
    green_vertical_px = fy * np.tan(np.deg2rad(20.0))
    assert red_horizontal_px == pytest.approx(194.0, abs=1.0)
    assert green_vertical_px == pytest.approx(151.0, abs=1.0)
