# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public raw-frame + camera-params APIs on the MuJoCo backend (issue #1537).

``get_camera_params`` is pure model math (no GL); ``get_frame`` and the
end-to-end HybridCompositor test need an offscreen GL context and are gated
behind the shared runtime probe.
"""

import os

import numpy as np
import pytest

pytest.importorskip("mujoco")

from tests.simulation.mujoco._gl_probe import requires_gl


def _make_sim(width: int = 64, height: int = 48):
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("front", position=[0.4, -0.5, 0.3], target=[0.0, 0.0, 0.1], width=width, height=height)
    sim.step(n_steps=5)
    return sim


# ----- get_camera_params (no GL required) ----- #


def test_get_camera_params_math_and_conventions() -> None:
    sim = _make_sim()
    try:
        cam = sim.get_camera_params("front")
        assert cam.width == 64 and cam.height == 48
        # K: square pixels, centered principal point, fy from the camera's
        # vertical FOV (add_camera default fov=60 deg).
        assert cam.K.shape == (3, 3)
        fy_expected = 0.5 * 48 / np.tan(np.deg2rad(60.0) / 2.0)
        assert cam.K[1, 1] == pytest.approx(fy_expected, rel=1e-6)
        assert cam.K[0, 0] == pytest.approx(cam.K[1, 1])
        assert cam.K[0, 2] == pytest.approx(32.0)
        assert cam.K[1, 2] == pytest.approx(24.0)
        # Pose: rotation is orthonormal, translation is the camera position.
        R = cam.T_world_cam[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.allclose(cam.T_world_cam[:3, 3], [0.4, -0.5, 0.3], atol=1e-6)
        # OpenGL optical convention: -Z (third column negated) points from
        # the eye towards the look-at target.
        fwd = -R[:, 2]
        expected = np.array([0.0, 0.0, 0.1]) - np.array([0.4, -0.5, 0.3])
        expected /= np.linalg.norm(expected)
        assert np.allclose(fwd, expected, atol=1e-6)
        assert 0.0 < cam.znear < cam.zfar
    finally:
        sim.destroy()


def test_get_camera_params_width_height_override() -> None:
    sim = _make_sim()
    try:
        cam = sim.get_camera_params("front", width=128, height=96)
        assert cam.width == 128 and cam.height == 96
        assert cam.K[0, 2] == pytest.approx(64.0)
    finally:
        sim.destroy()


def test_get_camera_params_rejects_unknown_names() -> None:
    sim = _make_sim()
    try:
        with pytest.raises(KeyError, match="not found"):
            sim.get_camera_params("nope")
    finally:
        sim.destroy()


def test_get_camera_params_free_camera_tokens_all_report_the_same_view() -> None:
    """Every free-camera token ``get_frame`` accepts also resolves here.

    ``get_frame`` documents ``None`` / ``""`` / ``"default"`` / ``"free"`` as
    interchangeable free-camera tokens and ``list_cameras()`` advertises
    ``"default"`` first, so the params API must answer for the same set --
    otherwise a caller that renders the default view (the hybrid compositor
    does, by default) cannot obtain its intrinsics/extrinsics.
    """
    sim = _make_sim()
    try:
        reference = sim.get_camera_params("default", width=64, height=48)
        assert reference.width == 64 and reference.height == 48
        for token in (None, "", "free"):
            cam = sim.get_camera_params(token, width=64, height=48)
            assert np.allclose(cam.K, reference.K)
            assert np.allclose(cam.T_world_cam, reference.T_world_cam)
            assert (cam.znear, cam.zfar) == (reference.znear, reference.zfar)
    finally:
        sim.destroy()


def test_get_camera_params_free_camera_math_and_conventions() -> None:
    """The free-camera pose/intrinsics follow MuJoCo's own free-camera defaults.

    MuJoCo derives the free view from the compiled model: ``vis.global_.fovy``
    for the vertical FOV and an ``azimuth``/``elevation`` orbit of
    ``stat.center`` at ``1.5 * stat.extent``. Pin those relations (and the
    OpenGL optical basis) so a refactor cannot quietly report a different
    camera than the one MuJoCo renders.
    """
    sim = _make_sim()
    try:
        model = sim.mj_model
        cam = sim.get_camera_params("default", width=64, height=48)

        fy_expected = 0.5 * 48 / np.tan(np.deg2rad(float(model.vis.global_.fovy)) / 2.0)
        assert cam.K[1, 1] == pytest.approx(fy_expected, rel=1e-9)
        assert cam.K[0, 0] == pytest.approx(cam.K[1, 1])
        assert (cam.K[0, 2], cam.K[1, 2]) == pytest.approx((32.0, 24.0))

        R = cam.T_world_cam[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)

        # The eye orbits stat.center at 1.5 * stat.extent, looking at it.
        eye = cam.T_world_cam[:3, 3]
        lookat = np.asarray(model.stat.center, dtype=float)
        assert np.linalg.norm(eye - lookat) == pytest.approx(1.5 * float(model.stat.extent), rel=1e-6)
        forward = -R[:, 2]  # OpenGL optical: -Z is the view direction
        to_target = lookat - eye
        assert np.allclose(forward, to_target / np.linalg.norm(to_target), atol=1e-9)
    finally:
        sim.destroy()


def test_get_camera_params_rejects_orthographic_free_camera() -> None:
    """An orthographic free camera is refused, not given a bogus pinhole ``K``.

    ``<visual><global orthographic="true"/>`` makes MuJoCo render a parallel
    projection, which no pinhole ``K`` can describe. Reporting perspective
    intrinsics anyway would misproject every compositing/unprojection consumer
    silently, so the call must fail loudly instead.
    """
    sim = _make_sim()
    try:
        sim.mj_model.vis.global_.orthographic = 1
        with pytest.raises(ValueError, match="orthographic"):
            sim.get_camera_params("default")
    finally:
        sim.destroy()


def test_get_camera_params_requires_world() -> None:
    from strands_robots.simulation import Simulation

    sim = Simulation()
    with pytest.raises(RuntimeError):
        sim.get_camera_params("front")


# ----- get_frame + compositor (GL required) ----- #


@requires_gl
def test_get_frame_returns_raw_rgb_and_metric_depth() -> None:
    sim = _make_sim()
    try:
        rgb, depth = sim.get_frame("front")
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8
        assert depth is not None
        assert depth.shape == (48, 64)
        assert depth.dtype == np.float32
        assert np.isfinite(depth).all()  # sanitized: no NaN/inf
        assert (depth >= 0).all()
        # The scene has geometry: some pixel is nearer than the far clip.
        cam = sim.get_camera_params("front")
        assert float(depth.min()) < cam.zfar * 0.999
    finally:
        sim.destroy()


@requires_gl
def test_get_frame_unknown_camera_raises() -> None:
    sim = _make_sim()
    try:
        with pytest.raises(KeyError, match="not found"):
            sim.get_frame("nope")
    finally:
        sim.destroy()


@requires_gl
def test_get_frame_rejects_bad_dimensions() -> None:
    sim = _make_sim()
    try:
        with pytest.raises(ValueError):
            sim.get_frame("front", width=0, height=48)
    finally:
        sim.destroy()


@requires_gl
def test_hybrid_compositor_end_to_end_over_mujoco() -> None:
    from strands_robots.rendering import HybridCompositor

    sim = _make_sim()
    try:
        frame = HybridCompositor(sim, feather_pixels=0).render("front")
        assert frame.rgb.shape == (48, 64, 3)
        assert frame.rgb.dtype == np.uint8
        # Both regimes present: some robot pixels won, some background shows.
        assert bool(frame.foreground_mask.any())
        assert not bool(frame.foreground_mask.all())
    finally:
        sim.destroy()


@requires_gl
def test_free_camera_params_describe_the_frame_get_frame_renders() -> None:
    """The reported free-camera pose/FOV reproduce the free view pixel-for-pixel.

    Params that do not match the rendered frame are worse than no params: the
    compositor would blend a backdrop rendered from a different viewpoint. So
    plant a *named* camera at the reported free-camera pose and FOV (the
    already-trusted ``cam_xpos``/``cam_xmat`` path) and require the two renders
    to agree.
    """
    sim = _make_sim(width=128, height=96)
    try:
        free = sim.get_camera_params("default", width=128, height=96)
        eye = free.T_world_cam[:3, 3]
        forward = -free.T_world_cam[:3, 2]
        fovy_deg = float(np.rad2deg(2.0 * np.arctan(0.5 * free.height / free.K[1, 1])))
        sim.add_camera(
            "free_mirror",
            position=list(eye),
            target=list(eye + forward),
            fov=fovy_deg,
            width=128,
            height=96,
        )

        rgb_free, depth_free = sim.get_frame("default", width=128, height=96)
        rgb_mirror, depth_mirror = sim.get_frame("free_mirror", width=128, height=96)

        # Same viewpoint => same pixels (allow only encoder-level rounding).
        assert np.abs(rgb_free.astype(np.int16) - rgb_mirror.astype(np.int16)).max() <= 2
        assert depth_free is not None and depth_mirror is not None
        assert np.abs(depth_free - depth_mirror).max() == pytest.approx(0.0, abs=1e-4)
    finally:
        sim.destroy()


@requires_gl
def test_hybrid_compositor_composites_the_default_free_camera() -> None:
    """``HybridCompositor(sim).render()`` works on its own default argument.

    ``render`` defaults to ``camera_name="default"``, so the zero-config path
    (no user-added camera, just the free view MuJoCo already renders) must
    composite rather than dead-end on the camera-params lookup.
    """
    from strands_robots.rendering import HybridCompositor
    from strands_robots.simulation import Simulation

    sim = Simulation()
    # ground_plane=False is the documented recipe for photoreal backdrops: with
    # MuJoCo's floor filling the frame, every pixel has geometry and nothing
    # would be left for the background to fill.
    sim.create_world(ground_plane=False)
    sim.add_object(name="cube", shape="box", position=[0.0, 0.0, 0.1], size=[0.05, 0.05, 0.05])
    sim.step(n_steps=5)
    try:
        frame = HybridCompositor(sim, default_width=96, default_height=72, feather_pixels=0).render()
        assert frame.rgb.shape == (72, 96, 3)
        assert frame.rgb.dtype == np.uint8
        assert frame.camera.width == 96 and frame.camera.height == 72
        # Both regimes present: the cube won some pixels, the backdrop shows elsewhere.
        assert bool(frame.foreground_mask.any())
        assert not bool(frame.foreground_mask.all())
        # And the depth test actually selected between the two sources.
        mask = frame.foreground_mask
        assert np.array_equal(frame.rgb[mask], frame.foreground_rgb[mask])
        assert np.array_equal(frame.rgb[~mask], frame.background_rgb[~mask])
    finally:
        sim.destroy()
