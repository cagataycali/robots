# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Newton's ray-traced render must reuse its resources, not rebuild them per frame.

``_render_rgb`` is called once per frame per camera (from ``render``, ``get_frame``,
and once per registered camera in ``get_observation``). Every call constructed a new
``sensors.SensorTiledCamera(model=self._model)``, re-created the directional light,
recomputed the whole per-pixel ray grid, and allocated a fresh color output buffer
and camera-transform array. Only the transform depends on simulation state.

Measured on an NVIDIA Thor at 224x224, so101 + ground plane::

    BEFORE get_frame: mean 31.5 ms  median 30.6
    reused resources: mean  0.7 ms  median  0.7   -> 44.8x
    AFTER  get_frame: mean  1.2 ms  median  1.2   -> 25x through the public path

That cost is per camera per control step, so it dominated the IL rollout loop: the
ledger measured a 30-step so101 + 224x224 wrist-camera rollout at 0.85 Hz against a
30 Hz target.

The risk of this fix is OVER-caching, so most of these tests are correctness rather
than counting: two camera poses at one resolution must still differ, a returned-to
pose must reproduce exactly, a different FOV must key separately, a scene mutation
must invalidate (a ``SensorTiledCamera`` binds the ``Model`` it was built with, and
``_rebuild`` finalizes a NEW one), and a ``randomize(randomize_lighting=True)`` must
still change the pixels even though the light lives on the cached camera.

``compute_pinhole_camera_rays`` is deprecated in newton 1.4.0; the cache builder now
prefers ``compute_camera_rays_pinhole``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

from strands_robots.simulation.newton.simulation import NewtonSimEngine  # noqa: E402

_EYE = (0.6, 0.6, 0.5)
_TARGET = (0.0, 0.0, 0.15)


@pytest.fixture
def engine():
    eng = NewtonSimEngine()
    eng.create_world()
    assert eng.add_robot("so101")["status"] == "success"
    # A visible object gives the frames something to differ about.
    assert (
        eng.add_object("box", position=[0.2, 0.0, 0.05], size=[0.04, 0.04, 0.04], color=[1.0, 0.0, 0.0])["status"]
        == "success"
    )
    try:
        yield eng
    finally:
        eng.destroy()


def _render(engine, w=96, h=96, eye=_EYE, target=_TARGET, fov_deg=50.0) -> np.ndarray:
    return engine._render_rgb(w, h, eye=eye, target=target, fov_deg=fov_deg)


class TestResourcesAreBuiltOnce:
    def test_five_frames_build_one_camera(self, monkeypatch):
        """Regression: this constructed 5 cameras, 5 ray grids, 5 buffers."""
        engine = NewtonSimEngine()
        engine.create_world()
        try:
            assert engine.add_robot("so101")["status"] == "success"
            original = engine._nt.sensors.SensorTiledCamera
            built: list[int] = []

            class _Counting(original):  # type: ignore[misc, valid-type]
                def __init__(self, *args, **kwargs):
                    built.append(1)
                    super().__init__(*args, **kwargs)

            monkeypatch.setattr(engine._nt.sensors, "SensorTiledCamera", _Counting)

            for _ in range(5):
                engine.get_frame("default", 64, 64)

            assert len(built) == 1, f"{len(built)} tiled cameras built for 5 frames"
        finally:
            engine.destroy()

    def test_two_resolutions_build_two_cameras(self, monkeypatch):
        """The ray grid encodes the resolution, so it cannot be shared."""
        engine = NewtonSimEngine()
        engine.create_world()
        try:
            assert engine.add_robot("so101")["status"] == "success"
            original = engine._nt.sensors.SensorTiledCamera
            built: list[int] = []

            class _Counting(original):  # type: ignore[misc, valid-type]
                def __init__(self, *args, **kwargs):
                    built.append(1)
                    super().__init__(*args, **kwargs)

            monkeypatch.setattr(engine._nt.sensors, "SensorTiledCamera", _Counting)

            engine.get_frame("default", 64, 64)
            engine.get_frame("default", 32, 32)
            engine.get_frame("default", 64, 64)
            engine.get_frame("default", 32, 32)

            assert len(built) == 2, f"{len(built)} cameras for 2 distinct resolutions"
        finally:
            engine.destroy()

    def test_the_cache_is_keyed_on_resolution_and_fov(self, engine):
        _render(engine, 96, 96, fov_deg=50.0)
        _render(engine, 96, 96, fov_deg=20.0)
        _render(engine, 64, 64, fov_deg=50.0)

        assert set(engine._render_cache) == {(96, 96, 50.0), (96, 96, 20.0), (64, 64, 50.0)}


class TestPixelsAreStillCorrect:
    def test_two_camera_poses_at_one_resolution_still_differ(self, engine):
        """The pose is written per frame; it must NOT be baked into the cache."""
        front = _render(engine, eye=(0.6, 0.6, 0.5))
        behind = _render(engine, eye=(-0.6, -0.6, 0.5))

        assert list(engine._render_cache) == [(96, 96, 50.0)], "the two poses did not share one entry"
        assert not np.array_equal(front, behind), "both poses rendered identical pixels"

    def test_returning_to_a_pose_reproduces_it_exactly(self, engine):
        """A reused buffer must not leak the previous frame's state."""
        first = _render(engine, eye=(0.6, 0.6, 0.5))
        _render(engine, eye=(-0.6, -0.6, 0.5))
        again = _render(engine, eye=(0.6, 0.6, 0.5))

        assert np.array_equal(first, again)

    def test_a_different_fov_renders_differently(self, engine):
        wide = _render(engine, fov_deg=50.0)
        narrow = _render(engine, fov_deg=20.0)

        assert not np.array_equal(wide, narrow)

    def test_frames_change_as_the_simulation_advances(self, engine):
        """Stepping must move pixels: bvh_refit_shapes still runs per frame."""
        before = _render(engine)
        # Newton reads the MJCF joint labels, which for so101 are "1".."6".
        joints = engine.robot_joint_names("so101")
        assert engine.send_action({joints[0]: 1.2, joints[1]: -0.8}, robot_name="so101")["status"] == "success"
        engine.step(60)
        after = _render(engine)

        assert not np.array_equal(before, after), "the arm moved but the frame did not"


class TestInvalidation:
    def test_a_scene_mutation_drops_the_cache(self, engine):
        """A cached camera binds the OLD model; _rebuild finalizes a new one."""
        first = _render(engine)
        assert engine._render_cache, "nothing was cached"

        assert (
            engine.add_object("sphere", position=[-0.2, 0.1, 0.05], size=[0.05], color=[0.0, 0.0, 1.0])["status"]
            == "success"
        )

        assert engine._render_cache == {}, "the cache survived a rebuild"
        after = _render(engine)
        assert not np.array_equal(first, after), "the new object is not in the frame"

    def test_destroy_drops_the_cache(self, engine):
        _render(engine)
        assert engine._render_cache

        engine.destroy()

        assert engine._render_cache == {}
        assert engine._render_cache_light_dir is None

    def test_reset_drops_the_cache(self, engine):
        """reset() goes through _rebuild, so it must invalidate too."""
        _render(engine)
        assert engine._render_cache

        assert engine.reset()["status"] == "success"

        assert engine._render_cache == {}

    def test_a_new_light_direction_is_visible_through_the_cache(self, engine):
        """The light lives on the cached camera's render context."""
        before = _render(engine)

        engine.randomize(randomize_colors=False, randomize_lighting=True, seed=3)

        assert engine._dr_light_dir is not None, "randomize did not set a light direction"
        after = _render(engine)
        assert not np.array_equal(before, after), "the light direction changed but the pixels did not"

    def test_a_second_light_direction_is_also_visible(self, engine):
        """One invalidation is not enough: every change must be tracked."""
        engine.randomize(randomize_colors=False, randomize_lighting=True, seed=3)
        first = _render(engine)
        engine.randomize(randomize_colors=False, randomize_lighting=True, seed=99)
        second = _render(engine)

        assert engine._render_cache_light_dir == engine._dr_light_dir
        assert not np.array_equal(first, second)


class TestRayHelper:
    def test_the_deprecated_ray_helper_is_not_called(self, engine, recwarn):
        """newton 1.4.0 deprecates compute_pinhole_camera_rays.

        The repo's own Newton test run emitted that DeprecationWarning once per
        rendered frame.
        """
        engine._invalidate_render_cache()

        _render(engine, 48, 48)

        deprecated = [
            warning
            for warning in recwarn
            if issubclass(warning.category, DeprecationWarning) and "pinhole" in str(warning.message).lower()
        ]
        assert not deprecated, [str(warning.message) for warning in deprecated]


class TestRenderStillWorksEndToEnd:
    def test_render_returns_a_png(self, engine):
        result = engine.render("default", 64, 64)

        assert result["status"] == "success", result
        assert any("image" in block for block in result["content"])

    def test_get_observation_renders_every_registered_camera(self, engine):
        assert (
            engine.add_camera("wrist", position=[0.3, 0.0, 0.3], target=[0.0, 0.0, 0.1], width=48, height=48)["status"]
            == "success"
        )

        obs = engine.get_observation("so101")

        images = [key for key in obs if "image" in key or "wrist" in key]
        assert images, f"no camera keys in observation: {sorted(obs)}"
