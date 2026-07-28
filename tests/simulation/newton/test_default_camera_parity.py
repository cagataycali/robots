"""A Newton world must expose the same default camera the MuJoCo backend does.

``MuJoCoSimEngine.create_world`` registers a built-in three-quarter view as
``world.cameras["default"]``, so ``get_observation`` on a freshly created world
returns a rendered frame. The Newton backend did not register it. Its
``render()`` and ``list_cameras()`` both already claimed ``"default"`` was
available -- ``list_cameras()`` literally returns ``["default", ...]`` and
``render()`` special-cases the name -- but ``get_observation`` iterates
``world.cameras``, which was empty.

So a Newton world that never called ``add_camera`` produced a pixel-less
observation while the identical MuJoCo world produced a frame, and nothing raised
to explain it: a vision policy simply found no image key on one backend only.

Gated on Newton + Warp; the cross-backend test additionally needs mujoco.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


def _make_engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    return NewtonSimEngine(solver="mujoco")


class TestDefaultCameraRegistered:
    def test_fresh_world_has_a_default_camera(self):
        sim = _make_engine()
        try:
            sim.create_world()
            assert "default" in sim._world.cameras
        finally:
            sim.destroy()

    def test_observation_carries_a_frame_without_add_camera(self):
        """The regression: no add_camera call, yet an image must be present."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=3)

            obs = sim.get_observation()

            assert "default" in obs
            frame = np.asarray(obs["default"])
            assert frame.ndim == 3 and frame.shape[2] == 3
            assert frame.dtype == np.uint8
        finally:
            sim.destroy()

    def test_the_default_frame_is_real_pixels_not_a_zero_buffer(self):
        """A plausible-looking empty allocation would pass a shape-only check."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=3)

            frame = np.asarray(sim.get_observation()["default"])

            assert frame.max() > 0, "frame is entirely black"
            # A rendered scene has many distinct colours; a fill has one or two.
            distinct = len(np.unique(frame.reshape(-1, 3), axis=0))
            assert distinct > 10, f"only {distinct} distinct colours - not a render"
        finally:
            sim.destroy()

    def test_skip_images_still_suppresses_the_default_frame(self):
        """The new camera must respect the existing skip_images contract."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=2)

            obs = sim.get_observation(skip_images=True)

            assert "default" not in obs
        finally:
            sim.destroy()

    def test_add_camera_still_works_alongside_the_default(self):
        """Registering the default must not displace user cameras."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            assert sim.add_camera("wrist", position=[0.4, 0.0, 0.3], target=[0.0, 0.0, 0.1])["status"] == "success"
            sim.step(n_steps=2)

            obs = sim.get_observation()

            assert "default" in obs
            assert "wrist" in obs
        finally:
            sim.destroy()


class TestCrossBackendParity:
    def test_same_world_yields_the_same_observation_keys_on_both_backends(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        newton_sim = _make_engine()
        mujoco_sim = MuJoCoSimEngine()
        try:
            for sim in (newton_sim, mujoco_sim):
                sim.create_world()
                sim.add_robot("so101")
                sim.step(n_steps=3)

            assert set(newton_sim.get_observation()) == set(mujoco_sim.get_observation())
        finally:
            newton_sim.destroy()
            mujoco_sim.destroy()

    def test_default_frames_have_the_same_shape_and_dtype(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        newton_sim = _make_engine()
        mujoco_sim = MuJoCoSimEngine()
        try:
            for sim in (newton_sim, mujoco_sim):
                sim.create_world()
                sim.add_robot("so101")
                sim.step(n_steps=3)
            newton_frame = np.asarray(newton_sim.get_observation()["default"])
            mujoco_frame = np.asarray(mujoco_sim.get_observation()["default"])

            assert newton_frame.shape == mujoco_frame.shape
            assert newton_frame.dtype == mujoco_frame.dtype
        finally:
            newton_sim.destroy()
            mujoco_sim.destroy()
