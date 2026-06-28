"""Named per-robot cameras on the Newton backend (parity with MuJoCo).

Exercises the multi-camera contract: registering named cameras via
``add_camera``, rendering each from its own viewpoint, body-mounted wrist
cameras that track the arm, and camera frames surfacing in
``get_observation``. These need the real Newton engine (a GPU model + the
ray-traced tiled camera sensor), so the module is skipped when Newton/Warp
are not installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


@pytest.fixture
def engine_with_so101():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    sim = NewtonSimEngine(solver="mujoco")
    sim.create_world()
    sim.add_robot("so101")
    yield sim
    sim.destroy()


def _pixel_mean(render_result: dict) -> float:
    for block in render_result["content"]:
        if "json" in block:
            return float(block["json"]["pixel_mean"])
    raise AssertionError("no json stats block in render result")


def _png_bytes(render_result: dict) -> bytes:
    for block in render_result["content"]:
        if "image" in block:
            return block["image"]["source"]["bytes"]
    raise AssertionError("no image block in render result")


class TestNamedCameraRegistration:
    def test_add_camera_appears_in_describe_and_list(self, engine_with_so101):
        sim = engine_with_so101
        assert sim.add_camera("front", position=[0.0, -0.6, 0.4], target=[0.0, 0.0, 0.1])["status"] == "success"
        assert "front" in sim.describe()["cameras"]
        assert "default" in sim.describe()["cameras"]
        assert sim.list_cameras() == ["default", "front"]

    def test_duplicate_camera_rejected(self, engine_with_so101):
        sim = engine_with_so101
        sim.add_camera("front", position=[0.0, -0.6, 0.4], target=[0.0, 0.0, 0.1])
        result = sim.add_camera("front", position=[0.0, -0.6, 0.4], target=[0.0, 0.0, 0.1])
        assert result["status"] == "error"
        assert "already exists" in result["content"][0]["text"]

    def test_reserved_name_rejected(self, engine_with_so101):
        result = engine_with_so101.add_camera("default", position=[0.0, -0.6, 0.4], target=[0.0, 0.0, 0.1])
        assert result["status"] == "error"
        assert "reserved" in result["content"][0]["text"]

    def test_degenerate_pose_rejected(self, engine_with_so101):
        result = engine_with_so101.add_camera("bad", position=[1.0, 1.0, 1.0], target=[1.0, 1.0, 1.0])
        assert result["status"] == "error"
        assert "look direction" in result["content"][0]["text"]

    def test_bad_position_shape_rejected(self, engine_with_so101):
        result = engine_with_so101.add_camera("bad", position=[1.0, 1.0], target=[0.0, 0.0, 0.0])
        assert result["status"] == "error"
        assert "3 elements" in result["content"][0]["text"]

    def test_remove_camera(self, engine_with_so101):
        sim = engine_with_so101
        sim.add_camera("front", position=[0.0, -0.6, 0.4], target=[0.0, 0.0, 0.1])
        assert sim.remove_camera("front")["status"] == "success"
        assert "front" not in sim.describe()["cameras"]
        assert sim.remove_camera("front")["status"] == "error"


class TestMultiCameraRender:
    def test_two_cameras_render_distinct_viewpoints(self, engine_with_so101):
        sim = engine_with_so101
        sim.add_camera("front", position=[0.0, -0.6, 0.4], target=[0.0, 0.0, 0.1])
        sim.add_camera("side", position=[0.6, 0.0, 0.4], target=[0.0, 0.0, 0.1])

        front = sim.render(camera_name="front", width=160, height=120)
        side = sim.render(camera_name="side", width=160, height=120)
        assert front["status"] == "success"
        assert side["status"] == "success"
        # Distinct viewpoints produce distinct frames.
        assert _png_bytes(front) != _png_bytes(side)
        assert front["content"][-1]["json"]["camera"] == "front"
        assert side["content"][-1]["json"]["camera"] == "side"

    def test_default_camera_still_renders(self, engine_with_so101):
        sim = engine_with_so101
        sim.add_camera("front", position=[0.0, -0.6, 0.4], target=[0.0, 0.0, 0.1])
        result = sim.render()  # default view
        assert result["status"] == "success"
        assert result["content"][-1]["json"]["camera"] == "default"

    def test_unknown_camera_errors_with_available_list(self, engine_with_so101):
        sim = engine_with_so101
        sim.add_camera("front", position=[0.0, -0.6, 0.4], target=[0.0, 0.0, 0.1])
        result = sim.render(camera_name="nope")
        assert result["status"] == "error"
        assert "front" in result["content"][0]["text"]

    def test_render_uses_camera_resolution_by_default(self, engine_with_so101):
        sim = engine_with_so101
        sim.add_camera("front", position=[0.0, -0.6, 0.4], target=[0.0, 0.0, 0.1], width=128, height=96)
        result = sim.render(camera_name="front")
        assert "128x96" in result["content"][0]["text"]


class TestWristCameraMounting:
    def test_body_mounted_camera_tracks_arm_motion(self, engine_with_so101):
        sim = engine_with_so101
        gripper = sim.list_bodies("so101")["content"][-1]["json"]["gripper_body"]
        assert gripper is not None
        sim.add_camera("wrist", position=[0.0, 0.0, 0.1], target=[0.05, 0.0, 0.0], parent_body=gripper)

        before = _pixel_mean(sim.render(camera_name="wrist", width=160, height=120))
        sim.send_action({"shoulder_pan": 0.8, "shoulder_lift": -0.6}, robot_name="so101", n_substeps=300)
        after = _pixel_mean(sim.render(camera_name="wrist", width=160, height=120))
        # A wrist-mounted camera moves with the arm, so the view changes.
        assert abs(before - after) > 0.3

    def test_unknown_parent_body_rejected(self, engine_with_so101):
        result = engine_with_so101.add_camera("wrist", parent_body="no_such_body")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]


class TestObservationCameras:
    def test_observation_includes_all_camera_frames(self, engine_with_so101):
        sim = engine_with_so101
        sim.add_camera("front", position=[0.0, -0.6, 0.4], target=[0.0, 0.0, 0.1], width=96, height=72)
        sim.add_camera("side", position=[0.6, 0.0, 0.4], target=[0.0, 0.0, 0.1], width=96, height=72)

        obs = sim.get_observation(skip_images=False)
        assert isinstance(obs["front"], np.ndarray)
        assert isinstance(obs["side"], np.ndarray)
        assert obs["front"].shape == (72, 96, 3)
        # Proprio joints still present alongside the camera frames.
        assert any(not isinstance(v, np.ndarray) for v in obs.values())

    def test_skip_images_omits_camera_frames(self, engine_with_so101):
        sim = engine_with_so101
        sim.add_camera("front", position=[0.0, -0.6, 0.4], target=[0.0, 0.0, 0.1])
        obs = sim.get_observation(skip_images=True)
        assert not any(isinstance(v, np.ndarray) for v in obs.values())


class TestRotateVecByQuat:
    def test_identity_quaternion_is_noop(self):
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        out = NewtonSimEngine._rotate_vec_by_quat(np.array([0.0, 0.0, 0.0, 1.0]), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0], atol=1e-9)

    def test_ninety_degree_z_rotation(self):
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        # +90 deg about Z maps +X -> +Y.
        s = np.sin(np.pi / 4)
        out = NewtonSimEngine._rotate_vec_by_quat(np.array([0.0, 0.0, s, s]), [1.0, 0.0, 0.0])
        np.testing.assert_allclose(out, [0.0, 1.0, 0.0], atol=1e-6)
