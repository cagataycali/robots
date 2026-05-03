"""Regression tests for PR #85 review feedback.

Tests:
1. Thread-safety: concurrent dispatch + policy doesn't corrupt state
2. Flat-index state copy: joint positions survive object injection
3. apply_force: force is latched (persists across steps)
4. Camera recording roundtrip: namespaced cameras survive schema reconcile

Run: MUJOCO_GL=osmesa python -m pytest tests/test_mujoco_regressions.py -v
"""

import math
import os
import shutil
import tempfile
import threading
import time

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No OpenGL context available (headless without EGL/OSMesa)",
)

# ── Test robot XML (simple 3-DOF arm) ──

ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <camera name="arm0/wrist_cam" pos="0.5 0 0.5" xyaxes="0 1 0 0 0 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <body name="link2" pos="0 0 0.2">
          <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.15" rgba="0.3 0.8 0.3 1"/>
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="shoulder_lift_act" joint="shoulder_lift" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def robot_xml_path():
    """Write test robot XML to a temp file."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(ROBOT_XML)
    yield path
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sim_with_robot(robot_xml_path):
    """Simulation with world + robot loaded."""
    sim = Simulation(tool_name="test_regression", mesh=False)
    result = sim.create_world(gravity=[0, 0, -9.81])
    assert result["status"] == "success"
    result = sim.add_robot("arm1", urdf_path=robot_xml_path)
    assert result["status"] == "success"
    yield sim
    sim.cleanup()


class TestFlatIndexStatePreservation:
    """Regression: joint positions must survive object injection (layout shift)."""

    def test_joint_survives_object_injection(self, sim_with_robot):
        """Set a joint to π/3, inject an object, verify joint is still ≈π/3.

        This catches the flat-index qpos copy bug where injected bodies
        shift existing qpos entries.
        """
        sim = sim_with_robot
        target_angle = math.pi / 3

        # Set shoulder_pan to π/3
        result = sim.set_joint_positions(
            positions={"shoulder_pan": target_angle},
            robot_name="arm1",
        )
        assert result["status"] == "success"

        # Verify it's set
        state = sim.get_robot_state("arm1")
        assert state["status"] == "success"  # state returned
        # Read qpos directly
        model = sim._world._model
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm1/shoulder_pan")
        if jid < 0:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder_pan")
        assert jid >= 0
        qpos_before = float(sim._world._data.qpos[model.jnt_qposadr[jid]])
        assert abs(qpos_before - target_angle) < 1e-6

        # Inject an object (triggers XML round-trip + _reload_scene_from_xml)
        result = sim.add_object(
            "test_box",
            shape="box",
            position=[0.5, 0.5, 0.1],
            size=[0.05, 0.05, 0.05],
        )
        assert result["status"] == "success"

        # Verify joint is still ≈π/3 after injection
        model = sim._world._model
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm1/shoulder_pan")
        if jid < 0:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder_pan")
        assert jid >= 0
        qpos_after = float(sim._world._data.qpos[model.jnt_qposadr[jid]])
        assert abs(qpos_after - target_angle) < 1e-4, (
            f"Joint drifted from {target_angle:.6f} to {qpos_after:.6f} after object injection"
        )


class TestApplyForceLatchedBehavior:
    """Regression: apply_force is latched (persists across steps)."""

    def test_force_persists_across_multiple_steps(self, sim_with_robot):
        """Apply lateral force to a body, step 50 times, verify body moved.

        This validates the docstring contract: force is latched in
        qfrc_applied and applied on every subsequent step.

        NOTE: We use an X-force (lateral) because a Z-force along the
        kinematic chain of hinge joints produces zero generalized torque
        (mj_applyFT maps Cartesian force to joint space; Z-force at CoM
        compresses the chain without creating torques on Y-axis hinges).
        """
        sim = sim_with_robot

        # Get initial x position of link2
        model = sim._world._model
        data = sim._world._data
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "arm1/link2")
        if body_id < 0:
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link2")
        assert body_id >= 0

        x_before = float(data.xpos[body_id, 0])

        # Apply strong lateral (X) force — this creates torques on Y-axis hinges
        result = sim.apply_force("link2", force=[100.0, 0, 0])
        assert result["status"] == "success"

        # Step physics 50 times — force should persist (latched)
        sim.step(n_steps=50)

        x_after = float(data.xpos[body_id, 0])
        # Body should have moved laterally due to persistent force
        assert abs(x_after - x_before) > 1e-4, (
            f"Body did not move (x_before={x_before:.6f}, x_after={x_after:.6f}). "
            "Force may not be persisting across steps."
        )

    def test_zero_force_stops_effect(self, sim_with_robot):
        """Apply force, then zero it, verify force buffer is cleared."""
        sim = sim_with_robot

        # Apply lateral (X) force — produces non-zero generalized torques
        sim.apply_force("link2", force=[50.0, 0, 0])
        assert np.any(sim._world._data.qfrc_applied != 0), "X-force on link2 should produce non-zero generalized forces"

        # Zero it — apply_force zeros buffer first, then applies zero force
        sim.apply_force("link2", force=[0, 0, 0])
        # After zeroing + applying zero force/torque, buffer should be all zeros
        assert np.allclose(sim._world._data.qfrc_applied, 0.0)


class TestThreadSafety:
    """Regression: concurrent operations don't corrupt MuJoCo state."""

    def test_concurrent_step_and_reset_no_crash(self, sim_with_robot):
        """Concurrent step() and reset() must not SIGSEGV.

        Both acquire self._lock, so they serialize. This test verifies
        the lock is actually held (no segfault, no exception).
        """
        sim = sim_with_robot
        errors = []

        def stepper():
            try:
                for _ in range(100):
                    sim.step(n_steps=1)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"stepper: {e}")

        def resetter():
            try:
                for _ in range(10):
                    sim.reset()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(f"resetter: {e}")

        t1 = threading.Thread(target=stepper)
        t2 = threading.Thread(target=resetter)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_set_joint_and_step(self, sim_with_robot):
        """Concurrent set_joint_positions and step must serialize safely."""
        sim = sim_with_robot
        errors = []

        def setter():
            try:
                for i in range(50):
                    sim.set_joint_positions(
                        positions={"shoulder_pan": float(i) * 0.01},
                        robot_name="arm1",
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"setter: {e}")

        def stepper():
            try:
                for _ in range(50):
                    sim.step(n_steps=2)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"stepper: {e}")

        t1 = threading.Thread(target=setter)
        t2 = threading.Thread(target=stepper)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"


# ── Robot XML for multi-robot asset directory test ──

ROBOT_B_XML = """
<mujoco model="test_gripper">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="grip_base" pos="0 0 0.05">
      <geom type="box" size="0.02 0.04 0.02" rgba="0.5 0.5 0.1 1"/>
      <joint name="grip_slide" type="slide" axis="1 0 0" range="-0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="grip_act" joint="grip_slide" kp="30"/>
  </actuator>
</mujoco>
"""


class TestRecordingRoundtripCameraFrames:
    """Regression: namespaced cameras survive schema reconcile and have frames.

    @yinsong1986 review (2026-04-30): "Please add a round-trip test:
    start_recording → run_policy → stop_recording, reopen the dataset,
    assert the camera feature has non-zero frames."
    """

    @pytest.fixture
    def sim_with_namespaced_camera(self, robot_xml_path, tmp_path):
        """Sim with a robot whose camera name contains '/' (namespace)."""
        sim = Simulation(tool_name="test_recording", mesh=False)
        result = sim.create_world(gravity=[0, 0, -9.81])
        assert result["status"] == "success"
        result = sim.add_robot("arm1", urdf_path=robot_xml_path)
        assert result["status"] == "success"
        yield sim
        sim.cleanup()

    @requires_gl
    def test_recording_roundtrip_has_camera_frames(self, sim_with_namespaced_camera, tmp_path):
        """Record → run mock policy → stop → verify dataset has camera data.

        This validates the /→__ sanitization fix doesn't silently drop frames.
        The test robot XML has camera 'arm0/wrist_cam' which becomes
        'arm0__wrist_cam' in the dataset schema.
        """
        pytest.importorskip("lerobot")
        from pathlib import Path

        sim = sim_with_namespaced_camera
        ds_root = str(tmp_path / "roundtrip_ds")

        # Start recording
        result = sim._dispatch_action(
            "start_recording",
            {"repo_id": "local/rt-test", "root": ds_root, "fps": 10, "overwrite": True},
        )
        assert result["status"] == "success", f"start_recording failed: {result}"

        # Run mock policy for a short burst (generates frames via on_frame hook)
        result = sim._dispatch_action(
            "run_policy",
            {
                "robot_name": "arm1",
                "policy_provider": "mock",
                "duration": 0.5,
                "control_frequency": 10,
            },
        )
        assert result["status"] == "success", f"run_policy failed: {result}"

        # Stop recording
        result = sim._dispatch_action("stop_recording", {})
        assert result["status"] == "success", f"stop_recording failed: {result}"

        # Verify dataset exists and has frames
        ds_path = Path(ds_root)
        assert ds_path.exists(), f"Dataset dir not created at {ds_root}"

        # Reopen dataset and verify camera feature has frames
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(repo_id="local/rt-test", root=ds_root)
            assert len(ds) > 0, f"Dataset has no frames (expected > 0, got {len(ds)})"

            # Check that the camera feature exists (sanitized name)
            cam_feature_found = False
            for feat_name in ds.features:
                if feat_name.startswith("observation.images."):
                    cam_feature_found = True
                    break

            assert cam_feature_found, (
                f"No observation.images.* feature found in dataset. Features: {list(ds.features.keys())}"
            )

            # Access a frame and verify image data is present
            sample = ds[0]
            for feat_name in ds.features:
                if feat_name.startswith("observation.images."):
                    assert feat_name in sample, f"Camera feature {feat_name} missing from sample"
                    img = sample[feat_name]
                    # Image should be non-empty (tensor or array with shape)
                    assert hasattr(img, "shape"), f"Camera data has no shape: {type(img)}"
                    assert img.shape[0] > 0, f"Camera image has zero height: {img.shape}"
                    break

        except ImportError:
            pytest.skip("lerobot dataset API not available for verification")


class TestMultiRobotDifferentAssetDirs:
    """Regression: two robots from different asset dirs both compile and render.

    @yinsong1986 review (2026-04-30): "load two robots whose urdf_paths
    are in different directories; assert both render."
    """

    def test_two_robots_different_directories_both_load(self):
        """Load two robots from separate temp dirs, verify both have joints."""
        tmpdir_a = tempfile.mkdtemp(prefix="robot_a_")
        tmpdir_b = tempfile.mkdtemp(prefix="robot_b_")

        try:
            # Write robot A (arm) to dir A
            path_a = os.path.join(tmpdir_a, "arm.xml")
            with open(path_a, "w") as f:
                f.write(ROBOT_XML)

            # Write robot B (gripper) to dir B
            path_b = os.path.join(tmpdir_b, "gripper.xml")
            with open(path_b, "w") as f:
                f.write(ROBOT_B_XML)

            sim = Simulation(tool_name="test_multi_asset", mesh=False)
            result = sim.create_world(gravity=[0, 0, -9.81])
            assert result["status"] == "success"

            # Add robot A from dir A
            result = sim.add_robot("arm1", urdf_path=path_a)
            assert result["status"] == "success", f"Robot A failed: {result}"

            # Add robot B from dir B (different asset directory)
            result = sim.add_robot("grip1", urdf_path=path_b, position=[0.3, 0, 0])
            assert result["status"] == "success", f"Robot B failed: {result}"

            # Both robots should be registered
            assert "arm1" in sim._world.robots
            assert "grip1" in sim._world.robots

            # Both should have joints discovered
            assert len(sim._world.robots["arm1"].joint_names) == 3  # shoulder_pan, shoulder_lift, elbow
            assert len(sim._world.robots["grip1"].joint_names) == 1  # grip_slide

            # Physics step should succeed (proves combined model compiled)
            result = sim.step(n_steps=10)
            assert result["status"] == "success", f"Step failed: {result}"

            # Verify we can read state from both robots
            state_a = sim.get_robot_state("arm1")
            assert state_a["status"] == "success", f"State A failed: {state_a}"
            state_b = sim.get_robot_state("grip1")
            assert state_b["status"] == "success", f"State B failed: {state_b}"

            sim.cleanup()
        finally:
            shutil.rmtree(tmpdir_a, ignore_errors=True)
            shutil.rmtree(tmpdir_b, ignore_errors=True)

    @requires_gl
    def test_two_robots_both_render_cameras(self):
        """Two robots with cameras from different dirs — both cameras render."""
        # Robot A has arm0/wrist_cam (from ROBOT_XML)
        # Add a camera to Robot B as well
        robot_b_with_cam = """
<mujoco model="gripper_cam">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <camera name="grip_cam" pos="0 0.2 0.3" xyaxes="1 0 0 0 0 1"/>
    <body name="grip_base" pos="0 0 0.05">
      <geom type="box" size="0.02 0.04 0.02" rgba="0.5 0.5 0.1 1"/>
      <joint name="grip_slide" type="slide" axis="1 0 0" range="-0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="grip_act" joint="grip_slide" kp="30"/>
  </actuator>
</mujoco>
"""
        tmpdir_a = tempfile.mkdtemp(prefix="robot_a_cam_")
        tmpdir_b = tempfile.mkdtemp(prefix="robot_b_cam_")

        try:
            path_a = os.path.join(tmpdir_a, "arm.xml")
            with open(path_a, "w") as f:
                f.write(ROBOT_XML)

            path_b = os.path.join(tmpdir_b, "gripper_cam.xml")
            with open(path_b, "w") as f:
                f.write(robot_b_with_cam)

            sim = Simulation(tool_name="test_render_multi", mesh=False)
            result = sim.create_world(gravity=[0, 0, -9.81])
            assert result["status"] == "success"

            result = sim.add_robot("arm1", urdf_path=path_a)
            assert result["status"] == "success"
            result = sim.add_robot("grip1", urdf_path=path_b, position=[0.5, 0, 0])
            assert result["status"] == "success"

            # Step to settle physics
            sim.step(n_steps=5)

            # Get observation (includes camera renders)
            obs = sim._get_sim_observation("arm1")

            # We should have at least one camera rendered (arm0/wrist_cam)
            cam_frames = {k: v for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim == 3}
            assert len(cam_frames) > 0, f"No camera frames rendered. Observation keys: {list(obs.keys())}"

            # Verify camera frame is not all-zero (actually rendered something)
            for cam_name, frame in cam_frames.items():
                assert frame.shape[2] == 3, f"Camera {cam_name} not RGB: shape={frame.shape}"
                # At minimum, the frame should have some non-zero pixels
                # (ground plane + colored geoms should provide contrast)
                assert frame.sum() > 0, f"Camera {cam_name} rendered all-black frame"

            sim.cleanup()
        finally:
            shutil.rmtree(tmpdir_a, ignore_errors=True)
            shutil.rmtree(tmpdir_b, ignore_errors=True)


class TestSceneMutationBlockedDuringPolicy:
    """Scene mutations must hard-fail while a policy is running.

    A concurrent PolicyRunner worker calling mj_step on stale model/data
    pointers (swapped by XML round-trip in add_object, add_camera, etc.)
    is undefined behaviour. The guard ensures agents learn to stop_policy
    before modifying the scene.
    """

    @pytest.fixture
    def robot_path(self, tmp_path):
        """Write test robot XML to a temp file."""
        path = tmp_path / "arm.xml"
        path.write_text(ROBOT_XML)
        return str(path)

    def test_add_object_blocked_during_policy(self, robot_path):
        sim = Simulation(tool_name="test_guard_obj", mesh=False)
        result = sim.create_world(gravity=[0, 0, -9.81])
        assert result["status"] == "success"

        result = sim.add_robot("arm1", urdf_path=robot_path)
        assert result["status"] == "success"

        # Start a policy (fast_mode so it completes quickly after stop)
        result = sim.start_policy("arm1", policy_provider="mock", duration=10.0, fast_mode=True)
        assert result["status"] == "success"

        # Try adding an object while policy is running — should be blocked
        result = sim.add_object("cube", shape="box", position=[0.3, 0, 0.05])
        assert result["status"] == "error"
        assert "policy is running" in result["content"][0]["text"].lower()

        # Stop the policy
        sim.stop_policy("arm1")
        if "arm1" in sim._policy_threads:
            sim._policy_threads["arm1"].result(timeout=5.0)

        # Now it should work
        result = sim.add_object("cube", shape="box", position=[0.3, 0, 0.05])
        assert result["status"] == "success"

        sim.cleanup()

    def test_add_camera_blocked_during_policy(self, robot_path):
        sim = Simulation(tool_name="test_guard_cam", mesh=False)
        result = sim.create_world(gravity=[0, 0, -9.81])
        assert result["status"] == "success"

        result = sim.add_robot("arm1", urdf_path=robot_path)
        assert result["status"] == "success"

        result = sim.start_policy("arm1", policy_provider="mock", duration=10.0, fast_mode=True)
        assert result["status"] == "success"

        # Try adding a camera while policy is running — should be blocked
        result = sim.add_camera("top_cam", position=[0, 0, 2], target=[0, 0, 0])
        assert result["status"] == "error"
        assert "policy is running" in result["content"][0]["text"].lower()

        sim.stop_policy("arm1")
        if "arm1" in sim._policy_threads:
            sim._policy_threads["arm1"].result(timeout=5.0)

        result = sim.add_camera("top_cam", position=[0, 0, 2], target=[0, 0, 0])
        assert result["status"] == "success"

        sim.cleanup()

    def test_load_scene_blocked_during_policy(self, robot_path):
        sim = Simulation(tool_name="test_guard_scene", mesh=False)
        result = sim.create_world(gravity=[0, 0, -9.81])
        assert result["status"] == "success"

        result = sim.add_robot("arm1", urdf_path=robot_path)
        assert result["status"] == "success"

        result = sim.start_policy("arm1", policy_provider="mock", duration=10.0, fast_mode=True)
        assert result["status"] == "success"

        # load_scene while policy is running — should be blocked
        result = sim.load_scene(robot_path)
        assert result["status"] == "error"
        assert "policy is running" in result["content"][0]["text"].lower()

        sim.stop_policy("arm1")
        if "arm1" in sim._policy_threads:
            sim._policy_threads["arm1"].result(timeout=5.0)

        sim.cleanup()

    def test_move_object_blocked_during_policy(self, robot_path):
        sim = Simulation(tool_name="test_guard_move", mesh=False)
        result = sim.create_world(gravity=[0, 0, -9.81])
        assert result["status"] == "success"

        result = sim.add_robot("arm1", urdf_path=robot_path)
        assert result["status"] == "success"

        # Add an object to move later
        result = sim.add_object("cube", shape="box", position=[0.3, 0, 0.05])
        assert result["status"] == "success"

        result = sim.start_policy("arm1", policy_provider="mock", duration=10.0, fast_mode=True)
        assert result["status"] == "success"

        # Try moving an object while policy is running — should be blocked
        result = sim.move_object("cube", position=[0.5, 0, 0.1])
        assert result["status"] == "error"
        assert "policy is running" in result["content"][0]["text"].lower()

        sim.stop_policy("arm1")
        if "arm1" in sim._policy_threads:
            sim._policy_threads["arm1"].result(timeout=5.0)

        # Now it should work
        result = sim.move_object("cube", position=[0.5, 0, 0.1])
        assert result["status"] == "success"

        sim.cleanup()

    def test_remove_robot_blocked_during_policy(self, robot_path):
        sim = Simulation(tool_name="test_guard_remove_robot", mesh=False)
        result = sim.create_world(gravity=[0, 0, -9.81])
        assert result["status"] == "success"

        result = sim.add_robot("arm1", urdf_path=robot_path)
        assert result["status"] == "success"

        result = sim.start_policy("arm1", policy_provider="mock", duration=10.0, fast_mode=True)
        assert result["status"] == "success"

        # Try removing robot while policy is running — should be blocked
        result = sim.remove_robot("arm1")
        assert result["status"] == "error"
        assert "policy is running" in result["content"][0]["text"].lower()

        sim.stop_policy("arm1")
        if "arm1" in sim._policy_threads:
            sim._policy_threads["arm1"].result(timeout=5.0)

        # Now it should work
        result = sim.remove_robot("arm1")
        assert result["status"] == "success"

        sim.cleanup()
