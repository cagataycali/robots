"""Pre-flight validation contracts for the plain-MP4 camera recorders.

Both plain-MP4 recorder entry points on
:class:`strands_robots.simulation.mujoco.rendering.RenderingMixin` -
``start_cameras_recording`` (daemon-thread) and
``start_cameras_recording_synchronous`` (``(on_frame, finalize)`` closures) -
run the same pre-flight guards before touching the filesystem or spawning a
capture thread:

* an empty resolved camera set fails loudly with ``"No cameras to record."``
  rather than silently starting a recording that would only ever write empty
  MP4 files, and
* an ``output_dir`` that fails path validation (traversal / metacharacters) is
  rejected with a ``"cameras_recording: ..."`` error instead of being passed
  through to ``os.makedirs``, and
* every frame/pixel-count option (``fps``, ``width``, ``height``,
  ``max_frames_per_camera``) must be a positive whole number, since each
  unusable value produced an empty recording that both ``start`` and ``stop``
  reported as ``status="success"``.

These are LLM-facing tool contracts, so the guards return the structured
``{"status": "error", ...}`` shape rather than raising. Pinned here so a
regression that lets a zero-camera or traversal-carrying request slip through
is caught immediately.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time

import pytest

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "glfw")

# Inline MJCF avoids a network-dependent model download and keeps the world
# deterministic: three hinge joints, matching the fixtures used by the other
# recorder tests in this package.
_ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="shoulder_lift_act" joint="shoulder_lift" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim():
    from strands_robots.simulation import Simulation

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(_ROBOT_XML)

    s = Simulation()
    s.create_world()
    s.add_robot("arm", urdf_path=path, position=[0.0, 0.0, 0.0])
    yield s
    s.destroy()
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestNoCamerasGuard:
    """An empty resolved camera set is a fail-loud error, not a silent no-op.

    Passing ``cameras=[]`` explicitly resolves to zero camera names even though
    the compiled world carries a ``default`` camera, so it exercises the guard
    without depending on the scene having no cameras at all.
    """

    def test_start_cameras_recording_empty_list_errors(self, sim):
        result = sim.start_cameras_recording(cameras=[])
        assert result["status"] == "error"
        assert result["content"][0]["text"] == "No cameras to record."
        # The guard fires before any recorder thread is registered.
        assert sim.get_cameras_recording_status()["status"] == "success"

    def test_start_cameras_recording_synchronous_empty_list_errors(self, sim):
        result = sim.start_cameras_recording_synchronous(cameras=[])
        assert result["status"] == "error"
        assert result["content"][0]["text"] == "No cameras to record."


class TestOutputDirValidation:
    """A traversal-carrying ``output_dir`` is rejected before ``os.makedirs``."""

    def test_start_cameras_recording_synchronous_rejects_traversal(self, sim):
        sim.add_camera("cam_a", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
        result = sim.start_cameras_recording_synchronous(cameras=["cam_a"], output_dir="../../etc/evil")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert text.startswith("cameras_recording:")
        assert "output_dir" in text
        # Rejected at the pre-flight stage: no synchronous session started.
        assert sim.get_cameras_recording_status()["status"] == "success"


class TestRecordingOptionValidation:
    """Frame/pixel-count options must be positive whole numbers on both entry points.

    Each of these values makes a recording impossible, and each one used to be
    accepted: ``fps=0`` killed the capture thread on its first ``1 / fps``,
    ``fps=-1``/``nan``/``inf`` were refused by the ffmpeg writer at flush time,
    ``fps="30"`` raised a ``TypeError`` on the capture thread, and a
    non-positive ``max_frames_per_camera``/``width``/``height`` dropped or
    failed every frame. In every case ``start_cameras_recording`` returned
    ``status="success"`` (announcing e.g. "Recording 1 camera(s) @ 0 FPS") and
    ``stop_cameras_recording`` also returned success with ``frames: 0`` and no
    MP4 on disk - so a caller had no signal that the recording never happened.
    """

    _BAD_VALUES = [0, -1, 2.5, float("nan"), float("inf"), "30", True, None]

    @pytest.fixture
    def cam_sim(self, sim):
        sim.add_camera("cam_a", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
        return sim

    @staticmethod
    def _assert_rejected(result, method, param):
        """The result is a structured error naming the method and the parameter."""
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert text.startswith(f"{method}: {param} must be a positive whole number"), text

    @pytest.mark.parametrize("bad", _BAD_VALUES)
    def test_daemon_recorder_rejects_unusable_fps(self, cam_sim, bad, tmp_path):
        result = cam_sim.start_cameras_recording(cameras=["cam_a"], output_dir=str(tmp_path), fps=bad)
        self._assert_rejected(result, "start_cameras_recording", "fps")
        # Rejected before any thread/state registration: nothing is recording.
        assert "No active camera recording" in cam_sim.get_cameras_recording_status()["content"][0]["text"]

    @pytest.mark.parametrize("bad", _BAD_VALUES)
    def test_synchronous_recorder_rejects_unusable_fps(self, cam_sim, bad, tmp_path):
        result = cam_sim.start_cameras_recording_synchronous(cameras=["cam_a"], output_dir=str(tmp_path), fps=bad)
        self._assert_rejected(result, "start_cameras_recording_synchronous", "fps")

    @pytest.mark.parametrize("bad", [0, -5, 1.5, None])
    def test_daemon_recorder_rejects_unusable_frame_cap(self, cam_sim, bad, tmp_path):
        result = cam_sim.start_cameras_recording(cameras=["cam_a"], output_dir=str(tmp_path), max_frames_per_camera=bad)
        self._assert_rejected(result, "start_cameras_recording", "max_frames_per_camera")

    @pytest.mark.parametrize("param", ["width", "height"])
    @pytest.mark.parametrize("bad", [0, -64, 12.5])
    def test_daemon_recorder_rejects_unusable_frame_size(self, cam_sim, param, bad, tmp_path):
        result = cam_sim.start_cameras_recording(cameras=["cam_a"], output_dir=str(tmp_path), **{param: bad})
        self._assert_rejected(result, "start_cameras_recording", param)

    def test_omitted_frame_size_is_accepted(self, cam_sim, tmp_path):
        """``width``/``height`` of ``None`` means "camera default", not an error."""
        result = cam_sim.start_cameras_recording(cameras=["cam_a"], output_dir=str(tmp_path), width=None, height=None)
        assert result["status"] == "success"
        cam_sim.stop_cameras_recording()

    def test_integral_float_and_numpy_options_are_accepted(self, cam_sim, tmp_path):
        """A ``30.0``/``np.int64`` option is usable and must not be rejected."""
        import numpy as np

        result = cam_sim.start_cameras_recording(
            cameras=["cam_a"],
            output_dir=str(tmp_path),
            fps=30.0,
            width=np.int64(64),
            height=np.int64(48),
            max_frames_per_camera=np.int64(10),
        )
        assert result["status"] == "success"
        cam_sim.stop_cameras_recording()

    def test_valid_options_still_write_an_mp4(self, cam_sim, tmp_path):
        """Round trip: the guard does not disturb a recording it should allow."""
        started = cam_sim.start_cameras_recording(
            cameras=["cam_a"], output_dir=str(tmp_path), name="roundtrip", fps=10, width=64, height=48
        )
        assert started["status"] == "success"
        time.sleep(0.6)
        stopped = cam_sim.stop_cameras_recording()
        assert stopped["status"] == "success"
        mp4 = tmp_path / "roundtrip__cam_a.mp4"
        assert mp4.exists() and mp4.stat().st_size > 0


class TestSharedPositiveWholeNumberDomain:
    """The recorder and ``run_policy(video=...)`` share one accepted domain.

    ``video={"fps": 0}`` has been rejected since the video-config schema landed;
    the plain-MP4 recorder accepted the same value. Both now bind the same
    predicate, so the two surfaces cannot drift on what a usable frame or pixel
    count is.
    """

    # ``None`` is deliberately absent: in the ``video`` dict it means "key not
    # supplied" (``_pick`` falls back to the field default), while on the
    # recorder it is an explicit argument with no such fallback. Every value
    # that is a real supplied number is rejected identically by both.
    @pytest.mark.parametrize("bad", [0, -1, 2.5, float("nan"), float("inf"), "30", True])
    def test_video_config_and_recorder_agree_on_rejection(self, bad):
        from strands_robots.simulation.policy_runner import VideoConfig
        from strands_robots.utils import positive_whole_number_error

        assert positive_whole_number_error(bad, "fps", "start_cameras_recording") is not None
        assert VideoConfig.validation_error({"path": "/tmp/a.mp4", "fps": bad}) is not None

    @pytest.mark.parametrize("good", [1, 30, 30.0])
    def test_video_config_and_recorder_agree_on_acceptance(self, good):
        from strands_robots.simulation.policy_runner import VideoConfig
        from strands_robots.utils import positive_whole_number_error

        assert positive_whole_number_error(good, "fps", "start_cameras_recording") is None
        assert VideoConfig.validation_error({"path": "/tmp/a.mp4", "fps": good}) is None

    def test_error_text_names_the_receiving_surface(self):
        from strands_robots.utils import positive_whole_number_error

        assert positive_whole_number_error(0, "fps", "video") == "video: fps must be a positive whole number, got 0."
        assert (
            positive_whole_number_error(0, "fps", "start_cameras_recording")
            == "start_cameras_recording: fps must be a positive whole number, got 0."
        )
