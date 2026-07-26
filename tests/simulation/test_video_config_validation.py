"""Rollout video configs must reject options the recorder cannot honor.

``run_policy`` / ``start_policy`` / ``eval_policy`` / ``evaluate_benchmark``
take recording options as a free-form ``video={...}`` dict, so a mistyped key
has no function signature to bounce off. Dropping one silently is the worst
outcome for the caller:

* ``video={"filename": "/tmp/a.mp4"}`` left ``path`` unset, so the rollout
  reported ``status="success"`` having written no MP4 at all.
* ``video={"path": p, "resolution": [320, 240]}`` recorded at the default
  640x480 while the caller believed the request had been honored.
* ``video={"path": p, "fps": 0}`` fell through an ``or`` chain to 30 fps.

Every one of those is now a structured error naming the offending key.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.policy_runner import VideoConfig  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No OpenGL context available (EGL/OSMesa required for offscreen rendering)",
)

ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
    <camera name="side" pos="0.8 -0.8 0.4" xyaxes="0.707 0.707 0 -0.2 0.2 0.96"/>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="30"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim_with_arm(tmp_path):
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text(ARM_XML)
    sim = Simulation(tool_name="video_schema", mesh=False)
    try:
        sim.create_world()
        result = sim.add_robot(name="arm1", urdf_path=str(xml_path))
        assert result["status"] == "success", result
        yield sim
    finally:
        sim.cleanup(policy_stop_timeout=0.5)


class TestVideoConfigSchema:
    """Unit contract of ``VideoConfig``: what it accepts and what it refuses."""

    @pytest.mark.parametrize(
        ("video", "expected"),
        [
            # A wrong name for the output file used to disable recording silently.
            ({"filename": "/tmp/a.mp4"}, "unknown key 'filename'"),
            # A wrong name for the frame size used to record at 640x480 silently.
            ({"path": "/tmp/a.mp4", "resolution": [320, 240]}, "unknown key 'resolution'"),
            # Zero/negative counts used to fall through an ``or`` chain to the default.
            ({"path": "/tmp/a.mp4", "fps": 0}, "fps must be a positive whole number"),
            ({"path": "/tmp/a.mp4", "width": -1}, "width must be a positive whole number"),
            # ``bool`` is an ``int`` subclass and would have acted as a silent 1px.
            ({"path": "/tmp/a.mp4", "height": True}, "height must be a positive whole number"),
            # A fractional fps cannot be honored by the writer.
            ({"path": "/tmp/a.mp4", "fps": 29.97}, "fps must be a positive whole number"),
            # Wrong value types for the string fields.
            ({"path": 5}, "path must be a string"),
            ({"path": "/tmp/a.mp4", "camera": ["cam"]}, "camera must be a string"),
        ],
    )
    def test_unhonorable_option_is_rejected(self, video, expected):
        error = VideoConfig.validation_error(video)
        assert error is not None, f"{video!r} was silently accepted"
        assert expected in error, error
        # from_dict is the construction path and must refuse the same input.
        with pytest.raises(ValueError, match="video"):
            VideoConfig.from_dict(video)

    def test_unknown_key_error_lists_the_accepted_keys(self):
        # An unknown key is the one case where the caller cannot guess the fix
        # from the message alone, so the whole accepted set is spelled out.
        error = VideoConfig.validation_error({"filename": "/tmp/a.mp4"}) or ""
        for key in ("path", "fps", "camera", "width", "height"):
            assert key in error, error

    def test_typo_of_a_real_key_suggests_the_canonical_spelling(self):
        assert "Did you mean 'path'?" in (VideoConfig.validation_error({"pathh": "/tmp/a.mp4"}) or "")
        # Case differences are a typo too, not an unrelated key.
        assert "Did you mean 'fps'?" in (VideoConfig.validation_error({"path": "/tmp/a.mp4", "FPS": 30}) or "")

    @pytest.mark.parametrize("video", [None, {}, {"path": "/tmp/a.mp4", "camera": None}])
    def test_valid_configs_are_not_rejected(self, video):
        assert VideoConfig.validation_error(video) is None

    def test_legacy_aliases_still_resolve(self):
        config = VideoConfig.from_dict(
            {
                "record_video": "/tmp/a.mp4",
                "video_fps": 20,
                "camera_name": "cam",
                "video_width": 160,
                "video_height": 120,
            }
        )
        assert config == VideoConfig(path="/tmp/a.mp4", fps=20, camera="cam", width=160, height=120)

    def test_absent_path_still_means_recording_disabled(self):
        # An options dict with no path is the documented "recording off" case
        # and must keep working - only unknown/unusable options are rejected.
        config = VideoConfig.from_dict({"fps": 30})
        assert config is not None
        assert config.enabled is False


class TestRolloutRejectsBadVideoConfig:
    """The public rollout entry points surface the schema error themselves."""

    @requires_gl
    def test_run_policy_rejects_unknown_key_instead_of_skipping_recording(self, sim_with_arm, tmp_path):
        video_path = tmp_path / "typo.mp4"
        result = sim_with_arm.run_policy(
            robot_name="arm1",
            policy_provider="mock",
            n_steps=4,
            control_frequency=30.0,
            fast_mode=True,
            video={"filename": str(video_path), "fps": 30},
        )
        assert result["status"] == "error", result
        assert "unknown key 'filename'" in result["content"][0]["text"]
        assert not video_path.exists()

    @requires_gl
    def test_run_policy_rejects_unknown_size_key_instead_of_using_the_default(self, sim_with_arm, tmp_path):
        video_path = tmp_path / "size.mp4"
        result = sim_with_arm.run_policy(
            robot_name="arm1",
            policy_provider="mock",
            n_steps=4,
            control_frequency=30.0,
            fast_mode=True,
            video={"path": str(video_path), "camera": "arm1/side", "resolution": [320, 240]},
        )
        assert result["status"] == "error", result
        assert "unknown key 'resolution'" in result["content"][0]["text"]
        # Nothing was recorded at the wrong resolution.
        assert not video_path.exists()

    def test_start_policy_reports_the_error_instead_of_a_false_started(self, sim_with_arm, tmp_path):
        result = sim_with_arm.start_policy(
            robot_name="arm1",
            policy_provider="mock",
            n_steps=4,
            control_frequency=30.0,
            video={"path": str(tmp_path / "bg.mp4"), "fps": 0},
        )
        assert result["status"] == "error", result
        assert "fps must be a positive whole number" in result["content"][0]["text"]
        # The rejected call must not have marked the robot as running: a
        # subsequent well-formed start is accepted.
        started = sim_with_arm.start_policy(
            robot_name="arm1", policy_provider="mock", n_steps=2, control_frequency=30.0
        )
        assert started["status"] == "success", started
        sim_with_arm.stop_policy(robot_name="arm1")

    @requires_gl
    def test_eval_policy_rejects_unknown_key_before_running_episodes(self, sim_with_arm, tmp_path):
        result = sim_with_arm.eval_policy(
            robot_name="arm1",
            policy_provider="mock",
            n_episodes=2,
            max_steps=4,
            control_frequency=30.0,
            video={"path": str(tmp_path / "eval.mp4"), "camera": "arm1/side", "fpss": 10},
        )
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "unknown key 'fpss'" in text and "Did you mean 'fps'?" in text
        assert not list(tmp_path.glob("eval*.mp4"))
