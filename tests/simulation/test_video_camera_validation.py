"""Regression for the 'video recording silently writes 0-frame MP4' DX bug.

Surfaced by /tmp/e2e_agentic_test_85 scenario S2 (LLM passed video.camera="side"
when add_robot() had compiled the camera as "arm1/side"). Before the fix,
sim.render() returned status=error, _extract_frame_ndarray() returned None,
and the rollout silently completed with writer.close() producing an empty
file. After the fix, PolicyRunner pre-validates the camera name up-front
and returns a clean error dict with a "cameras are namespaced" hint.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

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
    sim = Simulation(tool_name="video_guard", mesh=False)
    try:
        sim.create_world()
        r = sim.add_robot(name="arm1", urdf_path=str(xml_path))
        assert r["status"] == "success", r
        yield sim
    finally:
        sim.cleanup(policy_stop_timeout=0.5)


class TestVideoCameraPreValidation:
    def test_bad_camera_fails_fast_with_hint(self, sim_with_arm, tmp_path):
        """A wrong camera name must be caught BEFORE the rollout starts,
        not silently produce a 0-byte MP4 at the end."""
        video_path = tmp_path / "bad.mp4"
        # "side" is the raw camera name but the compiled scene has "arm1/side"
        r = sim_with_arm.run_policy(
            robot_name="arm1",
            policy_provider="mock",
            duration=0.5,
            fast_mode=False,
            video={"path": str(video_path), "camera": "side", "fps": 30},
        )
        assert r["status"] == "error", r
        text = r["content"][0]["text"].lower()
        assert "not renderable" in text or "not found" in text
        # The hint is the whole point of this fix - verify it's there.
        assert "namespaced" in text or "arm1/" in text, f"missing hint: {text}"
        # No stub MP4 should have been written
        assert not video_path.exists() or video_path.stat().st_size == 0

    @requires_gl
    def test_namespaced_camera_succeeds(self, sim_with_arm, tmp_path):
        """Happy path: the correctly-namespaced camera compiles, records, closes."""
        video_path = tmp_path / "ok.mp4"
        r = sim_with_arm.run_policy(
            robot_name="arm1",
            policy_provider="mock",
            duration=0.5,
            fast_mode=False,
            video={
                "path": str(video_path),
                "camera": "arm1/side",
                "fps": 30,
                "width": 160,
                "height": 120,
            },
        )
        assert r["status"] == "success", r
        text = r["content"][0]["text"]
        assert "🎬 Video" in text or "Video:" in text, text
        assert video_path.exists() and video_path.stat().st_size > 0


class TestCamerasRecordingSuffixResolution:
    """Regression for PR #85 follow-up review (yinsong1986, 2026-05-07 02:06):

    `start_cameras_recording` used to compare raw user inputs to the already-
    namespaced `names` list returned by `_active_camera_list`. Users who
    passed the short form (e.g. 'side' when the scene had 'arm1/side') hit
    a spurious "not found" error even though the suffix resolution had
    correctly resolved their input.

    The fix: `_active_camera_list` now returns (resolved, unresolved_inputs)
    so the strict check operates on actual user input, not the resolved set.
    """

    @requires_gl
    def test_short_name_resolves_to_namespaced(self, sim_with_arm, tmp_path):
        """start_cameras_recording must accept 'side' when scene has 'arm1/side'."""
        out_dir = tmp_path / "recs"
        r = sim_with_arm.start_cameras_recording(
            cameras=["side"],
            fps=5,
            width=64,
            height=48,
            output_dir=str(out_dir),
            name="suffix_test",
        )
        assert r["status"] == "success", r
        # Stop cleanly
        sim_with_arm.stop_cameras_recording()

    def test_bogus_name_still_errors(self, sim_with_arm):
        """A name that neither matches directly nor by suffix must still error."""
        r = sim_with_arm.start_cameras_recording(
            cameras=["definitely_not_a_camera"],
            fps=5,
            width=64,
            height=48,
        )
        assert r["status"] == "error", r
        text = r["content"][0]["text"]
        assert "not found" in text.lower()
        assert "definitely_not_a_camera" in text

    def test_mixed_resolvable_and_bogus(self, sim_with_arm):
        """If any input doesn't resolve, fail loudly - don't silently shrink."""
        r = sim_with_arm.start_cameras_recording(
            cameras=["side", "nope"],
            fps=5,
            width=64,
            height=48,
        )
        assert r["status"] == "error", r
        text = r["content"][0]["text"]
        assert "nope" in text
