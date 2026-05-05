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

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

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
