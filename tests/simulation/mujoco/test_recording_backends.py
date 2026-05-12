"""T12: Video recording backends.

* start_recording (LeRobotDataset) requires the lerobot extra; when it's
  not installed, the error message must point to start_cameras_recording
  for plain MP4 and to the [lerobot] extra for dataset recording.
* start_cameras_recording works under [sim-mujoco] alone (imageio-ffmpeg)
  and does not need lerobot.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No GL context available (headless CI without EGL/OSMesa)",
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

has_lerobot = importlib.util.find_spec("lerobot") is not None


@pytest.fixture
def sim():
    s = Simulation(tool_name="rec_backend_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


class TestStartRecordingErrorWithoutLerobot:
    @pytest.mark.skipif(has_lerobot, reason="test targets the no-lerobot code path")
    def test_error_message_points_to_start_cameras_recording(self, sim):
        result = sim.start_recording(repo_id="local/test_rec")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "start_cameras_recording" in text
        assert "lerobot" in text.lower()


@requires_gl
class TestCamerasRecordingWithoutLerobot:
    """start_cameras_recording must work under [sim-mujoco] alone."""

    def test_start_stop_writes_mp4(self, sim, tmp_path):
        # Ensure at least one camera exists.
        r = sim.add_camera(name="cam1", position=[0.5, 0.5, 0.5], target=[0.0, 0.0, 0.0])
        assert r["status"] == "success"

        out = tmp_path / "mp4out"
        r = sim.start_cameras_recording(
            cameras=["cam1"],
            output_dir=str(out),
            fps=10,
            width=160,
            height=120,
            name="t12_smoke",
        )
        assert r["status"] == "success", r

        # Capture a few frames via stepping the sim.
        for _ in range(10):
            sim.step(n_steps=1)
            # tiny sleep to let the background capture thread tick
            import time

            time.sleep(0.05)

        r = sim.stop_cameras_recording()
        assert r["status"] == "success", r

        # At least one .mp4 must have landed in output_dir.
        assert out.exists()
        files = [f for f in os.listdir(out) if f.endswith(".mp4")]
        assert files, f"no mp4 files in {out}"
