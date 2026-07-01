"""Daemon-thread plain-MP4 camera recorder contract.

Covers ``Simulation.start_cameras_recording`` / ``stop_cameras_recording`` -
the background-thread MP4 recorder that captures raw RGB frames into
in-memory buffers and flushes them to one MP4 per camera on stop. This is
the ``[sim-mujoco]``-only recording path (no lerobot/torchcodec dependency),
distinct from the synchronous ``on_frame``/``finalize`` variant pinned in
``test_recording_synchronous.py`` and the LeRobot-dataset recorder pinned in
``test_recording_paths.py``.

The recorder spawns a real daemon thread, but ``render`` is monkey-patched on
the instance to return synthetic PNG-encoded frames (mirroring
``RenderingMixin.render``'s wire format). This exercises the buffer
bookkeeping, the MP4 flush, and every input-validation branch without needing
a live GL context, so the module runs headless on CI/dev boxes.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed - pip install strands-robots[sim-mujoco]")
imageio = pytest.importorskip("imageio", reason="imageio not installed - pip install imageio imageio-ffmpeg")
pytest.importorskip("imageio.v2", reason="imageio.v2 not available - upgrade imageio (>=2.5)")

from strands_robots.simulation import Simulation  # noqa: E402


def _png_bytes(arr: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 ndarray as PNG bytes (render() wire format)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _make_render_result(width: int = 32, height: int = 24) -> dict[str, Any]:
    """A fake render() dict carrying a real PNG payload the recorder can decode.

    A gradient (not a flat fill) is used so ``_extract_frame_ndarray`` returns
    a frame the recorder buffers rather than discards.
    """
    row = np.linspace(0, 255, width, dtype=np.uint8)
    arr = np.repeat(row[None, :], height, axis=0)
    arr = np.stack([arr, arr[::-1], arr], axis=-1).astype(np.uint8)
    return {
        "status": "success",
        "content": [
            {"text": f"{width}x{height}"},
            {"image": {"format": "png", "source": {"bytes": _png_bytes(arr)}}},
        ],
    }


def _make_sim_with_fake_render() -> Simulation:
    """Real Simulation with two named cameras and a fake, GL-free ``render``."""
    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam_a", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
    sim.add_camera("cam_b", position=[0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])

    def _fake_render(camera_name: str, width: int | None = None, height: int | None = None, **_kw):
        return _make_render_result(width=width or 32, height=height or 24)

    sim.render = _fake_render  # type: ignore[assignment,method-assign]
    return sim


def _wait_for_frames(sim: Simulation, cam: str, timeout: float = 3.0) -> None:
    """Block until the daemon recorder has buffered at least one frame."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = getattr(sim, "_cams_rec_state", None)
        if state and state["buffers"].get(cam):
            return
        time.sleep(0.02)


class TestDaemonRecorderValidation:
    """Every input-validation branch fails loud with a structured error dict
    rather than silently recording nothing or raising past the tool boundary.
    """

    def test_start_before_create_world_errors(self):
        """No world -> structured error, not a traceback."""
        sim = Simulation()
        res = sim.start_cameras_recording(cameras=["cam_a"])
        assert res["status"] == "error"
        assert "world" in res["content"][0]["text"].lower()

    def test_unresolved_camera_name_errors(self):
        """An explicit unknown camera is rejected loudly (never silently
        dropped), and the error lists the cameras that do exist."""
        sim = _make_sim_with_fake_render()
        try:
            res = sim.start_cameras_recording(cameras=["ghost_cam"])
            assert res["status"] == "error"
            text = res["content"][0]["text"]
            assert "ghost_cam" in text
            assert "cam_a" in text
            # The recorder must not have started.
            assert getattr(sim, "_cams_rec_state", None) is None
        finally:
            sim.destroy()

    def test_output_dir_traversal_rejected(self):
        """A ``..`` traversal in the LLM-supplied output_dir is rejected before
        any directory is created or filename interpolated."""
        sim = _make_sim_with_fake_render()
        try:
            res = sim.start_cameras_recording(cameras=["cam_a"], output_dir="../../../etc/evil")
            assert res["status"] == "error"
            assert getattr(sim, "_cams_rec_state", None) is None
        finally:
            sim.destroy()

    def test_bad_name_component_rejected(self):
        """A ``name`` carrying a path separator is rejected (it is interpolated
        into the per-camera MP4 filename)."""
        sim = _make_sim_with_fake_render()
        try:
            res = sim.start_cameras_recording(cameras=["cam_a"], name="../escape")
            assert res["status"] == "error"
            assert getattr(sim, "_cams_rec_state", None) is None
        finally:
            sim.destroy()

    def test_double_start_rejected(self, tmp_path: Path):
        """A second start while one recording is live is refused with a clear
        'Already recording' error naming the active tag; the first recording is
        untouched."""
        sim = _make_sim_with_fake_render()
        try:
            r1 = sim.start_cameras_recording(cameras=["cam_a"], output_dir=str(tmp_path), fps=30, name="first")
            assert r1["status"] == "success", r1

            r2 = sim.start_cameras_recording(cameras=["cam_a"], output_dir=str(tmp_path), fps=30, name="second")
            assert r2["status"] == "error"
            assert "Already recording 'first'" in r2["content"][0]["text"]
        finally:
            sim.stop_cameras_recording()
            sim.destroy()


class TestDaemonRecorderRoundTrip:
    """start -> capture -> stop writes a readable MP4 per camera."""

    def test_round_trip_writes_nonempty_mp4(self, tmp_path: Path):
        sim = _make_sim_with_fake_render()
        try:
            start = sim.start_cameras_recording(cameras=["cam_a"], output_dir=str(tmp_path), fps=30, name="clip")
            assert start["status"] == "success", start

            _wait_for_frames(sim, "cam_a")
            stop = sim.stop_cameras_recording()
            assert stop["status"] == "success", stop

            artifacts = next(c["json"] for c in stop["content"] if "json" in c)["artifacts"]
            cam_a = next(a for a in artifacts if a["camera"] == "cam_a")
            assert cam_a["frames"] > 0, "recorder must have buffered and flushed at least one frame"
            assert cam_a["errors"] == 0

            mp4 = Path(cam_a["path"])
            assert mp4.exists() and mp4.stat().st_size > 0, "MP4 artifact must exist and be non-empty"

            # Reopen the artifact to prove it is a decodable video.
            reader = imageio.get_reader(str(mp4))
            try:
                first = reader.get_data(0)
            finally:
                reader.close()
            assert first.ndim == 3 and first.shape[2] == 3
        finally:
            sim.destroy()

    def test_default_output_dir_used_when_omitted(self):
        """With ``output_dir`` omitted, artifacts land under the temp-dir
        recordings fallback rather than erroring."""
        sim = _make_sim_with_fake_render()
        try:
            start = sim.start_cameras_recording(cameras=["cam_a"], fps=30, name="deftmp")
            assert start["status"] == "success", start
            assert "strands_robots" in start["content"][0]["text"]

            _wait_for_frames(sim, "cam_a")
            stop = sim.stop_cameras_recording()
            assert stop["status"] == "success", stop
            artifacts = next(c["json"] for c in stop["content"] if "json" in c)["artifacts"]
            mp4 = Path(artifacts[0]["path"])
            assert "strands_robots" in mp4.parts
            if mp4.exists():
                mp4.unlink()
        finally:
            sim.destroy()

    def test_render_failures_are_counted_not_fatal(self):
        """A camera whose render keeps raising does not crash the recorder;
        the failures are tallied in ``errors`` and stop still returns success
        with zero frames for that camera."""
        sim = _make_sim_with_fake_render()

        def _boom(camera_name, width=None, height=None, **_kw):
            raise RuntimeError("simulated GL failure")

        sim.render = _boom  # type: ignore[assignment,method-assign]
        try:
            start = sim.start_cameras_recording(cameras=["cam_a"], fps=60, name="boom")
            assert start["status"] == "success", start

            # Let the capture loop tick a few times so failures accumulate.
            deadline = time.time() + 2.0
            state = sim._cams_rec_state
            while time.time() < deadline and state["errors"]["cam_a"] == 0:
                time.sleep(0.02)

            stop = sim.stop_cameras_recording()
            assert stop["status"] == "success", stop
            artifacts = next(c["json"] for c in stop["content"] if "json" in c)["artifacts"]
            cam_a = next(a for a in artifacts if a["camera"] == "cam_a")
            assert cam_a["frames"] == 0
            assert cam_a["errors"] > 0
        finally:
            sim.destroy()

    def test_stop_without_start_is_idempotent(self):
        """Stopping when nothing is recording is a success, not an error."""
        sim = _make_sim_with_fake_render()
        try:
            res = sim.stop_cameras_recording()
            assert res["status"] == "success"
            assert "not recording" in res["content"][0]["text"].lower()
        finally:
            sim.destroy()
