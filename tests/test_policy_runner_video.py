"""Regression test: run_policy(video={...}) actually writes frames to disk.

This was silently broken — the recording loop used ``frame.get("image")`` on
the top-level render() result, but sim.render() nests the image under
``content[n].image.source.bytes``. Every rollout opened a writer, wrote zero
frames, closed it, and crashed on ``os.path.getsize`` of a non-existent file.

This test runs a short mock rollout with video enabled and asserts:
- the file is created,
- it has non-zero size,
- the run returns status=success.
"""

import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("CI") == "true" and not os.environ.get("ROBOT_TEST_MUJOCO"),
    reason="requires OpenGL; opt-in via ROBOT_TEST_MUJOCO=1",
)
def test_run_policy_video_writes_mp4(tmp_path: Path) -> None:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    video_path = tmp_path / "rollout.mp4"

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam", position=[0.0, 0.0, 0.8], target=[0.0, 0.2, 0.05])

    result = sim.run_policy(
        robot_name="arm",
        policy_provider="mock",
        policy_config={},
        duration=0.5,
        control_frequency=20.0,
        video={"path": str(video_path), "fps": 20, "camera": "cam"},
    )

    sim.destroy()

    assert result["status"] == "success", f"rollout failed: {result}"
    assert video_path.exists(), f"video not written: {video_path}"
    assert video_path.stat().st_size > 0, "video file is empty"

    text_blocks = [c.get("text", "") for c in result.get("content", []) if isinstance(c, dict)]
    summary = "\n".join(text_blocks)
    assert "🎬 Video:" in summary, f"no video summary in output: {summary}"
    assert "📹" in summary and "frames" in summary, f"frame count missing: {summary}"


def test_extract_frame_ndarray_handles_render_shape() -> None:
    """Unit test the helper directly against the real render() output shape."""
    import base64

    import numpy as np
    from PIL import Image

    from strands_robots.simulation.policy_runner import _extract_frame_ndarray

    # Synthetic PNG with bytes source (the common MuJoCo path)
    img = Image.new("RGB", (8, 8), color=(128, 64, 32))
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    result_bytes = {
        "status": "success",
        "content": [
            {"text": "📸 8x8 from 'cam'"},
            {"image": {"format": "png", "source": {"bytes": png_bytes}}},
        ],
    }
    arr = _extract_frame_ndarray(result_bytes)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (8, 8, 3)

    # Also accepts base64-encoded 'data' field
    result_b64 = {
        "status": "success",
        "content": [
            {"image": {"format": "png", "source": {"data": base64.b64encode(png_bytes).decode()}}},
        ],
    }
    arr2 = _extract_frame_ndarray(result_b64)
    assert isinstance(arr2, np.ndarray)
    assert arr2.shape == (8, 8, 3)

    # Rejects garbage
    assert _extract_frame_ndarray({}) is None
    assert _extract_frame_ndarray({"content": []}) is None
    assert _extract_frame_ndarray({"content": [{"text": "no image here"}]}) is None
