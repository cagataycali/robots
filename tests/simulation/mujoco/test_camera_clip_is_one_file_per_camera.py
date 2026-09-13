"""A raw camera recording writes one MP4 per camera, inside ``output_dir``.

``add_camera`` accepts a namespaced camera name (``arm0/wrist``) - that is how a
wrist camera on a namespaced robot is spelled, and
:func:`~strands_robots.utils.camera_schema_key` collapses the separator to
``__`` for the dataset column recorded from it. The plain-MP4 sinks
(:meth:`~strands_robots.simulation.Simulation.start_cameras_recording` and its
synchronous counterpart) name a file after the same camera, so they owe the same
collapse: a ``/`` in a file name is a directory separator, which puts the clip
below the directory the caller named and drops the recording tag from its name.

Pinned here for both sinks: the clip a namespaced camera produces is a file in
``output_dir``, the name the result reports composes with the ``output_dir`` it
reports into that file, and two cameras whose names collapse to one clip are
refused instead of one overwriting the other.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed - pip install strands-robots[sim-mujoco]")
imageio = pytest.importorskip("imageio", reason="imageio not installed - pip install imageio imageio-ffmpeg")

from strands_robots.simulation import Simulation  # noqa: E402

#: The two plain-MP4 entry points. Both name their clips the same way.
DAEMON = "start_cameras_recording"
SYNCHRONOUS = "start_cameras_recording_synchronous"


def _render_result(width: int = 32, height: int = 24) -> dict[str, Any]:
    """A ``render()``-shaped dict carrying a real PNG the recorder can decode."""
    from PIL import Image

    row = np.linspace(0, 255, width, dtype=np.uint8)
    arr = np.repeat(row[None, :], height, axis=0)
    arr = np.stack([arr, arr[::-1], arr], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return {
        "status": "success",
        "content": [
            {"text": f"{width}x{height}"},
            {"image": {"format": "png", "source": {"bytes": buf.getvalue()}}},
        ],
    }


def _sim_with_cameras(*names: str) -> Simulation:
    """A world holding ``names`` as cameras, rendering PNGs without a GL context."""
    sim = Simulation()
    sim.create_world()
    for index, name in enumerate(names):
        sim.add_camera(name, position=[0.3 + 0.1 * index, 0.3, 0.4], target=[0.0, 0.0, 0.1])

    def _fake_render(camera_name: str, width: int | None = None, height: int | None = None, **_kw):
        return _render_result(width=width or 32, height=height or 24)

    sim.render = _fake_render  # type: ignore[assignment,method-assign]
    return sim


def _record_one_clip(sim: Simulation, entry_point: str, camera: str, output_dir: Path) -> dict[str, Any]:
    """Record ``camera`` through ``entry_point`` and return the flushed result."""
    start = getattr(sim, entry_point)
    result = start(cameras=[camera], output_dir=str(output_dir), fps=10, name="clip")
    assert result["status"] == "success", result
    if entry_point == DAEMON:
        deadline_frames = 0
        while deadline_frames < 300:  # ~3 s, the recorder thread captures at fps
            state = getattr(sim, "_cams_rec_state", None)
            if state and state["buffers"].get(camera):
                break
            import time

            time.sleep(0.01)
            deadline_frames += 1
        return sim.stop_cameras_recording()
    controls = next(c["json"] for c in result["content"] if "json" in c)
    for step in range(3):
        controls["on_frame"](step, {}, {})
    return controls["finalize"]()


@pytest.mark.parametrize("entry_point", [DAEMON, SYNCHRONOUS])
@pytest.mark.parametrize(
    ("camera", "clip"),
    [
        ("wrist", "clip__wrist.mp4"),  # control: an unnamespaced camera is unchanged
        ("arm0/wrist", "clip__arm0__wrist.mp4"),
    ],
)
def test_a_camera_clip_is_a_file_in_the_output_dir(entry_point: str, camera: str, clip: str, tmp_path: Path):
    """The clip is a file directly in ``output_dir``, named for tag and camera.

    A namespaced camera used to write ``<tag>__arm0/wrist.mp4``, i.e. a
    ``wrist.mp4`` inside a ``<tag>__arm0`` directory the caller never named and
    the result never mentioned.
    """
    sim = _sim_with_cameras(camera)
    try:
        final = _record_one_clip(sim, entry_point, camera, tmp_path)
        assert final["status"] == "success", final
        artifacts = next(c["json"] for c in final["content"] if "json" in c)["artifacts"]
        written = Path(next(a["path"] for a in artifacts if a["camera"] == camera))

        assert written == tmp_path / clip
        assert written.is_file(), f"no clip at {written}"
        assert sorted(p.name for p in tmp_path.iterdir()) == [clip], "the clip is the only entry written"

        # The two lines an operator reads - "output_dir: X" and "-> Y" - compose
        # into the file that was written.
        reported = final["content"][0]["text"]
        assert f"output_dir: {tmp_path}" in reported
        assert f"-> {clip}" in reported
        assert (tmp_path / clip).is_file()
    finally:
        sim.destroy()


@pytest.mark.parametrize("entry_point", [DAEMON, SYNCHRONOUS])
def test_two_cameras_naming_one_clip_are_refused(entry_point: str, tmp_path: Path):
    """``arm0/wrist`` and ``arm0__wrist`` are two cameras and one file name.

    Recording both would have one clip overwrite the other while the result
    reported both as written, so the recording is refused before it starts -
    where the dataset sink refuses the same pair of names.
    """
    sim = _sim_with_cameras("arm0/wrist", "arm0__wrist")
    try:
        result = getattr(sim, entry_point)(
            cameras=["arm0/wrist", "arm0__wrist"], output_dir=str(tmp_path), fps=10, name="clip"
        )
        assert result["status"] == "error", result
        message = result["content"][0]["text"]
        assert entry_point in message
        assert "arm0/wrist" in message and "arm0__wrist" in message
        assert list(tmp_path.iterdir()) == [], "a refused recording writes nothing"
    finally:
        sim.destroy()
