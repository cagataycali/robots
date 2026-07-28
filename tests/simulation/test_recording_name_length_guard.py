"""Regression tests: an over-long recording tag is refused before the capture.

``start_cameras_recording(name=...)`` interpolates the tag into
``{name}__{camera}.mp4``. POSIX caps a path component at 255 bytes, so a long tag
produces a filename that only fails when ffmpeg finally opens it - which for this
API is AFTER the capture, when the buffered frames are flushed:

    name = "x" * 300
    start_cameras_recording(...)   -> status="success"
    ... capture the whole rollout ...
    stop_cameras_recording()       -> "0 frames  0.0 KB", flush failed:
                                      "Error opening output files: File name too long"

So the entire recording was thrown away for a reason knowable up front. Every other
unusable ``name`` (separators, traversal, metacharacters, empty) is rejected before
capture starts; length was the one that was not.

``sanitize_name_component`` now enforces a byte budget, leaving headroom for the
suffix its callers append.
"""

from __future__ import annotations

import os
import time

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("imageio")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.safe_output import (  # noqa: E402
    _MAX_NAME_COMPONENT_BYTES,
    sanitize_name_component,
)


@pytest.fixture
def sim(tmp_path):
    s = Simulation(tool_name="recording_name_length", mesh=False)
    s.create_world()
    assert s.add_camera(name="c", position=[1, 1, 1], target=[0, 0, 0])["status"] == "success"
    s._out_dir = str(tmp_path)
    yield s
    s.destroy()


def test_the_budget_leaves_room_for_the_suffix() -> None:
    """The limit must be below NAME_MAX by more than the appended suffix."""
    assert _MAX_NAME_COMPONENT_BYTES < 255
    # "__" + a generously long camera name + ".mp4" must still fit.
    assert 255 - _MAX_NAME_COMPONENT_BYTES >= len("__.mp4") + 32


def test_a_tag_at_the_limit_is_accepted() -> None:
    name = "x" * _MAX_NAME_COMPONENT_BYTES
    assert sanitize_name_component(name) == name


def test_a_tag_one_byte_over_is_refused() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        sanitize_name_component("x" * (_MAX_NAME_COMPONENT_BYTES + 1))


def test_the_budget_is_counted_in_bytes_not_characters() -> None:
    """A multi-byte name can fit the character count but blow the byte limit."""
    # Each of these is 3 bytes in UTF-8.
    name = "é" * _MAX_NAME_COMPONENT_BYTES
    assert len(name) == _MAX_NAME_COMPONENT_BYTES
    with pytest.raises(ValueError, match="exceeds"):
        sanitize_name_component(name)


def test_an_over_long_tag_is_refused_before_the_capture(sim) -> None:
    """The core defect: it used to cost the whole rollout."""
    result = sim.start_cameras_recording(output_dir=sim._out_dir, cameras=["c"], fps=10, name="x" * 300)
    assert result["status"] == "error"
    assert "exceeds" in result["content"][0]["text"]
    # Nothing was started, so there is nothing to stop and nothing on disk.
    assert sorted(os.listdir(sim._out_dir)) == []


def test_a_normal_tag_still_records(sim) -> None:
    """Guard against the fix degenerating into 'reject everything'."""
    assert (
        sim.start_cameras_recording(output_dir=sim._out_dir, cameras=["c"], fps=10, name="fine")["status"] == "success"
    )
    time.sleep(0.25)
    assert sim.stop_cameras_recording()["status"] == "success"
    assert any(f.startswith("fine__") for f in os.listdir(sim._out_dir)), os.listdir(sim._out_dir)


def test_a_tag_at_the_limit_still_records(sim) -> None:
    """The boundary must be usable, not merely accepted by the validator."""
    name = "x" * _MAX_NAME_COMPONENT_BYTES
    assert sim.start_cameras_recording(output_dir=sim._out_dir, cameras=["c"], fps=10, name=name)["status"] == "success"
    time.sleep(0.25)
    stopped = sim.stop_cameras_recording()
    assert stopped["status"] == "success"
    assert "flush failed" not in stopped["content"][0]["text"]
    assert any(f.startswith(name[:40]) for f in os.listdir(sim._out_dir))


def test_a_single_leading_dot_is_still_allowed(sim) -> None:
    """A dotfile tag is legal - only ``.``/``..`` traversal is refused."""
    assert (
        sim.start_cameras_recording(output_dir=sim._out_dir, cameras=["c"], fps=10, name=".hidden")["status"]
        == "success"
    )
    time.sleep(0.25)
    assert sim.stop_cameras_recording()["status"] == "success"
    assert any(f.startswith(".hidden__") for f in os.listdir(sim._out_dir))


@pytest.mark.parametrize("bad", ["../../pwned", "a/b", "a\\b", "..", "a;b", "$HOME", ""])
def test_the_pre_existing_guards_still_hold(bad) -> None:
    with pytest.raises(ValueError):
        sanitize_name_component(bad)
