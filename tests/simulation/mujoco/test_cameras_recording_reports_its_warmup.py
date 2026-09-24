"""The camera recorder says whether it is capturing yet, and why it captured nothing.

``start_cameras_recording`` waits for the recorder thread to warm its render
context, but whether that wait succeeded was a log-only warning: the caller read
the same "Recording N camera(s) @ FPS" sentence over a capturing recorder and
over one still coming up, saw ``[recording] ... 0 frames`` in status with no
reason, and stopped to a "no clip written (0 errors)" line that read as "nothing
went wrong". Each of the three surfaces now names the warmup.
"""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("mujoco")
import numpy as np

from strands_robots.simulation.mujoco import rendering
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


@pytest.fixture
def sim():
    engine = MuJoCoSimEngine(tool_name="cams_warmup", mesh=False)
    engine.create_world()
    assert engine.add_robot(name="so101", data_config="so101")["status"] == "success"
    yield engine
    engine._cams_rec_state = None
    engine.cleanup()


def _state(tmp_path, *, warm: bool, warmup_s=None, frames=0, errors=0, mode=None) -> dict:
    """A recorder state as ``start_cameras_recording`` publishes one, 6.2 s old."""
    ready = threading.Event()
    if warm:
        ready.set()
    state = {
        "running": True,
        "name": "rec_test",
        "cameras": ["default"],
        "fps": 10,
        "width": 64,
        "height": 48,
        "buffers": {"default": [np.zeros((48, 64, 3), dtype=np.uint8)] * frames},
        "paths": {"default": str(tmp_path / "rec_test__default.mp4")},
        "errors": {"default": errors},
        "output_dir": str(tmp_path),
        "started_mono": time.monotonic() - 6.2,
        "thread": None if mode == "synchronous" else threading.Thread(target=lambda: None),
        "max_frames": 3000,
    }
    if mode == "synchronous":
        state["mode"] = mode
    else:
        state["ready"] = ready
    if warmup_s is not None:
        state["warmup_s"] = warmup_s
    return state


# Each row is a way a window can end with an empty buffer. The frames row is the
# control: a recorder that captured still names its MP4.
@pytest.mark.parametrize(
    ("case", "kwargs", "expected"),
    [
        (
            "never warmed",
            {"warm": False},
            "0 frames - no clip written (the recorder thread was still warming its render context "
            "for the whole 6.2s window)",
        ),
        (
            "warmed too late",
            {"warm": True, "warmup_s": 6.15},
            "0 frames - no clip written (warmup took 6.2s of the 6.2s window and no capture tick landed after it)",
        ),
        (
            "every render failed",
            {"warm": True, "warmup_s": 0.5, "errors": 7},
            "0 frames - no clip written (every render after the 0.5s warmup failed (7 errors))",
        ),
        (
            "synchronous, never stepped",
            {"mode": "synchronous"},
            "0 frames - no clip written (no step() rendered into the synchronous recorder during the 6.2s window)",
        ),
    ],
)
def test_a_stop_with_nothing_buffered_names_the_cause(sim, tmp_path, case, kwargs, expected):
    result = sim._flush_cameras_recording_state(_state(tmp_path, **{"warm": True, **kwargs}))
    assert expected in _text(result), _text(result)
    artifact = _json(result)["artifacts"][0]
    assert artifact["frames"] == 0 and artifact["path"] is None
    assert not (tmp_path / "rec_test__default.mp4").exists()


def test_a_stop_that_captured_still_names_its_clip(sim, tmp_path):
    pytest.importorskip("imageio")
    result = sim._flush_cameras_recording_state(_state(tmp_path, warm=True, warmup_s=0.5, frames=5))
    assert "5 frames" in _text(result) and "-> rec_test__default.mp4" in _text(result)
    assert "no clip written" not in _text(result)
    assert _json(result)["artifacts"][0]["path"].endswith("rec_test__default.mp4")


def test_status_marks_the_warming_phase_until_the_recorder_is_warm(sim, tmp_path):
    sim._cams_rec_state = _state(tmp_path, warm=False)
    warming = _text(sim.get_cameras_recording_status())
    assert warming.startswith("[recording] 'rec_test' for 6."), warming
    assert "(recorder thread still warming its render context - no frames yet)" in warming
    sim._cams_rec_state["ready"].set()
    assert "warming" not in _text(sim.get_cameras_recording_status())


def test_start_says_it_is_not_capturing_yet_when_warmup_outlasts_the_wait(sim, tmp_path, monkeypatch):
    # A recorder whose thread never warms, with the wait collapsed to zero so
    # the branch is reached in milliseconds rather than the real timeout.
    state = _state(tmp_path, warm=False)
    monkeypatch.setattr(rendering, "_cams_rec_ready_timeout", lambda n: 0.0, raising=False)
    monkeypatch.setattr(MuJoCoSimEngine, "_start_cameras_recording_under_lock", lambda self, **kw: {"state": state})
    result = sim.start_cameras_recording(cameras=["default"], output_dir=str(tmp_path))
    assert result["status"] == "success"
    assert "NOT CAPTURING YET: the recorder thread is still warming its render context" in _text(result)
    assert "check get_cameras_recording_status reports frames before you stop" in _text(result)
    assert _json(result) == {
        "recording": "rec_test",
        "cameras": ["default"],
        "fps": 10,
        "output_dir": str(tmp_path),
        "capturing": False,
        "warmup_s": None,
    }


def test_start_reports_the_warmup_a_capturing_recorder_paid(sim, tmp_path):
    from strands_robots.simulation.mujoco.backend import _can_render

    if not _can_render():
        pytest.skip("no renderer")
    result = sim.start_cameras_recording(cameras=["default"], output_dir=str(tmp_path), fps=10, width=64, height=48)
    try:
        assert result["status"] == "success", _text(result)
        block = _json(result)
        assert block["capturing"] is True, _text(result)
        assert isinstance(block["warmup_s"], float) and block["warmup_s"] >= 0.0
        assert f"warmup: {block['warmup_s']:.1f}s - the recorder is capturing" in _text(result)
    finally:
        sim.stop_cameras_recording()
