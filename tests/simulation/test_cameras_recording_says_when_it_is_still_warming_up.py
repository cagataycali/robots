"""``start_cameras_recording`` / status / stop say when the recorder is still warming up.

On a machine where a fresh thread's GL context takes seconds to come up,
``start_cameras_recording`` returned the same "Recording N camera(s) @ FPS"
sentence whether the thread was capturing or not (the not-ready case was a
log-only warning), status showed ``[recording] … 0 frames`` with no reason, and a
stop a few seconds later listed ``0 frames  0.0 KB  -> rec__cam.mp4`` beside
status success - naming an MP4 that was never written. Each surface now says
which it is: start carries a ``capturing`` flag and either the warmup time or a
NOT CAPTURING YET line; status marks the warming phase; stop says "no MP4
written" and why, and the artifact's ``path`` is ``None``.
"""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


@pytest.fixture
def sim():
    s = MuJoCoSimEngine(tool_name="cams_warm", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    yield s
    s.cleanup()


def _state(sim, tmp_path, *, ready: bool, warmup_s=None, frames=0, errors=0, thread=True):
    ev = threading.Event()
    if ready:
        ev.set()
    import numpy as np

    state = {
        "running": True,
        "name": "rec_test",
        "cameras": ["default"],
        "fps": 10,
        "width": 64,
        "height": 48,
        "buffers": {"default": [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(frames)]},
        "paths": {"default": str(tmp_path / "rec_test__default.mp4")},
        "errors": {"default": errors},
        "output_dir": str(tmp_path),
        "started_mono": time.monotonic() - 6.2,
        "thread": threading.Thread(target=lambda: None) if thread else None,
        "max_frames": 3000,
        "ready": ev,
    }
    if warmup_s is not None:
        state["warmup_s"] = warmup_s
    return state


def test_status_marks_the_warming_phase(sim, tmp_path):
    sim._cams_rec_state = _state(sim, tmp_path, ready=False)
    text = _text(sim.get_cameras_recording_status())
    assert text.startswith("[recording] 'rec_test' for 6.")
    assert "(recorder thread still warming up - no frames yet)" in text
    sim._cams_rec_state["ready"].set()
    assert "warming up" not in _text(sim.get_cameras_recording_status())
    sim._cams_rec_state = None


def test_stop_with_nothing_captured_says_no_mp4_and_why_still_warming(sim, tmp_path):
    state = _state(sim, tmp_path, ready=False)
    r = sim._flush_cameras_recording_state(state)
    text = _text(r)
    assert "0 frames - no MP4 written (the recorder thread was still warming up for the whole 6.2s window)" in text
    assert "rec_test__default.mp4" not in text
    art = _json(r)["artifacts"][0]
    assert art["frames"] == 0 and art["path"] is None
    assert not (tmp_path / "rec_test__default.mp4").exists()


def test_stop_with_nothing_captured_after_a_late_warmup_names_the_window(sim, tmp_path):
    state = _state(sim, tmp_path, ready=True, warmup_s=6.15)
    text = _text(sim._flush_cameras_recording_state(state))
    assert "0 frames - no MP4 written (warmup took 6.2s of the 6.2s window and no capture tick landed after it)" in text


def test_stop_with_render_errors_only_names_them(sim, tmp_path):
    state = _state(sim, tmp_path, ready=True, warmup_s=0.5, errors=7)
    text = _text(sim._flush_cameras_recording_state(state))
    assert "0 frames - no MP4 written (every render after the 0.5s warmup failed (7 errors))" in text


def test_stop_with_frames_still_writes_the_mp4_line(sim, tmp_path):
    pytest.importorskip("imageio")
    state = _state(sim, tmp_path, ready=True, warmup_s=0.5, frames=5)
    r = sim._flush_cameras_recording_state(state)
    assert "5 frames" in _text(r) and "-> rec_test__default.mp4" in _text(r)
    assert _json(r)["artifacts"][0]["path"].endswith("rec_test__default.mp4")


def test_start_reports_whether_it_is_capturing_and_the_text_matches(sim, tmp_path):
    from strands_robots.simulation.mujoco.backend import _can_render

    if not _can_render():
        pytest.skip("no renderer")
    r = sim.start_cameras_recording(cameras=["default"], output_dir=str(tmp_path), fps=10, width=64, height=48)
    try:
        assert r["status"] == "success", _text(r)
        j = _json(r)
        assert j["cameras"] == ["default"] and j["fps"] == 10 and j["output_dir"] == str(tmp_path)
        assert isinstance(j["capturing"], bool)
        if j["capturing"]:
            assert "recorder warm after" in _text(r) and isinstance(j["warmup_s"], float)
        else:
            assert "NOT CAPTURING YET: the recorder thread is still warming its render context" in _text(r)
            assert "check get_cameras_recording_status shows frames before stopping" in _text(r)
    finally:
        sim.stop_cameras_recording()
