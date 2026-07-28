"""Regression tests: a starved camera recording says so instead of reporting success.

``start_cameras_recording`` paces capture by the **wall clock** on a daemon thread
that renders under the sim lock, so a tight ``step()`` / policy loop starves it.
Measured at ``fps=20`` on a GPU renderer:

    for _ in range(120): step(5)   (1.2 s simulated)  ->   1 frame,  24.1 KB
    step(600)                      (1.2 s simulated)  ->   1 frame,  24.1 KB
    time.sleep(1.0), no stepping                      ->  20 frames, 47.1 KB
    sleep(1.0) interleaved with steps                 ->  21 frames, 66.6 KB

Both starved cases reported ``status="success"`` with a near-empty MP4, and the
docstring never said capture was wall-clock paced - so an agent recording a fast
rollout got a one-frame video and no indication the footage did not cover it.

The pacing is a legitimate design choice for interactive viewing, so the fix is to
surface the shortfall (and document the pacing) rather than change it. Frames driven
by ``run_policy`` through ``start_recording`` are unaffected - those are per control
step, not wall clock.

The rate a machine can sustain is not a constant: one 160x120 frame costs
microseconds through a GPU driver and a large fraction of a second through the
software GL of a headless runner. Every test here that asserts an UNSTARVED
capture therefore calibrates first (see the ``keeps_up`` fixture) instead of
assuming the nominal 20 fps is attainable.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("imageio")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_FPS = 20
#: Probe rate for measuring throughput - far above any renderer, so the capture
#: loop runs flat out instead of pacing itself.
_PROBE_FPS = 1000


@pytest.fixture
def sim(tmp_path):
    s = Simulation(tool_name="camera_recording_starvation", mesh=False)
    s.create_world()
    assert s.add_robot(name="a", data_config="panda")["status"] == "success"
    assert s.add_camera(name="side", position=[1, 1, 0.8], target=[0, 0, 0.3])["status"] == "success"
    s._starvation_output_dir = str(tmp_path)
    yield s
    s.destroy()


def _record(sim, body, fps: int = _FPS) -> str:
    started = sim.start_cameras_recording(
        output_dir=sim._starvation_output_dir, cameras=["side"], fps=fps, width=160, height=120
    )
    assert started["status"] == "success", started
    body(sim)
    stopped = sim.stop_cameras_recording()
    assert stopped["status"] == "success", stopped
    return stopped["content"][0]["text"]


def _report(text: str) -> tuple[int, float]:
    """Frames written for camera ``side`` and the recording's wall-clock span."""
    frames = int(text.split("side")[1].split("frames")[0].strip())
    elapsed = float(text.split("after ")[1].split("s")[0])
    return frames, elapsed


def _flush_report(
    sim,
    tmp_path,
    *,
    frames: int,
    capture_window: float,
    span: float | None = None,
    fps: int = _FPS,
    include_capture_start: bool = True,
) -> str:
    """Report text for a hand-built recorder state - exact windows, no machine timing.

    Whether a live capture keeps up depends on the machine, but which windows the
    check is entitled to judge does not, so drive the flush directly for those.
    ``span`` is the whole recording (default: the capture window, which is what a
    recorder without a separate warmup phase reports), and
    ``include_capture_start=False`` drops the key such a recorder never sets.
    """
    now = time.time()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    state = {
        "name": "rec_flush",
        "output_dir": str(tmp_path),
        "cameras": ["side"],
        "buffers": {"side": [frame] * frames},
        "paths": {"side": str(tmp_path / "rec_flush__side.mp4")},
        "errors": {"side": 0},
        "fps": fps,
        "started_at": now - (capture_window if span is None else span),
        "running": False,
    }
    if include_capture_start:
        state["capture_started_at"] = now - capture_window
    result = sim._flush_cameras_recording_state(state)
    assert result["status"] == "success", result
    return result["content"][0]["text"]


@pytest.fixture
def render_throughput(sim) -> float:
    """Frames per second this machine's recorder can actually produce.

    Render cost is not a constant: one 160x120 frame is microseconds through a
    GPU driver and a large fraction of a second through software GL, so the same
    idle 1.0 s recording collects the full 20 frames at ``fps=20`` on the former
    and 3 on the latter. Every assertion about what the note should or should not
    say is relative to this ceiling, so measure it.

    Probed with a rate no renderer can serve (``_PROBE_FPS``) so the capture loop
    runs flat out - asking for ``_FPS`` would measure at most ``_FPS`` and reveal
    nothing about the headroom above it. Nothing steps during the probe, so only
    render throughput limits it. The figure is conservative: the recorder thread's
    GL warmup is inside the measured span but produces no frames.
    """
    frames, elapsed = _report(_record(sim, lambda s: time.sleep(1.0), fps=_PROBE_FPS))
    return frames / elapsed if elapsed > 0 else 0.0


@pytest.fixture
def keeps_up(render_throughput) -> tuple[int, float]:
    """A rate this machine comfortably serves, and a duration long enough to judge.

    Half the measured ceiling leaves 2x headroom. The duration is stretched so the
    clip still expects the ten frames
    :meth:`_flush_cameras_recording_state` requires before it judges a recording
    at all - otherwise a slow machine would pass by being too short to assess.
    """
    fps = max(1, min(_FPS, int(render_throughput / 2)))
    return fps, max(1.0, 12.0 / fps)


def test_a_starved_capture_is_flagged(sim, render_throughput) -> None:
    """The core defect: a near-empty video reported as plain success.

    How badly a tight ``step()`` loop starves the daemon depends on the machine,
    so make the shortfall certain rather than likely: ask for eight times the rate
    this machine was measured to produce AND compete for the sim lock while it
    tries. It cannot serve that, so the report has to say so.
    """
    fps = max(8, int(8 * render_throughput))
    duration = max(1.5, 12.0 / fps)

    def body(s):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            s.step(n_steps=40)

    text = _record(sim, body, fps=fps)
    frames, _ = _report(text)
    assert "expected at" in text, text
    assert "wall-clock paced" in text
    assert f"captured {frames} of" in text, text


def test_the_note_names_the_remedy() -> None:
    """Checked on the emitted text, not on a live capture.

    How badly a tight loop starves the daemon depends on machine load, so asserting
    a specific shortfall in a live run is inherently flaky. The note's wording is
    what has to stay actionable, so assert on the source of truth: the branch that
    builds it.
    """
    source = inspect.getsource(Simulation._flush_cameras_recording_state)
    assert "expected at" in source
    assert "wall-clock paced" in source
    assert "sleep" in source and "render()" in source


def test_a_wall_clock_paced_recording_is_not_flagged(sim, keeps_up) -> None:
    """Guard against noise on a healthy recording: nothing competes for the lock."""
    fps, duration = keeps_up
    text = _record(sim, lambda s: time.sleep(duration), fps=fps)
    assert "expected at" not in text, text


def test_interleaving_a_sleep_is_not_flagged(sim, keeps_up) -> None:
    """The suggested remedy must actually silence the note."""
    fps, duration = keeps_up

    def body(s):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            s.step(n_steps=40)
            # Hand back half of each frame period - the remedy the note names,
            # sized to the rate this machine was measured to sustain.
            time.sleep(0.5 / fps)

    text = _record(sim, body, fps=fps)
    assert "expected at" not in text, text


def test_the_shortfall_is_judged_from_when_capture_began(sim, tmp_path) -> None:
    """The recorder thread's GL warmup captures nothing, so it is not charged to the rate.

    Ten frames across a 0.5 s capture window IS the nominal 20 fps. The five
    seconds before capture began went on warming a cold GL context - which on
    software GL is most of a short recording, and charging it to the rate made
    every brief clip look starved.
    """
    text = _flush_report(sim, tmp_path, frames=10, capture_window=0.5, span=5.0)
    assert "expected at" not in text, text


def test_a_recorder_without_a_warmup_phase_is_judged_over_its_whole_span(sim, tmp_path) -> None:
    """The synchronous recorder captures from the first call, so nothing is discounted.

    Its state carries no ``capture_started_at``, and 10 frames of an expected ~100
    is a genuine shortfall that must still be reported.
    """
    text = _flush_report(sim, tmp_path, frames=10, capture_window=5.0, include_capture_start=False)
    assert "expected at" in text, text


def test_a_very_short_recording_is_not_flagged(sim, tmp_path) -> None:
    """Too few expected frames to judge; do not cry wolf on a brief clip.

    At 20 fps a 0.3 s capture nominally expects 6 frames - under the ten the check
    requires before it judges anything - so one frame is not evidence of
    starvation. Stretch the same single frame over a second and it is.
    """
    brief = _flush_report(sim, tmp_path, frames=1, capture_window=0.3)
    assert "expected at" not in brief, brief

    long_enough = _flush_report(sim, tmp_path, frames=1, capture_window=1.0)
    assert "expected at" in long_enough, long_enough


def test_the_recording_still_succeeds_and_reports_what_it_wrote(sim) -> None:
    """The note is advisory - it must not turn a capture into an error.

    How much a tight loop leaves the daemon is machine-dependent, so the pin is
    that the report matches the disk: the artifact is always reported, and the MP4
    exists with a non-zero size exactly when frames were written to it.
    """
    started = sim.start_cameras_recording(
        output_dir=sim._starvation_output_dir, cameras=["side"], fps=_FPS, width=160, height=120
    )
    assert started["status"] == "success"
    for _ in range(200):
        sim.step(n_steps=40)
    stopped = sim.stop_cameras_recording()
    assert stopped["status"] == "success"
    payloads = [b["json"] for b in stopped["content"] if "json" in b]
    assert payloads, stopped
    entries = payloads[0]["artifacts"]
    assert [e["camera"] for e in entries] == ["side"], entries
    entry = entries[0]
    assert entry["path"].endswith(".mp4"), entry
    if entry["frames"]:
        assert Path(entry["path"]).exists(), entry
        assert entry["size_kb"] > 0, entry


def test_the_docstring_documents_the_pacing(sim) -> None:
    """The absence of this is what made the shortfall invisible."""
    doc = type(sim).start_cameras_recording.__doc__ or ""
    assert "wall clock" in doc
    assert "starves" in doc
