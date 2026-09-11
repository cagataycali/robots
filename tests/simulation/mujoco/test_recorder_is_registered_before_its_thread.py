"""A camera recorder is registered before the thread that fills it exists.

``_cams_rec_state`` is the only route every other recorder verb has to a live
recording: ``get_cameras_recording_status`` reads it to name a phase,
``stop_cameras_recording`` reads it to find the buffers to flush, and both start
verbs read it to refuse replacing a recording that still holds frames. So the
order in which :meth:`start_cameras_recording` starts its capture thread and
publishes that registration is a contract, not an implementation detail:
publishing second leaves a window in which a thread is already rendering into
buffers that nothing can reach.

Every consequence pinned here was measured in that window - a status read
answering ``[idle]`` about a live capture, a stop returning ``status="success"``
with "Was not recording cameras" while that thread kept filling its buffers to
the ``max_frames`` cap, and a second start admitted onto the same cameras whose
own registration was then overwritten, orphaning its thread and its frames.
That last one is the outcome the start guard exists to prevent, so the window
also disabled the guard.

The window is a few bytecodes wide, so it is not reachable by sleeping. It is
entered deterministically here instead: the recorder's ``Thread`` class is
replaced by a subclass that runs a hook on the *calling* thread as the recorder
thread is constructed, which is exactly the moment the recording must already
be registered. The threads are real and ``render`` is faked, so the buffers,
the guard and the flush all run without a GL context.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("mujoco", reason="mujoco not installed - pip install strands-robots[sim-mujoco]")
pytest.importorskip("imageio", reason="imageio not installed - pip install imageio imageio-ffmpeg")

from .test_daemon_camera_recording import _make_sim_with_fake_render, _wait_for_frames  # noqa: E402


def _head(envelope: dict) -> str:
    """The first line of an envelope's text block."""
    return envelope["content"][0]["text"].split("\n")[0]


class _UnstartableThread(threading.Thread):
    """A thread the interpreter refuses to start (``RuntimeError`` from ``start``)."""

    def start(self) -> None:
        raise RuntimeError("can't start new thread")


class _GapProbe:
    """Enters the window between the recorder thread and its registration.

    ``run(hook)`` starts a recording with ``hook`` wired to fire once, on the
    calling thread, as the recorder thread is constructed. Whatever the hook
    leaves behind - an extra recording, an extra thread - is drained by
    :meth:`drain`, which the fixture always calls.
    """

    def __init__(self, sim, monkeypatch, out_dir: Path) -> None:
        self.sim = sim
        self.monkeypatch = monkeypatch
        self.out_dir = out_dir
        self.threads: list[threading.Thread] = []
        self.states: list[dict] = []
        self._fired = False

    def start(self, name: str) -> dict:
        """Start a recording on ``cam_a`` (no hook wiring of its own)."""
        return self.sim.start_cameras_recording(
            cameras=["cam_a"], fps=20, width=32, height=24, name=name, output_dir=str(self.out_dir)
        )

    def run(self, hook, *, name: str = "first") -> dict:
        probe = self

        class _Hooked(threading.Thread):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                probe.threads.append(self)
                if not probe._fired:
                    probe._fired = True
                    hook()

        self.monkeypatch.setattr(threading, "Thread", _Hooked)
        return self.start(name)

    def unstartable(self, name: str) -> dict:
        """Start a recording whose capture thread cannot be started at all."""
        self.monkeypatch.setattr(threading, "Thread", _UnstartableThread)
        return self.start(name)

    def alive(self) -> int:
        return sum(1 for t in self.threads if t.is_alive())

    def drain(self) -> None:
        for state in [*self.states, getattr(self.sim, "_cams_rec_state", None)]:
            if state:
                state["running"] = False
        for thread in self.threads:
            thread.join(timeout=5)


@pytest.fixture
def probe(monkeypatch, tmp_path: Path):
    sim = _make_sim_with_fake_render()
    gap = _GapProbe(sim, monkeypatch, tmp_path)
    try:
        yield gap
    finally:
        gap.drain()
        sim.destroy()


def test_a_status_read_in_the_gap_sees_the_live_recording(probe: _GapProbe) -> None:
    """``get_cameras_recording_status`` never answers ``[idle]`` about a capture
    that is already running: ``[idle]`` promises there is no buffer left to
    encode, which is the one reading that verb documents it must never give."""
    seen: list[str] = []
    assert probe.run(lambda: seen.append(_head(probe.sim.get_cameras_recording_status())))["status"] == "success"

    assert seen[0].startswith("[recording]"), seen
    assert "first" in seen[0]


def test_a_second_start_in_the_gap_is_refused(probe: _GapProbe) -> None:
    """The start guard reads the registration, so an unpublished recording
    disabled it: the second start was admitted onto the same camera, and the
    outer publish then overwrote its registration - two capture threads on one
    camera set, one of them unreachable, with its frames unflushable."""
    second: list[dict] = []

    def _start_again() -> None:
        second.append(probe.start("second"))
        state = getattr(probe.sim, "_cams_rec_state", None)
        if state is not None:
            probe.states.append(state)

    assert probe.run(_start_again)["status"] == "success"

    assert second[0]["status"] == "error"
    assert "first" in _head(second[0])
    registered = getattr(probe.sim, "_cams_rec_state", None)
    assert registered is not None and registered["name"] == "first"
    time.sleep(0.2)
    assert probe.alive() == 1, "a refused start must leave no second capture thread"


def test_a_stop_in_the_gap_really_stops_the_capture(probe: _GapProbe) -> None:
    """``stop_cameras_recording`` reports "was not recording" as a success when
    it finds nothing registered, so in the window it told the caller the
    recording had stopped and left the capture thread rendering into a buffer
    only the ``max_frames`` cap would ever bound."""
    stopped: list[dict] = []
    assert probe.run(lambda: stopped.append(probe.sim.stop_cameras_recording()))["status"] == "success"

    assert stopped[0]["status"] == "success"
    assert "first" in _head(stopped[0]), _head(stopped[0])
    time.sleep(0.3)
    assert _head(probe.sim.get_cameras_recording_status()).startswith("[idle]")
    assert getattr(probe.sim, "_cams_rec_state", None) is None
    assert probe.alive() == 0, "the stop returned success while a capture thread was still running"


def test_a_recorder_thread_that_cannot_start_is_reported_not_raised(probe: _GapProbe) -> None:
    """Publishing first means a thread that never starts would leave a
    registration no flush can clear - refusing every later start for the life of
    the world and handing ``stop`` an unstarted thread to join. It is
    deregistered and reported in the envelope instead of raising past the tool
    boundary."""
    refused = probe.unstartable("doomed")

    assert refused["status"] == "error"
    assert "doomed" in _head(refused)
    assert getattr(probe.sim, "_cams_rec_state", None) is None
    probe.monkeypatch.undo()
    later = probe.start("later")
    assert later["status"] == "success", "a failed thread start must not wedge the recorder"


def test_the_ordinary_recording_round_trip_still_records(probe: _GapProbe) -> None:
    """The control: publishing before the thread starts changes nothing about an
    uncontended recording, which still captures frames and encodes them."""
    assert probe.start("plain")["status"] == "success"
    _wait_for_frames(probe.sim, "cam_a")

    result = probe.sim.stop_cameras_recording()

    assert result["status"] == "success"
    artifacts = result["content"][1]["json"]["artifacts"]
    assert [a["frames"] for a in artifacts] > [0]
    assert (probe.out_dir / "plain__cam_a.mp4").exists()
