# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Every span ``lerobot_camera`` reports is measured on the clock that cannot step.

``lerobot_camera`` reports a span from four of its actions: ``capture`` reports a
connect time and a capture time, ``capture_batch`` a per-camera capture time and
a batch total, ``record`` the recording's achieved duration, and ``test`` a
connect time plus two ten-frame read windows. Those windows are durations, so the
tree's clock boundary puts them on ``time.monotonic()`` -- and ``preview``, the
fifth, already had them there, with the reason written above its bases. The other
four measured on ``time.time()``, which is not a clock but the current opinion
about the date: an NTP correction, a ``date -s`` or a resume from suspend lands
inside the window and the step is subtracted from the span, with nothing raised.

``test`` is why this is not only a cosmetic mis-report. It turns each span into a
verdict about the *camera*: ``Est. FPS: {1 / avg_sync_time}``, ``Connection:
Fast`` below one second, ``Sync capture: Good`` below 100 ms, ``Frame rate:
Stable`` within a 50 ms spread. A +30 s correction landing in the ten sync reads
makes one sample 30 s, so a camera reading in 20 ms is reported as ``Slow`` at
``0.3`` FPS; a -30 s correction makes the sample negative, and the tool then
reports a negative frame rate as a measurement, calls the camera ``Good`` because
a negative average is below the threshold, and can divide by a zero average.

These cells drive the real handlers with a wall clock that takes one step
mid-window and assert on what the tool *reports*, which is stronger than the
source shape: a span measured on the wall clock cannot pass them whatever it is
named. The step is a property of the clock double rather than of a read the
implementation performs, so it lands whether or not the code under test consults
that clock -- which a correct implementation does not. That is the same shape as
``tests/tools/test_tool_wait_budgets_survive_a_clock_step.py``, which pins the
budgets ``spin_for`` and the preview *decide* on; this file pins the spans this
tool *reports*, and the last cell pins that no wall-clock read is left in the
module for a later span to be built from.

The package-wide source scan cannot see any of this and is right not to:
``tests/test_expiry_gates_survive_a_clock_step.py`` grades a wall-clock read that
decides whether to keep waiting, and none of these four does. The recording is
bounded by a frame count and the ten-frame loops by ``range(10)``, so the tree
was clean by that predicate while four handlers reported spans off the wall
clock.
"""

from __future__ import annotations

import ast
import pathlib
import re
import time
from typing import Any

import numpy as np
import pytest

import strands_robots.tools.lerobot_camera as cam_mod

#: Real seconds each modelled device operation takes. Wide enough that a clock
#: step can be armed to land inside a specific window, and small enough to keep
#: the whole file under a second of real time.
_READ_SECONDS = 0.02


class _SlowCamera:
    """A camera stand-in whose every operation takes a measurable span.

    ``tests/tools/test_lerobot_camera_async_read_budget.py`` has a sibling double
    for the read *budget*; this one exists for the read *span*, so its reads are
    slow enough (``_READ_SECONDS``) for a clock step to be placed inside one and
    fast enough that a real camera's verdicts still hold: 20 ms per frame is
    ``Good`` (under 100 ms) at an estimated 50 FPS.
    """

    def __init__(self) -> None:
        self.width = 8
        self.height = 6
        self.fps = 30
        self.color_mode = type("_M", (), {"value": "RGB"})()
        self.rotation: Any = None

    def connect(self, warmup: bool = True) -> None:
        time.sleep(_READ_SECONDS)

    def disconnect(self) -> None:
        return None

    def _frame(self) -> np.ndarray:
        time.sleep(_READ_SECONDS)
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def read(self) -> np.ndarray:
        return self._frame()

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        return self._frame()


class _SteppingWallClock:
    """A wall clock that takes a single step, the way an NTP correction does.

    Elapsed real time comes from ``time.monotonic()``, so the double advances
    exactly as the true wall clock would, plus one discontinuity of ``step_by``
    seconds once ``step_after`` seconds of real time have passed. The trigger is
    real time rather than a read count, so the step lands whether or not the code
    under test reads this clock -- which is what lets the premise be asserted
    independently of the behaviour.
    """

    def __init__(self, step_after: float, step_by: float) -> None:
        self._epoch = 1_700_000_000.0
        self._origin = time.monotonic()
        self._step_after = step_after
        self._step_by = step_by
        self._stepped = False

    def __call__(self) -> float:
        elapsed = time.monotonic() - self._origin
        if not self._stepped and elapsed >= self._step_after:
            self._stepped = True
            self._epoch += self._step_by
        return self._epoch + elapsed


@pytest.fixture
def camera(monkeypatch: pytest.MonkeyPatch) -> _SlowCamera:
    """Substitute the camera factory and neutralise every sink a handler writes to."""
    cam = _SlowCamera()

    monkeypatch.setattr(cam_mod, "_create_camera", lambda *a, **k: cam)
    writer = type("_W", (), {"write": lambda self, f: None, "release": lambda self: None})()
    monkeypatch.setattr(cam_mod.cv2, "VideoWriter", lambda *a, **k: writer)
    monkeypatch.setattr(cam_mod.cv2, "VideoWriter_fourcc", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(cam_mod.cv2, "imwrite", lambda *a, **k: True)
    monkeypatch.setattr(cam_mod.os.path, "getsize", lambda p: 1234)
    return cam


def _install(monkeypatch: pytest.MonkeyPatch, clock: _SteppingWallClock) -> None:
    """Make ``clock`` the process wall clock for the duration of one cell."""
    monkeypatch.setattr(cam_mod.time, "time", clock)


def _text(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _reported(text: str, label: str) -> float:
    """The number the report states for ``label``, as the caller reads it."""
    match = re.search(rf"{re.escape(label)}[^-\d]*(-?\d+\.?\d*)", text)
    assert match is not None, f"the report states no {label!r}: {text}"
    return float(match.group(1))


def test_the_stepping_wall_clock_double_takes_the_step_it_advertises() -> None:
    """Pin the double: with no real discontinuity the cells below prove nothing."""
    clock = _SteppingWallClock(step_after=0.01, step_by=+30.0)
    before = clock()
    time.sleep(0.05)
    after = clock()

    assert after - before > 29.0, f"the double advanced {after - before:.3f}s, so it never stepped"


class TestASpanIsNotTheSizeOfAClockCorrection:
    """A reported span is the work's, so a step cannot appear in it."""

    def test_a_capture_reports_the_connect_and_read_spans_it_measured(
        self, camera: _SlowCamera, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A -30 s step inside the connect makes a wall-clock span negative."""
        _install(monkeypatch, _SteppingWallClock(step_after=_READ_SECONDS / 2, step_by=-30.0))

        result = cam_mod.lerobot_camera(
            action="capture", camera_type="opencv", camera_id=0, save_path=str(tmp_path), width=8, height=6
        )

        assert result["status"] == "success"
        text = _text(result)
        assert 0.0 <= _reported(text, "Connect time:") < 5.0, text
        assert 0.0 <= _reported(text, "Capture time:") < 5.0, text

    def test_a_batch_reports_a_total_that_is_the_work_rather_than_the_step(
        self, camera: _SlowCamera, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The batch total shared one variable with its own base, so it is graded too."""
        _install(monkeypatch, _SteppingWallClock(step_after=_READ_SECONDS / 2, step_by=+30.0))

        result = cam_mod.lerobot_camera(
            action="capture_batch", camera_type="opencv", camera_ids=[0], save_path=str(tmp_path), width=8, height=6
        )

        assert result["status"] == "success"
        assert 0.0 <= _reported(_text(result), "Total time:") < 5.0, _text(result)

    @pytest.mark.parametrize("step_by", (+30.0, -30.0))
    def test_a_recording_reports_the_duration_it_actually_took(
        self, camera: _SlowCamera, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, step_by: float
    ) -> None:
        """The recording is frame-bounded, so only its reported duration can be wrong.

        The step is armed for ``_READ_SECONDS * 2``, which is inside the four-frame
        loop rather than inside the connect that precedes it: a step landing
        *before* the duration base is taken moves the base and the end together
        and is invisible, so a window this cell can only grade from inside.
        """
        _install(monkeypatch, _SteppingWallClock(step_after=_READ_SECONDS * 2, step_by=step_by))

        result = cam_mod.lerobot_camera(
            action="record",
            camera_type="opencv",
            camera_id=0,
            save_path=str(tmp_path),
            width=8,
            height=6,
            fps=2,
            capture_duration=2.0,
        )

        assert result["status"] == "success"
        assert 0.0 <= _reported(_text(result), "Duration:") < 5.0, _text(result)


class TestAPerformanceVerdictIsAboutTheCameraNotTheClock:
    """``test`` turns each span into a verdict, so a step becomes a device claim."""

    def test_a_forward_step_does_not_report_a_fast_camera_as_slow(
        self, camera: _SlowCamera, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One +30 s sample among ten 20 ms reads averages 3 s: ``Slow`` at 0.3 FPS."""
        _install(monkeypatch, _SteppingWallClock(step_after=_READ_SECONDS * 3, step_by=+30.0))

        result = cam_mod.lerobot_camera(
            action="test", camera_type="opencv", camera_id=0, save_path=str(tmp_path), width=8, height=6
        )

        assert result["status"] == "success"
        text = _text(result)
        assert "Sync capture: Good" in text, text
        assert "Connection: Fast" in text, text
        assert _reported(text, "Est. FPS:") > 5.0, text

    def test_a_backward_step_does_not_report_a_negative_frame_rate(
        self, camera: _SlowCamera, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A negative average is also below every threshold, so the verdict flatters."""
        _install(monkeypatch, _SteppingWallClock(step_after=_READ_SECONDS * 3, step_by=-30.0))

        result = cam_mod.lerobot_camera(
            action="test", camera_type="opencv", camera_id=0, save_path=str(tmp_path), width=8, height=6
        )

        assert result["status"] == "success"
        text = _text(result)
        assert _reported(text, "Est. FPS:") > 5.0, text
        assert _reported(text, "Average:") >= 0.0, text
        assert _reported(text, "Min:") >= 0.0, text


def test_no_span_in_the_module_can_be_built_from_the_wall_clock_again() -> None:
    """The module reads no wall clock at all, so a new span cannot pick one up.

    Every clock read here is a duration; the absolute stamps this tool writes -- a
    filename's date, a report's ``Timestamp`` line -- come from ``datetime.now()``
    rather than from ``time.time()``, which is why the count that holds is zero
    rather than "the stamps only". The premise below states that the stamps are
    still written, so this is not read as a module that stopped stamping.
    """
    source = pathlib.Path(cam_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    wall_reads = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("time", "time_ns")
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "time"
    ]
    stamps = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "now"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "datetime"
    ]

    assert stamps, "premise: this module stamps its reports, so a stamp call must be present"
    assert not wall_reads, (
        f"these lines read the wall clock: {wall_reads}. Every span this tool reports is a "
        "duration and belongs on time.monotonic(); an absolute stamp here is written with "
        "datetime.now()"
    )
