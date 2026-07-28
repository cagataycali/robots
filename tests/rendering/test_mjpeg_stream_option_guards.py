# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``mjpeg_frames`` refuses a stream configuration it cannot produce.

``encode_clip``, the other media entry point in
:mod:`strands_robots.rendering.video`, validates its playback rate through the
shared pixel/frame-count domain. ``mjpeg_frames`` validated nothing, so every
one of its four stream options had a value that was accepted and then not
honored:

* ``fps`` of ``0``, a negative, ``nan`` or ``inf`` all disabled pacing outright
  and the generator emitted as fast as it could encode JPEGs;
* ``quality`` outside ``[1, 95]`` was silently substituted by Pillow;
* a malformed ``size`` raised ``ValueError``/``TypeError`` out of ``Image.resize``;
* ``max_frames`` of ``0`` or a negative emitted nothing, and ``2.7`` emitted 3.

The refusals also have to arrive at the call, not on the consumer's first
``next()``: the caller is an HTTP handler that has already written the
``multipart/x-mixed-replace`` response headers by then.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from strands_robots.rendering import mjpeg_frames

FRAME = np.zeros((8, 8, 3), dtype=np.uint8)


def _frame() -> np.ndarray:
    return FRAME


class TestPacingRate:
    """Only a positive finite ``fps`` can be slept to, so only that is accepted."""

    @pytest.mark.parametrize(
        "fps",
        [
            0,  # 1 / 0 is undefined; fell into the "no pacing" fallback
            0.0,
            -5,  # a negative period is not a rate
            -0.5,
            float("nan"),  # nan > 0 is False -> same fallback
            float("inf"),  # 1 / inf is 0.0 -> same fallback via the front door
            True,  # int subclass; would have acted as a silent 1 fps
            "12",  # compared against 0 as a str -> bare TypeError
            None,
            [12],
        ],
    )
    def test_a_rate_the_loop_cannot_pace_to_is_refused(self, fps: Any) -> None:
        with pytest.raises(ValueError, match="fps must be a positive finite number"):
            mjpeg_frames(_frame, fps=fps, max_frames=1)

    def test_the_refused_rate_is_named_in_the_message(self) -> None:
        with pytest.raises(ValueError, match=r"got 0\b"):
            mjpeg_frames(_frame, fps=0, max_frames=1)

    def test_an_unpaceable_rate_emits_no_chunk_at_all(self) -> None:
        """The pre-fix outcome was a full-speed stream, not an empty one."""
        with pytest.raises(ValueError):
            list(mjpeg_frames(_frame, fps=0, max_frames=200))

    def test_a_fractional_rate_is_honored(self) -> None:
        """Wider than ``encode_clip``: pacing is a sleep, not a container header."""
        chunks = list(mjpeg_frames(_frame, fps=12.5, max_frames=2))
        assert len(chunks) == 2

    def test_a_numpy_rate_is_honored(self) -> None:
        # ``numbers.Real`` covers every NumPy float; the annotation says ``float``.
        chunks = list(mjpeg_frames(_frame, fps=np.float32(500.0), max_frames=2))  # type: ignore[arg-type]
        assert len(chunks) == 2

    def test_the_accepted_rate_actually_paces_the_stream(self) -> None:
        """The rate is a real budget: N frames take at least (N-1)/fps seconds."""
        fps = 20.0
        n = 4
        started = time.monotonic()
        chunks = list(mjpeg_frames(_frame, fps=fps, max_frames=n))
        elapsed = time.monotonic() - started
        assert len(chunks) == n
        # (n - 1) intervals: the first chunk is emitted before any sleep.
        assert elapsed >= (n - 1) / fps


class TestJpegQuality:
    """Pillow clamps an out-of-range quality, so the stream is not what was asked for."""

    @pytest.mark.parametrize(
        "quality",
        [
            0,  # encoded identically to 1
            -5,
            96,  # above the documented ceiling
            100,  # accepted by Pillow; byte-identical to 500
            500,
            2.5,  # not a whole number
            float("nan"),
            True,
            "75",  # raised "Invalid quality setting" from Pillow
            None,
        ],
    )
    def test_a_quality_pillow_would_substitute_is_refused(self, quality: Any) -> None:
        with pytest.raises(ValueError, match="quality must be a whole number between 1 and 95"):
            mjpeg_frames(_frame, fps=1000.0, quality=quality, max_frames=1)

    @pytest.mark.parametrize("quality", [1, 75, 95, 95.0, np.int64(50)])
    def test_a_quality_inside_the_documented_range_is_honored(self, quality: Any) -> None:
        chunks = list(mjpeg_frames(_frame, fps=1000.0, quality=quality, max_frames=1))
        assert len(chunks) == 1


class TestFrameSize:
    """A malformed resize pair is refused instead of raising out of ``Image.resize``."""

    @pytest.mark.parametrize(
        "size",
        [
            (16,),  # TypeError: sequence of length 2
            (16, 12, 3),
            (),
            16,  # TypeError: not iterable
            "16x12",
            {"width": 16, "height": 12},
        ],
    )
    def test_a_pair_that_is_not_two_numbers_is_refused(self, size: Any) -> None:
        with pytest.raises(ValueError, match=r"size must be a \(width, height\) pair"):
            mjpeg_frames(_frame, fps=1000.0, size=size, max_frames=1)

    @pytest.mark.parametrize(
        ("size", "component"),
        [
            ((0, 12), "size width"),
            ((-4, 12), "size width"),
            ((16, 0), "size height"),
            ((16, -12), "size height"),
            ((16.5, 12), "size width"),
            ((16, None), "size height"),
        ],
    )
    def test_a_non_pixel_component_is_refused_and_named(self, size: Any, component: str) -> None:
        with pytest.raises(ValueError, match=f"{component} must be a positive whole number"):
            mjpeg_frames(_frame, fps=1000.0, size=size, max_frames=1)

    @pytest.mark.parametrize("size", [(16, 12), [16, 12], np.array([16, 12])])
    def test_a_pixel_pair_is_honored_whatever_container_carries_it(self, size: Any) -> None:
        chunks = list(mjpeg_frames(_frame, fps=1000.0, size=size, max_frames=1))
        assert len(chunks) == 1


class TestFrameBudget:
    """``max_frames`` counts frames, so it shares the frame-count domain."""

    @pytest.mark.parametrize(
        "max_frames",
        [
            0,  # emitted nothing while reporting nothing
            -5,
            2.7,  # emitted 3
            True,  # int subclass; acted as a silent 1
            "3",
            float("nan"),
        ],
    )
    def test_a_budget_that_is_not_a_frame_count_is_refused(self, max_frames: Any) -> None:
        with pytest.raises(ValueError, match="max_frames must be a positive whole number"):
            mjpeg_frames(_frame, fps=1000.0, max_frames=max_frames)

    @pytest.mark.parametrize("max_frames", [1, 3, np.int64(2)])
    def test_a_frame_count_is_honored_exactly(self, max_frames: Any) -> None:
        chunks = list(mjpeg_frames(_frame, fps=1000.0, max_frames=max_frames))
        assert len(chunks) == int(max_frames)

    def test_an_absent_budget_still_means_unbounded(self) -> None:
        stream = mjpeg_frames(_frame, fps=1000.0)
        assert next(stream).startswith(b"--frame")
        stream.close()  # type: ignore[attr-defined]


class TestRefusalArrivesAtTheCallSite:
    """A generator body runs on the first ``next()`` - too late for an HTTP handler."""

    def test_an_unusable_configuration_raises_from_the_call_itself(self) -> None:
        with pytest.raises(ValueError, match="fps must be a positive finite number"):
            mjpeg_frames(_frame, fps=0)  # no iteration at all

    def test_a_refused_configuration_never_asks_for_a_frame(self) -> None:
        calls = {"n": 0}

        def counting_frame() -> np.ndarray:
            calls["n"] += 1
            return FRAME

        with pytest.raises(ValueError):
            mjpeg_frames(counting_frame, fps=1000.0, quality=500, max_frames=1)
        assert calls["n"] == 0

    def test_an_accepted_configuration_still_renders_nothing_until_iterated(self) -> None:
        """Validation is eager; frame production stays lazy."""
        calls = {"n": 0}

        def counting_frame() -> np.ndarray:
            calls["n"] += 1
            return FRAME

        stream = mjpeg_frames(counting_frame, fps=1000.0, max_frames=1)
        assert calls["n"] == 0
        assert len(list(stream)) == 1
        assert calls["n"] == 1
