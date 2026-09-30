"""A recorded Isaac episode does not open on a camera frame that has not rendered.

Measured on one L40S (Isaac Sim 6.1, lerobot 0.6.1): an so101 wrist camera 0.26 m
above a cube recorded its first two frames of every episode as all zeros (std
0.0), frame 2 on real; MuJoCo never. The probe frame before ``start_recording``
was lit, and render-only ticks did not light the camera - only a physics step
did - so an ACT or SmolVLA trained on the dataset saw black inputs at every
episode start. After: the two unrendered frames are not written, and the
episode's frame 0 is a real image (std 0.075).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.recording import (  # noqa: E402
    _MAX_UNRENDERED_SKIPS,
    _opens_on_an_unrendered_frame,
)

_LIT = np.full((4, 6, 3), 90, dtype=np.uint8)
_BLACK = np.zeros((4, 6, 3), dtype=np.uint8)


class _Recorder:
    def __init__(self, frames: int = 0) -> None:
        self.episode_frame_count = frames


def _state() -> dict[str, Any]:
    return {"recording_cameras": [("wrist", "wrist", 6, 4), ("front", "front", 6, 4)]}


class TestTheEpisodeOpensOnARenderedFrame:
    def test_a_black_camera_at_the_start_is_skipped(self) -> None:
        state = _state()
        assert _opens_on_an_unrendered_frame(_Recorder(0), state, {"wrist": _BLACK, "front": _LIT}) is True
        assert state["unrendered_skips"] == 1

    def test_a_missing_camera_at_the_start_is_skipped(self) -> None:
        assert _opens_on_an_unrendered_frame(_Recorder(0), _state(), {"front": _LIT}) is True

    def test_every_camera_lit_is_written(self) -> None:
        assert _opens_on_an_unrendered_frame(_Recorder(0), _state(), {"wrist": _LIT, "front": _LIT}) is False

    def test_after_the_first_frame_a_black_frame_is_written(self) -> None:
        assert _opens_on_an_unrendered_frame(_Recorder(3), _state(), {"wrist": _BLACK, "front": _LIT}) is False

    def test_a_camera_that_really_sees_black_is_recorded_after_the_budget(self) -> None:
        state = _state()
        skipped = 0
        while _opens_on_an_unrendered_frame(_Recorder(0), state, {"wrist": _BLACK, "front": _LIT}):
            skipped += 1
            assert skipped <= _MAX_UNRENDERED_SKIPS
        assert skipped == _MAX_UNRENDERED_SKIPS

    def test_a_recording_without_cameras_is_unaffected(self) -> None:
        assert _opens_on_an_unrendered_frame(_Recorder(0), {"recording_cameras": []}, {}) is False
