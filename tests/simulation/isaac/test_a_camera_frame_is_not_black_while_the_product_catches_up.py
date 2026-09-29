"""A camera frame is not black while its RTX render product catches up.

Measured on one L40S (Isaac Sim 6.1): an so101 wrist camera 0.26 m above a red
cube recorded its first two rollout frames as all zeros (std 0.0), and frame 2 on
was real, on every recording (MuJoCo: never). The probe frame before
``start_recording`` was lit, so rendering first did not help: the product hands
back a correctly shaped buffer of zeros for a tick or two, and every caller took
it as the picture - a policy trained on the dataset saw black inputs at every
episode start.

Unit-level: the handle is a stand-in whose ``get_rgba`` returns zeros for a set
number of reads, and the render-only tick is counted.
"""

from __future__ import annotations

import threading
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    _BLANK_FRAME_RETRIES,
    IsaacConfig,
    IsaacSimulation,
    _CameraState,
)

_LIT = np.full((4, 6, 4), 90, dtype=np.uint8)
_BLACK = np.zeros((4, 6, 4), dtype=np.uint8)


class _Handle:
    """Zeros for ``black`` reads, then a lit frame (or zeros forever when ``black`` is None)."""

    def __init__(self, black: int | None) -> None:
        self.black = black
        self.reads = 0

    def get_rgba(self) -> np.ndarray:
        self.reads += 1
        if self.black is None:
            return _BLACK.copy()
        if self.black > 0:
            self.black -= 1
            return _BLACK.copy()
        return _LIT.copy()

    def get_depth(self) -> np.ndarray:
        return np.ones((4, 6), dtype=np.float32)


def _engine(handle: _Handle) -> Any:
    engine: Any = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(headless=True, render_mode="rtx_realtime")
    engine.ticks = 0

    def _update() -> None:
        engine.ticks += 1

    engine._app = types.SimpleNamespace(update=_update)
    engine._world = types.SimpleNamespace(step=lambda render=False: None)
    engine._world_created = True
    cam = _CameraState(name="wrist", prim_path="/World/Cameras/wrist", width=6, height=4)
    cam.handle = handle
    engine._cameras = {"wrist": cam}
    return engine


class TestABlackFrameIsReRendered:
    @pytest.mark.parametrize("black", [1, 2, _BLANK_FRAME_RETRIES])
    def test_zeros_for_a_few_ticks_come_back_lit(self, black: int) -> None:
        engine = _engine(_Handle(black))
        arr = np.asarray(engine._read_camera_rgb(engine._cameras["wrist"]))
        assert arr.max() == 90
        assert engine.ticks == black  # render-only ticks, one per black read

    def test_a_lit_frame_costs_no_tick(self) -> None:
        engine = _engine(_Handle(0))
        engine._read_camera_rgb(engine._cameras["wrist"])
        assert engine.ticks == 0

    def test_render_returns_the_lit_frame(self) -> None:
        engine = _engine(_Handle(2))
        engine._robots, engine._objects = {}, {}
        result = engine.render(camera_name="wrist")
        assert result["status"] == "success"
        assert engine._cameras["wrist"].handle.black == 0


class TestACameraThatReallySeesBlackPaysOnce:
    def test_after_one_budget_black_is_believed(self) -> None:
        engine = _engine(_Handle(None))
        cam = engine._cameras["wrist"]
        assert np.asarray(engine._read_camera_rgb(cam)).max() == 0
        assert engine.ticks == _BLANK_FRAME_RETRIES and cam.dark is True
        engine._read_camera_rgb(cam)
        assert engine.ticks == _BLANK_FRAME_RETRIES  # no second budget

    def test_colour_clears_dark(self) -> None:
        engine = _engine(_Handle(0))
        cam = engine._cameras["wrist"]
        cam.dark = True
        engine._read_camera_rgb(cam)
        assert cam.dark is False

    def test_no_renderer_ends_the_wait(self) -> None:
        engine = _engine(_Handle(None))
        engine._app = None
        engine._world = types.SimpleNamespace()  # no step: nothing to tick
        assert np.asarray(engine._read_camera_rgb(engine._cameras["wrist"])).max() == 0
