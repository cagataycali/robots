"""A camera warming up is not an error, so it does not log one.

``add_camera`` steps the world until the new RTX render product yields a
frame. The not-ready answer on those steps - ``get_rgba()`` returning a 0-D
buffer - is the very condition the loop waits out, but each attempt went
through the public ``render`` and logged::

    [Error] Failed to render camera 'front': camera 'front' returned a malformed
    RGB buffer (shape ()); the RTX render product likely hasn't accumulated ...

twice per camera on every healthy Isaac Sim 6.x run (measured on 6.1.0, L40S,
a live GPU probe), and the very next ``render`` succeeded. An
ERROR on the success path teaches operators to ignore the one that matters.

Outside warm-up the same failure is still an ERROR: a caller's own ``render``
of a camera that never accumulates is a real fault.
"""

from __future__ import annotations

import logging
import threading
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import IsaacConfig, IsaacSimulation  # noqa: E402

_LOGGER = "strands_robots.simulation.isaac.simulation"


class _Handle:
    """An RTX camera that needs ``not_ready`` reads before it yields a frame."""

    def __init__(self, not_ready: int) -> None:
        self.not_ready = not_ready

    def get_rgba(self) -> np.ndarray:
        if self.not_ready > 0:
            self.not_ready -= 1
            return np.array(0.0)  # the 0-D buffer 6.x returns
        return np.full((4, 6, 4), 128, dtype=np.uint8)

    def get_depth(self) -> np.ndarray:
        return np.ones((4, 6), dtype=np.float32)


def _engine(handle: _Handle) -> Any:
    engine: Any = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(headless=True, render_mode="rtx_realtime")
    engine._world = types.SimpleNamespace(step=lambda render=False: None)
    engine._world_created = True
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._cameras = {
        "front": types.SimpleNamespace(handle=handle, width=6, height=4, prim_path="/World/Cameras/front")
    }
    engine._objects = {}
    engine._robots = {}
    return engine


class TestWarmup:
    def test_the_not_ready_reads_log_no_error(self, caplog) -> None:
        engine = _engine(_Handle(not_ready=2))
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            assert engine._warmup_camera("front", n_steps=5) is True

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == [], [r.getMessage() for r in errors]
        # Still observable at DEBUG for whoever is chasing a slow product.
        assert any("malformed RGB buffer" in r.getMessage() for r in caplog.records)

    def test_render_after_warmup_errors_again(self, caplog) -> None:
        engine = _engine(_Handle(not_ready=0))
        assert engine._warmup_camera("front", n_steps=1) is True
        engine._cameras["front"].handle.not_ready = 1
        with caplog.at_level(logging.ERROR, logger=_LOGGER):
            result = engine.render(camera_name="front")

        assert result["status"] == "error"
        assert any("malformed RGB buffer" in r.getMessage() for r in caplog.records)


class TestOutsideWarmup:
    def test_a_callers_render_of_a_not_ready_camera_is_an_error(self, caplog) -> None:
        engine = _engine(_Handle(not_ready=1))
        with caplog.at_level(logging.ERROR, logger=_LOGGER):
            result = engine.render(camera_name="front")

        assert result["status"] == "error"
        assert any(r.levelno == logging.ERROR and "malformed RGB buffer" in r.getMessage() for r in caplog.records)
