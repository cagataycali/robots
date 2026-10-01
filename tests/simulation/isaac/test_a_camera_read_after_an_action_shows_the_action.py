"""A camera read after an action shows the action, not the moment before it.

Measured on one L40S (Isaac Sim 6.1, go2, one RTX camera): ``send_action``
folded the robot flat and ``render`` - twice - still showed it standing. The
render product delivers a tick behind; one render-only tick was not enough, two
were. With one camera ``get_observation`` refreshed nothing at all.

Unit-level: the renderer tick is counted.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    _RENDER_LAG_TICKS,
    IsaacConfig,
    _CameraState,
    _RobotState,
)
from tests.simulation._isaac_engine import isaac_engine  # noqa: E402


class _Handle:
    def get_rgba(self) -> np.ndarray:
        return np.full((4, 6, 4), 90, dtype=np.uint8)

    def get_depth(self) -> np.ndarray:
        return np.ones((4, 6), dtype=np.float32)


def _engine() -> Any:
    engine: Any = isaac_engine(IsaacConfig(headless=True, render_mode="rtx_realtime"))
    engine.ticks = 0

    def _update() -> None:
        engine.ticks += 1

    engine._app = types.SimpleNamespace(update=_update)
    engine._world = types.SimpleNamespace(step=lambda render=False: None)
    engine._world_created = True
    cam = _CameraState(name="side", prim_path="/World/Cameras/side", width=6, height=4)
    cam.handle = _Handle()
    engine._cameras = {"side": cam}
    return engine


class TestTheRendererCatchesUpOncePerStep:
    def test_two_ticks_after_physics_moved(self) -> None:
        engine = _engine()
        engine._step_count = 40
        engine._refresh_if_physics_moved()
        assert engine.ticks == _RENDER_LAG_TICKS == 2

    def test_repeated_reads_between_steps_cost_nothing(self) -> None:
        engine = _engine()
        engine._step_count = 40
        engine._refresh_if_physics_moved()
        engine._refresh_if_physics_moved()
        assert engine.ticks == 2
        engine._step_count = 41
        engine._refresh_if_physics_moved()
        assert engine.ticks == 4

    def test_render_refreshes_before_reading(self) -> None:
        engine = _engine()
        engine._step_count = 3
        assert engine.render(camera_name="side")["status"] == "success"
        assert engine.ticks == 2

    def test_get_observation_refreshes_a_single_camera(self) -> None:
        engine = _engine()
        engine._step_count = 3
        robot = _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=["j0"])
        robot.articulation = types.SimpleNamespace(
            get_joint_positions=lambda: np.zeros(1), get_joint_velocities=lambda: np.zeros(1)
        )
        engine._robots = {"arm": robot}
        obs = engine.get_observation("arm")
        assert "side" in obs and engine.ticks == 2

    def test_no_renderer_is_not_an_error(self) -> None:
        engine = _engine()
        engine._app = None
        engine._world = types.SimpleNamespace()
        engine._step_count = 1
        engine._refresh_if_physics_moved()  # no raise


class TestTheMarkerDoesNotSurviveARewind:
    """``_step_count`` is not monotonic: create_world, reset and destroy set it back to 0.

    A marker keyed on the step alone survives the rewind with a pre-rewind
    index, and the first camera read in the new world at the coinciding index
    skips the refresh: the never-warmed RTX product serves a blank or stale
    buffer under status="success". Keyed like ``_contact_cache``, on
    ``(_contact_epoch, _step_count)``, so ``_rewind_clock`` invalidates it.
    """

    def test_a_read_at_the_same_index_after_a_rewind_refreshes(self) -> None:
        engine = _engine()
        engine._step_count = 0
        engine._refresh_if_physics_moved()
        assert engine.ticks == 2
        engine._rewind_clock()  # what destroy() + create_world() and reset() call
        engine._refresh_if_physics_moved()
        assert engine.ticks == 4, "the first read in the new world at step 0 skipped the refresh"

    def test_without_a_rewind_the_same_index_is_still_free(self) -> None:
        engine = _engine()
        engine._step_count = 7
        engine._refresh_if_physics_moved()
        engine._refresh_if_physics_moved()
        assert engine.ticks == 2

    def test_reset_needs_no_manual_clear(self) -> None:
        """The rewind is the single owner of the invalidation; reset() carries no second line."""
        import inspect

        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        assert "_rendered_at_step = None" not in inspect.getsource(IsaacSimulation.reset)
