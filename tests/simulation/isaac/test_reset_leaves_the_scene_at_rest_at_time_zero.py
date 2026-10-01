"""``reset()`` leaves the Isaac scene at rest, and the clock starts at zero.

Measured on one L40S (Isaac Sim 6.1, ``create_world(timestep=1/500)``, so100 and
a dynamic cube), right after ``reset()``:

* the largest so100 joint velocity was 0.026 rad/s and the cube fell at
  0.039 m/s (MuJoCo: exact zeros) - ``World.reset()`` integrates warm-up steps
  from the authored pose, so every episode's first observation, and the first
  frame of every recorded episode, was already moving;
* ``step(10)`` reported ``sim_time=0.024`` and 500 steps 1.004 s: the World's
  clock was read as-is, and it had already advanced during the warm-up.

After: 0.0, 0.0, 0.020 and 1.000.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import IsaacConfig, _ObjectState, _RobotState  # noqa: E402
from tests.simulation._isaac_engine import isaac_engine  # noqa: E402


class _Articulation:
    def __init__(self) -> None:
        self.writes: dict[str, Any] = {}

    def set_joint_velocities(self, v: Any) -> None:
        self.writes["joint"] = np.asarray(v)

    def set_linear_velocity(self, v: Any) -> None:
        self.writes["lin"] = np.asarray(v)

    def set_angular_velocity(self, v: Any) -> None:
        self.writes["ang"] = np.asarray(v)


class _Body:
    def __init__(self) -> None:
        self.writes: dict[str, Any] = {}

    def set_linear_velocity(self, v: Any) -> None:
        self.writes["lin"] = np.asarray(v)

    def set_angular_velocity(self, v: Any) -> None:
        self.writes["ang"] = np.asarray(v)


def _engine() -> Any:
    engine: Any = isaac_engine(IsaacConfig(headless=True))
    robot = _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=["a", "b", "c"])
    robot.articulation = _Articulation()
    engine._robots = {"arm": robot}
    engine._objects = {
        "cube": _ObjectState(
            name="cube", prim_path="/World/Objects/cube", shape="box", is_static=False, handle=_Body()
        ),
        "wall": _ObjectState(name="wall", prim_path="/World/Objects/wall", shape="box", is_static=True, handle=_Body()),
    }
    return engine


class TestTheSceneIsAtRest:
    def test_every_robot_and_dynamic_body_is_zeroed(self) -> None:
        engine = _engine()
        engine._settle_after_reset()
        writes = engine._robots["arm"].articulation.writes
        assert writes["joint"].tolist() == [0.0, 0.0, 0.0]
        assert writes["lin"].tolist() == [0.0] * 3 and writes["ang"].tolist() == [0.0] * 3
        cube = engine._objects["cube"].handle.writes
        assert cube["lin"].tolist() == [0.0] * 3 and cube["ang"].tolist() == [0.0] * 3
        assert engine._objects["wall"].handle.writes == {}

    def test_a_handle_that_refuses_the_write_does_not_fail_the_reset(self) -> None:
        engine = _engine()

        def _boom(v: Any) -> None:
            raise RuntimeError("torn down")

        engine._robots["arm"].articulation.set_joint_velocities = _boom
        engine._settle_after_reset()  # no raise
        assert "lin" in engine._robots["arm"].articulation.writes


class TestTheClockStartsAtZero:
    def test_the_world_time_at_the_rewind_is_the_origin(self) -> None:
        engine = _engine()
        engine._world = types.SimpleNamespace(current_time=0.004)  # after World.reset()'s warm-up
        engine._rewind_clock()
        assert engine._sim_time == 0.0 and engine._world_clock() == pytest.approx(0.0)
        engine._world.current_time = 0.024  # ten 2 ms steps later
        assert engine._world_clock() == pytest.approx(0.020)

    def test_a_world_without_a_clock_counts_physics_dt(self) -> None:
        engine = _engine()
        engine._world = types.SimpleNamespace()
        engine._rewind_clock()
        assert engine._world_clock() == pytest.approx(engine._config.physics_dt)
