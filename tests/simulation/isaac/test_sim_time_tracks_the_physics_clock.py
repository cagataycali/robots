"""The Isaac backend's reported sim time is the physics time it drives.

Three shipped defects made ``_sim_time`` / ``physics_timestep()`` disagree with the
physics ``World`` integrated (measured on Isaac Sim 6.0.1 against ``World.current_time``):

* a rendering ``step()`` ran Kit's ``app.update()`` inside ``World.step(render=True)``,
  which integrates a whole ``rendering_dt`` (four ``physics_dt`` substeps at the
  defaults) while the clock was credited one constant ``physics_dt`` - so a rendering
  scene under-reported its elapsed time ~4x;
* a ``create_world(timestep=)`` override is honoured by ``World`` but never written to
  ``config.physics_dt``, so ``physics_timestep()`` and the accumulator over-reported it;
* the idle render pump advanced physics every tick while touching neither counter.

The fix reads the clock back from ``World.current_time`` after each tick, steps physics
exactly once per ``step()`` (refreshing the frame with a separate ``World.render()`` so
one ``step()`` is one ``physics_dt`` in every render mode), and reports the dt the World
integrates. These pins fail on the pre-fix accumulate-a-constant code.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from tests.simulation._isaac_engine import isaac_engine


class _ClockWorld:
    """A World whose physics clock advances by ``world_dt`` per physics tick.

    ``step(render=...)`` is the physics tick; it records the ``render`` argument and
    advances ``current_time`` by ``world_dt`` (the dt the World integrates, which a
    ``timestep=`` override can make differ from ``config.physics_dt``). ``render()`` is
    the frame refresh and advances no time - the real render-only path.
    """

    def __init__(self, world_dt: float) -> None:
        self._world_dt = world_dt
        self.current_time = 0.0
        self.step_render_args: list[bool] = []
        self.render_calls = 0

    def step(self, render: bool = False) -> None:
        self.step_render_args.append(render)
        self.current_time += self._world_dt

    def render(self) -> None:
        self.render_calls += 1

    def get_physics_dt(self) -> float:
        return self._world_dt


def _engine(*, render_mode: str, config_dt: float, world_dt: float) -> Any:
    engine = isaac_engine()
    engine._config = IsaacConfig(render_mode=render_mode, physics_dt=config_dt)
    engine._world = _ClockWorld(world_dt)
    engine._world_created = True
    engine._STEPS_PER_BATCH = IsaacSimulation._STEPS_PER_BATCH
    return engine


class TestSimTimeReadsTheWorldClock:
    def test_headless_timestep_override_is_not_over_reported(self) -> None:
        """The headless timestep=0.005 row: 30 steps integrate 0.150 s, not the
        0.250 s an accumulated default 1/120 credited."""
        engine = _engine(render_mode="headless", config_dt=1 / 120, world_dt=0.005)

        engine.step(30)

        assert engine._sim_time == pytest.approx(0.150)
        assert engine._sim_time == pytest.approx(engine._world.current_time)

    def test_a_rendering_step_advances_one_physics_tick_not_a_render_dt(self) -> None:
        """A rendering step must step physics ONCE (render=False) and refresh the
        frame separately, so it advances one physics_dt rather than a whole
        rendering_dt of physics folded into World.step(render=True)."""
        engine = _engine(render_mode="rtx_realtime", config_dt=1 / 120, world_dt=1 / 120)

        engine.step(3)

        assert engine._world.step_render_args == [False, False, False]
        assert engine._world.render_calls == 3
        assert engine._sim_time == pytest.approx(3 / 120)

    def test_send_action_reads_the_world_clock(self) -> None:
        engine = _engine(render_mode="headless", config_dt=1 / 120, world_dt=0.005)
        from strands_robots.simulation.isaac.simulation import _RobotState  # noqa: PLC0415

        engine._robots["arm"] = _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=["j0"])

        engine.send_action({"j0": 0.0}, robot_name="arm", n_substeps=4)

        assert engine._sim_time == pytest.approx(0.020)
        assert engine._sim_time == pytest.approx(engine._world.current_time)


class TestPhysicsTimestepReportsWhatTheWorldIntegrates:
    def test_it_reports_the_worlds_dt_over_a_stale_config(self) -> None:
        """create_world(timestep=0.005) is honoured by World but never written to
        config.physics_dt; physics_timestep must report the integrated 0.005."""
        engine = _engine(render_mode="headless", config_dt=1 / 120, world_dt=0.005)

        assert engine.physics_timestep() == pytest.approx(0.005)

    def test_it_falls_back_to_config_without_a_world(self) -> None:
        engine = IsaacSimulation(physics_dt=0.005)

        assert engine.physics_timestep() == pytest.approx(0.005)
