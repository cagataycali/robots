"""Regression tests: a runtime physics change survives the next scene mutation.

``set_timestep`` / ``set_gravity`` wrote ``model.opt`` (immediate effect) and
``world.timestep`` / ``world.gravity`` (bookkeeping), but NOT the live ``MjSpec``.
Every scene mutation recompiles from that spec via
``_recompile_preserving_state``, and the spec still carried whatever
``SpecBuilder.build`` baked in at ``create_world`` - so the setting was silently
reverted:

    set_timestep(0.001); set_gravity([0, 0, -3.0])   -> opt = 0.001 / -3.0
    add_object(...)                                  -> opt = 0.002 / -9.81

No error, no warning; a rollout that lowered gravity for a manipulation task
quietly ran the rest of the episode at -9.81 as soon as anything was added to the
scene. ``world.*`` still reported the requested values, so even an introspecting
caller could not see the divergence.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_TIMESTEP = 0.001
_GRAVITY_Z = -1.5


@pytest.fixture
def sim():
    s = Simulation(tool_name="option_persistence", mesh=False)
    s.create_world()
    assert s.set_timestep(timestep=_TIMESTEP)["status"] == "success"
    assert s.set_gravity(gravity=[0, 0, _GRAVITY_Z])["status"] == "success"
    yield s
    s.destroy()


def _assert_options_held(sim) -> None:
    opt = sim.mj_model.opt
    assert opt.timestep == pytest.approx(_TIMESTEP)
    assert float(opt.gravity[2]) == pytest.approx(_GRAVITY_Z)


def test_setters_take_effect_immediately(sim) -> None:
    _assert_options_held(sim)


def test_spec_is_updated_not_just_the_model(sim) -> None:
    """The spec is the recompile source of truth, so it must carry the values."""
    spec = sim._world._backend_state["spec"]
    assert spec.option.timestep == pytest.approx(_TIMESTEP)
    assert float(spec.option.gravity[2]) == pytest.approx(_GRAVITY_Z)


def test_survives_add_object(sim) -> None:
    assert sim.add_object(name="c", shape="box", size=[0.03] * 3, position=[0.3, 0, 0.05])["status"] == "success"
    _assert_options_held(sim)


def test_survives_add_robot(sim) -> None:
    assert sim.add_robot(name="panda")["status"] == "success"
    _assert_options_held(sim)


def test_survives_add_camera(sim) -> None:
    assert sim.add_camera(name="cam", position=[1, 1, 1], target=[0, 0, 0])["status"] == "success"
    _assert_options_held(sim)


def test_survives_remove_object(sim) -> None:
    sim.add_object(name="c", shape="box", size=[0.03] * 3, position=[0.3, 0, 0.05])
    assert sim.remove_object(name="c")["status"] == "success"
    _assert_options_held(sim)


def test_survives_reset(sim) -> None:
    assert sim.reset()["status"] == "success"
    _assert_options_held(sim)


def test_scalar_gravity_form_also_persists(sim) -> None:
    assert sim.set_gravity(gravity=-4.0)["status"] == "success"
    sim.add_object(name="c", shape="box", size=[0.03] * 3, position=[0.3, 0, 0.05])
    assert float(sim.mj_model.opt.gravity[2]) == pytest.approx(-4.0)


def test_lowered_gravity_actually_changes_the_fall(sim) -> None:
    """End-to-end: the physics must reflect the setting after a mutation."""
    sim.add_object(name="cube", shape="box", size=[0.03] * 3, position=[0, 0, 2.0], mass=0.2)
    model, data = sim.mj_model, sim.mj_data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    z0 = float(data.xpos[body][2])
    for _ in range(500):
        mujoco.mj_step(model, data)
    dropped = z0 - float(data.xpos[body][2])
    # 500 steps * 1 ms at 1.5 m/s^2 ~= 0.19 m; at the default 9.81 it would be
    # far more (and at the default 2 ms timestep, ~4x further still).
    assert dropped < 0.35, f"fell {dropped:.3f} m - gravity/timestep reverted to defaults"
