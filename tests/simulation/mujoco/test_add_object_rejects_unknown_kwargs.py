"""Regression tests: ``add_object`` must not silently swallow unknown kwargs.

The MuJoCo/Newton ``add_object`` signatures are identical and carry no
backend-specific parameters, yet both previously declared ``**kwargs`` that
were documented as "currently ignored". A caller reaching for MuJoCo's native
``rgba=`` (or making a typo such as ``colour=`` / ``radius=``) had the argument
silently dropped while the call still returned ``{"status": "success"}`` - the
object was created with the default grey ``color`` and no signal was given.
This is the "success contract, wrong physical effect" failure mode the project
explicitly forbids ("never warn-and-continue; no silent defaults").

These pin the corrected contract:

* the agent-dispatch path (``sim(action="add_object", ...)``) reports a
  structured error that names the offending parameter and lists the valid
  ones (so ``rgba`` -> use ``color``);
* a direct Python call with an unknown keyword fails loudly (``TypeError``)
  instead of returning a misleading success;
* the supported ``color=`` parameter actually reaches ``geom_rgba``.
"""

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_unknown_kwargs_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    yield s
    s.cleanup()


def _geom_rgba(world, geom_name):
    model = world._model
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
    assert gid >= 0, f"geom {geom_name!r} not found in compiled model"
    return list(model.geom_rgba[gid])


def test_router_rejects_rgba_and_lists_color(sim):
    """The agent-dispatch path names the bad kwarg and points at ``color``."""
    result = sim(
        action="add_object",
        name="cube",
        shape="box",
        size=[0.05, 0.05, 0.05],
        position=[0.2, 0.0, 0.05],
        rgba=[1, 0, 0, 1],
    )
    assert result["status"] == "error", result
    text = result["content"][0]["text"]
    assert "rgba" in text
    assert "color" in text  # the valid parameter to use instead
    # the rejected call must not have created a (grey) object
    assert "cube" not in sim._world.objects


def test_router_rejects_typo_kwarg(sim):
    """A plain typo (``colour``) is rejected, not silently swallowed."""
    result = sim(action="add_object", name="c2", shape="box", colour=[0, 0, 1, 1])
    assert result["status"] == "error", result
    assert "colour" in result["content"][0]["text"]
    assert "c2" not in sim._world.objects


def test_direct_call_raises_typeerror_on_unknown_kwarg(sim):
    """A direct library call fails loudly instead of returning success."""
    with pytest.raises(TypeError):
        sim.add_object("cube", shape="box", size=[0.05, 0.05, 0.05], rgba=[1, 0, 0, 1])


def test_color_parameter_applies_to_geom_rgba(sim):
    """The supported ``color=`` argument actually colours the geom."""
    result = sim.add_object("red", shape="box", size=[0.05, 0.05, 0.05], color=[1, 0, 0, 1])
    assert result["status"] == "success", result
    rgba = _geom_rgba(sim._world, "red_geom")
    assert rgba == pytest.approx([1.0, 0.0, 0.0, 1.0])
