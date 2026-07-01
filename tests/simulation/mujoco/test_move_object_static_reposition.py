"""Regression tests: ``move_object`` must actually move STATIC objects.

A static object (``is_static=True``) is welded to the worldbody with no
freejoint, so it has no ``data.qpos`` slice. The old ``move_object`` only wrote
``data.qpos`` when a ``<name>_joint`` freejoint existed and otherwise fell
through to ``return {"status": "success", ...}`` - reporting a successful move
while the body never budged and its stored ``SimObject.position`` stayed stale.
That is the "success contract, no physical effect" failure mode the project
forbids ("never warn-and-continue; no silent defaults").

These pin the corrected contract:

* moving a static object repositions the compiled body AND updates the stored
  ``SimObject`` pose (the previously-silent no-op);
* orientation-only moves of a static object are applied;
* the state of other (dynamic) objects survives the static-body recompile;
* dynamic objects still move through the cheap ``data.qpos`` path;
* an unknown object name is a loud error.
"""

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_move_static_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    yield s
    s.cleanup()


def _body_xpos(world, name):
    bid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"body {name!r} not found in compiled model"
    return [float(x) for x in world._data.xpos[bid]]


def _body_xquat(world, name):
    bid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"body {name!r} not found in compiled model"
    return [float(x) for x in world._data.xquat[bid]]


def test_move_static_object_repositions_body(sim):
    """Moving a static fixture actually relocates the compiled body."""
    sim.add_object("wall", shape="box", size=[0.2, 0.02, 0.1], position=[0.3, 0.0, 0.1], is_static=True)
    assert _body_xpos(sim._world, "wall") == pytest.approx([0.3, 0.0, 0.1])

    result = sim.move_object("wall", position=[0.5, 0.1, 0.2])
    assert result["status"] == "success", result

    # The previously-silent no-op: the body must have actually moved...
    assert _body_xpos(sim._world, "wall") == pytest.approx([0.5, 0.1, 0.2])
    # ...and the stored SimObject pose must reflect the new position.
    assert list(sim._world.objects["wall"].position) == pytest.approx([0.5, 0.1, 0.2])


def test_move_static_object_orientation_only(sim):
    """An orientation-only move of a static object is applied."""
    sim.add_object("wall", shape="box", size=[0.2, 0.02, 0.1], position=[0.3, 0.0, 0.1], is_static=True)
    quat = [0.9238795, 0.0, 0.0, 0.3826834]  # 45 deg about +z

    result = sim.move_object("wall", orientation=quat)
    assert result["status"] == "success", result

    assert _body_xquat(sim._world, "wall") == pytest.approx(quat, abs=1e-4)
    # position untouched when only orientation is supplied
    assert _body_xpos(sim._world, "wall") == pytest.approx([0.3, 0.0, 0.1])
    assert list(sim._world.objects["wall"].orientation) == pytest.approx(quat)


def test_static_move_preserves_dynamic_object_state(sim):
    """Repositioning a static body must not reset other objects' state.

    The static path recompiles the scene; a dynamic object mid-trajectory must
    keep its qpos (not snap back to its spawn pose).
    """
    sim.add_object("wall", shape="box", size=[0.2, 0.02, 0.1], position=[0.3, 0.0, 0.1], is_static=True)
    sim.add_object("cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.0, 0.0, 0.30], is_static=False)
    sim.step(60)  # cube falls away from its spawn height
    cube_before = _body_xpos(sim._world, "cube")
    assert cube_before[2] < 0.27  # it has actually left the spawn height

    result = sim.move_object("wall", position=[0.5, 0.1, 0.2])
    assert result["status"] == "success", result

    cube_after = _body_xpos(sim._world, "cube")
    # cube state preserved across the recompile (not reset to spawn 0.30)
    assert cube_after == pytest.approx(cube_before, abs=5e-3)


def test_move_dynamic_object_uses_qpos_path(sim):
    """Dynamic objects still move (via the cheap data.qpos path)."""
    sim.add_object("cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.0, 0.0, 0.05], is_static=False)
    result = sim.move_object("cube", position=[0.2, 0.2, 0.05])
    assert result["status"] == "success", result
    assert _body_xpos(sim._world, "cube") == pytest.approx([0.2, 0.2, 0.05])
    assert list(sim._world.objects["cube"].position) == pytest.approx([0.2, 0.2, 0.05])


def test_move_unknown_object_errors(sim):
    """An unknown object name is a loud error, not a false success."""
    result = sim.move_object("ghost", position=[0.0, 0.0, 0.0])
    assert result["status"] == "error", result
    assert "ghost" in result["content"][0]["text"]
