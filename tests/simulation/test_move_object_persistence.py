"""Regression tests: ``move_object`` on a dynamic object actually sticks.

A dynamic object carries its pose in its freejoint's ``qpos``, and ``move_object``
took the cheap path of writing that slice directly (no recompile). But ``qpos`` is
RE-DERIVED from compile-time state in two common situations:

* ``mj_resetData`` restores ``qpos`` from ``model.qpos0`` - so every ``reset()``
  undid the move.
* a recompile that shifts the qpos layout seeds a new freejoint from its body's
  ``pos``/``quat`` - and ``add_robot`` both shifts the layout (it inserts the new
  robot's bodies earlier in the tree) and then deliberately calls ``mj_resetData``.

Measured:

    move_object("dyn", [0.6, 0.1, 0.5])   -> xpos [0.6, 0.1, 0.5]
    reset()                               -> xpos [0.4, 0.0, 0.3]   spawn pose
    add_robot("b")                        -> xpos [0.4, 0.0, 0.3]
    list_objects()                        -> "dyn: box at [0.6, 0.1, 0.5]"

Both calls reported success, and the agent's own inventory kept claiming the new
pose - so an eval loop that arranged a scene and reset between episodes silently
ran every episode from the ORIGINAL layout.

The pose is now mirrored onto ``qpos0``, the body rest pose, and the spec, so all
three re-derivation paths agree. Static objects already went through
``reposition_body_in_scene`` (a spec edit + recompile) and are unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_SPAWN = [0.4, 0.0, 0.3]
_MOVED = [0.6, 0.1, 0.5]
_QUAT = [0.7071, 0.0, 0.7071, 0.0]


def _xpos(sim, name: str = "dyn") -> np.ndarray:
    model, data = sim.mj_model, sim.mj_data
    return np.asarray(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]).copy()


@pytest.fixture
def sim():
    s = Simulation(tool_name="move_object_persistence", mesh=False)
    s.create_world()
    assert s.add_robot(name="a", data_config="panda")["status"] == "success"
    assert s.add_object(name="dyn", shape="box", size=[0.05] * 3, position=_SPAWN, mass=0.2)["status"] == "success"
    yield s
    s.destroy()


def test_move_takes_effect(sim) -> None:
    assert sim.move_object(name="dyn", position=_MOVED)["status"] == "success"
    assert list(_xpos(sim)) == pytest.approx(_MOVED)


def test_move_survives_reset(sim) -> None:
    """The core defect: reset restored qpos from the stale qpos0."""
    assert sim.move_object(name="dyn", position=_MOVED)["status"] == "success"
    assert sim.reset()["status"] == "success"
    assert list(_xpos(sim)) == pytest.approx(_MOVED), "reset resurrected the spawn pose"


def test_move_survives_reset_after_settling(sim) -> None:
    """An episode loop settles, then resets; it must restart from the MOVED pose."""
    assert sim.move_object(name="dyn", position=_MOVED)["status"] == "success"
    assert sim.step(n_steps=1500)["status"] == "success"
    assert sim.reset()["status"] == "success"
    assert list(_xpos(sim)) == pytest.approx(_MOVED)


def test_move_survives_add_robot(sim) -> None:
    """add_robot shifts the qpos layout AND resets to a clean state."""
    assert sim.move_object(name="dyn", position=_MOVED)["status"] == "success"
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    assert list(_xpos(sim)) == pytest.approx(_MOVED)


def test_move_survives_the_full_rebuild_path(sim) -> None:
    """``remove_robot`` rebuilds declaratively from ``world.objects``."""
    assert sim.move_object(name="dyn", position=_MOVED)["status"] == "success"
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    assert sim.remove_robot(name="b")["status"] == "success"
    assert list(_xpos(sim)) == pytest.approx(_MOVED)


def test_orientation_survives_reset(sim) -> None:
    assert sim.move_object(name="dyn", orientation=_QUAT)["status"] == "success"
    assert sim.reset()["status"] == "success"
    model, data = sim.mj_model, sim.mj_data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dyn")
    assert [float(v) for v in data.xquat[body]] == pytest.approx(_QUAT, abs=1e-4)


def test_the_inventory_agrees_with_the_physics(sim) -> None:
    """``list_objects`` reported the new pose even when the physics did not."""
    assert sim.move_object(name="dyn", position=_MOVED)["status"] == "success"
    assert sim.reset()["status"] == "success"
    assert list(sim._world.objects["dyn"].position) == pytest.approx(_MOVED)
    assert list(_xpos(sim)) == pytest.approx(_MOVED)


def test_qpos0_carries_the_new_rest_pose(sim) -> None:
    """The direct invariant: mj_resetData restores qpos from qpos0."""
    assert sim.move_object(name="dyn", position=_MOVED)["status"] == "success"
    model = sim.mj_model
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dyn_joint")
    adr = int(model.jnt_qposadr[joint])
    assert [float(v) for v in model.qpos0[adr : adr + 3]] == pytest.approx(_MOVED)


def test_a_settled_object_is_not_teleported_by_a_move_of_another_object(sim) -> None:
    """Only the moved object's rest pose changes."""
    assert (
        sim.add_object(name="other", shape="box", size=[0.04] * 3, position=[1.0, 1.0, 0.3], mass=0.1)["status"]
        == "success"
    )
    assert sim.move_object(name="dyn", position=_MOVED)["status"] == "success"
    assert sim.reset()["status"] == "success"
    assert list(_xpos(sim, "other")) == pytest.approx([1.0, 1.0, 0.3])


def test_static_object_move_still_survives_reset(sim) -> None:
    """The static path (spec edit + recompile) must keep working."""
    assert (
        sim.add_object(name="stat", shape="box", size=[0.05] * 3, position=[0.4, 0.4, 0.3], is_static=True)["status"]
        == "success"
    )
    assert sim.move_object(name="stat", position=_MOVED)["status"] == "success"
    assert sim.reset()["status"] == "success"
    assert list(_xpos(sim, "stat")) == pytest.approx(_MOVED)


def test_repeated_moves_do_not_compound(sim) -> None:
    """Each move sets an absolute pose, not a delta."""
    for target in ([0.5, 0.0, 0.4], [0.55, 0.05, 0.45], _MOVED):
        assert sim.move_object(name="dyn", position=target)["status"] == "success"
        assert sim.reset()["status"] == "success"
        assert list(_xpos(sim)) == pytest.approx(target)
