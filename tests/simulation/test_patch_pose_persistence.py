"""Regression tests: a ``patch_scene_mjcf`` pose edit survives a full rebuild.

``patch_scene_mjcf`` edits the live ``MjSpec``. The incremental recompile it runs
keeps that edit, but a later FULL rebuild - ``eject_robot_from_scene``, triggered
by removing ANY robot - reconstructs object bodies from ``world.objects`` and so
silently reverted it:

    patch set_body_pos  cube -> [1.2, 0.3, 0.5]      applied
    remove_robot("b")   (an unrelated robot)
    cube body_pos       -> [0.4, 0.0, 0.3]           original, patch lost

Same for ``set_body_quat``. A scene an agent had arranged via patches came apart
as soon as any robot was removed, with no error.

Pose patches are now mirrored onto the tracked ``SimObject``, which is what the
declarative rebuild reads.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_POS = [1.2, 0.3, 0.5]
_QUAT = [0.7071, 0.0, 0.7071, 0.0]


@pytest.fixture
def sim():
    s = Simulation(tool_name="patch_pose_persistence", mesh=False)
    s.create_world()
    assert s.add_robot(name="a", data_config="panda")["status"] == "success"
    assert s.add_robot(name="b", data_config="panda", position=[1.5, 0, 0])["status"] == "success"
    assert (
        s.add_object(name="cube", shape="box", size=[0.04] * 3, position=[0.4, 0, 0.3], mass=0.2)["status"] == "success"
    )
    yield s
    s.destroy()


def _body_pose(sim):
    model = sim.mj_model
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    return np.array(model.body_pos[body]), np.array(model.body_quat[body])


def test_patched_position_survives_unrelated_robot_removal(sim) -> None:
    assert sim.patch_scene_mjcf(ops=[{"op": "set_body_pos", "name": "cube", "pos": _POS}])["status"] == "success"
    assert _body_pose(sim)[0] == pytest.approx(_POS)

    assert sim.remove_robot(name="b")["status"] == "success"
    assert _body_pose(sim)[0] == pytest.approx(_POS), "patch reverted on rebuild"


def test_patched_orientation_survives_unrelated_robot_removal(sim) -> None:
    assert sim.patch_scene_mjcf(ops=[{"op": "set_body_quat", "name": "cube", "quat": _QUAT}])["status"] == "success"
    assert _body_pose(sim)[1] == pytest.approx(_QUAT, abs=1e-4)

    assert sim.remove_robot(name="b")["status"] == "success"
    assert _body_pose(sim)[1] == pytest.approx(_QUAT, abs=1e-4)


def test_batched_pose_patch_survives(sim) -> None:
    ops = [
        {"op": "set_body_pos", "name": "cube", "pos": _POS},
        {"op": "set_body_quat", "name": "cube", "quat": _QUAT},
    ]
    assert sim.patch_scene_mjcf(ops=ops)["status"] == "success"
    assert sim.remove_robot(name="b")["status"] == "success"
    pos, quat = _body_pose(sim)
    assert pos == pytest.approx(_POS)
    assert quat == pytest.approx(_QUAT, abs=1e-4)


def test_sim_object_records_the_patched_pose(sim) -> None:
    """The declarative rebuild reads these fields, so they must be in step."""
    sim.patch_scene_mjcf(
        ops=[
            {"op": "set_body_pos", "name": "cube", "pos": _POS},
            {"op": "set_body_quat", "name": "cube", "quat": _QUAT},
        ]
    )
    obj = sim._world.objects["cube"]
    assert list(obj.position) == pytest.approx(_POS)
    assert list(obj.orientation) == pytest.approx(_QUAT)


def test_patch_against_a_robot_link_is_not_mirrored(sim) -> None:
    """Only tracked objects have a SimObject; a robot link must not break the mirror."""
    result = sim.patch_scene_mjcf(ops=[{"op": "set_body_pos", "name": "a/link1", "pos": [0.0, 0.0, 0.4]}])
    assert result["status"] == "success"
    # No SimObject for a robot link, and the scene stays mutable.
    assert "a/link1" not in sim._world.objects
    assert sim.add_object(name="later", shape="box", size=[0.02] * 3, position=[2, 2, 0.05])["status"] == "success"


def test_rejected_patch_leaves_the_object_untouched(sim) -> None:
    """A failed batch must not half-apply the mirror."""
    before = list(sim._world.objects["cube"].position)
    result = sim.patch_scene_mjcf(
        ops=[
            {"op": "set_body_pos", "name": "cube", "pos": _POS},
            {"op": "set_body_pos", "name": "no_such_body", "pos": [0, 0, 0]},
        ]
    )
    assert result["status"] == "error"
    # The spec batch is rejected wholesale, so the mirror must roll back with it -
    # otherwise the next full rebuild would apply a patch the caller was told failed.
    assert _body_pose(sim)[0] == pytest.approx(before)
    assert list(sim._world.objects["cube"].position) == pytest.approx(before)

    # And the rolled-back pose must survive a rebuild (not resurrect the refused one).
    assert sim.remove_robot(name="b")["status"] == "success"
    assert _body_pose(sim)[0] == pytest.approx(before)


def test_patched_pose_moves_a_dynamic_body_immediately(sim) -> None:
    """A pose patch must take effect NOW, not at the next reset.

    A free body's pose lives in its freejoint's ``qpos``, and the batch recompile
    faithfully preserves that qpos - so editing the body's rest pose in the spec
    moved a STATIC body but left a DYNAMIC one exactly where it was:

        patch set_body_pos cube -> [1.2, 0.3, 0.5]   status=success
        get_body_state("cube")  -> pos [0.4, 0.0, 0.3]      unmoved
        list_objects()          -> "cube: box at [1.2, 0.3, 0.5]"
        reset()                 -> pos [1.2, 0.3, 0.5]      moves only now

    so two tool actions disagreed about where the object was, and an agent that
    arranged a scene by patching saw nothing happen.
    """
    assert sim.patch_scene_mjcf(ops=[{"op": "set_body_pos", "name": "cube", "pos": _POS}])["status"] == "success"

    model, data = sim.mj_model, sim.mj_data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    assert [float(v) for v in data.xpos[body]] == pytest.approx(_POS), "dynamic body did not actually move"

    # The two read paths must agree with each other.
    state = sim.get_body_state(body_name="cube")
    assert state["status"] == "success"
    assert "1.2000, 0.3000, 0.5000" in state["content"][0]["text"]


def test_patched_orientation_rotates_a_dynamic_body_immediately(sim) -> None:
    assert sim.patch_scene_mjcf(ops=[{"op": "set_body_quat", "name": "cube", "quat": _QUAT}])["status"] == "success"
    model, data = sim.mj_model, sim.mj_data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    assert [float(v) for v in data.xquat[body]] == pytest.approx(_QUAT, abs=1e-4)


def test_patched_pose_is_at_rest(sim) -> None:
    """A teleport must not retain the body's prior momentum."""
    assert sim.step(n_steps=400)["status"] == "success"  # let it acquire some velocity
    assert sim.patch_scene_mjcf(ops=[{"op": "set_body_pos", "name": "cube", "pos": _POS}])["status"] == "success"
    model, data = sim.mj_model, sim.mj_data
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    dof = int(model.jnt_dofadr[joint])
    assert [float(v) for v in data.qvel[dof : dof + 6]] == pytest.approx([0.0] * 6)


def test_static_body_patch_still_works(sim) -> None:
    """Static bodies have no freejoint; the rest-pose edit alone positions them."""
    assert (
        sim.add_object(name="fixture", shape="box", size=[0.05] * 3, position=[0.4, 0.4, 0.3], is_static=True)["status"]
        == "success"
    )
    assert sim.patch_scene_mjcf(ops=[{"op": "set_body_pos", "name": "fixture", "pos": _POS}])["status"] == "success"
    model, data = sim.mj_model, sim.mj_data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fixture")
    assert [float(v) for v in data.xpos[body]] == pytest.approx(_POS)


def test_patch_against_a_robot_link_does_not_write_qpos(sim) -> None:
    """A robot link has no freejoint, so the qpos pass must skip it cleanly."""
    before = np.array(sim.mj_data.qpos).copy()
    result = sim.patch_scene_mjcf(ops=[{"op": "set_body_pos", "name": "a/link1", "pos": [0.0, 0.0, 0.4]}])
    assert result["status"] == "success"
    assert np.array(sim.mj_data.qpos).shape == before.shape
    assert bool(np.all(np.isfinite(sim.mj_data.qpos)))
