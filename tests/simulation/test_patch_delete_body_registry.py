"""Regression tests: a patch that deletes bodies re-resolves the robot registry.

``patch_scene_mjcf``'s ``delete_body`` op accepts ANY body name, including a robot
link, and deleting one removes that link's whole subtree. The per-robot
``joint_ids`` / ``actuator_ids`` are plain integer indices, and the patch path was
the one recompile path that never re-resolved them, so they were left describing a
robot that no longer existed:

    add_robot("a")                  nbody=12 njnt=9 nu=8
    patch delete_body "a/link5"     nbody=6  njnt=4 nu=4     status=success
    world.robots["a"].joint_ids     [0..8] - five of them past the end
    zero_dynamics(robot_name="a")   IndexError: index 4 is out of bounds

i.e. a stale index into the new, smaller arrays, raising straight out of the tool
contract instead of returning a structured error.

``joint_names`` was stale too, which made the two ``set_joint_positions`` forms
contradict each other: a 9-value list was rejected for naming 5 non-joints, while
a 4-value list was rejected for "not matching joint count 9".

Both are now re-resolved by ``scene_ops._rediscover_robot_ids`` (also used by the
incremental and full-rebuild paths, so all three agree). Deleting a robot link is
still allowed - ``patch_scene_mjcf`` is the documented escape hatch - but the
registry now describes the live scene.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="patch_delete_body_registry", mesh=False)
    s.create_world()
    assert s.add_robot(name="a", data_config="panda")["status"] == "success"
    yield s
    s.destroy()


def _robot(sim):
    return sim._world.robots["a"]


def _delete_link(sim) -> None:
    assert sim.patch_scene_mjcf(ops=[{"op": "delete_body", "name": "a/link5"}])["status"] == "success"


def test_ids_stay_within_the_model(sim) -> None:
    """The core defect: out-of-range indices into the new arrays."""
    _delete_link(sim)
    model = sim.mj_model
    robot = _robot(sim)
    assert all(0 <= i < model.njnt for i in robot.joint_ids), (robot.joint_ids, model.njnt)
    assert all(0 <= i < model.nu for i in robot.actuator_ids), (robot.actuator_ids, model.nu)


def test_zero_dynamics_does_not_raise(sim) -> None:
    """It indexed qacc with a stale joint id and raised IndexError."""
    _delete_link(sim)
    assert sim.zero_dynamics(robot_name="a")["status"] == "success"


def test_joint_names_describe_the_live_model(sim) -> None:
    _delete_link(sim)
    model = sim.mj_model
    robot = _robot(sim)
    for name in robot.joint_names:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"a/{name}") >= 0, name
    assert len(robot.joint_names) == len(robot.joint_ids)


def test_the_two_set_joint_positions_forms_agree(sim) -> None:
    """They used to reject each other's idea of the joint count."""
    _delete_link(sim)
    count = len(_robot(sim).joint_names)
    assert sim.set_joint_positions(robot_name="a", positions=[0.05] * count)["status"] == "success"
    assert sim.set_joint_positions(robot_name="a", positions=[0.05] * (count + 5))["status"] == "error"


def test_action_keys_match_the_surviving_actuators(sim) -> None:
    _delete_link(sim)
    keys = sim.robot_action_keys("a")
    assert len(keys) == int(sim.mj_model.nu)
    assert sim.send_action(action=[0.0] * len(keys), robot_name="a")["status"] == "success"


def test_the_world_is_still_steppable(sim) -> None:
    _delete_link(sim)
    assert sim.step(n_steps=50)["status"] == "success"
    assert sim.get_observation("a", skip_images=True)


def test_an_ordinary_patch_leaves_the_registry_intact(sim) -> None:
    """No pruning on the common path: every joint still resolves."""
    assert (
        sim.add_object(name="c", shape="box", size=[0.04] * 3, position=[0.4, 0, 0.3], mass=0.1)["status"] == "success"
    )
    before = list(_robot(sim).joint_names)
    assert sim.patch_scene_mjcf(ops=[{"op": "set_body_pos", "name": "c", "pos": [0.5, 0, 0.4]}])["status"] == "success"
    assert list(_robot(sim).joint_names) == before
    assert len(_robot(sim).joint_ids) == len(before)


def test_deleting_a_tracked_object_does_not_touch_the_robot(sim) -> None:
    assert (
        sim.add_object(name="c", shape="box", size=[0.04] * 3, position=[0.4, 0, 0.3], mass=0.1)["status"] == "success"
    )
    before = list(_robot(sim).joint_names)
    assert sim.patch_scene_mjcf(ops=[{"op": "delete_body", "name": "c"}])["status"] == "success"
    assert list(_robot(sim).joint_names) == before


def test_a_second_robot_is_unaffected(sim) -> None:
    """Pruning must be per-robot, not global."""
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    before_b = list(sim._world.robots["b"].joint_names)
    _delete_link(sim)
    assert list(sim._world.robots["b"].joint_names) == before_b
    model = sim.mj_model
    for name in sim._world.robots["b"].joint_names:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"b/{name}") >= 0, name
