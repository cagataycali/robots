"""The shared-world thesis (tests/test_shared_world.py) on Isaac: two robots, one PhysX world.

Same assertions where PhysX can mean the same thing; the gaps are tests too (a refusal, never a silent no-op).
Run on a CUDA box with Isaac Sim: ``pytest -s tests/test_isaac_shared_world.py``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from strands_robots import Robot, Simulation

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("isaacsim") is None, reason="isaacsim not installed (needs a CUDA box with Isaac Sim)"
)


@pytest.fixture
def pair():
    sim = Simulation("isaac")
    try:
        yield Robot("so101", sim=sim), Robot("g1", sim=sim, position=[1, 0, 0])
    finally:
        sim.close()


def test_two_robots_share_one_world_in_insertion_order(pair):
    arm, g1 = pair
    assert arm.sim is g1.sim and list(arm.sim.robots) == [arm.id, g1.id]


def test_each_robot_reads_its_own_joints_and_pose(pair):
    arm, g1 = pair
    a, g = arm.observe(), g1.observe()
    assert a.joints.names == arm.description.action_keys and len(g.joints.names) == 29
    assert np.allclose(g.pose.position[:2], [1, 0], atol=1e-3) and g.pose.position[2] > 0.5  # keyframe height
    assert np.allclose(a.pose.position, [0, 0, 0], atol=1e-6)


def test_act_advances_one_control_step_and_leaves_the_other_robot(pair):
    arm, g1 = pair
    sim, key = arm.sim, arm.description.action_keys[0]
    before = g1.observe().joints.positions.copy()
    t0 = sim.time
    arm.act({key: 0.3})
    assert sim.time == pytest.approx(t0 + sim.physics.control_dt, abs=1e-6)
    for _ in range(49):
        arm.act({key: 0.3})
    assert arm.observe()[key] == pytest.approx(0.3, abs=0.05)
    drift = np.abs(g1.observe().joints.positions - before).max()
    print(f"\n[measured] isaac g1 joint drift while the arm moved 50 steps: {np.degrees(drift):.2f} deg", flush=True)


def test_manual_time_hands_stepping_to_the_caller(pair):
    arm, _ = pair
    sim, key = arm.sim, arm.description.action_keys[0]
    t0 = sim.time
    with sim.manual_time():
        arm.act({key: 0.2})
        assert sim.time == t0 and sim.time_is_manual
        sim.step(10)
    assert sim.time == pytest.approx(t0 + 10 * sim.physics.timestep, abs=1e-6)


def test_set_pose_lands_for_a_floating_and_a_fixed_base(pair):
    arm, g1 = pair
    sim, key = arm.sim, arm.description.action_keys[0]
    g1.set_pose(position=[2, 0, 0.8])
    assert np.allclose(g1.pose.position, [2, 0, 0.8], atol=1e-3)
    for _ in range(5):
        arm.act({key: 0.3})
    moved, t = arm.observe()[key], sim.time
    arm.set_pose(position=[0, 0.5, 0])
    assert np.allclose(arm.pose.position, [0, 0.5, 0], atol=1e-3)
    assert arm.observe()[key] == pytest.approx(moved, abs=1e-3) and sim.time == pytest.approx(t)
    with pytest.raises(KeyError, match="ghost"):
        sim.set_pose("ghost", position=[0, 0, 0])


def test_one_render_shows_both_bodies(pair):
    img = pair[0].sim.render()
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8 and img.std() > 5.0


def test_state_and_time_survive_add_object_camera_and_remove(pair):
    arm, g1 = pair
    sim, key = arm.sim, arm.description.action_keys[2]
    for _ in range(30):
        arm.act({key: 0.4})
    before, t, r0 = arm.observe()[key], sim.time, sim.revision
    box = sim.add_object("box", shape="box", position=[0.3, 0, 0.1], size=[0.03, 0.03, 0.03])
    assert arm.observe()[key] == pytest.approx(before, abs=1e-3)
    assert np.allclose(box.pose.position, [0.3, 0, 0.1], atol=1e-3)
    sim.camera("cam", position=[1, -1, 1], target=[0, 0, 0.2])
    sim.remove(g1)
    assert list(sim.robots) == [arm.id] and g1.sim is None
    assert arm.observe()[key] == pytest.approx(before, abs=1e-3)
    assert sim.time == pytest.approx(t) and sim.revision == r0 + 3


def test_reset_zeroes_time_and_stands_a_teleported_g1_on_its_keyframe(pair):
    _, g1 = pair
    sim = g1.sim
    sim.step(50)
    sim.set_pose(g1, position=[2, 0, 0.8])
    sim.reset()
    assert sim.time == 0.0
    assert np.allclose(g1.pose.position[:2], [1, 0], atol=1e-3)


def test_lighting_randomizes_and_physics_randomization_is_refused_by_name(pair):
    sim = pair[0].sim
    sim.randomize(lighting=0.5)
    assert sim.render().std() > 2.0
    with pytest.raises(NotImplementedError, match="friction"):
        sim.randomize(friction=1.2)


def test_the_gaps_are_named(pair):
    arm, _ = pair
    assert arm.sim.contacts() == []  # PhysX contact reports need a contact sensor per prim (not wired)
    with pytest.raises(NotImplementedError, match="gains"):
        arm.sim.backend.set_gains(arm.id, np.ones(6), np.ones(6))
