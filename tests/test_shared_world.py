"""Part 2 gate: one world, many robots. A control step is one control step, a teleport lands, one render shows both.

``Robot`` is part 3; here a world member is the least the world asks for (an id, a Description, a
detach hook), and ``act`` is what ``Robot.act`` does on a simulated body: write targets, step once.
"""

import json
import threading
import time

import numpy as np
import pytest

from strands_robots import Description, Simulation

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class Member:
    def __init__(self, name: str, sim: Simulation | None = None, **at: object) -> None:
        self.description = Description.load(name)
        self.sim: Simulation | None = sim or Simulation.default()
        self.id = self.sim._unique_id(self.description.name)
        self.sim.add(self, **at)  # type: ignore[arg-type]

    def _detached(self) -> None:
        self.sim = None

    def joint(self, key: str) -> float:
        assert self.sim is not None
        return self.sim.backend.joint_state(self.id)[key]

    def act(self, **targets: float) -> None:
        assert self.sim is not None
        q = self.sim.backend.joint_state(self.id).as_dict() | targets
        self.sim.backend.set_targets(self.id, np.array([q[k] for k in self.description.action_keys]))
        self.sim.control_step()


@pytest.fixture
def pair():
    return Member("so101"), Member("g1", position=[1, 0, 0])


def test_two_robots_share_the_default_world(pair):
    arm, g1 = pair
    assert arm.sim is g1.sim is Simulation.default()
    assert list(arm.sim.robots) == ["so101", "unitree_g1"]
    assert Member("so101", position=[0, -0.5, 0]).id == "so101_2"
    assert json.loads(json.dumps(arm.sim.describe()))["robots"]["unitree_g1"]["position"] == [1, 0, 0]


def test_each_robot_reads_its_own_joints_and_pose(pair):
    arm, g1 = pair
    a, g = arm.sim.backend.joint_state(arm.id), g1.sim.backend.joint_state(g1.id)
    assert a.names == ARM_JOINTS
    assert len(g.names) == 29 and g.names[0] == "left_hip_pitch_joint"
    assert np.allclose(arm.sim.backend.body_pose(g1.id).position, [1, 0, 0.79], atol=1e-6)  # "stand", spawn frame
    assert np.allclose(arm.sim.backend.body_pose(arm.id).position, [0, 0, 0], atol=1e-9)


def test_act_advances_one_control_step_and_leaves_the_other_robot(pair):
    arm, g1 = pair
    sim = arm.sim
    before_g1 = sim.backend.joint_state(g1.id).positions.copy()
    t0 = sim.time
    arm.act(shoulder_pan=0.3)
    assert sim.physics.control_dt == pytest.approx(0.02)
    assert sim.time == pytest.approx(t0 + sim.physics.control_dt)
    for _ in range(49):
        arm.act(shoulder_pan=0.3)
    assert arm.joint("shoulder_pan") == pytest.approx(0.3, abs=0.02)
    assert np.abs(sim.backend.joint_state(g1.id).positions - before_g1).max() < np.radians(1.0)


def test_manual_time_hands_stepping_to_the_caller(pair):
    arm, _ = pair
    sim = arm.sim
    t0 = sim.time
    with sim.manual_time():
        arm.act(shoulder_pan=0.2)
        assert sim.time == t0 and sim.time_is_manual
        sim.step(10)
    assert sim.time == pytest.approx(t0 + 10 * sim.physics.timestep)
    arm.act(shoulder_pan=0.2)
    assert sim.time == pytest.approx(t0 + 10 * sim.physics.timestep + sim.physics.control_dt)


def test_set_pose_lands_for_a_floating_and_a_fixed_base(pair):
    arm, g1 = pair
    sim = arm.sim
    sim.set_pose(g1, position=[2, 0, 0.8])
    assert np.allclose(sim.backend.body_pose(g1.id).position, [2, 0, 0.8], atol=1e-9)
    assert np.allclose(sim.backend.body_pose(g1.id).linear_velocity, 0.0)
    for _ in range(5):
        arm.act(shoulder_pan=0.3)
    moved, t = arm.joint("shoulder_pan"), sim.time
    sim.set_pose(arm, position=[0, 0.5, 0])  # fixed base: spec edit + recompile, joint state and time survive
    assert np.allclose(sim.backend.body_pose(arm.id).position, [0, 0.5, 0], atol=1e-9)
    assert arm.joint("shoulder_pan") == pytest.approx(moved, abs=1e-9) and sim.time == t
    with pytest.raises(KeyError, match="ghost"):
        sim.set_pose("ghost", position=[0, 0, 0])


@pytest.mark.parametrize(
    ("spawn", "target"),
    [
        ({"position": [1, 0, 0]}, [0, 0.5, 0]),  # an offset spawn frame
        ({"position": [0.2, 0, 0], "orientation": [0.7071068, 0, 0, 0.7071068]}, [0.5, 0, 0]),  # a yawed one
    ],
)
def test_a_fixed_base_teleport_lands_in_world_coordinates_and_survives_a_reload(spawn, target):
    with Simulation() as sim:
        arm = Member("so101", sim=sim, **spawn)
        sim.set_pose(arm, position=target)
        assert np.allclose(sim.backend.body_pose(arm.id).position, target, atol=1e-6)
        sim.add_object("box", shape="box", position=[0.3, 0, 0.1])  # an unrelated rebuild does not move it
        assert np.allclose(sim.backend.body_pose(arm.id).position, target, atol=1e-6)


def test_a_locked_reader_never_sees_a_half_added_or_half_removed_robot():
    with Simulation() as sim:
        Member("so101", sim=sim)
        stop, errors = threading.Event(), []

        def reader() -> None:
            try:
                while not stop.is_set():
                    with sim.lock:
                        sim.describe()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
        for t in threads:
            t.start()
        for _ in range(15):
            sim.remove(Member("so101", sim=sim))
        stop.set()
        for t in threads:
            t.join(2)
        assert not errors, errors[:3]


def test_one_render_shows_both_bodies(pair):
    arm, _ = pair
    img = arm.sim.render()
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    overview = arm.sim.scene.cameras[-1]
    with Simulation() as empty:  # the same camera over an empty floor: what differs is a robot
        empty.camera("overview", position=overview.position, target=overview.target)
        diff = np.abs(img.astype(int) - empty.render("overview").astype(int)).sum(axis=2) > 60
    h, w, _ = img.shape
    left, right = int(diff[:, : w // 2].sum()), int(diff[:, w // 2 :].sum())
    assert left > 150 and right > 1000, (left, right)  # the 30 cm arm left of centre, the G1 right of it
    assert diff.sum() < 0.25 * h * w


def test_render_without_a_camera_is_the_overview_even_after_user_cameras(pair):
    arm, _ = pair
    sim = arm.sim
    sim.camera("wrist", position=[0.15, -0.25, 0.25], target=[0.2, 0, 0.05], size=(64, 48), attach=arm)
    assert sim.render().shape == (480, 640, 3)
    assert [c.name for c in sim.scene.cameras] == ["wrist", "overview"]
    assert sim.render("wrist").shape == (48, 64, 3)
    with pytest.raises(KeyError, match="no camera 'nope'"):
        sim.render("nope")


def test_state_survives_add_object_and_remove(pair):
    arm, g1 = pair
    sim = arm.sim
    for _ in range(30):
        arm.act(elbow_flex=0.4)
    before = arm.joint("elbow_flex")
    box = sim.add_object("box", shape="box", position=[0.3, 0, 0.1], size=[0.03, 0.03, 0.03])
    assert arm.joint("elbow_flex") == pytest.approx(before, abs=1e-6)
    assert np.allclose(box.pose.position, [0.3, 0, 0.1])
    sim.remove(g1)
    assert list(sim.robots) == ["so101"] and g1.sim is None
    assert arm.joint("elbow_flex") == pytest.approx(before, abs=1e-6)


def test_revision_bumps_on_every_scene_change_and_not_on_motion(pair):
    arm, g1 = pair
    sim = arm.sim
    r0 = sim.revision
    sim.add_object("box", shape="box", position=[0.3, 0, 0.1])
    sim.camera("cam", position=[1, -1, 1], target=[0, 0, 0.2])
    sim.set_pose(g1, position=[2, 0, 0.8])
    sim.randomize(friction=1.2)
    sim.remove(g1)
    assert sim.revision == r0 + 5
    arm.act(shoulder_pan=0.1)
    sim.step(5)
    assert sim.revision == r0 + 5


def test_the_world_lock_serialises_steps_and_reads_across_a_recompile(pair):
    arm, _ = pair
    sim = arm.sim
    errors: list[BaseException] = []
    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                with sim.lock:
                    sim.backend.joint_state(arm.id)
                    sim.describe()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    for _ in range(100):
        arm.act(shoulder_pan=0.3)
    Member("so101", position=[0, 0.5, 0])  # a recompile under the reader
    stop.set()
    t.join(2)
    assert not errors


def test_an_explicit_world_is_separate_and_closes():
    with Simulation(timestep=0.001, substeps=5) as sim:
        arm = Member("so101", sim=sim)
        assert arm.sim is sim and not sim.is_default and sim.physics.control_dt == pytest.approx(0.005)
    assert arm.sim is None and "so101" not in Simulation.default().robots


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"backend": "bullet"}, ValueError, "unknown simulation backend"),
        ({"physics": "warp"}, ValueError, "physics="),
        ({"substeps": 0}, ValueError, "substeps"),
        ({"num_envs": 4}, ValueError, "num_envs"),
        ({"bogus": 1}, TypeError, "unexpected keyword"),
    ],
)
def test_a_world_refuses_what_it_cannot_be(kwargs, error, match):
    with pytest.raises(error, match=match):
        Simulation(**kwargs)


def test_step_cost_is_measured(pair, capsys):
    sim = pair[0].sim
    n = 2000
    t = time.perf_counter()
    sim.step(n)
    dt = time.perf_counter() - t
    with capsys.disabled():
        print(f"\n[measured] {n} physics steps (so101 + g1) in {dt * 1e3:.1f} ms = {dt / n * 1e6:.1f} us/step")
    assert dt / n < 0.002  # slower than real time would be a regression


def test_contacts_name_both_bodies_and_reset_stands_a_teleported_g1_on_its_keyframe(pair):
    _, g1 = pair
    sim = g1.sim
    sim.step(50)
    feet = [c for c in sim.contacts() if "ground" in (c.body_a, c.body_b) or c.body_a == "world"]
    assert any(c.body_b.startswith("unitree_g1/") and c.force > 0 for c in feet), sim.contacts()[:3]
    sim.set_pose(g1, position=[2, 0, 0.8])
    sim.add_object("box", shape="box", position=[0.3, 0, 0.1])  # a reload keeps the teleported base where it is
    assert np.allclose(sim.backend.body_pose(g1.id).position, [2, 0, 0.8], atol=1e-9)
    sim.reset()
    assert sim.time == 0.0
    assert np.allclose(sim.backend.body_pose(g1.id).position, [1, 0, 0.79], atol=1e-6)


@pytest.mark.parametrize("name", ["g1", "panda"])  # a directional and a spot scene light
def test_the_world_owns_the_lighting(name):
    look = {"position": [1.9, -2.1, 1.3], "target": [0.45, 0, 0.45], "size": (160, 100)}
    with Simulation() as empty, Simulation() as sim:
        empty.camera("c", **look)
        Member(name, sim=sim)
        sim.camera("c", **look)
        assert sim.backend.model.nlight == empty.backend.model.nlight
        assert sim.render("c")[-10:].mean() == pytest.approx(empty.render("c")[-10:].mean(), abs=5)  # the floor
