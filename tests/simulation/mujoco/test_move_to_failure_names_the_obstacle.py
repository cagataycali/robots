"""A not-reached ``move_to`` names the contact or the joint limit that stopped the servo.

Measured with an agent on the bundled so100 (v0.5.2 devx replay, s09):
three ``move_to`` calls in a row ended in "the pose fights joint
limits/contacts" and only a separate ``get_contacts`` call named the cause.
On that robot a low target near the base stops the servo with the jaw pushing
into the floor (``'ground' <-> 'so100/Fixed_Jaw/geom_18'``). The refusal now
reads the engine at the final tick and names the active contacts on the robot
(at most three, nearest first, total reported) and the commanded joints
sitting at a bound, and carries the same facts as ``json.obstruction`` -
spelled the way ``get_contacts`` spells a contact. When nothing visible
blocks the arm it says so, which is the "raise max_steps" case.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

requires_mujoco = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")


def _world(sim: Any) -> Any:
    """The engine's world, asserted present - a created world is every cell's premise."""
    world = sim._world
    assert world is not None
    return world


def _text(obstruction: dict[str, object] | None) -> str:
    """The refusal clause the builder produces for ``obstruction``."""
    from strands_robots.simulation.motion_primitives_base import MotionPrimitivesCore

    return MotionPrimitivesCore._obstruction_text(obstruction)


class TestObstructionText:
    def test_engine_did_not_look_keeps_the_legacy_clause(self) -> None:
        assert _text(None) == "The servo may need more steps, or the pose fights joint limits/contacts."

    def test_contacts_are_named_with_distance_and_total(self) -> None:
        text = _text(
            {
                "contacts": [{"geom1": "ground", "geom2": "so100/Fixed_Jaw/geom_18", "dist": -0.0002}],
                "contacts_total": 5,
                "joints_at_limit": [],
            }
        )
        assert "the robot is in contact: 'ground' <-> 'so100/Fixed_Jaw/geom_18' (d=-0.0002 m) and 4 more" in text
        assert text.endswith("Clear what it touches, or command a value that avoids it - or loosen tol.")

    def test_joints_at_limit_are_named_with_side_and_bound(self) -> None:
        text = _text(
            {
                "contacts": [],
                "contacts_total": 0,
                "joints_at_limit": [{"joint": "arm/elbow", "pos": 1.5699, "limit": 1.57, "side": "upper"}],
            }
        )
        assert "commanded joint(s) at a limit: 'arm/elbow' at its upper limit (1.5699 vs 1.5700)" in text
        assert "Command a value that joint can reach inside its range, or loosen tol." in text

    def test_nothing_visible_points_at_max_steps(self) -> None:
        text = _text({"contacts": [], "contacts_total": 0, "joints_at_limit": []})
        assert "no contact involved the robot and no commanded joint was at a limit" in text
        assert "raise max_steps" in text

    def test_backend_that_did_not_read_contacts_does_not_claim_the_robot_is_free(self) -> None:
        text = _text({"contacts": [], "contacts_total": None, "joints_at_limit": []})
        assert "contacts were not read on this backend" in text
        assert "no contact involved" not in text


_YAW_ARM_XML = """
<mujoco model="yaw_arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-0.5 0.5"/>
      <body name="link1" pos="0.3 0 0">
        <geom type="capsule" size="0.02" fromto="-0.3 0 0 0 0 0"/>
        <body name="gripper" pos="0.1 0 0">
          <geom name="jaw" type="box" size="0.02 0.02 0.02"/>
        </body>
      </body>
    </body>
    {extra}
  </worldbody>
  <actuator><position name="shoulder_act" joint="shoulder" kp="200" kv="5"/></actuator>
</mujoco>
"""

# A pitch hinge with gravity on and a servo too weak to lift the link: the
# arm rests against its UPPER bound (gravity pushes it there) while the
# target asks it to point up. IK solves the target exactly, so the failure
# reaches the servo path - which is the path that reads the obstruction.
_PITCH_ARM_XML = """
<mujoco model="pitch_arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.6">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-0.2 1.0"/>
      <body name="link1" pos="0.3 0 0">
        <geom type="capsule" size="0.02" fromto="-0.3 0 0 0 0 0"/>
        <body name="gripper" pos="0.1 0 0">
          <geom name="jaw" type="box" size="0.02 0.02 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><position name="shoulder_act" joint="shoulder" kp="0.2" kv="0.05"/></actuator>
</mujoco>
"""


@pytest.fixture
def arm_with(tmp_path):
    from strands_robots.simulation import Simulation

    sims = []

    def make(extra: str = "", xml: str = _YAW_ARM_XML):
        path = tmp_path / f"arm_{len(sims)}.xml"
        path.write_text(xml.format(extra=extra))
        sim = Simulation()
        sim.create_world(timestep=0.002)
        assert sim.add_robot("arm", urdf_path=str(path), position=[0.0, 0.0, 0.0])["status"] == "success"
        sims.append(sim)
        return sim

    yield make
    for sim in sims:
        sim.destroy()


@requires_mujoco
class TestMoveToNamesTheJointLimit:
    def test_a_servo_parked_on_a_bound_names_the_joint(self, arm_with) -> None:
        import math

        sim = arm_with(xml=_PITCH_ARM_XML)
        ang = -0.15  # tip slightly UP; gravity holds the weak servo at the +1.0 bound
        target = [0.4 * math.cos(ang), 0.0, 0.6 - 0.4 * math.sin(ang)]
        res = sim.move_to(robot_name="arm", position=target, tol=0.005, max_steps=60)
        assert res["status"] == "error"
        text = res["content"][0]["text"]
        obstruction = res["content"][1]["json"]["obstruction"]
        assert obstruction["contacts"] == [] and obstruction["contacts_total"] == 0
        joints = obstruction["joints_at_limit"]
        assert [j["joint"] for j in joints] == ["arm/shoulder"]
        assert joints[0]["side"] == "upper" and joints[0]["limit"] == pytest.approx(1.0)
        from strands_robots.simulation.motion_primitives_base import JOINT_LIMIT_MARGIN_FRACTION

        assert joints[0]["pos"] >= 1.0 - max(1.2 * JOINT_LIMIT_MARGIN_FRACTION, 1e-3)
        assert "The servo was stopped: commanded joint(s) at a limit: 'arm/shoulder' at its upper limit" in text
        assert "fights joint limits/contacts" not in text

    def test_a_reached_move_carries_no_obstruction(self, arm_with) -> None:
        sim = arm_with()
        res = sim.move_to(robot_name="arm", position=[0.3894, 0.0919, 0.1], tol=0.02, max_steps=200)
        assert res["status"] == "success"
        assert "obstruction" not in res["content"][1]["json"]


_WALL = """
    <body name="wall" pos="0.35 0.12 0.1">
      <geom name="wall_geom" type="box" size="0.01 0.03 0.1"/>
    </body>
    <body name="loose_cube" pos="1.0 1.0 0.02">
      <freejoint/>
      <geom name="cube_geom" type="box" size="0.02 0.02 0.02" mass="0.1"/>
    </body>
"""


@requires_mujoco
class TestMoveToNamesTheContact:
    def test_a_wall_in_the_sweep_is_named_and_a_far_cube_is_not(self, arm_with) -> None:
        sim = arm_with(extra=_WALL)
        # Swing toward +Y into the wall standing at y=0.12 across the tip's arc;
        # the target sits ON the arc (0.45 rad), so IK solves it and the servo
        # is what gets stopped.
        from strands_robots.simulation.motion_primitives_base import OBSTRUCTION_MAX_CONTACTS

        res = sim.move_to(robot_name="arm", position=[0.3603, 0.1740, 0.1], tol=0.005, max_steps=60)
        assert res["status"] == "error"
        obstruction = res["content"][1]["json"]["obstruction"]
        pairs = {frozenset((c["geom1"], c["geom2"])) for c in obstruction["contacts"]}
        assert frozenset(("arm/jaw", "arm/wall_geom")) in pairs, obstruction
        assert not any("cube_geom" in geom for pair in pairs for geom in pair), (
            "the loose cube's own contacts are not the robot's - and it even shares the robot's name prefix"
        )
        assert len(obstruction["contacts"]) <= OBSTRUCTION_MAX_CONTACTS
        assert obstruction["contacts_total"] >= len(obstruction["contacts"])
        text = res["content"][0]["text"]
        assert "the robot is in contact:" in text and "'arm/jaw' <-> 'arm/wall_geom'" in text
        # One line per geom pair even when MuJoCo reports several contact points for it.
        assert len(pairs) == len(obstruction["contacts"])
        assert "Clear what it touches, or command a value that avoids it" in text

    def test_the_scoped_subtree_is_the_arm_only(self, arm_with) -> None:
        import mujoco as mj

        sim = arm_with(extra=_WALL)
        model = _world(sim)._model
        commanded = _world(sim).robots["arm"].joint_ids
        ids = sim._commanded_robot_body_ids(model, commanded)
        names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in ids}
        assert names == {"arm/base", "arm/link1", "arm/gripper"}

    def test_a_contact_report_is_spelled_like_get_contacts(self, arm_with) -> None:
        """One vocabulary for one fact: the row a caller reads in either report is the same row."""
        sim = arm_with(extra=_WALL)
        res = sim.move_to(robot_name="arm", position=[0.3603, 0.1740, 0.1], tol=0.005, max_steps=60)
        contacts = res["content"][1]["json"]["obstruction"]["contacts"]
        assert contacts
        listed = sim.get_contacts()["content"][1]["json"]["contacts"]
        assert listed
        assert all(set(c) <= set(listed[0]) for c in contacts), (contacts, listed[0])
        assert set(contacts[0]) == {"geom1", "geom2", "dist"}


_CURTAIN = """
    <body name="curtain" pos="0.40 0.15 0.1">
      <geom name="curtain_geom" type="box" size="0.02 0.02 0.1" margin="0.08" gap="0.08"/>
    </body>
"""


@requires_mujoco
class TestAProximityReportIsNotWhatStoppedTheServo:
    def test_only_the_pair_that_carries_force_is_named(self, arm_with) -> None:
        """A wide ``margin`` puts two pairs of one obstacle in the list; one pushes back.

        ``dist`` cannot sort them - the load-bearing pair here is at a
        *positive* distance too - so the report reads the solver's own
        admission, the reading
        :func:`~strands_robots.simulation.predicates.contact_is_active` owns
        for a ``get_contacts`` record.
        """
        sim = arm_with(extra=_CURTAIN)
        res = sim.move_to(robot_name="arm", position=[0.3603, 0.1740, 0.1], tol=0.005, max_steps=60)
        assert res["status"] == "error"
        named = {frozenset((c["geom1"], c["geom2"])) for c in res["content"][1]["json"]["obstruction"]["contacts"]}
        listed = sim.get_contacts()["content"][1]["json"]["contacts"]
        pushing = {frozenset((c["geom1"], c["geom2"])) for c in listed if c["active"]}
        in_the_gap = {frozenset((c["geom1"], c["geom2"])) for c in listed if not c["active"]} - pushing
        assert named == pushing & named, (named, pushing)
        assert in_the_gap, "the wide margin must put at least one pair in the gap"
        assert not (named & in_the_gap), (named, in_the_gap)
        # The pair that stopped the arm is named even though it is 0.0765 m
        # away: a wide margin is load-bearing at a positive distance.
        jaw = frozenset(("arm/jaw", "arm/curtain_geom"))
        assert jaw in named, named
        assert min(c["dist"] for c in res["content"][1]["json"]["obstruction"]["contacts"]) > 0.0


@requires_mujoco
class TestOnTheBundledSo100:
    def test_the_replay_target_names_what_the_jaw_touches(self) -> None:
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()
        try:
            assert sim.add_robot("arm", data_config="so100")["status"] == "success"
            # s09: a point low and close to the base, well inside IK reach.
            res = sim.move_to(robot_name="arm", position=[0.0, -0.10, 0.03], tol=0.005, max_steps=60)
            assert res["status"] == "error"
            text = res["content"][0]["text"]
            obstruction = res["content"][1]["json"]["obstruction"]
            assert obstruction["contacts"], text
            assert any("Fixed_Jaw" in c["geom1"] or "Fixed_Jaw" in c["geom2"] for c in obstruction["contacts"]), text
            assert "The servo was stopped: the robot is in contact:" in text
            assert "fights joint limits/contacts" not in text
        finally:
            sim.destroy()

    def test_the_scope_reaches_above_the_first_commanded_joint(self) -> None:
        """The base link carries no commanded joint, and a jaw can still hit it."""
        import mujoco as mj

        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()
        try:
            assert sim.add_robot("arm", data_config="so100")["status"] == "success"
            model = _world(sim)._model
            commanded = _world(sim).robots["arm"].joint_ids
            first = min(int(j) for j in commanded)
            assert mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[first])) != "arm/Base"
            names = {
                mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in sim._commanded_robot_body_ids(model, commanded)
            }
            assert "arm/Base" in names, names
        finally:
            sim.destroy()
