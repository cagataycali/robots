"""A not-reached ``rotate_wrist`` names the contact or the joint bound that stopped it.

``move_to`` learned to read the engine at the final servo tick and name what
stopped it. ``rotate_wrist`` servos to a set-point the same way and reported
only its residual: measured on the bundled so100, the wrist turned 0.003 of
2.500 rad while 15 contact pairs held the gripper inside its own base, and the
refusal named none of them - the cause needed a separate ``get_contacts``
call.

Both primitives now answer through one reader, so this file pins what is new
rather than re-pinning it: the reader's contact scoping, its ``get_contacts``
vocabulary and its active-vs-in-the-gap reading are already pinned on
``move_to`` and are shared code. What is new here is that the wrist reports at
all, that the clause's remedy no longer assumes the commanded value is a
position, that the report is scoped to the joint this primitive commands (it
HOLDS every other one, so a bound one of those did not stop the call), and
that a backend which did not look still says nothing rather than claiming the
wrist is clear.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

from strands_robots.simulation.motion_primitives_base import MotionPrimitivesCore

requires_mujoco = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")


def _obstruction_of(result: dict[str, Any]) -> dict[str, Any]:
    """The obstruction report a refusal carries, asserted present."""
    payload = result["content"][1]["json"]
    assert "obstruction" in payload, payload
    return payload["obstruction"]


class TestTheRemedyFitsEveryPrimitiveThatSharesIt:
    """``rotate_wrist`` commands an angle, so "move the target away" is not a remedy it can offer."""

    def test_the_contact_remedy_names_the_commanded_value_not_a_target_position(self) -> None:
        text = MotionPrimitivesCore._obstruction_text(
            {
                "contacts": [{"geom1": "arm/Base/geom_3", "geom2": "arm/Fixed_Jaw/geom_19", "dist": -0.0456}],
                "contacts_total": 15,
                "joints_at_limit": [],
            }
        )
        assert "the robot is in contact: 'arm/Base/geom_3' <-> 'arm/Fixed_Jaw/geom_19' (d=-0.0456 m)" in text
        assert "and 14 more" in text
        assert text.endswith("Clear what it touches, or command a value that avoids it - or loosen tol.")
        assert "target" not in text

    def test_the_joint_remedy_names_the_commanded_value_not_a_target_position(self) -> None:
        text = MotionPrimitivesCore._obstruction_text(
            {
                "contacts": [],
                "contacts_total": 0,
                "joints_at_limit": [{"joint": "arm/wrist_roll", "pos": 1.0004, "limit": 1.0, "side": "upper"}],
            }
        )
        assert "commanded joint(s) at a limit: 'arm/wrist_roll' at its upper limit (1.0004 vs 1.0000)" in text
        assert text.endswith("Command a value that joint can reach inside its range, or loosen tol.")
        assert "target" not in text


@pytest.mark.parametrize(
    ("pos", "lo", "hi", "side"),
    [
        (0.0, -1.0, 1.0, None),
        (0.99, -1.0, 1.0, "upper"),
        (-0.99, -1.0, 1.0, "lower"),
        (0.97, -1.0, 1.0, None),
        # The 1e-3 floor: a range this narrow has a proportional margin of
        # 1e-4, too tight to ever call a servo "parked on the bound".
        (0.0009, 0.0, 0.01, "lower"),
    ],
)
def test_one_owner_decides_whether_a_joint_sits_on_a_bound(pos, lo, hi, side) -> None:
    """Both backends ask this, by different routes, so the threshold lives in one place."""
    record = MotionPrimitivesCore._joint_limit_record("j", pos, lo, hi)
    if side is None:
        assert record is None
        return
    assert record is not None
    assert record["side"] == side
    assert record["joint"] == "j" and record["pos"] == pos
    assert record["limit"] == pytest.approx(lo if side == "lower" else hi)


# A wrist hinge whose axis lies across gravity, driven by a servo too weak to
# lift the link: it rests on its UPPER bound while the set-point asks it to
# come up. The target is inside the range, so the refusal is the servo's, not
# the range check's.
_WEAK_WRIST_XML = """
<mujoco model="weak_wrist">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.6">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="wrist_roll" type="hinge" axis="0 1 0" range="-0.2 1.0"/>
      <body name="link1" pos="0.3 0 0">
        <geom type="capsule" size="0.02" fromto="-0.3 0 0 0 0 0"/>
        <body name="gripper" pos="0.1 0 0">
          <geom name="jaw" type="box" size="0.02 0.02 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><position name="wrist_act" joint="wrist_roll" kp="0.2" kv="0.05"/></actuator>
</mujoco>
"""

# A shoulder that can be parked ON its bound plus a healthy wrist roll about
# the link axis (gravity exerts no torque about it, so it stays put).
_HELD_BOUND_XML = """
<mujoco model="held_bound">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.6">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-0.2 1.0"/>
      <body name="link1" pos="0.3 0 0">
        <geom type="capsule" size="0.02" fromto="-0.3 0 0 0 0 0"/>
        <body name="gripper" pos="0.1 0 0">
          <joint name="wrist_roll" type="hinge" axis="1 0 0" range="-3.0 3.0"/>
          <geom name="jaw" type="box" size="0.04 0.04 0.04" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="0.2" kv="0.05"/>
    <position name="wrist_act" joint="wrist_roll" kp="5" kv="0.5"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def wrist_arm(tmp_path):
    from strands_robots.simulation import Simulation

    sims = []

    def make(xml: str = _WEAK_WRIST_XML):
        path = tmp_path / f"arm_{len(sims)}.xml"
        path.write_text(xml)
        sim = Simulation()
        sim.create_world(timestep=0.002)
        assert sim.add_robot("arm", urdf_path=str(path), position=[0.0, 0.0, 0.0])["status"] == "success"
        sims.append(sim)
        return sim

    yield make
    for sim in sims:
        sim.destroy()


@requires_mujoco
class TestRotateWristNamesWhatStoppedIt:
    def test_a_wrist_resting_on_its_bound_names_the_joint(self, wrist_arm) -> None:
        from strands_robots.simulation.motion_primitives_base import JOINT_LIMIT_MARGIN_FRACTION

        sim = wrist_arm()
        res = sim.rotate_wrist(robot_name="arm", target_yaw=-0.15, tol=0.01, max_steps=60)
        assert res["status"] == "error"
        obstruction = _obstruction_of(res)
        assert obstruction["contacts"] == [] and obstruction["contacts_total"] == 0
        joints = obstruction["joints_at_limit"]
        assert [j["joint"] for j in joints] == ["arm/wrist_roll"], obstruction
        assert joints[0]["side"] == "upper" and joints[0]["limit"] == pytest.approx(1.0)
        assert joints[0]["pos"] >= 1.0 - max(1.2 * JOINT_LIMIT_MARGIN_FRACTION, 1e-3)
        text = res["content"][0]["text"]
        assert "The servo was stopped: commanded joint(s) at a limit: 'arm/wrist_roll' at its upper limit" in text

    def test_a_bound_joint_the_call_only_holds_is_not_offered_as_the_cause(self, wrist_arm) -> None:
        """The wrist is what was commanded; the shoulder was asked to stay where it already was."""
        sim = wrist_arm(_HELD_BOUND_XML)
        assert sim.set_joint_positions(robot_name="arm", positions={"shoulder": 1.0})["status"] == "success"
        res = sim.rotate_wrist(robot_name="arm", target_yaw=2.5, tol=0.01, max_steps=3)
        assert res["status"] == "error"
        obstruction = _obstruction_of(res)
        assert obstruction["joints_at_limit"] == [], obstruction
        # ... and the shoulder really was sitting on its bound, so a report
        # scoped to every joint the call touched WOULD have named it.
        import mujoco as mj

        world = sim._world
        assert world is not None
        every_joint = MotionPrimitivesCore._joints_at_limit(
            mj, world._model, world._data.qpos, world.robots["arm"].joint_ids
        )
        assert [j["joint"] for j in every_joint] == ["arm/shoulder"], every_joint
        text = res["content"][0]["text"]
        assert "no contact involved the robot and no commanded joint was at a limit" in text
        assert "raise max_steps" in text

    def test_a_reached_rotation_carries_no_obstruction(self, wrist_arm) -> None:
        sim = wrist_arm(_HELD_BOUND_XML)
        res = sim.rotate_wrist(robot_name="arm", target_yaw=0.4, tol=0.02, max_steps=200)
        assert res["status"] == "success"
        assert "obstruction" not in res["content"][1]["json"]


@requires_mujoco
class TestOnTheBundledSo100:
    def test_the_folded_gripper_that_blocks_the_wrist_is_named(self) -> None:
        """The measured case: the wrist barely turns because the gripper is inside the base."""
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world(timestep=0.002)
        try:
            assert sim.add_robot("arm", data_config="so100")["status"] == "success"
            # Fold the gripper back down into the base, then ask for a big turn.
            assert (
                sim.set_joint_positions(robot_name="arm", positions={"Pitch": -0.5, "Elbow": 3.0})["status"]
                == "success"
            )
            res = sim.rotate_wrist(robot_name="arm", target_yaw=2.5, tol=0.02, max_steps=200)
            assert res["status"] == "error"
            payload = res["content"][1]["json"]
            # The wrist hardly moved - this is a blocked servo, not a slow one.
            assert abs(payload["final_yaw"]) < 0.05 and payload["yaw_error_rad"] > 2.0
            obstruction = _obstruction_of(res)
            contacts = obstruction["contacts"]
            assert contacts, obstruction
            assert any("Base" in c["geom1"] or "Base" in c["geom2"] for c in contacts), obstruction
            assert obstruction["contacts_total"] >= len(contacts)
            text = res["content"][0]["text"]
            assert "The servo was stopped: the robot is in contact:" in text
            assert f"(d={contacts[0]['dist']:.4f} m)" in text
        finally:
            sim.destroy()


class TestABackendThatDidNotLook:
    """``None`` and an empty report are different answers, so they read differently."""

    @staticmethod
    def _result(obstruction: dict[str, Any] | None) -> dict[str, Any]:
        return MotionPrimitivesCore._rotate_wrist_result(
            "arm",
            0.02,
            200,
            reached=False,
            steps_used=200,
            wrist_name="wrist_roll",
            target_yaw=2.5,
            final_yaw=0.003,
            yaw_error=2.497,
            obstruction=obstruction,
        )

    def test_an_engine_that_did_not_look_reports_only_the_residual(self) -> None:
        result = self._result(None)
        assert "obstruction" not in result["content"][1]["json"]
        # Byte-identical to the reply before the wrist could read anything: a
        # null field would read as "looked, found nothing".
        assert result["content"][0]["text"].endswith("(residual 2.4970 rad).")

    def test_an_engine_that_looked_and_found_nothing_says_so(self) -> None:
        result = self._result({"contacts": [], "contacts_total": 0, "joints_at_limit": []})
        assert result["content"][1]["json"]["obstruction"]["contacts_total"] == 0
        assert "nothing visible is blocking it" in result["content"][0]["text"]
