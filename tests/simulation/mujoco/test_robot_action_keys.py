"""Policies must be keyed by a robot's actuators, not its joints.

A robot's actuator set is not always its joint set. Two common cases break a
naive joint-name keying:

* passive/mimic finger joints have no driving actuator, so a policy keyed by
  those joint names emits keys that ``send_action`` resolves to nothing, and
* a tendon-driven gripper is an *actuator* with no matching joint name, so it
  is never commanded at all.

Before the fix the simulation keyed the policy via
``set_robot_state_keys(robot_joint_names(...))``. On robots whose actuators
differ from their joints (``xarm7``, ``aloha``, ``unitree_g1``, ``stretch``,
tendon grippers) the mock policy's actions were silently dropped: the robot
never moved while ``run_policy`` still reported a partial/total action failure
that previously even read as ``status=success``. The fix introduces
``robot_action_keys`` (the keys ``send_action`` resolves) and keys policies by
it instead.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim():
    s = Simulation(mesh=False)
    s.create_world()
    yield s
    s.cleanup()


class TestRobotActionKeys:
    def test_actuators_diverge_from_joints_on_tendon_gripper(self, sim):
        """xarm7: 13 joints (6 passive finger joints) but 8 actuators (tendon gripper)."""
        sim.add_robot("xarm7")
        joints = sim.robot_joint_names("xarm7")
        actions = sim.robot_action_keys("xarm7")
        # The bug's precondition: the two sets genuinely differ.
        assert set(actions) != set(joints)
        # Action keys are the actuators send_action resolves (act1-7 + tendon).
        assert actions == ["act1", "act2", "act3", "act4", "act5", "act6", "act7", "gripper"]
        # Passive finger joints (no driving actuator) must NOT be action keys.
        assert "left_finger_joint" not in actions
        assert "right_inner_knuckle_joint" not in actions

    def test_action_keys_resolve_via_send_action(self, sim):
        """Every key from robot_action_keys must resolve in send_action."""
        sim.add_robot("aloha")
        keys = sim.robot_action_keys("aloha")
        # aloha grippers are tendon actuators (left/gripper, right/gripper) with
        # no matching joint name; the finger joints have no actuator.
        assert "left/gripper" in keys
        assert "right/gripper" in keys
        result = sim.send_action({k: 0.0 for k in keys}, robot_name="aloha")
        assert result["status"] == "success", result

    def test_joint_keys_leave_actuators_unresolved(self, sim):
        """Keying by joints (the old behaviour) leaves actuators unresolved."""
        sim.add_robot("aloha")
        joints = sim.robot_joint_names("aloha")
        result = sim.send_action({j: 0.0 for j in joints}, robot_name="aloha")
        # The passive finger joints cannot resolve to any actuator -> error.
        assert result["status"] == "error", result
        unresolved = next(
            c["json"]["unresolved_keys"] for c in result["content"] if isinstance(c, dict) and "json" in c
        )
        # At least one passive finger joint is dropped (it has no actuator), and
        # every dropped key is a finger joint -- i.e. exactly the joints that are
        # not action keys. This is the silent-no-op precondition.
        assert unresolved, "expected joint-keyed action to leave some keys unresolved"
        assert all("finger" in k for k in unresolved)
        assert set(unresolved).isdisjoint(sim.robot_action_keys("aloha"))

    @pytest.mark.parametrize("robot", ["xarm7", "aloha", "unitree_g1", "stretch"])
    def test_mock_policy_drives_every_actuator(self, sim, robot):
        """Mock policy moves every actuator (no silent no-op) on divergent robots."""
        sim.add_robot(robot)
        result = sim.run_policy(
            robot_name=robot,
            policy_provider="mock",
            instruction="move",
            n_steps=20,
            control_frequency=50.0,
            fast_mode=True,
        )
        assert result["status"] == "success", result
        payload = next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)
        assert payload["action_errors"] == 0
        assert payload["partial_action_failure_rate"] == 0.0
        rates = payload["action_resolution_rate"]
        # Stats are keyed by actuators and every one is driven every step.
        assert rates == {k: 1.0 for k in sim.robot_action_keys(robot)}


class TestRunMultiPolicyKeying:
    def test_each_policy_keyed_by_its_actuators(self, sim):
        """run_multi_policy keys every policy by that robot's action keys."""
        from strands_robots.policies.mock import MockPolicy

        sim.add_robot("xarm7")
        sim.add_robot("so101")
        pols = {"xarm7": MockPolicy(), "so101": MockPolicy()}
        result = sim.run_multi_policy(
            policies=pols,
            instructions="move",
            n_steps=10,
            control_frequency=50.0,
        )
        assert result["status"] == "success", result
        # Each policy was told its robot's actuator short-names, NOT the joints.
        # For xarm7 these genuinely differ (tendon gripper + passive joints).
        assert pols["xarm7"].robot_state_keys == sim.robot_action_keys("xarm7")
        assert pols["xarm7"].robot_state_keys != sim.robot_joint_names("xarm7")
        assert pols["so101"].robot_state_keys == sim.robot_action_keys("so101")


class TestRobotActionKeysDefault:
    def test_abc_default_mirrors_joint_names(self):
        """A backend that does not override robot_action_keys mirrors joints."""
        from strands_robots.simulation.base import SimEngine

        class _Stub:
            def robot_joint_names(self, robot_name: str) -> list[str]:
                return ["a", "b", "c"]

        # The ABC default delegates to robot_joint_names, so a backend whose
        # actuators match its joints needs no override.
        assert SimEngine.robot_action_keys(_Stub(), "anything") == ["a", "b", "c"]

    def test_missing_robot_returns_empty(self, sim):
        assert sim.robot_action_keys("does_not_exist") == []


class TestValidActionKeyHint:
    """The valid-key hint shown when an action key is dropped must return the
    short (prefix-stripped) form callers pass to ``send_action``.

    When a key cannot be applied, ``_warn_unresolved_action_key`` surfaces the
    actuators the scene accepts via ``_get_valid_action_keys(robot_name)``. In a
    multi-robot world actuators are namespaced (``armA/shoulder``); the hint
    must strip the active robot's prefix so the operator sees exactly the keys
    ``send_action`` expects, not the internal fully-qualified names. Unnamed
    actuators have no addressable key and are omitted from the hint.
    """

    _XML = """
    <mujoco>
      <worldbody>
        <body name="l">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom type="box" size="0.02 0.02 0.02"/>
          <body name="l2" pos="0 0 0.1">
            <joint name="j2" type="hinge" axis="0 1 0"/>
            <geom type="box" size="0.02 0.02 0.02"/>
            <body name="l3" pos="0 0 0.1">
              <joint name="j3" type="hinge" axis="1 0 0"/>
              <geom type="box" size="0.02 0.02 0.02"/>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position name="armA/shoulder" joint="j1"/>
        <position name="armA/elbow" joint="j2"/>
        <position joint="j3"/>
      </actuator>
    </mujoco>
    """

    def _mixin(self, namespace: str) -> Any:
        """A mixin over a one-robot world whose actuators carry ``namespace``."""
        import mujoco

        from strands_robots.simulation.models import SimRobot, SimWorld
        from strands_robots.simulation.mujoco.rendering import RenderingMixin

        model = mujoco.MjModel.from_xml_string(self._XML)
        robot = SimRobot(name="armA", urdf_path="armA.xml")
        robot.namespace = namespace
        robot.actuator_ids = list(range(model.nu))

        world = SimWorld()
        world.robots["armA"] = robot
        world._model = model

        mixin = RenderingMixin()
        mixin._world = world
        return mixin

    def test_prefix_is_stripped_for_matching_robot(self):
        """A robot prefix yields the short keys send_action resolves."""
        assert self._mixin("armA/")._get_valid_action_keys("armA") == ["shoulder", "elbow"]

    def test_no_prefix_returns_fully_qualified_names(self):
        """Without a prefix the raw namespaced actuator names are returned."""
        assert self._mixin("")._get_valid_action_keys("armA") == ["armA/shoulder", "armA/elbow"]


class TestActionKeysFollowJointOrder:
    """The keys a policy binds are ordered by the joints, not by the MJCF.

    ``robot_action_keys`` names the actuators, and the engine hands that list to
    ``Policy.set_robot_state_keys`` -- which orders the ``observation.state``
    vector the policy reads. A recording writes that vector in the robot's JOINT
    order. A model may declare its actuators in any order (``dynamixel_2r``
    declares ``R2`` before ``R1``), so where the two rosters were the same names
    in a different order a checkpoint was evaluated on a transposed state
    vector, silently: every guard passed, the robot moved, the numbers were
    wrong. The keys now follow the joint order, and an actuator that drives no
    single joint (a tendon gripper) keeps the slot the model declared it in.
    """

    _XML = """
    <mujoco>
      <worldbody>
        <body name="l1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom type="box" size="0.02 0.02 0.02"/>
          <body name="l2" pos="0 0 0.1">
            <joint name="j2" type="hinge" axis="0 1 0"/>
            <geom type="box" size="0.02 0.02 0.02"/>
            <body name="l3" pos="0 0 0.1">
              <joint name="j3" type="hinge" axis="1 0 0"/>
              <geom type="box" size="0.02 0.02 0.02"/>
            </body>
          </body>
        </body>
      </worldbody>
      <tendon>
        <fixed name="t"><joint joint="j3" coef="1"/></fixed>
      </tendon>
      <actuator>
        <position name="a2" joint="j2"/>
        <position name="grip" tendon="t"/>
        <position name="a1" joint="j1"/>
        <position name="a2b" joint="j2"/>
      </actuator>
    </mujoco>
    """

    def _mixin(self, joint_names: list[str]) -> Any:
        """A one-robot world whose robot declares ``joint_names`` in that order."""
        import mujoco

        from strands_robots.simulation.models import SimRobot, SimWorld
        from strands_robots.simulation.mujoco.rendering import RenderingMixin

        model = mujoco.MjModel.from_xml_string(self._XML)
        robot = SimRobot(name="armA", urdf_path="armA.xml")
        robot.joint_names = list(joint_names)
        robot.actuator_ids = list(range(model.nu))
        world = SimWorld()
        world.robots["armA"] = robot
        world._model = model
        mixin = RenderingMixin()
        mixin._world = world
        return mixin

    @pytest.mark.parametrize(
        ("joint_names", "expected"),
        [
            # The MJCF declares a2 before a1; the joint roster decides.
            (["j1", "j2", "j3"], ["a1", "grip", "a2", "a2b"]),
            # A roster that agrees with the declaration order is left alone -
            # the rule is the joint order, not a sort of the key names.
            (["j2", "j1", "j3"], ["a2", "grip", "a2b", "a1"]),
            # Two actuators on one joint keep their declared order between them.
            (["j2"], ["a2", "grip", "a1", "a2b"]),
            # A backend that knows no joint roster reports declaration order.
            ([], ["a2", "grip", "a1", "a2b"]),
        ],
    )
    def test_joint_driving_actuators_are_ranked_by_their_joint(self, joint_names, expected):
        assert self._mixin(joint_names)._get_valid_action_keys("armA") == expected

    def test_a_reversed_model_binds_the_roster_the_recording_writes(self, sim):
        """dynamixel_2r declares R2 first; state is recorded R1, R2."""
        from strands_robots.policies.mock import MockPolicy

        sim.add_robot("dynamixel_2r")
        assert sim.robot_action_keys("dynamixel_2r") == sim.robot_joint_names("dynamixel_2r") == ["R1", "R2"]
        policy = MockPolicy()
        result = sim.run_policy(robot_name="dynamixel_2r", policy_object=policy, instruction="move", n_steps=5)
        assert result["status"] == "success", result
        # The state vector the policy reads is ordered like the recorded columns.
        assert policy.robot_state_keys == sim.robot_joint_names("dynamixel_2r")
