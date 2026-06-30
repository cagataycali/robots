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
