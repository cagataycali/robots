"""A robot's action keys are its owned actuators, whatever those are named.

``actuate_robot`` exists to make an actuator-less (URDF-loaded) arm drivable:
its docstring promises that afterwards "``send_action`` / ``run_policy`` can
drive them". It injects one position servo per joint, named
``"<robot>_act_<joint>"`` -- deliberately WITHOUT the robot's namespace
prefix, which is why
:func:`~strands_robots.simulation.mujoco.scene_ops.robot_owned_actuator_ids`
carries a second rule that recognizes them by the joint they drive.

Every surface that NAMES those actuators re-derived ownership from the
namespace prefix instead of reading that answer, so all of them reported none
of the injected servos:

* ``robot_action_keys`` returned ``[]``, so a policy bound to it could only
  ever emit an empty action and ``run_policy`` refused the rollout -- half of
  ``actuate_robot``'s promise was false,
* the refusal named ``Valid actuator/joint names: []`` and the ``send_action``
  / dropped-key hints listed nothing, on a robot with 14 actuators, and
* ``get_features`` printed ``"Actuators (14): "`` with nothing after the colon,
  contradicting its own count on the same line.

``send_action`` resolved every one of them the whole time (it falls back to the
raw name), which is what made the disagreement silent: the robot was drivable,
and only the surfaces that advertise how to drive it said otherwise.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation

# A registry humanoid whose pack ships no <actuator> block at all, so it
# compiles with nu=0 and actuate_robot is the only way to drive it.
ROBOT = "asimov_v0"


@pytest.fixture
def actuated():
    """An ``asimov_v0`` repaired by ``actuate_robot``: 14 injected servos."""
    sim = Simulation(mesh=False)
    sim.create_world()
    assert sim.add_robot(ROBOT)["status"] == "success"
    # The precondition: the model itself declares no actuator.
    assert sim.robot_action_keys(ROBOT) == []
    assert sim.actuate_robot(robot_name=ROBOT)["status"] == "success"
    yield sim
    sim.cleanup()


class TestActuatedRobotAdvertisesTheKeysThatDriveIt:
    def test_action_keys_name_every_injected_actuator(self, actuated):
        """The keys are the owned actuators, not the namespace-matching ones."""
        keys = actuated.robot_action_keys(ROBOT)
        robot = actuated._world.robots[ROBOT]
        # Ownership is the rule: one key per owned actuator, none dropped.
        assert len(keys) == len(robot.actuator_ids) == 14
        # None of them carries the robot's namespace, which is exactly why a
        # prefix scan reported zero of them.
        assert robot.namespace == f"{ROBOT}/"
        assert not any(k.startswith(robot.namespace) for k in keys)
        assert "asimov_v0_act_left_knee_joint" in keys

    def test_every_advertised_key_moves_its_actuator(self, actuated):
        """``send_action`` resolves every key, which is the documented contract."""
        keys = actuated.robot_action_keys(ROBOT)
        result = actuated.send_action(robot_name=ROBOT, action=dict.fromkeys(keys, 0.0))
        assert result["status"] == "success"
        # Not just "accepted": each key reaches a distinct ctrl slot. Writing a
        # per-key value and reading ctrl back proves the routing, so a key list
        # that merely parsed cannot pass.
        commanded = {k: 0.01 * (i + 1) for i, k in enumerate(keys)}
        assert actuated.send_action(robot_name=ROBOT, action=commanded)["status"] == "success"
        ctrl = actuated.mj_data.ctrl
        assert sorted(round(float(c), 4) for c in ctrl) == sorted(round(v, 4) for v in commanded.values())

    def test_a_policy_bound_to_those_keys_completes_a_rollout(self, actuated):
        """``run_policy`` drives the repaired robot instead of refusing it."""
        result = actuated.run_policy(robot_name=ROBOT, policy_provider="mock", instruction="stand", max_steps=20)
        # Pre-fix this was status="error": the binding produced no key, so every
        # step commanded nothing and the rollout was refused as uncommanded.
        assert result["status"] == "success", result["content"][0]["text"]

    def test_get_features_lists_the_actuators_it_counts(self, actuated):
        """``Actuators (N):`` cannot report a count with an empty name list."""
        payload = actuated.get_features(robot_name=ROBOT)["content"][1]["json"]["features"]
        assert len(payload["actuator_names"]) == payload["robots"][ROBOT]["n_actuators"] == 14
        line = next(
            ln
            for ln in actuated.get_features(robot_name=ROBOT)["content"][0]["text"].splitlines()
            if ln.startswith("Actuators (")
        )
        assert "asimov_v0_act_left_hip_pitch_joint" in line

    def test_the_dropped_key_hint_names_the_keys_that_would_work(self, actuated):
        """An unresolved key is reported with the robot's real key list."""
        result = actuated.send_action(robot_name=ROBOT, action={"joint_0": 0.1})
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        # The remedy, not an empty list: pre-fix the hint was suppressed
        # entirely because there were no keys to show.
        assert "asimov_v0_act_left_knee_joint" in text
        assert "Valid keys: []" not in text


class TestNamespacedActuatorsAreUnchanged:
    """The namespaced path keeps reporting the short keys it always did.

    Ownership and a prefix scan agree for a robot whose actuators are merged in
    by ``spec.attach(prefix=...)``, so this is the no-change half of the fix.
    """

    def test_tendon_gripper_arm_keeps_its_short_keys(self):
        sim = Simulation(mesh=False)
        sim.create_world()
        try:
            sim.add_robot("xarm7")
            keys = sim.robot_action_keys("xarm7")
            assert keys == ["act1", "act2", "act3", "act4", "act5", "act6", "act7", "gripper"]
            assert sim.send_action(robot_name="xarm7", action=dict.fromkeys(keys, 0.0))["status"] == "success"
        finally:
            sim.cleanup()
