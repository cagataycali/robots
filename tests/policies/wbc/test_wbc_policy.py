"""Unit tests for WBCPolicy (mocked ONNX, no network)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np

from strands_robots.policies.wbc.wbc_policy import (
    _DEFAULT_CONFIG,
    WBCPolicy,
    _quat_rotate_inverse,
)


class TestQuatRotateInverse:
    """Test the quaternion rotation utility."""

    def test_identity_quaternion(self):
        """Identity quat [1,0,0,0] should not rotate the vector."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 3.0])
        result = _quat_rotate_inverse(q, v)
        np.testing.assert_allclose(result, v, atol=1e-6)

    def test_gravity_orientation_upright(self):
        """Upright robot: gravity in body frame should be [0, 0, -1]."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        gravity_world = np.array([0.0, 0.0, -1.0])
        result = _quat_rotate_inverse(q, gravity_world)
        np.testing.assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-6)


class TestWBCPolicyInit:
    """Test WBCPolicy initialization."""

    def test_default_init(self):
        policy = WBCPolicy()
        assert policy.provider_name == "wbc"
        assert policy.requires_images is False
        assert not policy._loaded
        np.testing.assert_array_equal(policy._target_velocity, [0.0, 0.0, 0.0])

    def test_custom_velocity(self):
        policy = WBCPolicy(target_velocity=[0.5, 0.0, 0.3])
        np.testing.assert_allclose(policy._target_velocity, [0.5, 0.0, 0.3])

    def test_reset(self):
        policy = WBCPolicy()
        policy._last_action = np.ones(15)
        policy._initialized = True
        policy.reset()
        np.testing.assert_array_equal(policy._last_action, np.zeros(15))
        assert not policy._initialized

    def test_set_target_velocity(self):
        policy = WBCPolicy()
        policy.set_target_velocity([1.0, -0.5, 0.2])
        np.testing.assert_allclose(policy._target_velocity, [1.0, -0.5, 0.2])

    def test_set_robot_state_keys(self):
        policy = WBCPolicy()
        keys = ["joint_a", "joint_b"]
        policy.set_robot_state_keys(keys)
        assert policy._robot_state_keys == keys


class TestWBCPolicyObservation:
    """Test observation computation."""

    def test_compute_single_obs_shape(self):
        policy = WBCPolicy(target_velocity=[0.5, 0.0, 0.0])
        # 29 DOF G1: qpos = 7 (free) + 29 = 36, qvel = 6 + 29 = 35
        n_joints = 29
        qpos = np.zeros(7 + n_joints, dtype=np.float32)
        qpos[3] = 1.0  # w of quaternion
        qvel = np.zeros(6 + n_joints, dtype=np.float32)

        obs = policy._compute_single_obs(qpos, qvel, n_joints)
        assert obs.shape == (86,)
        assert obs.dtype == np.float32

    def test_compute_single_obs_command_scaling(self):
        """Verify command is scaled by cmd_scale."""
        policy = WBCPolicy(target_velocity=[1.0, 0.5, 0.3])
        n_joints = 29
        qpos = np.zeros(7 + n_joints, dtype=np.float32)
        qpos[3] = 1.0
        qvel = np.zeros(6 + n_joints, dtype=np.float32)

        obs = policy._compute_single_obs(qpos, qvel, n_joints)
        # cmd_scale = [2.0, 2.0, 0.5]
        np.testing.assert_allclose(obs[0], 1.0 * 2.0)  # vx * 2
        np.testing.assert_allclose(obs[1], 0.5 * 2.0)  # vy * 2
        np.testing.assert_allclose(obs[2], 0.3 * 0.5)  # wz * 0.5


class TestWBCPolicyInference:
    """Test inference with mocked ONNX sessions."""

    def test_get_actions_without_wbc_keys(self):
        """Without _wbc_qpos/qvel, returns current target_dof_pos."""
        policy = WBCPolicy()
        # Mock the ONNX loading
        policy._loaded = True
        policy._balance_session = MagicMock()
        policy._walk_session = MagicMock()

        obs_dict = {"some_joint": 0.1}
        actions = asyncio.run(policy.get_actions(obs_dict, "walk"))

        assert len(actions) == 1
        assert "_target_dof_pos" in actions[0]
        np.testing.assert_allclose(
            actions[0]["_target_dof_pos"],
            np.array(_DEFAULT_CONFIG["default_angles"], dtype=np.float32),
        )

    def test_get_actions_with_wbc_keys(self):
        """With _wbc_qpos/qvel, runs inference and updates targets."""
        policy = WBCPolicy(target_velocity=[0.5, 0.0, 0.0])
        policy._loaded = True

        # Mock ONNX sessions
        mock_output = np.zeros((1, 15), dtype=np.float32)
        mock_output[0, 0] = 0.1  # Small action for first joint

        mock_walk = MagicMock()
        mock_walk.run.return_value = [mock_output]
        mock_walk.get_inputs.return_value = [MagicMock(name="obs")]

        policy._walk_session = mock_walk
        policy._walk_input_name = "obs"
        policy._balance_session = MagicMock()
        policy._balance_input_name = "obs"

        n_joints = 29
        qpos = np.zeros(7 + n_joints, dtype=np.float32)
        qpos[3] = 1.0
        qpos[2] = 0.793  # standing height
        qvel = np.zeros(6 + n_joints, dtype=np.float32)

        obs_dict = {
            "_wbc_qpos": qpos,
            "_wbc_qvel": qvel,
            "_wbc_n_joints": n_joints,
        }

        actions = asyncio.run(policy.get_actions(obs_dict, "walk forward"))

        assert len(actions) == 1
        target = actions[0]["_target_dof_pos"]
        # First joint: 0.1 * action_scale(0.25) + default(-0.1) = -0.075
        expected_first = 0.1 * 0.25 + (-0.1)
        np.testing.assert_allclose(target[0], expected_first, atol=1e-6)

        # Walk session should have been called (velocity > 0.05)
        mock_walk.run.assert_called_once()

    def test_balance_vs_walk_switch(self):
        """Zero velocity uses balance; non-zero uses walk."""
        policy = WBCPolicy(target_velocity=[0.0, 0.0, 0.0])
        policy._loaded = True

        mock_balance = MagicMock()
        mock_balance.run.return_value = [np.zeros((1, 15), dtype=np.float32)]
        mock_walk = MagicMock()
        mock_walk.run.return_value = [np.zeros((1, 15), dtype=np.float32)]

        policy._balance_session = mock_balance
        policy._walk_session = mock_walk
        policy._balance_input_name = "obs"
        policy._walk_input_name = "obs"

        n_joints = 29
        qpos = np.zeros(7 + n_joints, dtype=np.float32)
        qpos[3] = 1.0
        qvel = np.zeros(6 + n_joints, dtype=np.float32)
        obs_dict = {"_wbc_qpos": qpos, "_wbc_qvel": qvel, "_wbc_n_joints": n_joints}

        # Zero velocity -> balance
        asyncio.run(policy.get_actions(obs_dict, ""))
        mock_balance.run.assert_called_once()
        mock_walk.run.assert_not_called()

        # Non-zero velocity -> walk
        mock_balance.run.reset_mock()
        policy.set_target_velocity([0.5, 0.0, 0.0])
        asyncio.run(policy.get_actions(obs_dict, ""))
        mock_walk.run.assert_called_once()


class TestWBCActionController:
    """Test the action controller."""

    def test_controller_creation(self):
        policy = WBCPolicy()
        controller = policy.get_action_controller()
        assert controller.owns_stepping is True
        assert controller.num_actions == 15
        assert controller.control_decimation == 4


class TestWBCPolicyFactory:
    """Test that WBC is accessible via create_policy."""

    def test_create_policy_wbc(self):
        from strands_robots.policies import create_policy

        policy = create_policy("wbc")
        assert policy.provider_name == "wbc"
        assert isinstance(policy, WBCPolicy)

    def test_create_policy_wbc_with_velocity(self):
        from strands_robots.policies import create_policy

        policy = create_policy("wbc", target_velocity=[0.3, 0.0, 0.0])
        np.testing.assert_allclose(policy._target_velocity, [0.3, 0.0, 0.0])
