"""Tests for ``strands_robots.policies.wbc.WBCPolicy``.

Tests the WBC policy with mocked ONNX sessions to verify:
- Correct observation encoding (86-dim per step, 516-dim history)
- Action shape (15-dim) and no-NaN guarantee
- Balance vs Walk auto-switch logic
- History buffer management and reset behaviour
- Integration with the policy factory
"""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np

from strands_robots.policies.wbc import WBC_JOINT_NAMES, WBCPolicy
from strands_robots.policies.wbc.wbc_policy import (
    _ACTION_DIM,
    _DEFAULT_ANGLES_15,
    _HISTORY_LEN,
    _N_JOINTS_FULL,
    _OBS_INPUT_DIM,
    _SINGLE_OBS_DIM,
    _WALK_THRESHOLD,
    _quat_rotate_inverse,
)


def _make_mock_session(output_dim: int = _ACTION_DIM) -> MagicMock:
    """Create a mock ONNX InferenceSession that returns a fixed action."""
    session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "obs"
    session.get_inputs.return_value = [mock_input]
    # Return a deterministic non-zero action
    action = np.linspace(-0.5, 0.5, output_dim).astype(np.float32).reshape(1, -1)
    session.run.return_value = [action]
    return session


def _make_obs(n_joints: int = _N_JOINTS_FULL) -> dict:
    """Create a minimal observation dict for testing."""
    return {
        "observation.state": np.zeros(n_joints, dtype=np.float32),
        "observation.velocity": np.zeros(n_joints, dtype=np.float32),
        "observation.base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "observation.base_angular_velocity": np.zeros(3, dtype=np.float32),
    }


class TestWBCPolicyConstants:
    """Verify WBC constants match the documented schema."""

    def test_joint_names_count(self):
        assert len(WBC_JOINT_NAMES) == 15

    def test_joint_names_structure(self):
        """12 leg joints + 3 waist joints."""
        leg_joints = [j for j in WBC_JOINT_NAMES if "hip" in j or "knee" in j or "ankle" in j]
        waist_joints = [j for j in WBC_JOINT_NAMES if "waist" in j]
        assert len(leg_joints) == 12
        assert len(waist_joints) == 3

    def test_default_angles_shape(self):
        assert _DEFAULT_ANGLES_15.shape == (15,)
        assert _DEFAULT_ANGLES_15.dtype == np.float32

    def test_obs_dimensions(self):
        assert _SINGLE_OBS_DIM == 86
        assert _HISTORY_LEN == 6
        assert _OBS_INPUT_DIM == 516


class TestQuatRotateInverse:
    """Test quaternion rotation utility."""

    def test_identity_quaternion(self):
        """Identity quat should not change the vector."""
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        vec = np.array([0.0, 0.0, -1.0])
        result = _quat_rotate_inverse(quat, vec)
        np.testing.assert_allclose(result, vec, atol=1e-6)

    def test_gravity_vector_upright(self):
        """Upright robot: gravity in body frame should be [0, 0, -1]."""
        quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        result = _quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        np.testing.assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-6)

    def test_90deg_rotation_about_z(self):
        """90-degree rotation about z-axis: world [1,0,0] in body = [0,-1,0]."""
        angle = np.pi / 2
        quat = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        vec = np.array([1.0, 0.0, 0.0])
        result = _quat_rotate_inverse(quat, vec)
        # Inverse of 90-deg about z applied to [1,0,0] -> [0, -1, 0]
        np.testing.assert_allclose(result, [0.0, -1.0, 0.0], atol=1e-6)

    def test_180deg_rotation_about_x(self):
        """180-degree rotation about x-axis: gravity [0,0,-1] in body = [0,0,1]."""
        angle = np.pi
        quat = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        vec = np.array([0.0, 0.0, -1.0])
        result = _quat_rotate_inverse(quat, vec)
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-6)


class TestWBCPolicyInit:
    """Test WBCPolicy initialization."""

    @patch("strands_robots.policies.wbc.wbc_policy.WBCPolicy._load_models")
    def test_default_init(self, mock_load):
        """Default construction should not raise."""
        policy = WBCPolicy()
        assert policy.provider_name == "wbc"
        assert policy.requires_images is False
        assert policy._variant == "auto"
        assert policy._device == "cpu"

    @patch("strands_robots.policies.wbc.wbc_policy.WBCPolicy._load_models")
    def test_custom_default_angles(self, mock_load):
        """Custom default angles should be accepted."""
        custom = np.ones(15, dtype=np.float32) * 0.1
        policy = WBCPolicy(default_angles=custom)
        np.testing.assert_allclose(policy._default_angles, custom)

    @patch("strands_robots.policies.wbc.wbc_policy.WBCPolicy._load_models")
    def test_variant_selection(self, mock_load):
        """Variant should be stored correctly."""
        for v in ("auto", "balance", "walk"):
            policy = WBCPolicy(variant=v)
            assert policy._variant == v


class TestWBCPolicyInference:
    """Test WBCPolicy inference with mocked ONNX sessions."""

    def _make_policy_with_mocks(self, variant: str = "auto") -> WBCPolicy:
        """Create a WBCPolicy with mocked ONNX sessions."""
        with patch("strands_robots.policies.wbc.wbc_policy.WBCPolicy._load_models"):
            policy = WBCPolicy(variant=variant)
        policy._balance_session = _make_mock_session()
        policy._walk_session = _make_mock_session()
        return policy

    def test_action_shape(self):
        """get_actions should return a list with one dict of 15 joint targets."""
        policy = self._make_policy_with_mocks()
        obs = _make_obs()
        actions = asyncio.run(policy.get_actions(obs, ""))
        assert len(actions) == 1
        assert len(actions[0]) == _ACTION_DIM

    def test_action_keys_default(self):
        """Without set_robot_state_keys, should use WBC_JOINT_NAMES."""
        policy = self._make_policy_with_mocks()
        obs = _make_obs()
        actions = asyncio.run(policy.get_actions(obs, ""))
        assert set(actions[0].keys()) == set(WBC_JOINT_NAMES)

    def test_action_keys_custom(self):
        """With set_robot_state_keys, should use those keys."""
        policy = self._make_policy_with_mocks()
        custom_keys = [f"joint_{i}" for i in range(15)]
        policy.set_robot_state_keys(custom_keys)
        obs = _make_obs()
        actions = asyncio.run(policy.get_actions(obs, ""))
        assert set(actions[0].keys()) == set(custom_keys)

    def test_no_nan_in_actions(self):
        """Actions should never contain NaN."""
        policy = self._make_policy_with_mocks()
        obs = _make_obs()
        for _ in range(20):
            actions = asyncio.run(policy.get_actions(obs, "", target_velocity=[0.5, 0, 0]))
            for action_dict in actions:
                for v in action_dict.values():
                    assert not np.isnan(v), f"NaN in action: {action_dict}"

    def test_balance_vs_walk_auto_switch(self):
        """Auto mode should select balance for zero vel, walk for non-zero."""
        policy = self._make_policy_with_mocks()
        obs = _make_obs()

        # Zero velocity -> balance session
        asyncio.run(policy.get_actions(obs, "", target_velocity=[0, 0, 0]))
        policy._balance_session.run.assert_called()

        # Reset call counts
        policy._balance_session.reset_mock()
        policy._walk_session.reset_mock()

        # Non-zero velocity -> walk session
        asyncio.run(policy.get_actions(obs, "", target_velocity=[0.5, 0, 0]))
        policy._walk_session.run.assert_called()

    def test_fixed_variant_ignores_velocity(self):
        """Fixed variant='balance' should always use balance session."""
        policy = self._make_policy_with_mocks(variant="balance")
        obs = _make_obs()

        asyncio.run(policy.get_actions(obs, "", target_velocity=[1.0, 0, 0]))
        policy._balance_session.run.assert_called()
        policy._walk_session.run.assert_not_called()

    def test_walk_threshold(self):
        """Velocity below threshold should select balance."""
        policy = self._make_policy_with_mocks()
        obs = _make_obs()

        # Just below threshold
        vel_below = [_WALK_THRESHOLD * 0.5, 0, 0]
        asyncio.run(policy.get_actions(obs, "", target_velocity=vel_below))
        policy._balance_session.run.assert_called()

        policy._balance_session.reset_mock()
        policy._walk_session.reset_mock()

        # Just above threshold
        vel_above = [_WALK_THRESHOLD * 2.0, 0, 0]
        asyncio.run(policy.get_actions(obs, "", target_velocity=vel_above))
        policy._walk_session.run.assert_called()

    def test_history_fills_correctly(self):
        """History should fill to 6 frames and maintain FIFO order."""
        policy = self._make_policy_with_mocks()
        obs = _make_obs()

        # First call: history padded from 1 to 6
        asyncio.run(policy.get_actions(obs, ""))
        assert len(policy._obs_history) == _HISTORY_LEN

        # After 10 more calls, still 6
        for _ in range(10):
            asyncio.run(policy.get_actions(obs, ""))
        assert len(policy._obs_history) == _HISTORY_LEN

    def test_reset_clears_history(self):
        """reset() should clear observation history and previous action."""
        policy = self._make_policy_with_mocks()
        obs = _make_obs()

        # Fill some history
        asyncio.run(policy.get_actions(obs, ""))
        assert len(policy._obs_history) == _HISTORY_LEN

        policy.reset()
        assert len(policy._obs_history) == 0
        np.testing.assert_array_equal(policy._prev_action, np.zeros(_ACTION_DIM))

    def test_obs_input_dimension(self):
        """ONNX should receive a 516-dim input."""
        policy = self._make_policy_with_mocks()
        obs = _make_obs()
        asyncio.run(policy.get_actions(obs, "", target_velocity=[0, 0, 0]))

        # session.run(None, {input_name: obs_array}) - positional args
        call_args = policy._balance_session.run.call_args
        input_dict = call_args[0][1]  # second positional arg is the feed dict
        obs_array = input_dict["obs"]
        assert obs_array.shape == (1, _OBS_INPUT_DIM)

    def test_sync_wrapper_works(self):
        """get_actions_sync should work from synchronous context."""
        policy = self._make_policy_with_mocks()
        obs = _make_obs()
        actions = policy.get_actions_sync(obs, "", target_velocity=[0.3, 0, 0])
        assert len(actions) == 1
        assert len(actions[0]) == _ACTION_DIM

    def test_action_values_are_offset_from_defaults(self):
        """Actions should be raw_output * scale + default_angles."""
        policy = self._make_policy_with_mocks()
        obs = _make_obs()
        actions = asyncio.run(policy.get_actions(obs, ""))

        # The mock returns linspace(-0.5, 0.5, 15)
        raw = np.linspace(-0.5, 0.5, 15).astype(np.float32)
        expected = raw * 0.25 + _DEFAULT_ANGLES_15
        for i, key in enumerate(WBC_JOINT_NAMES):
            np.testing.assert_allclose(actions[0][key], float(expected[i]), atol=1e-5, err_msg=f"Joint {key} mismatch")

    def test_no_models_returns_zero_actions(self):
        """When no ONNX models loaded, actions should be default angles."""
        with patch("strands_robots.policies.wbc.wbc_policy.WBCPolicy._load_models"):
            policy = WBCPolicy()
        # Both sessions are None (no models loaded)
        obs = _make_obs()
        actions = asyncio.run(policy.get_actions(obs, "", target_velocity=[0.5, 0, 0]))
        # Should return default angles (zero raw action * scale + defaults)
        for i, key in enumerate(WBC_JOINT_NAMES):
            np.testing.assert_allclose(actions[0][key], float(_DEFAULT_ANGLES_15[i]), atol=1e-5)


class TestWBCObservationEncoding:
    """Test the observation encoding matches the verified schema."""

    def _make_policy(self) -> WBCPolicy:
        with patch("strands_robots.policies.wbc.wbc_policy.WBCPolicy._load_models"):
            return WBCPolicy()

    def test_command_scaling(self):
        """Command vector should be scaled by [2, 2, 0.5, 1, 1, 1, 1]."""
        policy = self._make_policy()
        obs = policy._compute_single_obs(
            joint_positions=np.zeros(_N_JOINTS_FULL),
            joint_velocities=np.zeros(_N_JOINTS_FULL),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_angular_velocity=np.zeros(3),
            target_velocity=np.array([1.0, 2.0, 3.0]),
            target_height=0.74,
            target_rpy=np.array([0.1, 0.2, 0.3]),
        )
        # obs[0:7] = [vx*2, vy*2, wz*0.5, h*1, r*1, p*1, y*1]
        np.testing.assert_allclose(obs[0], 2.0, atol=1e-6)  # vx * 2
        np.testing.assert_allclose(obs[1], 4.0, atol=1e-6)  # vy * 2
        np.testing.assert_allclose(obs[2], 1.5, atol=1e-6)  # wz * 0.5
        np.testing.assert_allclose(obs[3], 0.74, atol=1e-6)  # height
        np.testing.assert_allclose(obs[4], 0.1, atol=1e-6)  # roll
        np.testing.assert_allclose(obs[5], 0.2, atol=1e-6)  # pitch
        np.testing.assert_allclose(obs[6], 0.3, atol=1e-6)  # yaw

    def test_angular_velocity_scaling(self):
        """Base angular velocity should be scaled by 0.5."""
        policy = self._make_policy()
        obs = policy._compute_single_obs(
            joint_positions=np.zeros(_N_JOINTS_FULL),
            joint_velocities=np.zeros(_N_JOINTS_FULL),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_angular_velocity=np.array([1.0, 2.0, 3.0]),
            target_velocity=np.zeros(3),
            target_height=0.74,
            target_rpy=np.zeros(3),
        )
        np.testing.assert_allclose(obs[7:10], [0.5, 1.0, 1.5], atol=1e-6)

    def test_gravity_direction_upright(self):
        """Upright robot: gravity in body frame = [0, 0, -1]."""
        policy = self._make_policy()
        obs = policy._compute_single_obs(
            joint_positions=np.zeros(_N_JOINTS_FULL),
            joint_velocities=np.zeros(_N_JOINTS_FULL),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_angular_velocity=np.zeros(3),
            target_velocity=np.zeros(3),
            target_height=0.74,
            target_rpy=np.zeros(3),
        )
        np.testing.assert_allclose(obs[10:13], [0.0, 0.0, -1.0], atol=1e-6)

    def test_joint_position_error(self):
        """Joint positions should be (pos - default) * 1.0."""
        policy = self._make_policy()
        pos = np.ones(_N_JOINTS_FULL, dtype=np.float32) * 0.5
        obs = policy._compute_single_obs(
            joint_positions=pos,
            joint_velocities=np.zeros(_N_JOINTS_FULL),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_angular_velocity=np.zeros(3),
            target_velocity=np.zeros(3),
            target_height=0.74,
            target_rpy=np.zeros(3),
        )
        expected = pos - policy._default_angles_padded
        np.testing.assert_allclose(obs[13:42], expected, atol=1e-6)

    def test_joint_velocity_scaling(self):
        """Joint velocities should be vel * 0.05."""
        policy = self._make_policy()
        vel = np.ones(_N_JOINTS_FULL, dtype=np.float32) * 2.0
        obs = policy._compute_single_obs(
            joint_positions=np.zeros(_N_JOINTS_FULL),
            joint_velocities=vel,
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_angular_velocity=np.zeros(3),
            target_velocity=np.zeros(3),
            target_height=0.74,
            target_rpy=np.zeros(3),
        )
        np.testing.assert_allclose(obs[42:71], vel * 0.05, atol=1e-6)

    def test_prev_action_slot(self):
        """Previous action should appear in obs[71:86]."""
        policy = self._make_policy()
        policy._prev_action = np.ones(_ACTION_DIM, dtype=np.float32) * 0.3
        obs = policy._compute_single_obs(
            joint_positions=np.zeros(_N_JOINTS_FULL),
            joint_velocities=np.zeros(_N_JOINTS_FULL),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_angular_velocity=np.zeros(3),
            target_velocity=np.zeros(3),
            target_height=0.74,
            target_rpy=np.zeros(3),
        )
        np.testing.assert_allclose(obs[71:86], 0.3, atol=1e-6)

    def test_obs_total_dim(self):
        """Single observation should be exactly 86-dim."""
        policy = self._make_policy()
        obs = policy._compute_single_obs(
            joint_positions=np.zeros(_N_JOINTS_FULL),
            joint_velocities=np.zeros(_N_JOINTS_FULL),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            base_angular_velocity=np.zeros(3),
            target_velocity=np.zeros(3),
            target_height=0.74,
            target_rpy=np.zeros(3),
        )
        assert obs.shape == (_SINGLE_OBS_DIM,)


class TestWBCFactoryIntegration:
    """Test WBC registration in the policy factory."""

    def test_create_wbc_from_factory(self):
        """create_policy('wbc') should return a WBCPolicy instance."""
        from strands_robots.policies import create_policy

        # This will try to load ONNX models and warn (no models available)
        # but should not raise
        policy = create_policy("wbc")
        assert isinstance(policy, WBCPolicy)
        assert policy.provider_name == "wbc"

    def test_wbc_in_providers_list(self):
        """'wbc' should appear in list_providers()."""
        from strands_robots.policies import list_providers

        providers = list_providers()
        assert "wbc" in providers

    def test_groot_wbc_shorthand(self):
        """'groot_wbc' shorthand should resolve to WBCPolicy."""
        from strands_robots.policies import create_policy

        policy = create_policy("groot_wbc")
        assert isinstance(policy, WBCPolicy)
