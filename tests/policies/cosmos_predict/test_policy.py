"""Unit tests for CosmosPredictPolicy.

Tests use mocks for cosmos-predict2 to avoid GPU/model dependencies.
"""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from strands_robots.policies.cosmos_predict.policy import CosmosPredictPolicy


class TestCosmosPredictPolicyInit:
    """Test initialization and configuration."""

    def test_init_defaults(self) -> None:
        """Policy initializes with default parameters."""
        policy = CosmosPredictPolicy(server_url="http://localhost:8000")
        assert policy.provider_name == "cosmos_predict"
        assert policy._suite == "libero"
        assert policy._chunk_size == 16
        assert policy._num_denoising_steps == 5
        assert policy._action_dim == 7

    def test_init_custom_suite(self) -> None:
        """Policy accepts valid suite names."""
        for suite in ("libero", "robocasa", "aloha"):
            policy = CosmosPredictPolicy(suite=suite, server_url="http://x")
            assert policy._suite == suite

    def test_init_invalid_suite_raises(self) -> None:
        """Policy rejects invalid suite names with ValueError."""
        with pytest.raises(ValueError, match="Unknown suite 'invalid'"):
            CosmosPredictPolicy(suite="invalid")

    def test_set_robot_state_keys(self) -> None:
        """Robot state keys are stored correctly."""
        policy = CosmosPredictPolicy(server_url="http://x")
        keys = ["joint_0", "joint_1", "joint_2"]
        policy.set_robot_state_keys(keys)
        assert policy._robot_state_keys == keys

    def test_provider_name(self) -> None:
        """Provider name is cosmos_predict."""
        policy = CosmosPredictPolicy(server_url="http://x")
        assert policy.provider_name == "cosmos_predict"


class TestCosmosPredictPolicyBuildObservation:
    """Test observation format conversion."""

    def test_direct_key_mapping(self) -> None:
        """Direct camera key names are mapped without pattern search."""
        policy = CosmosPredictPolicy(server_url="http://x")
        obs_in = {
            "primary_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_image": np.ones((224, 224, 3), dtype=np.uint8) * 128,
            "proprio": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        }
        obs_out = policy._build_observation(obs_in)
        assert "primary_image" in obs_out
        assert "wrist_image" in obs_out
        assert "proprio" in obs_out
        np.testing.assert_array_equal(obs_out["primary_image"], obs_in["primary_image"])

    def test_pattern_based_search(self) -> None:
        """Camera keys are found by pattern matching."""
        policy = CosmosPredictPolicy(server_url="http://x")
        obs_in = {
            "cam_high_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "gripper_cam": np.ones((224, 224, 3), dtype=np.uint8) * 64,
        }
        obs_out = policy._build_observation(obs_in)
        # "gripper" matches wrist_image pattern
        assert "wrist_image" in obs_out

    def test_proprio_from_state_keys(self) -> None:
        """Proprioception is built from individual robot_state_keys."""
        policy = CosmosPredictPolicy(server_url="http://x")
        policy.set_robot_state_keys(["j0", "j1", "j2"])
        obs_in = {"j0": 0.1, "j1": 0.2, "j2": 0.3}
        obs_out = policy._build_observation(obs_in)
        assert "proprio" in obs_out
        np.testing.assert_allclose(obs_out["proprio"], [0.1, 0.2, 0.3])

    def test_rgba_to_rgb_conversion(self) -> None:
        """RGBA images are truncated to RGB."""
        policy = CosmosPredictPolicy(server_url="http://x")
        obs_in = {
            "primary_image": np.zeros((224, 224, 4), dtype=np.uint8),
        }
        obs_out = policy._build_observation(obs_in)
        assert obs_out["primary_image"].shape == (224, 224, 3)


class TestCosmosPredictPolicyDecodeActions:
    """Test action decoding."""

    def test_default_labels(self) -> None:
        """Actions use default 7-DoF labels when no robot_state_keys set."""
        policy = CosmosPredictPolicy(server_url="http://x")
        result = {"actions": [np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5])]}
        actions = policy._decode_actions(result)
        assert len(actions) == 1
        assert actions[0]["x"] == 1.0
        assert actions[0]["gripper"] == 0.5

    def test_custom_state_keys(self) -> None:
        """Actions use robot_state_keys when configured."""
        policy = CosmosPredictPolicy(server_url="http://x")
        policy.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4", "j5"])
        result = {"actions": [np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])]}
        actions = policy._decode_actions(result)
        assert actions[0]["j0"] == pytest.approx(0.1)
        assert actions[0]["gripper"] == pytest.approx(1.0)

    def test_multiple_actions_in_chunk(self) -> None:
        """A chunk of 16 actions is decoded correctly."""
        policy = CosmosPredictPolicy(server_url="http://x")
        raw = [np.random.randn(7).astype(np.float32) for _ in range(16)]
        result = {"actions": raw}
        actions = policy._decode_actions(result)
        assert len(actions) == 16


class TestCosmosPredictPolicyServerMode:
    """Test server-based inference."""

    @patch("requests.get")
    def test_server_health_check(self, mock_get: MagicMock) -> None:
        """Server health check is called on first use."""
        mock_get.return_value = MagicMock(status_code=200)
        policy = CosmosPredictPolicy(server_url="http://localhost:8000")
        policy._ensure_loaded()
        mock_get.assert_called_once_with("http://localhost:8000/health", timeout=5)

    @patch("requests.post")
    @patch("requests.get")
    def test_server_inference(self, mock_get: MagicMock, mock_post: MagicMock) -> None:
        """Server inference returns action dicts."""
        mock_get.return_value = MagicMock(status_code=200)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"actions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]]}
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        policy = CosmosPredictPolicy(server_url="http://localhost:8000")
        obs = {
            "primary_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "proprio": np.zeros(7, dtype=np.float32),
        }
        actions = asyncio.run(policy.get_actions(obs, "pick up cube"))
        assert len(actions) == 1
        assert actions[0]["x"] == pytest.approx(0.1)
        assert actions[0]["gripper"] == pytest.approx(0.8)


class TestCosmosPredictPolicyRegistry:
    """Test that the policy is discoverable via the registry."""

    def test_registry_import(self) -> None:
        """Policy class can be imported via the registry module path."""
        from strands_robots.policies.cosmos_predict import CosmosPredictPolicy as Cls

        assert Cls is not None
        assert Cls.__name__ == "CosmosPredictPolicy"

    def test_create_policy_with_server_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """create_policy resolves cosmos_predict provider with trust gate."""
        monkeypatch.setenv("STRANDS_TRUST_REMOTE_CODE", "1")

        from strands_robots.policies import create_policy

        policy = create_policy("cosmos_predict", server_url="http://localhost:9999")
        assert isinstance(policy, CosmosPredictPolicy)
        assert policy._server_url == "http://localhost:9999"

    def test_create_policy_blocked_without_trust(self) -> None:
        """create_policy raises UntrustedRemoteCodeError without env var."""
        import os

        # Ensure env var is NOT set
        os.environ.pop("STRANDS_TRUST_REMOTE_CODE", None)

        from strands_robots.policies.factory import UntrustedRemoteCodeError

        with pytest.raises(UntrustedRemoteCodeError):
            from strands_robots.policies import create_policy

            create_policy("cosmos_predict", server_url="http://localhost:9999")


class TestCosmosPredictPolicyReset:
    """Test reset behavior."""

    def test_reset_clears_step_counter(self) -> None:
        """reset() zeroes the step counter."""
        policy = CosmosPredictPolicy(server_url="http://x")
        policy._step = 42
        policy.reset()
        assert policy._step == 0
