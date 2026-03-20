"""Integration tests for lerobot_local policy — requires real model downloads.

Run explicitly: hatch run test-integ
Or: pytest tests_integ/lerobot_local/ -v --timeout=300

Requirements: lerobot>=0.5.0, internet access (HuggingFace Hub model downloads)

These tests download real models from HuggingFace Hub and run actual inference.
They are NOT run in CI by default — they require ~2GB disk for model weights
and several minutes for first-run downloads.

Models tested:
- ACT: lerobot/act_aloha_sim_transfer_cube_human (14-DOF, ~300MB)
- Diffusion: lerobot/diffusion_pusht (2-DOF, ~100MB)
"""

import asyncio
import logging
import os
import time

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# Models to test — override with env vars for custom models
ACT_MODEL = os.getenv("LEROBOT_ACT_MODEL", "lerobot/act_aloha_sim_transfer_cube_human")
DIFFUSION_MODEL = os.getenv("LEROBOT_DIFFUSION_MODEL", "lerobot/diffusion_pusht")

# Timeout for model downloads (first run can be slow)
DOWNLOAD_TIMEOUT = int(os.getenv("LEROBOT_DOWNLOAD_TIMEOUT", "300"))

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def act_policy():
    """Load ACT policy once for the entire module."""
    from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

    logger.info("Loading ACT model: %s", ACT_MODEL)
    start = time.time()
    policy = LerobotLocalPolicy(pretrained_name_or_path=ACT_MODEL)
    elapsed = time.time() - start
    logger.info("ACT model loaded in %.1fs", elapsed)

    assert policy._loaded, "ACT policy failed to load"
    assert policy.policy_type is not None, "Policy type not detected"

    yield policy


@pytest.fixture(scope="module")
def diffusion_policy():
    """Load Diffusion policy once for the entire module."""
    from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

    logger.info("Loading Diffusion model: %s", DIFFUSION_MODEL)
    start = time.time()
    policy = LerobotLocalPolicy(pretrained_name_or_path=DIFFUSION_MODEL)
    elapsed = time.time() - start
    logger.info("Diffusion model loaded in %.1fs", elapsed)

    assert policy._loaded, "Diffusion policy failed to load"
    assert policy.policy_type is not None, "Policy type not detected"

    yield policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_zero_observation(policy):
    """Build a zero observation dict matching the policy's expected features."""
    action_dim = policy._output_features["action"].shape[0]
    observation = {
        "observation.state": np.zeros(action_dim, dtype=np.float32),
    }
    for feat_name, feat_info in policy._input_features.items():
        if "image" in feat_name and hasattr(feat_info, "shape"):
            observation[feat_name] = np.zeros(feat_info.shape, dtype=np.float32)
    return observation


# ---------------------------------------------------------------------------
# Tests: ACT Policy (14-DOF, aloha sim)
# ---------------------------------------------------------------------------


class TestACTPolicy:
    """Integration tests for ACT policy with real model inference."""

    def test_model_loads_successfully(self, act_policy):
        """Model should load with correct internal state."""
        assert act_policy._loaded is True
        assert act_policy.provider_name == "lerobot_local"
        assert act_policy.pretrained_name_or_path == ACT_MODEL
        assert act_policy._policy is not None
        # Should have parameters
        n_params = sum(p.numel() for p in act_policy._policy.parameters())
        assert n_params > 0
        logger.info("ACT model: %d parameters", n_params)

    def test_policy_type_detected(self, act_policy):
        """Policy type should be auto-detected from config."""
        assert act_policy.policy_type is not None
        assert len(act_policy.policy_type) > 0
        logger.info("ACT policy type: %s", act_policy.policy_type)

    def test_output_features_detected(self, act_policy):
        """Output features (action dim) should be auto-detected."""
        assert "action" in act_policy._output_features
        action_feat = act_policy._output_features["action"]
        assert hasattr(action_feat, "shape")
        assert action_feat.shape[0] > 0
        logger.info("ACT action dim: %s", action_feat.shape)

    def test_robot_state_keys_auto_generated(self, act_policy):
        """Robot state keys should be auto-generated from action dim."""
        assert len(act_policy.robot_state_keys) > 0
        assert act_policy.robot_state_keys[0] == "joint_0"
        logger.info("ACT auto-generated %d state keys", len(act_policy.robot_state_keys))

    def test_get_actions_sync_returns_action_dicts(self, act_policy):
        """get_actions_sync should return list of action dicts with correct keys."""
        observation = _build_zero_observation(act_policy)
        actions = act_policy.get_actions_sync(observation, "pick up the cube")

        assert isinstance(actions, list), f"Expected list, got {type(actions)}"
        assert len(actions) >= 1, "Expected at least 1 action"
        assert isinstance(actions[0], dict), f"Expected dict, got {type(actions[0])}"
        assert len(actions[0]) == len(
            act_policy.robot_state_keys
        ), f"Expected {len(act_policy.robot_state_keys)} keys, got {len(actions[0])}"
        # All values should be finite floats
        for key, value in actions[0].items():
            assert isinstance(value, float), f"Key '{key}' is not float: {type(value)}"
            assert np.isfinite(value), f"Key '{key}' is not finite: {value}"
        logger.info("ACT action: %d keys, sample values: %s", len(actions[0]), list(actions[0].values())[:4])

    def test_action_values_valid(self, act_policy):
        """Action values from inference should be finite and bounded."""
        observation = _build_zero_observation(act_policy)
        actions = act_policy.get_actions_sync(observation, "pick up the cube")

        values = np.array(list(actions[0].values()))
        assert not np.any(np.isnan(values)), "Action contains NaN values"
        assert not np.any(np.isinf(values)), "Action contains Inf values"
        logger.info("ACT action range: [%.4f, %.4f]", values.min(), values.max())

    def test_async_get_actions(self, act_policy):
        """Async get_actions should work via event loop."""
        observation = _build_zero_observation(act_policy)
        actions = asyncio.run(act_policy.get_actions(observation, "test"))

        assert isinstance(actions, list)
        assert len(actions) >= 1
        assert isinstance(actions[0], dict)

    def test_explicit_robot_state_keys(self, act_policy):
        """Setting explicit state keys should override auto-generated ones."""
        action_dim = act_policy._output_features["action"].shape[0]
        custom_keys = [f"motor_{i}" for i in range(action_dim)]
        act_policy.set_robot_state_keys(custom_keys)

        observation = _build_zero_observation(act_policy)
        actions = act_policy.get_actions_sync(observation, "test")
        assert set(actions[0].keys()) == set(custom_keys)

        # Restore auto-generated keys
        act_policy.set_robot_state_keys([])

    def test_strands_format_observation(self, act_policy):
        """Policy should accept strands-robots native observation format."""
        action_dim = act_policy._output_features["action"].shape[0]
        act_policy.set_robot_state_keys([f"joint_{i}" for i in range(action_dim)])

        # Individual joint key format
        observation = {f"joint_{i}": 0.0 for i in range(action_dim)}

        # Add a dummy image for each image feature
        for feat_name, feat_info in act_policy._input_features.items():
            if "image" in feat_name and hasattr(feat_info, "shape"):
                h, w = feat_info.shape[-2], feat_info.shape[-1]
                observation["camera_top"] = np.zeros((h, w, 3), dtype=np.uint8)
                break

        actions = act_policy.get_actions_sync(observation, "test")
        assert isinstance(actions, list)
        assert len(actions) >= 1
        values = np.array(list(actions[0].values()))
        assert not np.any(np.isnan(values))

    def test_multiple_inference_calls_stable(self, act_policy):
        """Multiple inference calls should produce stable (bounded, non-NaN) results."""
        observation = _build_zero_observation(act_policy)

        for _ in range(3):
            actions = act_policy.get_actions_sync(observation, "test")
            values = np.array(list(actions[0].values()))
            assert not np.any(np.isnan(values)), "Action contains NaN"
            assert not np.any(np.isinf(values)), "Action contains Inf"
            assert np.all(np.abs(values) < 100), "Action values unreasonably large"


# ---------------------------------------------------------------------------
# Tests: Diffusion Policy (2-DOF, pusht)
# ---------------------------------------------------------------------------


class TestDiffusionPolicy:
    """Integration tests for Diffusion policy with real model inference."""

    def test_model_loads_successfully(self, diffusion_policy):
        """Model should load with correct internal state."""
        assert diffusion_policy._loaded is True
        assert diffusion_policy.provider_name == "lerobot_local"
        assert diffusion_policy.pretrained_name_or_path == DIFFUSION_MODEL
        assert diffusion_policy._policy is not None
        n_params = sum(p.numel() for p in diffusion_policy._policy.parameters())
        assert n_params > 0
        logger.info("Diffusion model: %d parameters", n_params)

    def test_policy_type_detected(self, diffusion_policy):
        """Policy type should be auto-detected from config."""
        assert diffusion_policy.policy_type is not None
        logger.info("Diffusion policy type: %s", diffusion_policy.policy_type)

    def test_output_features_detected(self, diffusion_policy):
        """Output features (action dim) should be auto-detected."""
        assert "action" in diffusion_policy._output_features
        action_feat = diffusion_policy._output_features["action"]
        assert hasattr(action_feat, "shape")
        assert action_feat.shape[0] == 2, f"Expected 2-DOF for pusht, got {action_feat.shape[0]}"
        logger.info("Diffusion action dim: %s", action_feat.shape)

    def test_get_actions_sync_returns_action_dicts(self, diffusion_policy):
        """get_actions_sync should return list of action dicts."""
        observation = _build_zero_observation(diffusion_policy)
        actions = diffusion_policy.get_actions_sync(observation, "push the T block")

        assert isinstance(actions, list)
        assert len(actions) >= 1
        assert isinstance(actions[0], dict)
        for value in actions[0].values():
            assert isinstance(value, float)
            assert np.isfinite(value)

    def test_action_values_valid(self, diffusion_policy):
        """Action values from inference should be finite and bounded."""
        observation = _build_zero_observation(diffusion_policy)
        actions = diffusion_policy.get_actions_sync(observation, "push the T block")

        values = np.array(list(actions[0].values()))
        assert not np.any(np.isnan(values)), "Action contains NaN values"
        assert not np.any(np.isinf(values)), "Action contains Inf values"
        # Diffusion policy actions should be bounded
        assert np.all(np.abs(values) < 100), f"Actions seem unreasonably large: {values}"
        logger.info("Diffusion action range: [%.4f, %.4f]", values.min(), values.max())

    def test_action_values_in_reasonable_range(self, diffusion_policy):
        """Action values should be in a reasonable range (not exploding)."""
        observation = _build_zero_observation(diffusion_policy)
        actions = diffusion_policy.get_actions_sync(observation, "push")

        values = np.array(list(actions[0].values()))
        assert np.all(np.abs(values) < 100), f"Actions seem unreasonably large: {values}"


# ---------------------------------------------------------------------------
# Tests: Factory / Smart-String Resolution
# ---------------------------------------------------------------------------


class TestFactoryResolution:
    """Test that smart-string resolution works end-to-end with real models."""

    def test_create_policy_from_smart_string(self):
        """create_policy('lerobot/act_aloha_sim_...') should auto-resolve to lerobot_local."""
        from strands_robots.policies import create_policy

        policy = create_policy(ACT_MODEL)
        assert policy.provider_name == "lerobot_local"
        assert policy._loaded is True
        assert policy.policy_type is not None

    def test_create_policy_explicit_provider(self):
        """create_policy('lerobot_local', ...) should work with explicit provider."""
        from strands_robots.policies import create_policy

        policy = create_policy("lerobot_local", pretrained_name_or_path=DIFFUSION_MODEL)
        assert policy.provider_name == "lerobot_local"
        assert policy._loaded is True

    def test_loaded_policy_metadata_complete(self):
        """Loaded policy should have all expected metadata attributes."""
        from strands_robots.policies import create_policy

        policy = create_policy(DIFFUSION_MODEL)

        assert policy._loaded is True
        assert policy.provider_name == "lerobot_local"
        assert policy.pretrained_name_or_path == DIFFUSION_MODEL
        assert policy.policy_type is not None
        assert policy._device is not None
        assert len(policy._input_features) > 0
        assert len(policy._output_features) > 0
        assert policy._policy is not None

        n_params = sum(p.numel() for p in policy._policy.parameters())
        assert n_params > 0
        logger.info("Metadata: type=%s, device=%s, params=%d", policy.policy_type, policy._device, n_params)


# ---------------------------------------------------------------------------
# Tests: Processor Bridge (real model)
# ---------------------------------------------------------------------------


class TestProcessorBridgeIntegration:
    """Test ProcessorBridge with real model configs."""

    def test_processor_bridge_from_pretrained(self):
        """ProcessorBridge should load (or gracefully skip) from real model."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        bridge = ProcessorBridge.from_pretrained(ACT_MODEL)
        info = bridge.get_info()

        # ACT may or may not have processor configs — either is valid
        assert "has_preprocessor" in info
        assert "has_postprocessor" in info
        logger.info("ACT processor bridge: %s", info)

    def test_processor_bridge_passthrough_when_no_configs(self):
        """If model has no processor configs, bridge should pass data through."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        bridge = ProcessorBridge.from_pretrained(DIFFUSION_MODEL)

        observation = {"observation.state": np.array([1.0, 2.0])}
        result = bridge.preprocess(observation)

        # If no preprocessor, should return observation unchanged
        if not bridge.has_preprocessor:
            assert result == observation


# ---------------------------------------------------------------------------
# Tests: Error Handling (real model)
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling with real loaded models."""

    def test_invalid_model_path_raises(self):
        """Loading a nonexistent model should raise."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        with pytest.raises((ValueError, ImportError, OSError, RuntimeError)):
            LerobotLocalPolicy(pretrained_name_or_path="completely/nonexistent-model-path-xyz")

    def test_inference_error_propagates(self, act_policy):
        """Inference errors should propagate immediately."""
        original_select_action = act_policy._policy.select_action
        act_policy._policy.select_action = lambda batch: (_ for _ in ()).throw(RuntimeError("test failure"))

        observation = {"observation.state": np.zeros(14, dtype=np.float32)}

        with pytest.raises(RuntimeError, match="test failure"):
            act_policy.get_actions_sync(observation, "test")

        # Restore
        act_policy._policy.select_action = original_select_action
