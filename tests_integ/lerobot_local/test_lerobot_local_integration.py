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

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.lerobot_local.processor import ProcessorBridge

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


def _assert_valid_actions(actions, expected_key_count):
    """Assert that actions list is well-formed: list of dicts with finite float values."""
    assert isinstance(actions, list), f"Expected list, got {type(actions)}"
    assert len(actions) >= 1, "Expected at least 1 action"
    assert isinstance(actions[0], dict), f"Expected dict, got {type(actions[0])}"
    assert len(actions[0]) == expected_key_count, f"Expected {expected_key_count} keys, got {len(actions[0])}"
    values = np.array([v for a in actions for v in a.values()])
    assert np.all(np.isfinite(values)), f"Non-finite values in actions: {values}"
    assert np.all(np.abs(values) < 100), f"Unreasonably large action values: {values}"


# ---------------------------------------------------------------------------
# Tests: Full ACT Pipeline (load → configure → infer → validate)
# ---------------------------------------------------------------------------


class TestACTFullPipeline:
    """Behavioral e2e tests for ACT policy: load a real model, run inference, validate output."""

    def test_load_and_infer_zero_observation(self, act_policy):
        """Full pipeline: loaded model → zero obs → get actions → valid output."""
        assert act_policy._loaded is True
        assert act_policy.provider_name == "lerobot_local"
        assert act_policy._policy is not None
        n_params = sum(p.numel() for p in act_policy._policy.parameters())
        assert n_params > 0

        observation = _build_zero_observation(act_policy)
        actions = act_policy.get_actions_sync(observation, "pick up the cube")
        _assert_valid_actions(actions, len(act_policy.robot_state_keys))
        logger.info(
            "ACT: %d params, type=%s, action_dim=%d, got %d actions",
            n_params,
            act_policy.policy_type,
            len(act_policy.robot_state_keys),
            len(actions),
        )

    def test_custom_state_keys_respected(self, act_policy):
        """Setting explicit state keys should change action dict keys."""
        action_dim = act_policy._output_features["action"].shape[0]
        custom_keys = [f"motor_{i}" for i in range(action_dim)]
        act_policy.set_robot_state_keys(custom_keys)

        observation = _build_zero_observation(act_policy)
        actions = act_policy.get_actions_sync(observation, "test")
        assert set(actions[0].keys()) == set(custom_keys)

        # Restore auto-generated keys
        act_policy.set_robot_state_keys([])

    def test_strands_format_observation(self, act_policy):
        """Policy should accept strands-robots native observation format (individual keys)."""
        action_dim = act_policy._output_features["action"].shape[0]
        act_policy.set_robot_state_keys([f"joint_{i}" for i in range(action_dim)])

        observation = {f"joint_{i}": 0.0 for i in range(action_dim)}

        # Add a dummy image for each image feature
        for feat_name, feat_info in act_policy._input_features.items():
            if "image" in feat_name and hasattr(feat_info, "shape"):
                h, w = feat_info.shape[-2], feat_info.shape[-1]
                observation["camera_top"] = np.zeros((h, w, 3), dtype=np.uint8)
                break

        actions = act_policy.get_actions_sync(observation, "test")
        assert isinstance(actions, list) and len(actions) >= 1
        values = np.array(list(actions[0].values()))
        assert np.all(np.isfinite(values))

    def test_async_interface(self, act_policy):
        """Async get_actions should produce the same kind of output."""
        observation = _build_zero_observation(act_policy)
        actions = asyncio.run(act_policy.get_actions(observation, "test"))
        _assert_valid_actions(actions, len(act_policy.robot_state_keys))

    def test_multiple_calls_stable(self, act_policy):
        """Multiple inference calls should produce stable (bounded, non-NaN) results."""
        observation = _build_zero_observation(act_policy)
        for _ in range(3):
            actions = act_policy.get_actions_sync(observation, "test")
            values = np.array(list(actions[0].values()))
            assert np.all(np.isfinite(values))
            assert np.all(np.abs(values) < 100)


# ---------------------------------------------------------------------------
# Tests: Full Diffusion Pipeline
# ---------------------------------------------------------------------------


class TestDiffusionFullPipeline:
    """Behavioral e2e tests for Diffusion policy."""

    def test_load_and_infer_zero_observation(self, diffusion_policy):
        """Full pipeline: loaded model → zero obs → get actions → valid 2-DOF output."""
        assert diffusion_policy._loaded is True
        assert diffusion_policy._output_features["action"].shape[0] == 2

        observation = _build_zero_observation(diffusion_policy)
        actions = diffusion_policy.get_actions_sync(observation, "push the T block")
        _assert_valid_actions(actions, len(diffusion_policy.robot_state_keys))
        logger.info(
            "Diffusion: type=%s, action_dim=%d, got %d actions",
            diffusion_policy.policy_type,
            len(diffusion_policy.robot_state_keys),
            len(actions),
        )


# ---------------------------------------------------------------------------
# Tests: Factory / Smart-String Resolution (e2e)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: ProcessorBridge (e2e with real model configs)
# ---------------------------------------------------------------------------


class TestProcessorBridgeIntegration:
    """Test ProcessorBridge with real model configs."""

    def test_processor_bridge_loads_from_real_model(self):
        """ProcessorBridge should load (or gracefully skip) from a real model path."""
        bridge = ProcessorBridge.from_pretrained(ACT_MODEL)
        info = bridge.get_info()

        # ACT may or may not have processor configs — either is valid
        assert "has_preprocessor" in info
        assert "has_postprocessor" in info
        logger.info("ACT processor bridge: %s", info)

    def test_processor_bridge_passthrough_when_no_configs(self):
        """If model has no processor configs, bridge should pass data through unchanged."""
        bridge = ProcessorBridge.from_pretrained(DIFFUSION_MODEL)

        observation = {"observation.state": np.array([1.0, 2.0])}
        result = bridge.preprocess(observation)

        # If no preprocessor, should return observation unchanged
        if not bridge.has_preprocessor:
            assert result == observation

    def test_processor_bridge_active_model(self):
        """If a model ships processor configs, the bridge should be active and functional.

        NOTE: This test is a placeholder — currently ACT and Diffusion don't ship
        processor configs. When a model that does is added (e.g., a VLA model),
        update this test with that model ID.
        """
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        # Try ACT — if it happens to have processor configs, test them
        bridge = ProcessorBridge.from_pretrained(ACT_MODEL)
        if bridge.is_active:
            logger.info("ACT has active processor bridge — testing round-trip")
            observation = _build_zero_observation(
                # Need a policy to get features — create one
                __import__(
                    "strands_robots.policies.lerobot_local.policy",
                    fromlist=["LerobotLocalPolicy"],
                ).LerobotLocalPolicy(pretrained_name_or_path=ACT_MODEL)
            )
            result = bridge.preprocess(observation)
            assert result is not None
        else:
            pytest.skip("ACT model does not ship processor configs")


# ---------------------------------------------------------------------------
# Tests: RTC (Real-Time Chunking) e2e
# ---------------------------------------------------------------------------


class TestRTCIntegration:
    """Integration tests for Real-Time Chunking with real models.

    RTC requires a model that supports predict_action_chunk (flow-matching models
    like Pi0, Pi0.5, SmolVLA). Standard ACT/Diffusion models do NOT support RTC.

    This test validates that:
    1. RTC correctly auto-detects support from model capabilities
    2. When supported, the RTC inference path produces valid actions
    3. When not supported, RTC gracefully disables itself
    """

    # Override with env var if you have a flow-matching model to test
    RTC_MODEL = os.getenv("LEROBOT_RTC_MODEL", "")

    def test_rtc_auto_disabled_for_act(self, act_policy):
        """ACT has no rtc_config — RTC should be auto-disabled."""
        assert act_policy._rtc_enabled is False
        logger.info("ACT RTC status: disabled (expected — no rtc_config)")

    def test_rtc_auto_disabled_for_diffusion(self, diffusion_policy):
        """Diffusion has no rtc_config — RTC should be auto-disabled."""
        assert diffusion_policy._rtc_enabled is False
        logger.info("Diffusion RTC status: disabled (expected — no rtc_config)")

    @pytest.mark.skipif(
        not os.getenv("LEROBOT_RTC_MODEL"),
        reason="Set LEROBOT_RTC_MODEL env var to test RTC with a real flow-matching model",
    )
    def test_rtc_full_pipeline_with_real_model(self):
        """Full RTC pipeline: load flow-matching model → infer with RTC → validate."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        logger.info("Loading RTC model: %s", self.RTC_MODEL)
        policy = LerobotLocalPolicy(
            pretrained_name_or_path=self.RTC_MODEL,
            rtc_enabled=True,
        )
        assert policy._loaded is True
        assert policy._rtc_enabled is True, "RTC should be enabled for flow-matching model"

        observation = _build_zero_observation(policy)
        actions = policy.get_actions_sync(observation, "pick up the object")
        _assert_valid_actions(actions, len(policy.robot_state_keys))

        # Verify RTC state was populated
        assert policy._rtc_prev_chunk is not None, "RTC should store leftover chunk"
        assert len(policy._rtc_latency_history) == 1, "RTC should track latency"

        # Second call should use prev_chunk
        actions2 = policy.get_actions_sync(observation, "pick up the object")
        _assert_valid_actions(actions2, len(policy.robot_state_keys))
        assert len(policy._rtc_latency_history) == 2

        logger.info("RTC: 2 calls successful, latencies: %s", policy._rtc_latency_history)


# ---------------------------------------------------------------------------
# Tests: Error Handling (e2e)
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling with real loaded models."""

    def test_invalid_model_path_raises(self):
        """Loading a nonexistent model should raise immediately."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        with pytest.raises((ValueError, ImportError, OSError, RuntimeError)):
            LerobotLocalPolicy(pretrained_name_or_path="completely/nonexistent-model-path-xyz")

    def test_inference_error_propagates(self, act_policy):
        """Inference errors should propagate, not be silently swallowed."""
        original_select_action = act_policy._policy.select_action
        act_policy._policy.select_action = lambda batch: (_ for _ in ()).throw(RuntimeError("test failure"))

        # Build a complete observation (state + required images) so the error
        # reaches select_action rather than failing at batch-building.
        observation = {"observation.state": np.zeros(14, dtype=np.float32)}
        for feat_name in act_policy._input_features:
            if "image" in feat_name and feat_name not in observation:
                observation[feat_name] = np.zeros((3, 480, 640), dtype=np.float32)

        with pytest.raises(RuntimeError, match="test failure"):
            act_policy.get_actions_sync(observation, "test")

        # Restore
        act_policy._policy.select_action = original_select_action
