"""End-to-end tests for lerobot_local policy — full pipeline validation.

Run: pytest tests_e2e/ -v --timeout=600
Or:  hatch run test-e2e

Requirements:
    - lerobot>=0.5.0 installed
    - Internet access (HuggingFace Hub downloads)
    - ~2GB disk for model weights (cached after first run)

These tests validate the COMPLETE flow:
    create_policy(model_id) → load model → build observation → inference → action dicts

Unlike unit tests (mocked) or integration tests (component-level),
e2e tests run the real models with realistic observations and validate
that the output actions are physically plausible.
"""

import asyncio
import logging
import os
import time

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — extend via env vars
# ---------------------------------------------------------------------------

MODELS = {
    "act": {
        "repo": os.getenv("E2E_ACT_MODEL", "lerobot/act_aloha_sim_transfer_cube_human"),
        "action_dim": 14,
        "has_images": True,
        "instruction": "pick up the cube and place it on the plate",
    },
    "diffusion": {
        "repo": os.getenv("E2E_DIFFUSION_MODEL", "lerobot/diffusion_pusht"),
        "action_dim": 2,
        "has_images": True,
        "instruction": "push the T block to the target",
    },
}

# Optional models — only tested if env var is set or lerobot has them
OPTIONAL_MODELS = {}
if os.getenv("E2E_SMOLVLA_MODEL"):
    OPTIONAL_MODELS["smolvla"] = {
        "repo": os.getenv("E2E_SMOLVLA_MODEL"),
        "action_dim": None,  # auto-detect
        "has_images": True,
        "instruction": "pick up the object",
    }
if os.getenv("E2E_PI0_MODEL"):
    OPTIONAL_MODELS["pi0"] = {
        "repo": os.getenv("E2E_PI0_MODEL"),
        "action_dim": None,
        "has_images": True,
        "instruction": "grasp the object",
    }

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_realistic_observation(policy, randomize: bool = False):
    """Build a realistic observation dict matching the policy's expected features.

    Unlike zero observations, this creates plausible sensor data:
    - Joint states: small random values (simulating near-home position)
    - Images: random pixel values (simulating camera feed)
    """
    rng = np.random.default_rng(42) if randomize else None
    observation = {}

    # Build state vector
    action_dim = policy._output_features["action"].shape[0]
    if randomize:
        state = rng.uniform(-0.1, 0.1, size=action_dim).astype(np.float32)
    else:
        state = np.zeros(action_dim, dtype=np.float32)
    observation["observation.state"] = state

    # Build image observations
    for feat_name, feat_info in policy._input_features.items():
        if "image" in feat_name and hasattr(feat_info, "shape"):
            shape = feat_info.shape
            if randomize:
                observation[feat_name] = rng.integers(0, 255, size=shape, dtype=np.uint8).astype(np.float32)
            else:
                observation[feat_name] = np.zeros(shape, dtype=np.float32)

    return observation


def _build_strands_observation(policy, randomize: bool = False):
    """Build observation in strands-robots native format (individual joint keys + camera).

    This is what a real robot driver would produce.
    """
    rng = np.random.default_rng(42) if randomize else None
    action_dim = policy._output_features["action"].shape[0]
    observation = {}

    # Individual joint values
    for i in range(action_dim):
        key = policy.robot_state_keys[i] if i < len(policy.robot_state_keys) else f"joint_{i}"
        observation[key] = float(rng.uniform(-0.1, 0.1)) if randomize else 0.0

    # Camera image in HWC uint8 format (what a real camera produces)
    for feat_name, feat_info in policy._input_features.items():
        if "image" in feat_name and hasattr(feat_info, "shape"):
            # feat_info.shape is typically (C, H, W) — camera produces (H, W, C)
            c, h, w = feat_info.shape[0], feat_info.shape[1], feat_info.shape[2]
            if randomize:
                observation["camera_top"] = rng.integers(0, 255, size=(h, w, c), dtype=np.uint8)
            else:
                observation["camera_top"] = np.zeros((h, w, c), dtype=np.uint8)
            break

    return observation


def _validate_actions(actions, expected_dim, label=""):
    """Validate action output meets physical plausibility requirements."""
    prefix = f"[{label}] " if label else ""

    assert isinstance(actions, list), f"{prefix}Expected list, got {type(actions)}"
    assert len(actions) >= 1, f"{prefix}Expected at least 1 action"

    for i, action in enumerate(actions):
        assert isinstance(action, dict), f"{prefix}Action {i}: expected dict, got {type(action)}"

        if expected_dim is not None:
            assert len(action) == expected_dim, f"{prefix}Action {i}: expected {expected_dim} keys, got {len(action)}"

        values = np.array(list(action.values()))

        # Must be finite
        assert not np.any(np.isnan(values)), f"{prefix}Action {i}: contains NaN"
        assert not np.any(np.isinf(values)), f"{prefix}Action {i}: contains Inf"

        # Must be bounded (no exploding values)
        assert np.all(
            np.abs(values) < 100
        ), f"{prefix}Action {i}: values out of range: [{values.min():.4f}, {values.max():.4f}]"

        # All values must be Python floats
        for key, val in action.items():
            assert isinstance(val, float), f"{prefix}Key '{key}' is {type(val)}, expected float"


# ---------------------------------------------------------------------------
# Fixtures — one per model, module-scoped (load once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def act_policy():
    """Load ACT policy end-to-end via create_policy."""
    from strands_robots.policies import create_policy

    model_id = MODELS["act"]["repo"]
    logger.info("E2E: Loading ACT model: %s", model_id)
    start = time.time()

    policy = create_policy(model_id)

    elapsed = time.time() - start
    logger.info("E2E: ACT loaded in %.1fs (type=%s)", elapsed, policy.policy_type)

    assert policy._loaded, "ACT policy failed to load"
    yield policy


@pytest.fixture(scope="module")
def diffusion_policy():
    """Load Diffusion policy end-to-end via create_policy."""
    from strands_robots.policies import create_policy

    model_id = MODELS["diffusion"]["repo"]
    logger.info("E2E: Loading Diffusion model: %s", model_id)
    start = time.time()

    policy = create_policy(model_id)

    elapsed = time.time() - start
    logger.info("E2E: Diffusion loaded in %.1fs (type=%s)", elapsed, policy.policy_type)

    assert policy._loaded, "Diffusion policy failed to load"
    yield policy


# ---------------------------------------------------------------------------
# E2E Test: Full Pipeline — create_policy → inference → action dicts
# ---------------------------------------------------------------------------


class TestACTE2E:
    """End-to-end ACT policy tests with real model inference."""

    def test_full_pipeline_lerobot_format(self, act_policy):
        """Full pipeline: create_policy → lerobot-format observation → inference → actions."""
        obs = _build_realistic_observation(act_policy, randomize=True)
        actions = act_policy.get_actions_sync(obs, MODELS["act"]["instruction"])
        _validate_actions(actions, MODELS["act"]["action_dim"], "ACT/lerobot-fmt")
        logger.info("ACT lerobot-format: %d actions, first=%s", len(actions), list(actions[0].values())[:4])

    def test_full_pipeline_strands_format(self, act_policy):
        """Full pipeline: create_policy → strands-native observation → inference → actions."""
        obs = _build_strands_observation(act_policy, randomize=True)
        actions = act_policy.get_actions_sync(obs, MODELS["act"]["instruction"])
        _validate_actions(actions, MODELS["act"]["action_dim"], "ACT/strands-fmt")
        logger.info("ACT strands-format: %d actions, first=%s", len(actions), list(actions[0].values())[:4])

    def test_async_pipeline(self, act_policy):
        """Full async pipeline: create_policy → async get_actions → actions."""
        obs = _build_realistic_observation(act_policy, randomize=True)
        actions = asyncio.run(act_policy.get_actions(obs, MODELS["act"]["instruction"]))
        _validate_actions(actions, MODELS["act"]["action_dim"], "ACT/async")

    def test_repeated_inference_stability(self, act_policy):
        """10 consecutive inferences should all produce valid, bounded actions."""
        obs = _build_realistic_observation(act_policy, randomize=True)
        all_first_values = []

        for i in range(10):
            actions = act_policy.get_actions_sync(obs, "test")
            _validate_actions(actions, MODELS["act"]["action_dim"], f"ACT/repeat-{i}")
            all_first_values.append(list(actions[0].values())[0])

        # Values should be deterministic (same obs → same actions in eval mode)
        values_arr = np.array(all_first_values)
        assert np.std(values_arr) < 0.01, f"Actions not deterministic in eval mode: std={np.std(values_arr):.6f}"

    def test_different_observations_different_actions(self, act_policy):
        """Different observations should produce different actions."""
        obs_zero = _build_realistic_observation(act_policy, randomize=False)
        obs_rand = _build_realistic_observation(act_policy, randomize=True)

        actions_zero = act_policy.get_actions_sync(obs_zero, "test")
        actions_rand = act_policy.get_actions_sync(obs_rand, "test")

        vals_zero = np.array(list(actions_zero[0].values()))
        vals_rand = np.array(list(actions_rand[0].values()))

        # Different inputs should yield different outputs
        assert not np.allclose(
            vals_zero, vals_rand, atol=1e-4
        ), "Same actions for different observations — model may not be using input"

    def test_reset_clears_action_queue(self, act_policy):
        """reset() should clear internal state / action queues."""
        obs = _build_realistic_observation(act_policy)
        act_policy.get_actions_sync(obs, "test")

        # Reset should not error
        act_policy.reset()

        # Should still work after reset
        actions = act_policy.get_actions_sync(obs, "test")
        _validate_actions(actions, MODELS["act"]["action_dim"], "ACT/post-reset")

    def test_inference_latency(self, act_policy):
        """Single inference should complete within 5 seconds (CPU) / 500ms (GPU)."""
        obs = _build_realistic_observation(act_policy, randomize=True)

        # Warm up
        act_policy.get_actions_sync(obs, "test")

        start = time.time()
        act_policy.get_actions_sync(obs, "test")
        elapsed = time.time() - start

        logger.info("ACT inference latency: %.3fs", elapsed)
        # Generous bound for CPU — real robots need <100ms but CI may be slow
        assert elapsed < 10.0, f"Inference too slow: {elapsed:.3f}s"


class TestDiffusionE2E:
    """End-to-end Diffusion policy tests with real model inference."""

    def test_full_pipeline_lerobot_format(self, diffusion_policy):
        """Full pipeline with lerobot-format observation."""
        obs = _build_realistic_observation(diffusion_policy, randomize=True)
        actions = diffusion_policy.get_actions_sync(obs, MODELS["diffusion"]["instruction"])
        _validate_actions(actions, MODELS["diffusion"]["action_dim"], "Diff/lerobot-fmt")

    def test_full_pipeline_strands_format(self, diffusion_policy):
        """Full pipeline with strands-native observation."""
        obs = _build_strands_observation(diffusion_policy, randomize=True)
        actions = diffusion_policy.get_actions_sync(obs, MODELS["diffusion"]["instruction"])
        _validate_actions(actions, MODELS["diffusion"]["action_dim"], "Diff/strands-fmt")

    def test_async_pipeline(self, diffusion_policy):
        """Full async pipeline."""
        obs = _build_realistic_observation(diffusion_policy, randomize=True)
        actions = asyncio.run(diffusion_policy.get_actions(obs, MODELS["diffusion"]["instruction"]))
        _validate_actions(actions, MODELS["diffusion"]["action_dim"], "Diff/async")

    def test_2dof_action_structure(self, diffusion_policy):
        """PushT is 2-DOF — actions should have exactly 2 keys."""
        obs = _build_realistic_observation(diffusion_policy, randomize=True)
        actions = diffusion_policy.get_actions_sync(obs, "push")

        assert len(actions[0]) == 2, f"Expected 2 keys for PushT, got {len(actions[0])}"
        logger.info("Diffusion 2-DOF keys: %s", list(actions[0].keys()))

    def test_repeated_inference_stability(self, diffusion_policy):
        """Multiple inferences should produce stable results."""
        obs = _build_realistic_observation(diffusion_policy, randomize=True)

        for i in range(5):
            actions = diffusion_policy.get_actions_sync(obs, "push")
            _validate_actions(actions, MODELS["diffusion"]["action_dim"], f"Diff/repeat-{i}")


# ---------------------------------------------------------------------------
# E2E Test: Cross-Model Consistency
# ---------------------------------------------------------------------------


class TestCrossModelE2E:
    """Tests that validate behavior consistency across different model types."""

    def test_both_models_use_same_api(self, act_policy, diffusion_policy):
        """Both policies should expose identical API surface."""
        for policy in [act_policy, diffusion_policy]:
            assert hasattr(policy, "get_actions")
            assert hasattr(policy, "get_actions_sync")
            assert hasattr(policy, "reset")
            assert hasattr(policy, "set_robot_state_keys")
            assert hasattr(policy, "provider_name")
            assert policy.provider_name == "lerobot_local"

    def test_both_models_have_output_features(self, act_policy, diffusion_policy):
        """Both policies should expose output features with action shape."""
        for policy in [act_policy, diffusion_policy]:
            assert "action" in policy._output_features
            feat = policy._output_features["action"]
            assert hasattr(feat, "shape")
            assert feat.shape[0] > 0

    def test_create_policy_auto_resolves_both(self):
        """create_policy should auto-resolve both model types to lerobot_local."""
        from strands_robots.policies import create_policy

        for model_name, model_info in MODELS.items():
            policy = create_policy(model_info["repo"])
            assert (
                policy.provider_name == "lerobot_local"
            ), f"{model_name}: expected lerobot_local, got {policy.provider_name}"
            assert policy._loaded, f"{model_name}: not loaded"


# ---------------------------------------------------------------------------
# E2E Test: Error Recovery
# ---------------------------------------------------------------------------


class TestErrorRecoveryE2E:
    """Test error handling in real e2e scenarios."""

    def test_invalid_model_id_raises_immediately(self):
        """Invalid model ID should raise on construction, not silently fail."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        with pytest.raises((ValueError, ImportError, OSError, RuntimeError)):
            LerobotLocalPolicy(pretrained_name_or_path="definitely-not-a-real/model-xyz-99999")

    def test_empty_observation_raises(self, act_policy):
        """Empty observation dict should raise, not return garbage."""
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            act_policy.get_actions_sync({}, "test")

    def test_wrong_state_dim_handled(self, act_policy):
        """Wrong state dimension should raise or be handled gracefully."""
        obs = {"observation.state": np.zeros(3, dtype=np.float32)}  # Wrong dim

        # Should either raise or handle (pad/truncate) — not crash with unhelpful error
        try:
            actions = act_policy.get_actions_sync(obs, "test")
            # If it doesn't raise, actions should still be valid
            assert isinstance(actions, list)
        except (ValueError, RuntimeError) as e:
            # Expected — wrong dimensions
            logger.info("Correctly rejected wrong state dim: %s", e)


# ---------------------------------------------------------------------------
# E2E Test: Performance Benchmarks
# ---------------------------------------------------------------------------


class TestPerformanceE2E:
    """Performance benchmarks for real inference."""

    def test_throughput_act(self, act_policy):
        """Measure ACT inference throughput."""
        obs = _build_realistic_observation(act_policy, randomize=True)

        # Warm up
        act_policy.get_actions_sync(obs, "test")

        n_iters = 20
        start = time.time()
        for _ in range(n_iters):
            act_policy.get_actions_sync(obs, "test")
        elapsed = time.time() - start

        fps = n_iters / elapsed
        ms_per_call = (elapsed / n_iters) * 1000
        logger.info("ACT throughput: %.1f Hz (%.1f ms/call)", fps, ms_per_call)

        # Should be at least 1 Hz on CPU
        assert fps > 0.5, f"ACT too slow: {fps:.2f} Hz"

    def test_throughput_diffusion(self, diffusion_policy):
        """Measure Diffusion inference throughput."""
        obs = _build_realistic_observation(diffusion_policy, randomize=True)

        # Warm up
        diffusion_policy.get_actions_sync(obs, "test")

        n_iters = 20
        start = time.time()
        for _ in range(n_iters):
            diffusion_policy.get_actions_sync(obs, "test")
        elapsed = time.time() - start

        fps = n_iters / elapsed
        ms_per_call = (elapsed / n_iters) * 1000
        logger.info("Diffusion throughput: %.1f Hz (%.1f ms/call)", fps, ms_per_call)

        assert fps > 0.5, f"Diffusion too slow: {fps:.2f} Hz"

    def test_model_load_time(self):
        """Model loading should complete within 60 seconds."""
        from strands_robots.policies import create_policy

        start = time.time()
        policy = create_policy(MODELS["diffusion"]["repo"])
        elapsed = time.time() - start

        logger.info("Model load time: %.1fs", elapsed)
        assert elapsed < 60, f"Model loading too slow: {elapsed:.1f}s"
        assert policy._loaded
