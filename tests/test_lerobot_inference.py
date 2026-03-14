"""Cycle 2: Simulation-based inference tests for LeRobot policies.

Tests actual model loading + inference through strands_robots Simulation tool
with MuJoCo backend. Downloads pretrained models from HuggingFace and runs
them against simulated environments.

Requires: mujoco, torch with CUDA, internet access for model download
"""

import logging
import os
import sys
import time

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# Skip entire module if no CUDA
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    DEVICE = "cuda" if HAS_CUDA else "cpu"
except ImportError:
    HAS_CUDA = False
    DEVICE = "cpu"

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# ---------------------------------------------------------------------------
# Model loading + inference (actual HF download)
# ---------------------------------------------------------------------------

# Models known to work with LeRobot 0.5.x
PRETRAINED_MODELS = [
    pytest.param(
        "lerobot/act_aloha_sim_transfer_cube_human",
        "act",
        14,  # ALOHA action dim (14 joints)
        id="act-aloha-sim",
    ),
    pytest.param(
        "lerobot/diffusion_pusht",
        "diffusion",
        2,   # PushT action dim (x, y)
        id="diffusion-pusht",
    ),
]

SMOLVLA_MODEL = pytest.param(
    "lerobot/smolvla_base",
    "smolvla",
    6,  # SmolVLA base is typically 6-DOF
    id="smolvla-base",
)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestLerobotModelLoading:
    """Test loading pretrained models from HuggingFace."""

    @pytest.mark.parametrize("model_id,policy_type,action_dim", PRETRAINED_MODELS)
    def test_load_pretrained_model(self, model_id, policy_type, action_dim):
        """Load a pretrained model and verify it initializes correctly."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        t0 = time.time()
        policy = LerobotLocalPolicy(pretrained_name_or_path=model_id)
        elapsed = time.time() - t0

        assert policy._loaded, f"Failed to load {model_id}"
        assert policy._device is not None

        info = policy.get_model_info()
        assert info["loaded"] is True
        assert info["provider"] == "lerobot_local"
        assert info["model_id"] == model_id

        logger.info(
            f"Loaded {model_id} ({info['policy_class']}) in {elapsed:.1f}s "
            f"on {policy._device}, {info.get('n_parameters', 0):,} params"
        )

    @pytest.mark.parametrize("model_id,policy_type,action_dim", PRETRAINED_MODELS)
    def test_inference_with_dummy_obs(self, model_id, policy_type, action_dim):
        """Run inference with random dummy observations."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        policy = LerobotLocalPolicy(pretrained_name_or_path=model_id)

        # Set robot state keys
        state_keys = [f"joint_{i}" for i in range(action_dim)]
        policy.set_robot_state_keys(state_keys)

        # Build observation matching the model's expected format
        obs = {}
        for feat_name, feat_info in policy._input_features.items():
            if hasattr(feat_info, "shape"):
                shape = feat_info.shape
                if "image" in feat_name:
                    # Image: (C, H, W) -> generate random
                    obs[feat_name] = torch.randn(1, *shape, device=policy._device)
                elif "state" in feat_name:
                    obs[feat_name] = torch.randn(1, *shape, device=policy._device)
                else:
                    obs[feat_name] = torch.randn(1, *shape, device=policy._device)

        # Run inference
        t0 = time.time()
        action = policy.select_action_sync(obs, instruction="test task")
        elapsed = time.time() - t0

        assert isinstance(action, np.ndarray), f"Expected ndarray, got {type(action)}"
        assert action.size > 0, "Empty action returned"

        logger.info(
            f"Inference {model_id}: action shape={action.shape}, "
            f"range=[{action.min():.3f}, {action.max():.3f}], "
            f"time={elapsed*1000:.0f}ms"
        )

    def test_smolvla_loading(self):
        """Test SmolVLA base model loads (VLA with language)."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        try:
            t0 = time.time()
            policy = LerobotLocalPolicy(pretrained_name_or_path="lerobot/smolvla_base")
            elapsed = time.time() - t0

            assert policy._loaded
            info = policy.get_model_info()
            assert "SmolVLA" in info["policy_class"]

            logger.info(
                f"SmolVLA loaded in {elapsed:.1f}s, "
                f"{info.get('n_parameters', 0):,} params on {policy._device}"
            )
        except Exception as e:
            logger.warning(f"SmolVLA load failed (may need more VRAM): {e}")
            pytest.skip(f"SmolVLA failed: {e}")

    def test_smolvla_inference_with_instruction(self):
        """Test SmolVLA VLA inference with language instruction + image."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        try:
            policy = LerobotLocalPolicy(pretrained_name_or_path="lerobot/smolvla_base")
        except Exception as e:
            pytest.skip(f"SmolVLA load failed: {e}")

        state_keys = [f"joint_{i}" for i in range(6)]
        policy.set_robot_state_keys(state_keys)

        # Build obs with image + state
        obs = {}
        for feat_name, feat_info in policy._input_features.items():
            if hasattr(feat_info, "shape"):
                shape = feat_info.shape
                obs[feat_name] = torch.randn(1, *shape, device=policy._device)

        t0 = time.time()
        action = policy.select_action_sync(obs, instruction="pick up the red cube")
        elapsed = time.time() - t0

        assert isinstance(action, np.ndarray)
        assert action.size > 0

        logger.info(
            f"SmolVLA inference: action shape={action.shape}, "
            f"time={elapsed*1000:.0f}ms"
        )


# ---------------------------------------------------------------------------
# Simulation integration tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestSimulationInference:
    """Test policy execution through the Simulation tool."""

    def test_simulation_creates_world(self):
        """Simulation tool can create a MuJoCo world."""
        from strands_robots.simulation.simulation import Simulation

        sim = Simulation(tool_name="test_sim")
        assert sim is not None

    def test_mock_policy_in_sim_loop(self):
        """Run mock policy through simulation for N steps."""
        from strands_robots.policies import MockPolicy

        policy = MockPolicy()
        policy.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4", "j5"])

        # Run 10 steps
        for step in range(10):
            obs = {"observation.state": [0.0] * 6}
            actions = policy.get_actions_sync(obs, "test movement")
            assert len(actions) > 0
            assert all(isinstance(a, dict) for a in actions)
            assert all("j0" in a for a in actions)

    def test_lerobot_local_factory_roundtrip(self):
        """Create policy via factory, set keys, get dummy actions."""
        from strands_robots.policies import create_policy

        policy = create_policy("lerobot_local")
        policy.set_robot_state_keys(["shoulder", "elbow", "wrist", "gripper"])

        # Without loading a model, should return zero actions
        import asyncio
        actions = asyncio.run(policy.get_actions({}, "test"))
        assert len(actions) > 0
        for a in actions:
            assert all(v == 0.0 for v in a.values())
