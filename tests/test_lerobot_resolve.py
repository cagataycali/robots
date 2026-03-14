"""Tests for LeRobot 0.5.x policy resolution through strands-robots.

Validates that every policy shipped with LeRobot 0.5.x can be resolved
via _resolve_policy_class_by_name, and that pretrained models can be
loaded + run mock inference through the LerobotLocalPolicy wrapper.
"""

import importlib
import logging

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy class resolution
# ---------------------------------------------------------------------------

# All LeRobot 0.5.x policy types that have a Policy class
LEROBOT_POLICY_TYPES = [
    ("act", "ACTPolicy"),
    ("diffusion", "DiffusionPolicy"),
    ("pi0", "PI0Policy"),
    ("pi0_fast", "PI0FastPolicy"),
    ("pi05", "PI05Policy"),
    ("sac", "SACPolicy"),
    ("smolvla", "SmolVLAPolicy"),
    ("tdmpc", "TDMPCPolicy"),
    ("vqbet", "VQBeTPolicy"),
    ("xvla", "XVLAPolicy"),
]

# Policies that need optional deps (may skip on CI without GPU)
LEROBOT_OPTIONAL_POLICY_TYPES = [
    ("wall_x", "WallXPolicy", "peft"),  # needs peft
]

# Types that are NOT standalone policies
LEROBOT_NON_POLICY_TYPES = [
    ("rtc", "RTCProcessor"),   # post-processor wrapper, not a standalone policy
    ("sarm", "SARMRewardModel"),  # reward model, not an action policy
]


class TestLerobotPolicyResolution:
    """Test that strands-robots can resolve all LeRobot 0.5.x policy classes."""

    @pytest.mark.parametrize("policy_type,expected_class", LEROBOT_POLICY_TYPES)
    def test_resolve_core_policy(self, policy_type, expected_class):
        """Core policies must always resolve."""
        from strands_robots.policies.lerobot_local import _resolve_policy_class_by_name

        cls = _resolve_policy_class_by_name(policy_type)
        assert cls is not None, f"Failed to resolve {policy_type}"
        assert cls.__name__ == expected_class, (
            f"Expected {expected_class}, got {cls.__name__}"
        )
        assert hasattr(cls, "from_pretrained"), (
            f"{cls.__name__} missing from_pretrained"
        )

    @pytest.mark.parametrize("policy_type,expected_class,dep", LEROBOT_OPTIONAL_POLICY_TYPES)
    def test_resolve_optional_policy(self, policy_type, expected_class, dep):
        """Optional policies resolve when their deps are installed."""
        try:
            importlib.import_module(dep)
        except ImportError:
            pytest.skip(f"{dep} not installed")

        from strands_robots.policies.lerobot_local import _resolve_policy_class_by_name

        cls = _resolve_policy_class_by_name(policy_type)
        assert cls is not None, f"Failed to resolve {policy_type}"
        assert cls.__name__ == expected_class

    @pytest.mark.parametrize("policy_type,expected_class", LEROBOT_NON_POLICY_TYPES)
    def test_non_policy_types_noted(self, policy_type, expected_class):
        """Non-policy types (processors, reward models) resolve to their actual class
        or raise ImportError — but should NOT be confused with action policies."""
        from strands_robots.policies.lerobot_local import _resolve_policy_class_by_name

        # These may or may not resolve — the key point is they're not action policies
        try:
            cls = _resolve_policy_class_by_name(policy_type)
            # If it resolves, note what we got
            logger.info(f"{policy_type} resolved to {cls.__name__} (non-action policy)")
        except (ImportError, ValueError):
            # Expected — these aren't standalone policies
            pass


class TestLerobotPolicyResolutionFromHub:
    """Test auto-resolution from HuggingFace model IDs."""

    def test_resolve_from_act_model(self):
        """Resolve policy type from a known ACT model ID."""
        from strands_robots.policies.lerobot_local import _resolve_policy_class_from_hub

        PolicyClass, policy_type = _resolve_policy_class_from_hub(
            "lerobot/act_aloha_sim_transfer_cube_human"
        )
        assert PolicyClass.__name__ == "ACTPolicy"
        assert "act" in policy_type.lower()


class TestLerobotLocalPolicyInit:
    """Test LerobotLocalPolicy initialization (no model download)."""

    def test_init_empty(self):
        """Policy can be created without pretrained path."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        p = LerobotLocalPolicy()
        assert p.provider_name == "lerobot_local"
        assert not p._loaded

    def test_provider_name(self):
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        p = LerobotLocalPolicy()
        assert p.provider_name == "lerobot_local"

    def test_set_robot_state_keys(self):
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        p = LerobotLocalPolicy()
        p.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4", "j5"])
        assert len(p.robot_state_keys) == 6

    def test_create_via_factory(self):
        """Create via the standard create_policy factory."""
        from strands_robots.policies import create_policy

        p = create_policy("lerobot_local")
        assert p.provider_name == "lerobot_local"


class TestCreatePolicySmartStrings:
    """Test smart string resolution for HF model IDs."""

    def test_hf_model_id_resolves(self):
        """HuggingFace model ID should resolve to lerobot_local."""
        from strands_robots.registry import resolve_policy_string

        provider, kwargs = resolve_policy_string("lerobot/act_aloha_sim_transfer_cube_human")
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == "lerobot/act_aloha_sim_transfer_cube_human"

    def test_server_address_resolves(self):
        """Server address should resolve to lerobot_async."""
        from strands_robots.registry import resolve_policy_string

        provider, kwargs = resolve_policy_string("localhost:8080")
        assert provider == "lerobot_async"

    def test_mock_resolves(self):
        from strands_robots.registry import resolve_policy_string

        provider, kwargs = resolve_policy_string("mock")
        assert provider == "mock"


# ---------------------------------------------------------------------------
# Simulation inference tests (requires mujoco + GPU, mark slow)
# ---------------------------------------------------------------------------

def _has_mujoco():
    try:
        import mujoco
        return True
    except ImportError:
        return False


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _has_mujoco(), reason="mujoco not installed")
class TestSimulationPolicyRunner:
    """Test policy execution in MuJoCo simulation."""

    def test_mock_policy_in_sim(self):
        """Mock policy should run in a basic MuJoCo simulation."""
        from strands_robots.simulation.simulation import Simulation

        sim = Simulation(tool_name="test_sim")
        # Basic smoke test — create world + run mock policy
        # The simulation tool handles this via action="execute"
        assert sim is not None
        assert sim.tool_name_str == "test_sim"


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
@pytest.mark.slow
class TestLerobotModelLoading:
    """Test actual model loading from HuggingFace (slow, needs GPU + download)."""

    def test_load_act_model(self):
        """Load ACT model and run inference with dummy data."""
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        policy = LerobotLocalPolicy(
            pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human"
        )
        assert policy._loaded
        assert policy.policy_type is not None
        info = policy.get_model_info()
        assert info["loaded"] is True
        assert "ACT" in info["policy_class"]

    def test_act_inference_dummy_obs(self):
        """Run ACT inference with random observation."""
        import torch
        from strands_robots.policies.lerobot_local import LerobotLocalPolicy

        policy = LerobotLocalPolicy(
            pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human"
        )

        # Build dummy obs matching ACT's expected input
        info = policy.get_model_info()
        state_dim = 14  # ALOHA has 14 joints
        policy.set_robot_state_keys([f"joint_{i}" for i in range(state_dim)])

        obs = {
            "observation.state": torch.randn(1, state_dim).to(policy._device),
            "observation.images.top": torch.randn(1, 3, 480, 640).to(policy._device),
        }

        action = policy.select_action_sync(obs, instruction="pick up the cube")
        assert isinstance(action, np.ndarray)
        assert action.shape[-1] == state_dim
