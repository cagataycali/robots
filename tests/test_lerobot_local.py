"""Tests for strands_robots.policies.lerobot_local — LerobotLocalPolicy.

All tests run WITHOUT lerobot installed (pure mock/unit testing).
Integration tests that load real models are in tests_integ/.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy(**kwargs):
    """Create a LerobotLocalPolicy with model loading disabled."""
    from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(**kwargs)
    return policy


def _make_loaded_policy(action_dim=6, state_dim=6, device="cpu"):
    """Create a LerobotLocalPolicy that appears loaded (mocked internals)."""
    import torch
    from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path="test/model")

    policy._loaded = True
    policy._device = torch.device(device)

    action_feat = MagicMock()
    action_feat.shape = (action_dim,)
    policy._output_features = {"action": action_feat}

    state_feat = MagicMock()
    state_feat.shape = (state_dim,)
    policy._input_features = {
        "observation.state": state_feat,
        "observation.images.top": MagicMock(shape=(3, 480, 640)),
    }

    mock_lerobot_policy = MagicMock()
    mock_param = torch.nn.Parameter(torch.zeros(1))
    mock_lerobot_policy.parameters.return_value = [mock_param]
    mock_lerobot_policy.select_action.return_value = torch.zeros(action_dim)
    policy._policy = mock_lerobot_policy

    return policy


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestLerobotLocalInit:
    def test_init_without_path(self):
        """Creating without pretrained_name_or_path should not load model."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        policy = LerobotLocalPolicy()
        assert policy._loaded is False
        assert policy.provider_name == "lerobot_local"
        assert policy.robot_state_keys == []

    def test_init_with_path_triggers_load(self):
        """Creating with pretrained_name_or_path should call _load_model."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        with patch.object(LerobotLocalPolicy, "_load_model") as mock_load:
            LerobotLocalPolicy(pretrained_name_or_path="lerobot/act_aloha_sim")
            mock_load.assert_called_once()

    def test_provider_name(self):
        policy = _make_policy()
        assert policy.provider_name == "lerobot_local"

    def test_default_actions_per_step(self):
        policy = _make_policy()
        assert policy.actions_per_step == 1

    def test_custom_actions_per_step(self):
        policy = _make_policy(actions_per_step=5)
        assert policy.actions_per_step == 5

    def test_is_policy_subclass(self):
        from strands_robots.policies import Policy
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        assert issubclass(LerobotLocalPolicy, Policy)


# ---------------------------------------------------------------------------
# Tests: set_robot_state_keys
# ---------------------------------------------------------------------------


class TestSetRobotStateKeys:
    def test_explicit_keys(self):
        policy = _make_policy()
        policy.set_robot_state_keys(["shoulder", "elbow", "wrist"])
        assert policy.robot_state_keys == ["shoulder", "elbow", "wrist"]

    def test_empty_keys_auto_detect_from_output_features(self):
        policy = _make_loaded_policy(action_dim=7)
        policy.robot_state_keys = []
        policy.set_robot_state_keys([])
        assert len(policy.robot_state_keys) == 7
        assert policy.robot_state_keys[0] == "joint_0"

    def test_empty_keys_fallback_to_input_features(self):
        policy = _make_loaded_policy(state_dim=4)
        policy._output_features = {}
        policy.robot_state_keys = []
        policy.set_robot_state_keys([])
        assert len(policy.robot_state_keys) == 4

    def test_empty_keys_no_features_warns(self):
        policy = _make_policy()
        policy._loaded = True
        policy._output_features = {}
        policy._input_features = {}
        policy.set_robot_state_keys([])
        assert policy.robot_state_keys == []


# ---------------------------------------------------------------------------
# Tests: get_actions (async)
# ---------------------------------------------------------------------------


class TestGetActions:
    def test_not_loaded_triggers_load(self):
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        with patch.object(LerobotLocalPolicy, "_load_model") as mock_load:
            policy = LerobotLocalPolicy()
            policy.pretrained_name_or_path = "test/model"

            def fake_load():
                policy._loaded = True
                policy._device = "cpu"
                import torch

                mock_inner = MagicMock()
                mock_inner.select_action.return_value = torch.zeros(6)
                policy._policy = mock_inner
                policy._output_features = {}
                policy._input_features = {}
                policy.robot_state_keys = [f"j{i}" for i in range(6)]

            mock_load.side_effect = fake_load
            actions = policy.get_actions_sync({}, "test")
            mock_load.assert_called()

    def test_returns_list_of_dicts(self):
        policy = _make_loaded_policy(action_dim=3)
        policy.set_robot_state_keys(["a", "b", "c"])
        actions = policy.get_actions_sync({}, "test")
        assert isinstance(actions, list)
        assert all(isinstance(action, dict) for action in actions)

    def test_action_keys_match_state_keys(self):
        policy = _make_loaded_policy(action_dim=3)
        policy.set_robot_state_keys(["shoulder", "elbow", "gripper"])
        actions = policy.get_actions_sync({}, "pick up")
        assert set(actions[0].keys()) == {"shoulder", "elbow", "gripper"}

    def test_zero_actions_on_no_path(self):
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        policy = LerobotLocalPolicy()
        policy.robot_state_keys = ["a", "b"]
        actions = policy.get_actions_sync({}, "test")
        assert actions == [{"a": 0.0, "b": 0.0}]

    def test_consecutive_failure_tracking(self):
        policy = _make_loaded_policy(action_dim=3)
        policy.set_robot_state_keys(["a", "b", "c"])
        policy._max_consecutive_failures = 2

        policy._policy.select_action.side_effect = RuntimeError("boom")

        actions = policy.get_actions_sync({}, "test")
        assert actions == [{"a": 0.0, "b": 0.0, "c": 0.0}]
        assert policy._consecutive_failures == 1

        with pytest.raises(RuntimeError, match="failed 2 consecutive"):
            policy.get_actions_sync({}, "test")

    def test_success_resets_failure_counter(self):
        policy = _make_loaded_policy(action_dim=2)
        policy.set_robot_state_keys(["a", "b"])
        policy._consecutive_failures = 4

        actions = policy.get_actions_sync({}, "test")
        assert policy._consecutive_failures == 0


# ---------------------------------------------------------------------------
# Tests: _build_observation_batch
# ---------------------------------------------------------------------------


class TestBuildObservationBatch:
    def test_lerobot_format_passthrough(self):
        """Keys starting with 'observation.' should pass through."""
        import torch

        policy = _make_loaded_policy(state_dim=3)
        observation = {"observation.state": torch.tensor([1.0, 2.0, 3.0])}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.state" in batch

    def test_numpy_state_conversion(self):
        import torch

        policy = _make_loaded_policy(state_dim=3)
        observation = {"observation.state": np.array([1.0, 2.0, 3.0])}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.state" in batch
        assert isinstance(batch["observation.state"], torch.Tensor)

    def test_image_hwc_to_chw_conversion(self):
        policy = _make_loaded_policy()
        observation = {
            "observation.images.top": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.images.top" in batch
        assert batch["observation.images.top"].shape == (1, 3, 480, 640)
        assert batch["observation.images.top"].max() <= 1.0

    def test_strands_format_state_mapping(self):
        policy = _make_loaded_policy(state_dim=3)
        policy.set_robot_state_keys(["shoulder", "elbow", "gripper"])
        observation = {"shoulder": 0.5, "elbow": -0.3, "gripper": 1.0}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.state" in batch
        state = batch["observation.state"]
        assert state.shape == (1, 3)
        assert abs(state[0, 0].item() - 0.5) < 1e-6

    def test_list_state_conversion(self):
        """Lists in observation.state should convert to tensors."""
        import torch

        policy = _make_loaded_policy(state_dim=3)
        observation = {"observation.state": [1.0, 2.0, 3.0]}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.state" in batch
        assert isinstance(batch["observation.state"], torch.Tensor)

    def test_missing_image_features_filled_with_zeros(self):
        policy = _make_loaded_policy()
        batch = policy._build_observation_batch({}, "test")
        assert "observation.images.top" in batch
        assert batch["observation.images.top"].shape[-2:] == (480, 640)


# ---------------------------------------------------------------------------
# Tests: _tensor_to_action_dicts
# ---------------------------------------------------------------------------


class TestTensorToActionDicts:
    def test_1d_tensor(self):
        import torch

        policy = _make_loaded_policy(action_dim=3)
        policy.set_robot_state_keys(["a", "b", "c"])
        result = policy._tensor_to_action_dicts(torch.tensor([1.0, 2.0, 3.0]))
        assert len(result) == 1
        assert result[0] == {"a": 1.0, "b": 2.0, "c": 3.0}

    def test_2d_tensor_respects_actions_per_step(self):
        import torch

        policy = _make_loaded_policy(action_dim=2)
        policy.set_robot_state_keys(["x", "y"])
        policy.actions_per_step = 2
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = policy._tensor_to_action_dicts(tensor)
        assert len(result) == 2
        assert result[0] == {"x": 1.0, "y": 2.0}
        assert result[1] == {"x": 3.0, "y": 4.0}

    def test_3d_tensor_batch(self):
        import torch

        policy = _make_loaded_policy(action_dim=2)
        policy.set_robot_state_keys(["x", "y"])
        policy.actions_per_step = 1
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        result = policy._tensor_to_action_dicts(tensor)
        assert len(result) == 1
        assert result[0] == {"x": 1.0, "y": 2.0}

    def test_short_tensor_pads_with_zeros(self):
        import torch

        policy = _make_loaded_policy(action_dim=3)
        policy.set_robot_state_keys(["a", "b", "c"])
        result = policy._tensor_to_action_dicts(torch.tensor([1.0]))
        assert result[0] == {"a": 1.0, "b": 0.0, "c": 0.0}


# ---------------------------------------------------------------------------
# Tests: _zero_actions
# ---------------------------------------------------------------------------


class TestZeroActions:
    def test_returns_zero_dict(self):
        policy = _make_loaded_policy()
        policy.set_robot_state_keys(["a", "b", "c"])
        zeros = policy._zero_actions()
        assert zeros == [{"a": 0.0, "b": 0.0, "c": 0.0}]

    def test_empty_keys_returns_empty_dict(self):
        policy = _make_loaded_policy()
        policy.robot_state_keys = []
        zeros = policy._zero_actions()
        assert zeros == [{}]


# ---------------------------------------------------------------------------
# Tests: get_model_info
# ---------------------------------------------------------------------------


class TestGetModelInfo:
    def test_info_before_load(self):
        policy = _make_policy(pretrained_name_or_path="test/model")
        info = policy.get_model_info()
        assert info["provider"] == "lerobot_local"
        assert info["loaded"] is False
        assert info["model_id"] == "test/model"

    def test_info_after_load(self):
        import torch

        policy = _make_loaded_policy(action_dim=6)
        policy.pretrained_name_or_path = "test/model"
        policy.policy_type = "act"
        policy._policy.__class__.__name__ = "ACTPolicy"

        mock_param = torch.nn.Parameter(torch.zeros(10, 5))
        policy._policy.parameters.return_value = [mock_param]

        info = policy.get_model_info()
        assert info["loaded"] is True
        assert info["policy_class"] == "ACTPolicy"
        assert info["n_parameters"] == 50


# ---------------------------------------------------------------------------
# Tests: select_action_sync
# ---------------------------------------------------------------------------


class TestSelectActionSync:
    def test_returns_numpy_array(self):
        import torch

        policy = _make_loaded_policy(action_dim=4)
        policy._policy.select_action.return_value = torch.tensor([1.0, 2.0, 3.0, 4.0])

        result = policy.select_action_sync({}, "test")
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])

    def test_instruction_from_task_key(self):
        import torch

        policy = _make_loaded_policy(action_dim=2)
        policy._policy.select_action.return_value = torch.tensor([0.0, 0.0])

        result = policy.select_action_sync({"task": "pick up the cube"}, "")
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Tests: Policy resolution helpers
# ---------------------------------------------------------------------------


class TestPolicyResolution:
    def test_resolve_policy_class_by_name_raises_for_unknown(self):
        from strands_robots.policies.lerobot_local.resolution import resolve_policy_class_by_name

        with pytest.raises((ImportError, ValueError)):
            resolve_policy_class_by_name("nonexistent_policy_type_xyz")

    def test_resolve_from_hub_raises_without_type(self):
        from strands_robots.policies.lerobot_local.resolution import resolve_policy_class_from_hub

        with pytest.raises((ValueError, ImportError, Exception)):
            resolve_policy_class_from_hub("completely/fake-model-path-that-does-not-exist")


# ---------------------------------------------------------------------------
# Tests: Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_lerobot_local_in_registry(self):
        from strands_robots.registry import list_policy_providers

        providers = list_policy_providers()
        assert "lerobot_local" in providers

    def test_lerobot_alias_resolves(self):
        from strands_robots.registry import get_policy_provider

        config = get_policy_provider("lerobot")
        assert config is not None
        assert config["class"] == "LerobotLocalPolicy"

    def test_hf_org_resolution(self):
        from strands_robots.registry import resolve_policy

        provider, kwargs = resolve_policy("lerobot/act_aloha_sim")
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == "lerobot/act_aloha_sim"

    def test_create_policy_lerobot_local_without_model(self):
        """create_policy('lerobot_local') without a path should not crash."""
        from strands_robots.policies import create_policy

        policy = create_policy("lerobot_local")
        assert policy.provider_name == "lerobot_local"
        assert policy._loaded is False


# ---------------------------------------------------------------------------
# Tests: ProcessorBridge
# ---------------------------------------------------------------------------


class TestProcessorBridge:
    def test_import(self):
        """ProcessorBridge module should be importable."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        assert ProcessorBridge is not None

    def test_inactive_bridge(self):
        """A bridge with no configs should report inactive."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        bridge = ProcessorBridge()
        assert not bridge.is_active
        assert not bridge.has_preprocessor
        assert not bridge.has_postprocessor

    def test_passthrough_preprocess(self):
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        bridge = ProcessorBridge()
        observation = {"key": "value"}
        assert bridge.preprocess(observation) == observation

    def test_passthrough_postprocess(self):
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        bridge = ProcessorBridge()
        action = np.array([1.0, 2.0])
        result = bridge.postprocess(action)
        np.testing.assert_array_equal(result, action)

    def test_get_info(self):
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        bridge = ProcessorBridge()
        info = bridge.get_info()
        assert info["has_preprocessor"] is False
        assert info["has_postprocessor"] is False
        assert info["is_active"] is False

    def test_repr(self):
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        bridge = ProcessorBridge()
        assert "pre=None" in repr(bridge)
        assert "post=None" in repr(bridge)
