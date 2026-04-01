"""Tests for strands_robots.policies.lerobot_local — LerobotLocalPolicy.

All tests run WITHOUT lerobot installed (pure mock/unit testing).
"""

import json
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch  # real or conftest mock — both work

from strands_robots.policies import create_policy
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.lerobot_local.processor import ProcessorBridge
from strands_robots.policies.lerobot_local.resolution import (
    _read_policy_type_from_config,
    resolve_policy_class_by_name,
    resolve_policy_class_from_hub,
)
from strands_robots.registry import list_policy_providers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy(**kwargs):
    """Create a LerobotLocalPolicy with model loading disabled."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(**kwargs)
    return policy


def _make_loaded_policy(action_dim=6, state_dim=6, device="cpu", include_images=True):
    """Create a LerobotLocalPolicy that appears loaded (mocked internals).

    Args:
        action_dim: Output action dimensions.
        state_dim: Input state dimensions.
        device: Torch device string.
        include_images: If True, include observation.images.top in input features.
            Set to False when tests don't provide image data to avoid the
            missing-image validation error.
    """
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path="test/model")

    policy._loaded = True
    policy._device = torch.device(device)

    action_feat = MagicMock()
    action_feat.shape = (action_dim,)
    policy._output_features = {"action": action_feat}

    state_feat = MagicMock()
    state_feat.shape = (state_dim,)
    input_features = {
        "observation.state": state_feat,
    }
    if include_images:
        input_features["observation.images.top"] = MagicMock(shape=(3, 480, 640))
    policy._input_features = input_features

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

        policy = LerobotLocalPolicy()
        assert policy._loaded is False
        assert policy.provider_name == "lerobot_local"
        assert policy.robot_state_keys == []

    def test_init_with_path_triggers_load(self):
        """Creating with pretrained_name_or_path should call _load_model."""

        with patch.object(LerobotLocalPolicy, "_load_model") as mock_load:
            LerobotLocalPolicy(pretrained_name_or_path="lerobot/act_aloha_sim")
            mock_load.assert_called_once()

    def test_custom_actions_per_step(self):
        policy = _make_policy(actions_per_step=5)
        assert policy.actions_per_step == 5


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

    def test_empty_keys_no_features_raises(self):
        """Empty keys with no model features should raise ValueError."""
        policy = _make_policy()
        policy._loaded = True
        policy._output_features = {}
        policy._input_features = {}
        with pytest.raises(ValueError, match="robot_state_keys is empty"):
            policy.set_robot_state_keys([])


# ---------------------------------------------------------------------------
# Tests: Tokenizer resolution (VLA support)
# ---------------------------------------------------------------------------


class TestResolveTokenizer:
    def test_tokenizer_from_tokenizer_name_falls_to_processor(self):
        """Strategy 1 (tokenizer_name) falls through when transformers missing, lands on Strategy 3."""
        policy = _make_loaded_policy()
        mock_tok = MagicMock()
        policy._policy.config = MagicMock(
            tokenizer_name="mock-tokenizer",
            vlm_model_name=None,
            tokenizer_max_length=64,
            tokenizer_padding_side="left",
        )
        policy._policy.processor = MagicMock()
        policy._policy.processor.tokenizer = mock_tok
        result = policy._resolve_tokenizer()
        assert result is mock_tok
        assert policy._tokenizer is mock_tok

    def test_tokenizer_from_processor_builtin(self):
        """Strategy 3: policy.processor.tokenizer."""
        policy = _make_loaded_policy()
        mock_tok = MagicMock()
        policy._policy.config = MagicMock(
            tokenizer_name=None,
            vlm_model_name=None,
            tokenizer_max_length=48,
            tokenizer_padding_side="right",
        )
        policy._policy.processor = MagicMock()
        policy._policy.processor.tokenizer = mock_tok
        result = policy._resolve_tokenizer()
        assert result is mock_tok
        assert mock_tok.padding_side == "right"

    def test_returns_none_when_no_tokenizer_available(self):
        """No tokenizer_name, no vlm_model_name, no processor.tokenizer."""
        policy = _make_loaded_policy()
        policy._policy.config = MagicMock(
            tokenizer_name=None,
            vlm_model_name=None,
            tokenizer_max_length=48,
            tokenizer_padding_side="right",
        )
        policy._policy.processor = None
        result = policy._resolve_tokenizer()
        assert result is None


class TestTokenizeInstruction:
    def test_returns_none_without_tokenizer(self):
        policy = _make_loaded_policy()
        policy._policy.config = None
        assert policy._tokenize_instruction("pick up") is None

    def test_returns_none_for_empty_instruction(self):
        policy = _make_loaded_policy()
        policy._tokenizer = MagicMock()
        assert policy._tokenize_instruction("") is None

    def test_tokenizes_and_transfers_to_device(self):
        policy = _make_loaded_policy()
        policy._device = torch.device("cpu")
        policy._tokenizer_max_length = 32

        mock_ids = MagicMock()
        mock_ids.to.return_value = mock_ids
        mock_mask = MagicMock()
        mock_mask.bool.return_value = mock_mask
        mock_mask.to.return_value = mock_mask

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": mock_ids, "attention_mask": mock_mask}
        policy._tokenizer = mock_tok

        result = policy._tokenize_instruction("pick up the cube")
        assert result is not None
        tokens, mask = result
        assert tokens is mock_ids
        assert mask is mock_mask
        mock_tok.assert_called_once_with(
            "pick up the cube",
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True,
        )

    def test_handles_missing_attention_mask(self):
        policy = _make_loaded_policy()
        policy._device = torch.device("cpu")
        mock_ids = MagicMock()
        mock_ids.to.return_value = mock_ids
        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": mock_ids}
        policy._tokenizer = mock_tok

        tokens, mask = policy._tokenize_instruction("test")
        assert mask is None


class TestNeedsLanguageTokens:
    def test_tokenizer_name_returns_true(self):
        policy = _make_loaded_policy()
        policy._policy.config = MagicMock(tokenizer_name="gpt2", vlm_model_name=None)
        assert policy._needs_language_tokens() is True

    def test_vlm_model_name_returns_true(self):
        policy = _make_loaded_policy()
        policy._policy.config = MagicMock(tokenizer_name=None, vlm_model_name="smolvlm")
        assert policy._needs_language_tokens() is True

    def test_language_input_feature_returns_true(self):
        policy = _make_loaded_policy()
        policy._policy.config = MagicMock(tokenizer_name=None, vlm_model_name=None)
        policy._input_features["observation.language.tokens"] = MagicMock()
        assert policy._needs_language_tokens() is True

    def test_no_language_indicators_returns_false(self):
        policy = _make_loaded_policy()
        policy._policy.config = MagicMock(tokenizer_name=None, vlm_model_name=None)
        assert policy._needs_language_tokens() is False


# ---------------------------------------------------------------------------
# Tests: _load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_load_with_explicit_policy_type(self):

        mock_policy_cls = MagicMock()
        mock_inner = MagicMock()
        mock_inner.config = MagicMock(
            input_features={"observation.state": MagicMock(shape=(6,))},
            output_features={"action": MagicMock(shape=(6,))},
            device="cpu",
        )
        mock_inner.eval.return_value = None
        mock_policy_cls.from_pretrained.return_value = mock_inner

        policy = LerobotLocalPolicy()
        policy.pretrained_name_or_path = "test/model"
        policy.policy_type = "act"

        with patch(
            "strands_robots.policies.lerobot_local.policy.resolve_policy_class_by_name",
            return_value=mock_policy_cls,
        ):
            with patch(
                "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
                return_value=MagicMock(is_active=False),
            ):
                policy._load_model()

        assert policy._loaded is True
        assert policy._device == torch.device("cpu")

    def test_load_without_policy_type_resolves_from_hub(self):

        mock_policy_cls = MagicMock()
        mock_inner = MagicMock()
        mock_inner.config = MagicMock(spec=[])
        mock_inner.eval.return_value = None
        mock_inner.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
        mock_policy_cls.from_pretrained.return_value = mock_inner

        policy = LerobotLocalPolicy()
        policy.pretrained_name_or_path = "test/model"

        with patch(
            "strands_robots.policies.lerobot_local.policy.resolve_policy_class_from_hub",
            return_value=(mock_policy_cls, "diffusion"),
        ):
            with patch(
                "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
                return_value=MagicMock(is_active=False),
            ):
                policy._load_model()

        assert policy.policy_type == "diffusion"
        assert policy._loaded is True

    def test_device_from_config(self):

        mock_policy_cls = MagicMock()
        mock_inner = MagicMock()
        mock_inner.config = MagicMock(device="cpu", spec=["device"])
        mock_inner.eval.return_value = None
        mock_policy_cls.from_pretrained.return_value = mock_inner

        policy = LerobotLocalPolicy()
        policy.pretrained_name_or_path = "test/model"
        policy.policy_type = "act"

        with patch(
            "strands_robots.policies.lerobot_local.policy.resolve_policy_class_by_name",
            return_value=mock_policy_cls,
        ):
            with patch(
                "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
                return_value=MagicMock(is_active=False),
            ):
                policy._load_model()

        assert policy._device == torch.device("cpu")

    def test_auto_generates_state_keys_from_output(self):

        action_feat = MagicMock()
        action_feat.shape = (4,)
        mock_policy_cls = MagicMock()
        mock_inner = MagicMock()
        mock_inner.config = MagicMock(
            device="cpu",
            input_features={},
            output_features={"action": action_feat},
        )
        mock_inner.eval.return_value = None
        mock_policy_cls.from_pretrained.return_value = mock_inner

        policy = LerobotLocalPolicy()
        policy.pretrained_name_or_path = "test/model"
        policy.policy_type = "act"

        with patch(
            "strands_robots.policies.lerobot_local.policy.resolve_policy_class_by_name",
            return_value=mock_policy_cls,
        ):
            with patch(
                "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
                return_value=MagicMock(is_active=False),
            ):
                policy._load_model()

        assert policy.robot_state_keys == ["joint_0", "joint_1", "joint_2", "joint_3"]


# ---------------------------------------------------------------------------
# Tests: get_actions (async)
# ---------------------------------------------------------------------------


class TestGetActions:
    def test_not_loaded_triggers_load(self):

        with patch.object(LerobotLocalPolicy, "_load_model") as mock_load:
            policy = LerobotLocalPolicy()
            policy.pretrained_name_or_path = "test/model"

            def fake_load():
                policy._loaded = True
                policy._device = "cpu"
                mock_inner = MagicMock()
                mock_inner.select_action.return_value = torch.zeros(6)
                policy._policy = mock_inner
                policy._output_features = {}
                policy._input_features = {}
                policy.robot_state_keys = [f"j{i}" for i in range(6)]

            mock_load.side_effect = fake_load
            policy.get_actions_sync({}, "test")
            mock_load.assert_called()

    def test_returns_list_of_dicts(self):
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        actions = policy.get_actions_sync({}, "test")
        assert isinstance(actions, list)
        assert all(isinstance(action, dict) for action in actions)

    def test_action_keys_match_state_keys(self):
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["shoulder", "elbow", "gripper"])
        actions = policy.get_actions_sync({}, "pick up")
        assert set(actions[0].keys()) == {"shoulder", "elbow", "gripper"}

    def test_no_path_raises_runtime_error(self):

        policy = LerobotLocalPolicy()
        policy.robot_state_keys = ["a", "b"]
        with pytest.raises(RuntimeError, match="No model loaded"):
            policy.get_actions_sync({}, "test")

    def test_inference_error_raises(self):
        """Inference errors should propagate immediately."""
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        policy._policy.select_action.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            policy.get_actions_sync({}, "test")

    def test_multi_step_uses_predict_action_chunk(self):
        """actions_per_step > 1 should call predict_action_chunk for full chunk."""
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        policy.actions_per_step = 4

        # predict_action_chunk returns (batch, horizon, action_dim)
        policy._policy.predict_action_chunk.return_value = torch.zeros(1, 10, 3)

        actions = policy.get_actions_sync({}, "test")

        # Should have called predict_action_chunk, NOT select_action
        policy._policy.predict_action_chunk.assert_called_once()
        policy._policy.select_action.assert_not_called()
        assert len(actions) == 4

    def test_single_step_uses_select_action(self):
        """actions_per_step=1 should use select_action for temporal ensemble."""
        policy = _make_loaded_policy(action_dim=3, include_images=False)
        policy.set_robot_state_keys(["a", "b", "c"])
        policy.actions_per_step = 1

        actions = policy.get_actions_sync({}, "test")

        policy._policy.select_action.assert_called_once()
        assert len(actions) == 1

    def test_processor_bridge_preprocess_bypasses_batch_builder(self):
        policy = _make_loaded_policy(action_dim=3)
        policy.set_robot_state_keys(["a", "b", "c"])

        mock_bridge = MagicMock()
        mock_bridge.has_preprocessor = True
        mock_bridge.has_postprocessor = False
        mock_bridge.preprocess.return_value = {
            "observation.state": torch.zeros(1, 3),
        }
        policy._processor_bridge = mock_bridge

        with patch.object(policy, "_build_observation_batch") as mock_build:
            policy.get_actions_sync({"state": [0, 0, 0]}, "test")
            mock_build.assert_not_called()

        mock_bridge.preprocess.assert_called_once()

    def test_processor_bridge_postprocess_applied(self):
        policy = _make_loaded_policy(action_dim=2, include_images=False)
        policy.set_robot_state_keys(["a", "b"])

        mock_bridge = MagicMock()
        mock_bridge.has_preprocessor = False
        mock_bridge.has_postprocessor = True
        mock_bridge.postprocess.return_value = torch.tensor([10.0, 20.0])
        policy._processor_bridge = mock_bridge

        actions = policy.get_actions_sync({}, "test")
        mock_bridge.postprocess.assert_called_once()
        assert actions[0]["a"] == 10.0
        assert actions[0]["b"] == 20.0


# ---------------------------------------------------------------------------
# Tests: _build_observation_batch
# ---------------------------------------------------------------------------


class TestBuildObservationBatch:
    def test_lerobot_format_passthrough(self):
        policy = _make_loaded_policy(state_dim=3, include_images=False)
        observation = {"observation.state": torch.tensor([1.0, 2.0, 3.0])}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.state" in batch

    def test_numpy_state_conversion(self):
        policy = _make_loaded_policy(state_dim=3, include_images=False)
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
        policy = _make_loaded_policy(state_dim=3, include_images=False)
        policy.set_robot_state_keys(["shoulder", "elbow", "gripper"])
        observation = {"shoulder": 0.5, "elbow": -0.3, "gripper": 1.0}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.state" in batch
        state = batch["observation.state"]
        assert state.shape == (1, 3)
        assert abs(state[0, 0].item() - 0.5) < 1e-6

    def test_missing_image_features_filled_with_zeros(self):
        """Missing camera images should raise ValueError (fail-fast)."""
        policy = _make_loaded_policy()  # includes images in _input_features
        # Use lerobot-format keys so it goes through _build_batch_from_lerobot_format
        observation = {"observation.state": torch.zeros(6)}
        with pytest.raises(ValueError, match="Missing required image feature"):
            policy._build_observation_batch(observation, "test")

    def test_scalar_int_conversion(self):
        policy = _make_loaded_policy(include_images=False)
        observation = {"observation.gripper": 1}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.gripper" in batch
        assert batch["observation.gripper"].shape == (1, 1)

    def test_float64_tensor_auto_cast_to_float32(self):
        """float64 tensors from ROS/dynamixel drivers should be auto-cast to float32."""
        policy = _make_loaded_policy(state_dim=3, include_images=False)
        observation = {"observation.state": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)}
        batch = policy._build_observation_batch(observation, "test")
        assert batch["observation.state"].dtype == torch.float32

    def test_float64_numpy_auto_cast_to_float32(self):
        """float64 numpy arrays should be auto-cast to float32."""
        policy = _make_loaded_policy(state_dim=3, include_images=False)
        observation = {"observation.state": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        batch = policy._build_observation_batch(observation, "test")
        assert batch["observation.state"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests: _build_batch_from_strands_format
# ---------------------------------------------------------------------------


class TestBuildBatchFromStrandsFormat:
    def test_numpy_floating_state(self):
        policy = _make_loaded_policy(state_dim=2)
        policy.set_robot_state_keys(["a", "b"])
        observation = {"a": np.float32(1.5), "b": np.float64(2.5)}
        batch = policy._build_batch_from_strands_format(observation, {})
        assert "observation.state" in batch
        np.testing.assert_allclose(batch["observation.state"][0].numpy(), [1.5, 2.5], atol=1e-5)

    def test_state_padded_to_expected_dim(self):
        """State dimension mismatch should auto-pad (not raise)."""
        policy = _make_loaded_policy(state_dim=4, include_images=False)
        policy.set_robot_state_keys(["a", "b"])
        observation = {"a": 1.0, "b": 2.0}
        # After bug fix: auto-pads with zeros instead of raising
        batch = policy._build_batch_from_strands_format(observation, {})
        state = batch["observation.state"][0].numpy()
        assert len(state) == 4
        np.testing.assert_allclose(state[:2], [1.0, 2.0], atol=1e-5)
        np.testing.assert_allclose(state[2:], [0.0, 0.0], atol=1e-5)

    def test_empty_state_keys_raises(self):
        """Empty robot_state_keys should raise ValueError."""
        policy = _make_loaded_policy()
        policy.robot_state_keys = []
        with pytest.raises(ValueError, match="robot_state_keys is empty"):
            policy._build_batch_from_strands_format({"x": 1.0}, {})


# ---------------------------------------------------------------------------
# Tests: _tensor_to_action_dicts
# ---------------------------------------------------------------------------


class TestTensorToActionDicts:
    def test_1d_tensor(self):
        policy = _make_loaded_policy(action_dim=3)
        policy.set_robot_state_keys(["a", "b", "c"])
        result = policy._tensor_to_action_dicts(torch.tensor([1.0, 2.0, 3.0]))
        assert len(result) == 1
        assert result[0] == {"a": 1.0, "b": 2.0, "c": 3.0}

    def test_2d_tensor_respects_actions_per_step(self):
        policy = _make_loaded_policy(action_dim=2)
        policy.set_robot_state_keys(["x", "y"])
        policy.actions_per_step = 2
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = policy._tensor_to_action_dicts(tensor)
        assert len(result) == 2
        assert result[0] == {"x": 1.0, "y": 2.0}
        assert result[1] == {"x": 3.0, "y": 4.0}

    def test_empty_state_keys_raises(self):
        policy = _make_loaded_policy(action_dim=2)
        policy.robot_state_keys = []
        with pytest.raises(RuntimeError, match="robot_state_keys is empty"):
            policy._tensor_to_action_dicts(torch.tensor([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Tests: reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_delegates_to_inner_policy(self):
        policy = _make_policy()
        policy._loaded = True
        policy._policy = MagicMock()
        policy._processor_bridge = None
        policy.reset()
        policy._policy.reset.assert_called_once()

    def test_reset_safe_when_not_loaded(self):
        policy = _make_policy()
        assert policy._policy is None
        policy.reset()  # Should not raise


# ---------------------------------------------------------------------------
# Tests: Policy resolution helpers
# ---------------------------------------------------------------------------


class TestPolicyResolution:
    def test_resolve_policy_class_by_name_raises_for_unknown(self):

        with pytest.raises((ImportError, ValueError)):
            resolve_policy_class_by_name("nonexistent_policy_type_xyz")

    def test_resolve_from_hub_raises_without_type(self):

        with pytest.raises((ValueError, ImportError, Exception)):
            resolve_policy_class_from_hub("completely/fake-model-path-that-does-not-exist")

    def test_resolve_by_name_modeling_submodule(self):

        mock_policy_class = type("ACTPolicy", (), {"from_pretrained": classmethod(lambda cls: None)})
        mock_module = types.ModuleType("lerobot.policies.act.modeling_act")
        mock_module.ACTPolicy = mock_policy_class

        with patch("importlib.import_module", return_value=mock_module):
            result = resolve_policy_class_by_name("act")
            assert result is mock_policy_class

    def test_read_policy_type_from_local_config(self, tmp_path):

        config_dir = tmp_path / "model"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({"type": "act"}))

        result = _read_policy_type_from_config(str(config_dir))
        assert result == "act"


# ---------------------------------------------------------------------------
# Tests: Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_lerobot_local_in_registry(self):

        providers = list_policy_providers()
        assert "lerobot_local" in providers

    def test_create_policy_lerobot_local_without_model(self):

        policy = create_policy("lerobot_local")
        assert policy.provider_name == "lerobot_local"
        assert policy._loaded is False


# ---------------------------------------------------------------------------
# Tests: ProcessorBridge
# ---------------------------------------------------------------------------


class TestProcessorBridge:
    def test_preprocess_raises_on_pipeline_error(self):
        """preprocess() wraps pipeline exceptions in RuntimeError.

        The production code calls _preprocessor._forward(transition) after
        building a transition via create_transition(). We mock _forward to
        raise and patch the lerobot imports so the transition-building path
        is exercised regardless of whether lerobot is installed.
        """
        mock_pre = MagicMock()
        mock_pre._forward.side_effect = ValueError("bad data")
        bridge = ProcessorBridge(preprocessor=mock_pre)

        # Patch the lerobot imports used inside preprocess()
        mock_create_transition = MagicMock(return_value={"observation": {}})
        mock_transition_key = MagicMock()
        mock_transition_key.OBSERVATION = "observation"

        with patch(
            "strands_robots.policies.lerobot_local.processor.create_transition",
            mock_create_transition,
            create=True,
        ):
            with patch(
                "strands_robots.policies.lerobot_local.processor.TransitionKey",
                mock_transition_key,
                create=True,
            ):
                with pytest.raises(RuntimeError, match="Preprocessor pipeline failed"):
                    bridge.preprocess({})

    def test_postprocess_raises_on_pipeline_error(self):

        mock_post = MagicMock()
        mock_post.process_action.side_effect = ValueError("bad action")
        bridge = ProcessorBridge(postprocessor=mock_post)

        with pytest.raises(RuntimeError, match="Postprocessor pipeline failed"):
            bridge.postprocess(torch.zeros(2))

    def test_from_pretrained_passthrough_when_no_lerobot(self):

        with patch("strands_robots.policies.lerobot_local.processor._try_import_processor", return_value=None):
            bridge = ProcessorBridge.from_pretrained("test/model")
            assert not bridge.is_active


# ===========================================================================
# Tests: RTC (Real-Time Chunking)
# ===========================================================================


class TestRTCInit:
    """Tests for RTC initialization and auto-detection."""

    def test_rtc_default_disabled_when_no_predict_chunk(self):
        """RTC should be disabled when policy has no predict_action_chunk."""
        policy = _make_policy()
        mock_policy = MagicMock(spec=["reset", "eval", "parameters", "select_action"])
        mock_policy.config = MagicMock()
        mock_policy.config.device = "cpu"
        mock_policy.config.input_features = {}
        mock_policy.config.output_features = {}
        policy._policy = mock_policy
        policy._loaded = True
        policy._init_rtc()
        assert policy._rtc_enabled is False

    def test_rtc_auto_detect_from_config(self):
        """RTC should auto-detect from model config.rtc_config.enabled."""
        policy = _make_policy()
        mock_policy = MagicMock()
        mock_policy.predict_action_chunk = MagicMock()
        rtc_cfg = MagicMock()
        rtc_cfg.enabled = True
        rtc_cfg.execution_horizon = 15
        rtc_cfg.max_guidance_weight = 8.0
        mock_policy.config.rtc_config = rtc_cfg
        policy._policy = mock_policy
        policy._loaded = True
        policy._rtc_requested = None  # auto-detect
        policy._init_rtc()
        assert policy._rtc_enabled is True
        assert policy._rtc_execution_horizon == 15
        assert policy._rtc_max_guidance_weight == 8.0

    def test_rtc_explicit_enable(self):
        """rtc_enabled=True should enable when policy has rtc_config."""
        policy = _make_policy()
        mock_policy = MagicMock()
        mock_policy.predict_action_chunk = MagicMock()
        rtc_cfg = MagicMock()
        rtc_cfg.enabled = False  # config says disabled, but user forced True
        rtc_cfg.execution_horizon = 10
        rtc_cfg.max_guidance_weight = 10.0
        mock_policy.config.rtc_config = rtc_cfg
        policy._policy = mock_policy
        policy._loaded = True
        policy._rtc_requested = True
        policy._init_rtc()
        assert policy._rtc_enabled is True
        assert policy._rtc_execution_horizon == 10
        assert policy._rtc_max_guidance_weight == 10.0

    def test_rtc_explicit_enable_without_rtc_config_falls_back(self):
        """rtc_enabled=True without rtc_config should warn and disable."""
        policy = _make_policy()
        mock_policy = MagicMock()
        mock_policy.predict_action_chunk = MagicMock()
        mock_policy.config = MagicMock(spec=[])  # no rtc_config attr
        policy._policy = mock_policy
        policy._loaded = True
        policy._rtc_requested = True
        with patch("strands_robots.policies.lerobot_local.policy.logger") as mock_logger:
            policy._init_rtc()
            mock_logger.warning.assert_called_once()
            assert "no rtc_config" in mock_logger.warning.call_args[0][0]
        assert policy._rtc_enabled is False

    def test_rtc_explicit_disable(self):
        """rtc_enabled=False should disable even if config says enabled."""
        policy = _make_policy()
        mock_policy = MagicMock()
        mock_policy.predict_action_chunk = MagicMock()
        rtc_cfg = MagicMock()
        rtc_cfg.enabled = True
        mock_policy.config.rtc_config = rtc_cfg
        policy._policy = mock_policy
        policy._loaded = True
        policy._rtc_requested = False
        policy._init_rtc()
        assert policy._rtc_enabled is False

    def test_rtc_user_overrides_config_values(self):
        """User-provided execution_horizon/max_guidance_weight override config."""
        policy = _make_policy()
        mock_policy = MagicMock()
        mock_policy.predict_action_chunk = MagicMock()
        rtc_cfg = MagicMock()
        rtc_cfg.enabled = True
        rtc_cfg.execution_horizon = 15
        rtc_cfg.max_guidance_weight = 8.0
        mock_policy.config.rtc_config = rtc_cfg
        policy._policy = mock_policy
        policy._loaded = True
        policy._rtc_requested = True
        policy._rtc_execution_horizon = 20
        policy._rtc_max_guidance_weight = 5.0
        policy._init_rtc()
        assert policy._rtc_execution_horizon == 20
        assert policy._rtc_max_guidance_weight == 5.0

    def test_rtc_warning_when_requested_but_unsupported(self):
        """Should warn when RTC requested but policy lacks predict_action_chunk."""
        policy = _make_policy()
        mock_policy = MagicMock(spec=["reset", "eval", "parameters", "select_action"])
        policy._policy = mock_policy
        policy._loaded = True
        policy._rtc_requested = True
        with patch("strands_robots.policies.lerobot_local.policy.logger") as mock_logger:
            policy._init_rtc()
            mock_logger.warning.assert_called_once()
        assert policy._rtc_enabled is False


class TestRTCInference:
    """Tests for RTC inference path."""

    def test_predict_with_rtc_first_call_no_prev_chunk(self):
        """First RTC call should have no prev_chunk_left_over."""
        policy = _make_policy()
        policy._rtc_enabled = True
        policy._rtc_execution_horizon = 10
        policy._rtc_prev_chunk = None
        policy.actions_per_step = 1

        mock_policy = MagicMock()
        mock_policy.predict_action_chunk.return_value = torch.randn(1, 20, 6)
        policy._policy = mock_policy

        result = policy._predict_with_rtc({})

        call_kwargs = mock_policy.predict_action_chunk.call_args[1]
        assert "prev_chunk_left_over" not in call_kwargs
        assert call_kwargs["execution_horizon"] == 10
        assert result.dim() == 2  # (T, A) after squeeze

    def test_predict_with_rtc_passes_prev_chunk(self):
        """Subsequent RTC calls should pass prev_chunk_left_over."""
        policy = _make_policy()
        policy._rtc_enabled = True
        policy._rtc_execution_horizon = 10
        policy._rtc_prev_chunk = torch.randn(15, 6)
        policy.actions_per_step = 1

        mock_policy = MagicMock()
        mock_policy.predict_action_chunk.return_value = torch.randn(1, 20, 6)
        policy._policy = mock_policy

        policy._predict_with_rtc({})

        call_kwargs = mock_policy.predict_action_chunk.call_args[1]
        assert "prev_chunk_left_over" in call_kwargs
        assert call_kwargs["prev_chunk_left_over"].shape == (15, 6)

    def test_predict_with_rtc_stores_leftover(self):
        """After RTC inference, leftover should be stored for next call."""
        policy = _make_policy()
        policy._rtc_enabled = True
        policy._rtc_execution_horizon = 10
        policy._rtc_prev_chunk = None
        policy.actions_per_step = 1

        mock_policy = MagicMock()
        mock_policy.predict_action_chunk.return_value = torch.randn(1, 20, 6)
        policy._policy = mock_policy

        policy._predict_with_rtc({})

        assert policy._rtc_prev_chunk is not None
        assert policy._rtc_prev_chunk.dim() == 2

    def test_predict_with_rtc_tracks_latency(self):
        """RTC should track inference latency for delay estimation."""
        policy = _make_policy()
        policy._rtc_enabled = True
        policy._rtc_execution_horizon = 10
        policy._rtc_prev_chunk = None
        policy.actions_per_step = 1

        mock_policy = MagicMock()
        mock_policy.predict_action_chunk.return_value = torch.randn(1, 20, 6)
        policy._policy = mock_policy

        policy._predict_with_rtc({})

        assert len(policy._rtc_latency_history) == 1
        assert policy._rtc_latency_history[0] > 0


class TestRTCDelayEstimation:
    """Tests for inference delay estimation."""

    def test_estimate_delay_empty_history(self):
        """No history should return delay=0."""
        policy = _make_policy()
        assert policy._estimate_inference_delay(fps=30.0) == 0

    def test_estimate_delay_single_sample(self):
        """Single latency sample should give correct delay."""
        policy = _make_policy()
        policy._rtc_latency_history.append(0.1)  # 100ms
        assert policy._estimate_inference_delay(fps=30.0) == 3

    def test_estimate_delay_uses_p95(self):
        """Should use p95 latency, not mean or max."""
        policy = _make_policy()
        for _ in range(99):
            policy._rtc_latency_history.append(0.033)
        policy._rtc_latency_history.append(1.0)  # outlier
        delay = policy._estimate_inference_delay(fps=30.0)
        assert delay < 5


class TestRTCReset:
    """Tests for RTC state clearing on reset."""

    def test_reset_clears_rtc_state(self):
        """reset() should clear all RTC state."""
        policy = _make_policy()
        policy._policy = MagicMock()
        policy._processor_bridge = None

        policy._rtc_prev_chunk = torch.randn(10, 6)
        policy._rtc_action_queue.extend([torch.randn(6) for _ in range(5)])
        policy._rtc_latency_history.extend([0.1, 0.2, 0.3])
        policy._rtc_last_inference_time = 1.5

        policy.reset()

        assert policy._rtc_prev_chunk is None
        assert len(policy._rtc_action_queue) == 0
        assert len(policy._rtc_latency_history) == 0
        assert policy._rtc_last_inference_time == 0.0


class TestRTCGetActionsIntegration:
    """Tests for RTC integration in get_actions flow."""

    def test_get_actions_uses_rtc_when_enabled(self):
        """get_actions should use predict_action_chunk when RTC enabled."""
        policy = _make_loaded_policy(action_dim=6, include_images=False)
        policy._rtc_enabled = True
        policy._rtc_execution_horizon = 10
        policy._rtc_prev_chunk = None
        policy.actions_per_step = 1
        policy._processor_bridge = None
        policy.set_robot_state_keys([f"joint_{i}" for i in range(6)])

        policy._policy.predict_action_chunk = MagicMock(return_value=torch.randn(1, 20, 6))

        actions = policy.get_actions_sync({}, "test")

        policy._policy.predict_action_chunk.assert_called_once()
        policy._policy.select_action.assert_not_called()
        assert isinstance(actions, list)
        assert len(actions) >= 1

    def test_get_actions_uses_select_action_when_rtc_disabled(self):
        """get_actions should use select_action when RTC disabled."""
        policy = _make_loaded_policy(action_dim=6, include_images=False)
        policy._rtc_enabled = False
        policy._processor_bridge = None
        policy.set_robot_state_keys([f"joint_{i}" for i in range(6)])

        result = policy.get_actions_sync({}, "test")

        policy._policy.select_action.assert_called_once()
        assert isinstance(result, list)
