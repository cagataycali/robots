"""Tests for strands_robots.policies.lerobot_local — LerobotLocalPolicy.

All tests run WITHOUT lerobot installed (pure mock/unit testing).
torch is mocked via conftest.py when not available.
Integration tests that load real models are in tests_integ/.
"""

import json
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch  # real or conftest mock — both work

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

    def test_custom_max_consecutive_failures(self):
        policy = _make_policy(max_consecutive_failures=10)
        assert policy._max_consecutive_failures == 10

    def test_is_policy_subclass(self):
        from strands_robots.policies import Policy
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        assert issubclass(LerobotLocalPolicy, Policy)

    def test_processor_overrides_stored(self):
        policy = _make_policy(processor_overrides={"step1": {"key": "val"}})
        assert policy.processor_overrides == {"step1": {"key": "val"}}

    def test_use_processor_flag(self):
        policy = _make_policy(use_processor=False)
        assert policy.use_processor is False


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
# Tests: Tokenizer resolution (VLA support)
# ---------------------------------------------------------------------------


class TestResolveTokenizer:
    def test_returns_none_when_not_loaded(self):
        policy = _make_policy()
        assert policy._resolve_tokenizer() is None

    def test_returns_none_when_no_config(self):
        policy = _make_loaded_policy()
        policy._policy.config = None
        assert policy._resolve_tokenizer() is None

    def test_cached_tokenizer_returned(self):
        policy = _make_loaded_policy()
        sentinel = MagicMock()
        policy._tokenizer = sentinel
        assert policy._resolve_tokenizer() is sentinel

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
        # transformers not installed -> Strategy 1 fails with ImportError
        # -> falls through to Strategy 3 (processor.tokenizer)
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
        policy._policy.config = None  # No config -> _resolve_tokenizer returns None
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
    def test_no_config_returns_false(self):
        policy = _make_loaded_policy()
        policy._policy.config = None
        assert policy._needs_language_tokens() is False

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
        """When policy_type is set, should use resolve_policy_class_by_name."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

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
        ) as mock_resolve:
            with patch(
                "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
                return_value=MagicMock(is_active=False),
            ):
                policy._load_model()

        mock_resolve.assert_called_once_with("act")
        assert policy._loaded is True
        assert policy._device == torch.device("cpu")

    def test_load_without_policy_type_resolves_from_hub(self):
        """When policy_type is not set, should use resolve_policy_class_from_hub."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        mock_policy_cls = MagicMock()
        mock_inner = MagicMock()
        mock_inner.config = MagicMock(spec=[])  # No input_features/output_features
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
        """Device should be resolved from config.device if available."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

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

    def test_device_fallback_to_parameters(self):
        """Without config.device, should fallback to next(parameters()).device."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        mock_policy_cls = MagicMock()
        mock_inner = MagicMock(spec=["eval", "parameters", "from_pretrained"])
        mock_inner.eval.return_value = None
        param = torch.nn.Parameter(torch.zeros(1))
        mock_inner.parameters.return_value = iter([param])
        mock_policy_cls.from_pretrained.return_value = mock_inner

        # No config attribute at all
        del mock_inner.config

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

        assert policy._device == param.device

    def test_auto_generates_state_keys_from_output(self):
        """Should auto-generate joint_N keys when output_features.action has shape."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

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

    def test_processor_bridge_loaded_when_active(self):
        """Processor bridge should be kept when is_active is True."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        mock_bridge = MagicMock(is_active=True)
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
                return_value=mock_bridge,
            ):
                policy._load_model()

        assert policy._processor_bridge is mock_bridge

    def test_processor_bridge_none_when_inactive(self):
        """Processor bridge should be set to None when is_active is False."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        mock_bridge = MagicMock(is_active=False)
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
                return_value=mock_bridge,
            ):
                policy._load_model()

        assert policy._processor_bridge is None

    def test_processor_bridge_exception_handled(self):
        """Processor bridge load failure should not crash model loading."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

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
                side_effect=RuntimeError("load failed"),
            ):
                policy._load_model()

        assert policy._processor_bridge is None
        assert policy._loaded is True

    def test_skip_processor_when_disabled(self):
        """Processor bridge should not be loaded when use_processor=False."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        mock_policy_cls = MagicMock()
        mock_inner = MagicMock()
        mock_inner.config = MagicMock(device="cpu", spec=["device"])
        mock_inner.eval.return_value = None
        mock_policy_cls.from_pretrained.return_value = mock_inner

        policy = LerobotLocalPolicy()
        policy.pretrained_name_or_path = "test/model"
        policy.policy_type = "act"
        policy.use_processor = False

        with patch(
            "strands_robots.policies.lerobot_local.policy.resolve_policy_class_by_name",
            return_value=mock_policy_cls,
        ):
            with patch(
                "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
            ) as mock_pb:
                policy._load_model()
                mock_pb.assert_not_called()

        assert policy._processor_bridge is None


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

    def test_no_path_raises_runtime_error(self):
        """Calling get_actions without a model path should raise RuntimeError."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        policy = LerobotLocalPolicy()
        policy.robot_state_keys = ["a", "b"]
        with pytest.raises(RuntimeError, match="No model loaded"):
            policy.get_actions_sync({}, "test")

    def test_consecutive_failure_raises_after_threshold(self):
        policy = _make_loaded_policy(action_dim=3)
        policy.set_robot_state_keys(["a", "b", "c"])
        policy._max_consecutive_failures = 2

        policy._policy.select_action.side_effect = RuntimeError("boom")

        # First failure should raise (no silent zero-return)
        with pytest.raises(RuntimeError, match="boom"):
            policy.get_actions_sync({}, "test")
        assert policy._consecutive_failures == 1

        # Second failure should raise with "failed N consecutive" message
        with pytest.raises(RuntimeError, match="failed 2 consecutive"):
            policy.get_actions_sync({}, "test")

    def test_success_resets_failure_counter(self):
        policy = _make_loaded_policy(action_dim=2)
        policy.set_robot_state_keys(["a", "b"])
        policy._consecutive_failures = 4

        policy.get_actions_sync({}, "test")
        assert policy._consecutive_failures == 0

    def test_processor_bridge_preprocess_bypasses_batch_builder(self):
        """When processor bridge has preprocessor, _build_observation_batch should NOT be called."""
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
        """When processor bridge has postprocessor, action should be postprocessed."""
        policy = _make_loaded_policy(action_dim=2)
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

    def test_instruction_injected_as_task(self):
        """instruction param should be added as 'task' key to observation."""
        policy = _make_loaded_policy(action_dim=2)
        policy.set_robot_state_keys(["a", "b"])

        # We spy on select_action to capture the batch

        def capture_batch(batch):
            # The observation dict passed to _build_observation_batch should have 'task'
            return torch.zeros(2)

        policy._policy.select_action.side_effect = capture_batch
        policy.get_actions_sync({}, "pick up the cube")


# ---------------------------------------------------------------------------
# Tests: _build_observation_batch
# ---------------------------------------------------------------------------


class TestBuildObservationBatch:
    def test_lerobot_format_passthrough(self):
        """Keys starting with 'observation.' should pass through."""
        policy = _make_loaded_policy(state_dim=3)
        observation = {"observation.state": torch.tensor([1.0, 2.0, 3.0])}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.state" in batch

    def test_numpy_state_conversion(self):
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

    def test_scalar_int_conversion(self):
        """Integer scalars in observation should become tensors."""
        policy = _make_loaded_policy()
        observation = {"observation.gripper": 1}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.gripper" in batch
        assert batch["observation.gripper"].shape == (1, 1)

    def test_scalar_float_conversion(self):
        """Float scalars in observation should become tensors."""
        policy = _make_loaded_policy()
        observation = {"observation.effort": 0.75}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.effort" in batch

    def test_tensor_image_hwc_to_chw(self):
        """Torch tensor images in HWC format should be converted to CHW."""
        policy = _make_loaded_policy()
        observation = {
            "observation.images.top": torch.randint(0, 255, (480, 640, 3)).float(),
        }
        batch = policy._build_observation_batch(observation, "test")
        assert batch["observation.images.top"].shape == (1, 3, 480, 640)

    def test_tensor_auto_detect_image_from_shape(self):
        """3D tensor with last dim in (1,3,4) should be auto-detected as image even without 'image' in key."""
        policy = _make_loaded_policy()
        # Add a feature with >= 2D shape to trigger is_image heuristic
        policy._input_features["observation.cam"] = MagicMock(shape=(3, 480, 640))
        observation = {"observation.cam": torch.randint(0, 255, (480, 640, 3)).float()}
        batch = policy._build_observation_batch(observation, "test")
        assert batch["observation.cam"].shape == (1, 3, 480, 640)

    def test_non_numeric_list_skipped(self):
        """Lists that can't be converted to numpy should be skipped."""
        policy = _make_loaded_policy()
        observation = {"observation.labels": ["cat", "dog"]}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.labels" not in batch

    def test_list_2d_image(self):
        """2D+ list values should be handled (e.g. nested list image)."""
        policy = _make_loaded_policy()
        # 2x2 "image" as nested list
        observation = {"observation.images.top": [[[0.5, 0.3, 0.1]], [[0.2, 0.4, 0.6]]]}
        batch = policy._build_observation_batch(observation, "test")
        assert "observation.images.top" in batch

    def test_vla_token_injection(self):
        """VLA models should get tokenized instruction injected."""
        policy = _make_loaded_policy()
        policy._policy.config = MagicMock(tokenizer_name="gpt2", vlm_model_name=None)

        mock_ids = torch.zeros(1, 32, dtype=torch.long)
        mock_mask = torch.ones(1, 32, dtype=torch.bool)
        policy._tokenizer = MagicMock()
        policy._tokenizer.return_value = {"input_ids": mock_ids, "attention_mask": mock_mask}

        policy._input_features["observation.language.tokens"] = MagicMock()
        observation = {"observation.state": torch.zeros(6)}
        batch = policy._build_observation_batch(observation, "pick up the cube")
        assert "observation.language.tokens" in batch

    def test_task_key_filled_for_task_features(self):
        """Models with task in input_features should get task filled."""
        policy = _make_loaded_policy()
        policy._input_features["task"] = MagicMock()
        observation = {"observation.state": torch.zeros(6)}
        batch = policy._build_observation_batch(observation, "pick up")
        assert batch["task"] == "pick up"


# ---------------------------------------------------------------------------
# Tests: _build_batch_from_strands_format
# ---------------------------------------------------------------------------


class TestBuildBatchFromStrandsFormat:
    def test_numpy_floating_state(self):
        """np.float32/64 state values should be collected."""
        policy = _make_loaded_policy(state_dim=2)
        policy.set_robot_state_keys(["a", "b"])
        observation = {"a": np.float32(1.5), "b": np.float64(2.5)}
        batch = policy._build_batch_from_strands_format(observation, {})
        assert "observation.state" in batch
        np.testing.assert_allclose(batch["observation.state"][0].numpy(), [1.5, 2.5], atol=1e-5)

    def test_numpy_integer_state(self):
        """np.int32/64 state values should be collected."""
        policy = _make_loaded_policy(state_dim=2)
        policy.set_robot_state_keys(["a", "b"])
        observation = {"a": np.int32(1), "b": np.int64(2)}
        batch = policy._build_batch_from_strands_format(observation, {})
        assert "observation.state" in batch

    def test_numpy_0d_array_state(self):
        """0-dimensional numpy arrays should be treated as scalars."""
        policy = _make_loaded_policy(state_dim=1)
        policy.set_robot_state_keys(["a"])
        observation = {"a": np.array(3.14)}
        batch = policy._build_batch_from_strands_format(observation, {})
        assert "observation.state" in batch
        assert abs(batch["observation.state"][0, 0].item() - 3.14) < 1e-5

    def test_state_padded_to_expected_dim(self):
        """State values should be zero-padded to match expected dimension."""
        policy = _make_loaded_policy(state_dim=4)
        policy.set_robot_state_keys(["a", "b"])
        observation = {"a": 1.0, "b": 2.0}
        batch = policy._build_batch_from_strands_format(observation, {})
        state = batch["observation.state"]
        assert state.shape == (1, 4)
        assert state[0, 2].item() == 0.0
        assert state[0, 3].item() == 0.0

    def test_state_truncated_to_expected_dim(self):
        """State values beyond expected dimension should be truncated."""
        policy = _make_loaded_policy(state_dim=2)
        policy.set_robot_state_keys(["a", "b", "c"])
        observation = {"a": 1.0, "b": 2.0, "c": 3.0}
        batch = policy._build_batch_from_strands_format(observation, {})
        state = batch["observation.state"]
        assert state.shape == (1, 2)

    def test_image_mapped_to_first_free_slot(self):
        """Camera images should map to the first available image feature slot."""
        policy = _make_loaded_policy()
        policy.set_robot_state_keys([])
        observation = {"cam": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)}
        batch = policy._build_batch_from_strands_format(observation, {})
        assert "observation.images.top" in batch
        assert batch["observation.images.top"].shape == (1, 3, 480, 640)
        assert batch["observation.images.top"].max() <= 1.0

    def test_empty_state_keys_warns(self):
        """Empty robot_state_keys should not crash."""
        policy = _make_loaded_policy()
        policy.robot_state_keys = []
        batch = policy._build_batch_from_strands_format({"x": 1.0}, {})
        assert "observation.state" not in batch

    def test_state_keys_not_in_observation_skipped(self):
        """Missing keys from observation should be silently skipped."""
        policy = _make_loaded_policy(state_dim=3)
        policy.set_robot_state_keys(["a", "b", "c"])
        observation = {"a": 1.0}  # b and c are missing
        batch = policy._build_batch_from_strands_format(observation, {})
        assert "observation.state" in batch
        assert batch["observation.state"].shape == (1, 3)  # padded


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

    def test_3d_tensor_batch(self):
        policy = _make_loaded_policy(action_dim=2)
        policy.set_robot_state_keys(["x", "y"])
        policy.actions_per_step = 1
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        result = policy._tensor_to_action_dicts(tensor)
        assert len(result) == 1
        assert result[0] == {"x": 1.0, "y": 2.0}

    def test_short_tensor_pads_with_zeros(self):
        policy = _make_loaded_policy(action_dim=3)
        policy.set_robot_state_keys(["a", "b", "c"])
        result = policy._tensor_to_action_dicts(torch.tensor([1.0]))
        assert result[0] == {"a": 1.0, "b": 0.0, "c": 0.0}

    def test_empty_state_keys_raises(self):
        """With no robot_state_keys, tensor_to_action_dicts should raise."""
        policy = _make_loaded_policy(action_dim=2)
        policy.robot_state_keys = []
        with pytest.raises(RuntimeError, match="robot_state_keys is empty"):
            policy._tensor_to_action_dicts(torch.tensor([1.0, 2.0]))

    def test_4d_tensor_flattened(self):
        """4D+ tensors should be flattened."""
        policy = _make_loaded_policy(action_dim=4)
        policy.set_robot_state_keys(["a", "b", "c", "d"])
        tensor = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        result = policy._tensor_to_action_dicts(tensor)
        assert len(result) == 1
        assert result[0] == {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}


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

    def test_info_includes_processor(self):
        """When processor bridge is active, info should include processor details."""
        policy = _make_loaded_policy()
        policy.pretrained_name_or_path = "test/model"
        mock_bridge = MagicMock()
        mock_bridge.get_info.return_value = {"has_preprocessor": True, "steps": 3}
        policy._processor_bridge = mock_bridge

        info = policy.get_model_info()
        assert "processor" in info
        assert info["processor"]["has_preprocessor"] is True

    def test_info_device_none_when_not_loaded(self):
        policy = _make_policy()
        info = policy.get_model_info()
        assert info["device"] is None

    def test_info_features_serialized(self):
        """Feature shapes should be serialized as strings."""
        policy = _make_loaded_policy()
        policy.pretrained_name_or_path = "test/model"
        info = policy.get_model_info()
        assert "input_features" in info
        # Values should be string representations
        for v in info["input_features"].values():
            assert isinstance(v, str)


# ---------------------------------------------------------------------------
# Tests: reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_delegates_to_inner_policy(self):
        """reset() must call the inner LeRobot policy's reset()."""
        policy = _make_policy()
        policy._loaded = True
        policy._policy = MagicMock()
        policy._policy.reset = MagicMock()
        policy._processor_bridge = None

        policy.reset()
        policy._policy.reset.assert_called_once()

    def test_reset_resets_processor_bridge(self):
        """reset() must also reset the processor bridge if present."""
        policy = _make_policy()
        policy._loaded = True
        policy._policy = MagicMock()
        mock_bridge = MagicMock()
        policy._processor_bridge = mock_bridge

        policy.reset()
        mock_bridge.reset.assert_called_once()

    def test_reset_clears_failure_counter(self):
        """reset() must clear the consecutive failures counter."""
        policy = _make_policy()
        policy._loaded = True
        policy._policy = MagicMock()
        policy._processor_bridge = None
        policy._consecutive_failures = 3

        policy.reset()
        assert policy._consecutive_failures == 0

    def test_reset_safe_when_not_loaded(self):
        """reset() should not raise when policy not loaded yet."""
        policy = _make_policy()
        assert policy._policy is None
        policy.reset()  # Should not raise

    def test_reset_handles_policy_without_reset_method(self):
        """If inner policy has no reset(), should not crash."""
        policy = _make_policy()
        policy._loaded = True
        policy._policy = MagicMock(spec=[])  # No reset method
        policy._processor_bridge = None
        policy.reset()  # Should not raise


# ---------------------------------------------------------------------------
# Tests: select_action_sync
# ---------------------------------------------------------------------------


class TestSelectActionSync:
    def test_returns_numpy_array(self):
        policy = _make_loaded_policy(action_dim=4)
        policy._policy.select_action.return_value = torch.tensor([1.0, 2.0, 3.0, 4.0])

        result = policy.select_action_sync({}, "test")
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])

    def test_instruction_from_task_key(self):
        policy = _make_loaded_policy(action_dim=2)
        policy._policy.select_action.return_value = torch.tensor([0.0, 0.0])

        result = policy.select_action_sync({"task": "pick up the cube"}, "")
        assert isinstance(result, np.ndarray)

    def test_processor_bridge_preprocess_bypasses_batch_builder(self):
        """select_action_sync should also bypass _build_observation_batch with preprocessor."""
        policy = _make_loaded_policy(action_dim=2)

        mock_bridge = MagicMock()
        mock_bridge.has_preprocessor = True
        mock_bridge.has_postprocessor = False
        mock_bridge.preprocess.return_value = {
            "observation.state": torch.zeros(1, 2),
        }
        policy._processor_bridge = mock_bridge

        with patch.object(policy, "_build_observation_batch") as mock_build:
            result = policy.select_action_sync({}, "test")
            mock_build.assert_not_called()

        assert isinstance(result, np.ndarray)

    def test_processor_bridge_postprocess_applied(self):
        """select_action_sync should apply postprocessor."""
        policy = _make_loaded_policy(action_dim=2)

        mock_bridge = MagicMock()
        mock_bridge.has_preprocessor = False
        mock_bridge.has_postprocessor = True
        mock_bridge.postprocess.return_value = torch.tensor([5.0, 6.0])
        policy._processor_bridge = mock_bridge

        result = policy.select_action_sync({}, "test")
        np.testing.assert_array_almost_equal(result, [5.0, 6.0])

    def test_non_tensor_result_converted(self):
        """If select_action returns non-tensor, should still work."""
        policy = _make_loaded_policy(action_dim=2)
        policy._policy.select_action.return_value = np.array([1.0, 2.0])

        result = policy.select_action_sync({}, "test")
        assert isinstance(result, np.ndarray)

    def test_auto_loads_model(self):
        """select_action_sync should auto-load model if not loaded."""
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        policy = LerobotLocalPolicy()
        policy.pretrained_name_or_path = "test/model"

        def fake_load():
            policy._loaded = True
            policy._device = torch.device("cpu")
            mock_inner = MagicMock()
            mock_inner.select_action.return_value = torch.zeros(2)
            policy._policy = mock_inner
            policy._input_features = {}
            policy._output_features = {}

        with patch.object(policy, "_load_model", side_effect=fake_load):
            result = policy.select_action_sync({}, "test")
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

    def test_resolve_by_name_modeling_submodule(self):
        """Strategy 1: should find PolicyClass in lerobot.policies.{type}.modeling_{type}."""
        from strands_robots.policies.lerobot_local.resolution import resolve_policy_class_by_name

        mock_policy_class = type("ACTPolicy", (), {"from_pretrained": classmethod(lambda cls: None)})

        mock_module = types.ModuleType("lerobot.policies.act.modeling_act")
        mock_module.ACTPolicy = mock_policy_class

        with patch("importlib.import_module", return_value=mock_module):
            result = resolve_policy_class_by_name("act")
            assert result is mock_policy_class

    def test_resolve_by_name_package_level(self):
        """Strategy 2: should find PolicyClass in lerobot.policies.{type}."""
        from strands_robots.policies.lerobot_local.resolution import resolve_policy_class_by_name

        mock_policy_class = type("DiffusionPolicy", (), {"from_pretrained": classmethod(lambda cls: None)})
        mock_module = types.ModuleType("lerobot.policies.diffusion")
        mock_module.DiffusionPolicy = mock_policy_class

        call_count = [0]

        def mock_import(name, *args, **kwargs):
            call_count[0] += 1
            # Fail for modeling_* submodules (strategy 1)
            if "modeling_" in name or call_count[0] <= 2:
                raise ImportError(f"No module named '{name}'")
            return mock_module

        with patch("importlib.import_module", side_effect=mock_import):
            result = resolve_policy_class_by_name("diffusion")
            assert result is mock_policy_class

    def test_resolve_by_name_legacy_factory(self):
        """Strategy 3: should use lerobot.policies.factory.get_policy_class."""
        import sys

        from strands_robots.policies.lerobot_local.resolution import resolve_policy_class_by_name

        mock_policy_class = type("TDMPCPolicy", (), {"from_pretrained": classmethod(lambda cls: None)})

        # Make all import_module calls fail (strategies 1 & 2)
        # but inject lerobot.policies.factory into sys.modules for strategy 3
        mock_factory = MagicMock()
        mock_factory.get_policy_class.return_value = mock_policy_class
        sys.modules["lerobot.policies.factory"] = mock_factory
        try:
            with patch(
                "strands_robots.policies.lerobot_local.resolution.importlib.import_module",
                side_effect=ImportError("mocked"),
            ):
                result = resolve_policy_class_by_name("tdmpc")
                assert result is mock_policy_class
        finally:
            sys.modules.pop("lerobot.policies.factory", None)

    def test_resolve_from_hub_pretrained_config_success(self):
        """Strategy 1: PreTrainedConfig draccus resolution should work."""
        from strands_robots.policies.lerobot_local.resolution import resolve_policy_class_from_hub

        mock_config = MagicMock()
        mock_config.type = "act"

        mock_policy_class = type("ACTPolicy", (), {"from_pretrained": classmethod(lambda cls: None)})

        with patch.dict(
            "sys.modules",
            {
                "lerobot.configs.policies": MagicMock(
                    PreTrainedConfig=MagicMock(from_pretrained=MagicMock(return_value=mock_config))
                ),
            },
        ):
            with patch(
                "strands_robots.policies.lerobot_local.resolution.resolve_policy_class_by_name",
                return_value=mock_policy_class,
            ):
                cls, policy_type = resolve_policy_class_from_hub("test/model")
                assert cls is mock_policy_class
                assert policy_type == "act"

    def test_read_policy_type_from_local_config(self, tmp_path):
        """Should read 'type' from local config.json."""
        from strands_robots.policies.lerobot_local.resolution import _read_policy_type_from_config

        config_dir = tmp_path / "model"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({"type": "act"}))

        result = _read_policy_type_from_config(str(config_dir))
        assert result == "act"

    def test_read_policy_type_from_local_config_missing_type(self, tmp_path):
        """Should return None if config.json has no 'type' field."""
        from strands_robots.policies.lerobot_local.resolution import _read_policy_type_from_config

        config_dir = tmp_path / "model"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({"version": "1.0"}))

        result = _read_policy_type_from_config(str(config_dir))
        assert result is None

    def test_read_policy_type_from_hub(self):
        """Should try huggingface_hub download as fallback."""
        from strands_robots.policies.lerobot_local.resolution import _read_policy_type_from_config

        with patch.dict(
            "sys.modules",
            {
                "huggingface_hub": MagicMock(hf_hub_download=MagicMock(side_effect=OSError("network error"))),
            },
        ):
            result = _read_policy_type_from_config("nonexistent/model")
            assert result is None


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

    def test_active_bridge_preprocess(self):
        """Active preprocessor should delegate to pipeline.process_observation."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        mock_pre = MagicMock()
        mock_pre.process_observation.return_value = {"processed": True}
        bridge = ProcessorBridge(preprocessor=mock_pre)
        # Inject mock modules so bridge thinks lerobot is available
        bridge._modules = {"DataProcessorPipeline": MagicMock()}

        result = bridge.preprocess({"raw": True})
        assert result == {"processed": True}
        mock_pre.process_observation.assert_called_once_with({"raw": True})

    def test_active_bridge_postprocess(self):
        """Active postprocessor should delegate to pipeline.process_action."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        mock_post = MagicMock()
        mock_post.process_action.return_value = torch.tensor([1.0, 2.0])
        bridge = ProcessorBridge(postprocessor=mock_post)
        bridge._modules = {"DataProcessorPipeline": MagicMock()}

        bridge.postprocess(torch.tensor([0.5, 0.5]))
        mock_post.process_action.assert_called_once()

    def test_preprocess_raises_on_pipeline_error(self):
        """Preprocessor pipeline failure should raise RuntimeError."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        mock_pre = MagicMock()
        mock_pre.process_observation.side_effect = ValueError("bad data")
        bridge = ProcessorBridge(preprocessor=mock_pre)
        bridge._modules = {"DataProcessorPipeline": MagicMock()}

        with pytest.raises(RuntimeError, match="Preprocessor pipeline failed"):
            bridge.preprocess({})

    def test_postprocess_raises_on_pipeline_error(self):
        """Postprocessor pipeline failure should raise RuntimeError."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        mock_post = MagicMock()
        mock_post.process_action.side_effect = ValueError("bad action")
        bridge = ProcessorBridge(postprocessor=mock_post)
        bridge._modules = {"DataProcessorPipeline": MagicMock()}

        with pytest.raises(RuntimeError, match="Postprocessor pipeline failed"):
            bridge.postprocess(torch.zeros(2))

    def test_reset_with_active_pipelines(self):
        """Reset should delegate to both preprocessor and postprocessor."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        mock_pre = MagicMock()
        mock_post = MagicMock()
        bridge = ProcessorBridge(preprocessor=mock_pre, postprocessor=mock_post)

        bridge.reset()
        mock_pre.reset.assert_called_once()
        mock_post.reset.assert_called_once()

    def test_reset_with_no_pipelines(self):
        """Reset should not crash when no pipelines are loaded."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        bridge = ProcessorBridge()
        bridge.reset()  # Should not raise

    def test_get_info_with_active_pipelines(self):
        """Info should include step counts and names for active pipelines."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        mock_step1 = type("NormalizeStep", (), {})()
        mock_step2 = type("DeviceStep", (), {})()
        mock_pre = MagicMock()
        mock_pre.__len__ = MagicMock(return_value=2)
        mock_pre.steps = [mock_step1, mock_step2]

        mock_post = MagicMock()
        mock_post.__len__ = MagicMock(return_value=1)
        mock_post.steps = [mock_step1]

        bridge = ProcessorBridge(preprocessor=mock_pre, postprocessor=mock_post)
        info = bridge.get_info()

        assert info["has_preprocessor"] is True
        assert info["has_postprocessor"] is True
        assert info["is_active"] is True
        assert info["preprocessor_steps"] == 2
        assert info["postprocessor_steps"] == 1
        assert "NormalizeStep" in info["preprocessor_step_names"]

    def test_repr_with_active_pipelines(self):
        """Repr should show step counts for active pipelines."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        mock_pre = MagicMock()
        mock_pre.__len__ = MagicMock(return_value=3)
        mock_post = MagicMock()
        mock_post.__len__ = MagicMock(return_value=1)

        bridge = ProcessorBridge(preprocessor=mock_pre, postprocessor=mock_post)
        r = repr(bridge)
        assert "pre=3steps" in r
        assert "post=1steps" in r

    def test_from_pretrained_passthrough_when_no_lerobot(self):
        """from_pretrained should return passthrough bridge when lerobot is unavailable."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        with patch("strands_robots.policies.lerobot_local.processor._try_import_processor", return_value=None):
            bridge = ProcessorBridge.from_pretrained("test/model")
            assert not bridge.is_active

    def test_from_pretrained_loads_both_pipelines(self):
        """from_pretrained should load both preprocessor and postprocessor."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        mock_pipeline = MagicMock()
        mock_pipeline.from_pretrained.return_value = MagicMock(__len__=MagicMock(return_value=2))

        mock_modules = {
            "DataProcessorPipeline": mock_pipeline,
        }

        with patch("strands_robots.policies.lerobot_local.processor._try_import_processor", return_value=mock_modules):
            bridge = ProcessorBridge.from_pretrained("test/model", device="cpu")

        assert bridge.has_preprocessor
        assert bridge.has_postprocessor
        assert mock_pipeline.from_pretrained.call_count == 2

    def test_from_pretrained_handles_missing_preprocessor(self):
        """Missing preprocessor config should result in preprocessor=None."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        mock_pipeline = MagicMock()

        def selective_load(path, config_filename=None, overrides=None):
            if "preprocessor" in config_filename:
                raise FileNotFoundError("no preprocessor")
            return MagicMock(__len__=MagicMock(return_value=1))

        mock_pipeline.from_pretrained.side_effect = selective_load
        mock_modules = {"DataProcessorPipeline": mock_pipeline}

        with patch("strands_robots.policies.lerobot_local.processor._try_import_processor", return_value=mock_modules):
            bridge = ProcessorBridge.from_pretrained("test/model")

        assert not bridge.has_preprocessor
        assert bridge.has_postprocessor

    def test_from_pretrained_handles_missing_postprocessor(self):
        """Missing postprocessor config should result in postprocessor=None."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        mock_pipeline = MagicMock()

        def selective_load(path, config_filename=None, overrides=None):
            if "postprocessor" in config_filename:
                raise FileNotFoundError("no postprocessor")
            return MagicMock(__len__=MagicMock(return_value=1))

        mock_pipeline.from_pretrained.side_effect = selective_load
        mock_modules = {"DataProcessorPipeline": mock_pipeline}

        with patch("strands_robots.policies.lerobot_local.processor._try_import_processor", return_value=mock_modules):
            bridge = ProcessorBridge.from_pretrained("test/model")

        assert bridge.has_preprocessor
        assert not bridge.has_postprocessor

    def test_from_pretrained_handles_oserror(self):
        """OSError during pipeline load should be handled gracefully."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        mock_pipeline = MagicMock()
        mock_pipeline.from_pretrained.side_effect = OSError("disk error")
        mock_modules = {"DataProcessorPipeline": mock_pipeline}

        with patch("strands_robots.policies.lerobot_local.processor._try_import_processor", return_value=mock_modules):
            bridge = ProcessorBridge.from_pretrained("test/model")

        assert not bridge.has_preprocessor
        assert not bridge.has_postprocessor

    def test_from_pretrained_with_overrides(self):
        """Overrides should be passed to both pipeline loads."""
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        mock_pipeline = MagicMock()
        mock_pipeline.from_pretrained.return_value = MagicMock(__len__=MagicMock(return_value=1))
        mock_modules = {"DataProcessorPipeline": mock_pipeline}

        overrides = {"normalize": {"mean": [0.5]}}
        with patch("strands_robots.policies.lerobot_local.processor._try_import_processor", return_value=mock_modules):
            ProcessorBridge.from_pretrained("test/model", overrides=overrides)

        # Both calls should have received overrides
        for c in mock_pipeline.from_pretrained.call_args_list:
            assert c.kwargs.get("overrides") == overrides or c[1].get("overrides") == overrides


# ---------------------------------------------------------------------------
# Tests: _try_import_processor
# ---------------------------------------------------------------------------


class TestTryImportProcessor:
    def test_returns_none_when_lerobot_missing(self):
        """Should return None when lerobot is not installed."""
        from strands_robots.policies.lerobot_local.processor import _try_import_processor

        with patch.dict("sys.modules", {"lerobot.processor.pipeline": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = _try_import_processor()
                assert result is None

    def test_returns_dict_when_lerobot_available(self):
        """Should return dict of processor classes when lerobot is available."""
        from strands_robots.policies.lerobot_local.processor import _try_import_processor

        # Mock the entire lerobot.processor module tree
        mock_pipeline = MagicMock()
        mock_converters = MagicMock()
        mock_core = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.processor": MagicMock(),
                "lerobot.processor.pipeline": mock_pipeline,
                "lerobot.processor.converters": mock_converters,
                "lerobot.processor.core": mock_core,
            },
        ):
            result = _try_import_processor()
            if result is not None:
                assert "DataProcessorPipeline" in result
