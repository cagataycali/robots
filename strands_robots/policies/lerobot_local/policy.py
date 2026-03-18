"""LeRobot Local Policy — Direct HuggingFace model inference (no server needed).

Uses LeRobot's own factory for auto-detection.
No hardcoded policy classes — any model LeRobot supports, we support.

Architecture:
    Observation (dict)
        → ProcessorBridge.preprocess (normalize, device, crop, ...)
        → LeRobot PreTrainedPolicy.select_action
        → ProcessorBridge.postprocess (unnormalize, delta-action, ...)
        → Action dict
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .. import Policy
from .resolution import resolve_policy_class_from_hub, resolve_policy_class_by_name
from .processor import ProcessorBridge

logger = logging.getLogger(__name__)


class LerobotLocalPolicy(Policy):
    """Policy that loads and runs LeRobot models directly (no server).

    Auto-detects policy type from HF config.json → delegates to LeRobot's
    own class registry. Zero hardcoded policy mappings.

    Optionally loads the model's processor pipeline (preprocessor.json /
    postprocessor.json) for automatic normalization, device transfer,
    observation formatting, and action unnormalization.
    """

    def __init__(
        self,
        pretrained_name_or_path: str = "",
        policy_type: str = None,
        device: str = None,
        actions_per_step: int = 1,
        use_processor: bool = True,
        processor_overrides: dict = None,
        **kwargs,
    ):
        self.pretrained_name_or_path = pretrained_name_or_path
        self.policy_type = policy_type
        self.requested_device = device
        self.actions_per_step = actions_per_step
        self.use_processor = use_processor
        self.processor_overrides = processor_overrides
        self.robot_state_keys: List[str] = []

        self._policy = None
        self._device = None
        self._input_features = {}
        self._output_features = {}
        self._loaded = False
        self._processor_bridge = None
        self._tokenizer = None
        self._tokenizer_max_length: int = 48
        self._tokenizer_padding_side: str = "right"
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

        if pretrained_name_or_path:
            self._load_model()

    @property
    def provider_name(self) -> str:
        return "lerobot_local"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        """Set robot state keys for observation→tensor mapping.

        If robot_state_keys is provided and non-empty, uses those directly.
        If empty, auto-detects from the model's output_features (action dimension).

        Auto-detection:
        1. Uses provided keys if non-empty
        2. Falls back to model output_features action dimension
        3. Falls back to model input_features state dimension
        4. Last resort: generic joint_0..joint_N keys
        """
        if robot_state_keys:
            self.robot_state_keys = robot_state_keys
            logger.info(
                "LeRobot local state keys set: %d keys = %s%s",
                len(self.robot_state_keys),
                self.robot_state_keys[:5],
                "..." if len(self.robot_state_keys) > 5 else "",
            )
            return

        # Auto-detect from model config
        if self._loaded and self._output_features:
            action_feat = self._output_features.get("action")
            if action_feat and hasattr(action_feat, "shape") and action_feat.shape:
                action_dim = action_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(action_dim)]
                logger.warning(
                    "robot_state_keys empty — auto-detected %d keys from "
                    "model output_features.action.shape=%s. "
                    "For meaningful names, pass the robot's actual joint names.",
                    action_dim,
                    action_feat.shape,
                )
                return

        if self._loaded and self._input_features:
            state_feat = self._input_features.get("observation.state")
            if state_feat and hasattr(state_feat, "shape") and state_feat.shape:
                state_dim = state_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(state_dim)]
                logger.warning(
                    "robot_state_keys empty — auto-detected %d keys from "
                    "model input_features.observation.state.shape=%s.",
                    state_dim,
                    state_feat.shape,
                )
                return

        logger.warning(
            "robot_state_keys empty and no model features available for auto-detection. "
            "Actions will be empty dicts. Call set_robot_state_keys() with actual joint names."
        )

    # ------------------------------------------------------------------
    # Tokenizer resolution (VLA language token injection)
    # ------------------------------------------------------------------

    def _resolve_tokenizer(self):
        """Resolve and cache the tokenizer for VLA language token injection.

        Resolution order:
            1. Explicit ``tokenizer_name`` on policy config (e.g. xvla)
            2. ``vlm_model_name`` on policy config (e.g. SmolVLA → SmolVLM2)
            3. Policy's own ``.processor.tokenizer`` (e.g. Paligemma-based)

        Returns the tokenizer or ``None``.
        """
        if self._tokenizer is not None:
            return self._tokenizer

        if not self._loaded or not self._policy:
            return None

        config = getattr(self._policy, "config", None)
        if config is None:
            return None

        self._tokenizer_max_length = getattr(config, "tokenizer_max_length", 48)
        self._tokenizer_padding_side = getattr(config, "tokenizer_padding_side", "right")

        # 1. tokenizer_name (explicit)
        tokenizer_id = getattr(config, "tokenizer_name", None)

        # 2. vlm_model_name (SmolVLA, etc.)
        if not tokenizer_id:
            tokenizer_id = getattr(config, "vlm_model_name", None)

        if tokenizer_id:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
                self._tokenizer.padding_side = self._tokenizer_padding_side
                logger.info("Auto-resolved tokenizer from '%s' (%s)", tokenizer_id, type(self._tokenizer).__name__)
                return self._tokenizer
            except Exception as exc:
                logger.warning("Failed to load tokenizer from '%s': %s", tokenizer_id, exc)

        # 3. policy.processor.tokenizer
        processor = getattr(self._policy, "processor", None)
        if processor and hasattr(processor, "tokenizer"):
            self._tokenizer = processor.tokenizer
            self._tokenizer.padding_side = self._tokenizer_padding_side
            logger.info("Using policy's built-in processor tokenizer (%s)", type(self._tokenizer).__name__)
            return self._tokenizer

        return None

    def _tokenize_instruction(self, instruction: str):
        """Tokenize an instruction into ``(input_ids, attention_mask)`` tensors.

        Returns ``None`` if no tokenizer is available.
        """
        tokenizer = self._resolve_tokenizer()
        if tokenizer is None or not instruction:
            return None

        encoded = tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self._tokenizer_max_length,
            truncation=True,
        )
        tokens = encoded["input_ids"].to(self._device)
        mask = encoded.get("attention_mask")
        if mask is not None:
            mask = mask.bool().to(self._device)
        return tokens, mask

    def _needs_language_tokens(self) -> bool:
        """Check whether this policy requires observation.language.tokens."""
        config = getattr(self._policy, "config", None)
        if config is None:
            return False

        if getattr(config, "tokenizer_name", None):
            return True
        if getattr(config, "vlm_model_name", None):
            return True
        if any("language" in key for key in self._input_features):
            return True

        return False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the LeRobot model from pretrained path."""
        import warnings

        import torch

        warnings.filterwarnings("ignore", message=".*Device.*")

        # XVLA compat: Florence2LanguageConfig.forced_bos_token_id missing in transformers 5.x
        try:
            from transformers.models.florence2.configuration_florence2 import Florence2LanguageConfig

            if not hasattr(Florence2LanguageConfig, "forced_bos_token_id"):
                Florence2LanguageConfig.forced_bos_token_id = None
                logger.debug("Patched Florence2LanguageConfig.forced_bos_token_id")
        except (ImportError, Exception):
            pass

        logger.info("Loading %s...", self.pretrained_name_or_path)
        start = time.time()

        # Resolve the correct policy class — zero hardcoding
        if self.policy_type:
            PolicyClass = resolve_policy_class_by_name(self.policy_type)
        else:
            PolicyClass, self.policy_type = resolve_policy_class_from_hub(self.pretrained_name_or_path)

        # Suppress stale state-dict key warnings from old checkpoints
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.ERROR)
        try:
            self._policy = PolicyClass.from_pretrained(self.pretrained_name_or_path)
        finally:
            root_logger.setLevel(previous_level)

        self._policy.eval()
        self._device = next(self._policy.parameters()).device

        if hasattr(self._policy, "config"):
            config = self._policy.config
            if hasattr(config, "input_features"):
                self._input_features = config.input_features
            if hasattr(config, "output_features"):
                self._output_features = config.output_features

        elapsed = time.time() - start
        logger.info("Loaded %s (type='%s') in %.1fs on %s", type(self._policy).__name__, self.policy_type, elapsed, self._device)
        self._loaded = True

        # Auto-detect robot_state_keys from model config if not set
        if not self.robot_state_keys and self._output_features:
            action_feat = self._output_features.get("action")
            if action_feat and hasattr(action_feat, "shape") and action_feat.shape:
                action_dim = action_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(action_dim)]
                logger.warning(
                    "robot_state_keys not set — auto-generated %d generic keys "
                    "(joint_0..joint_%d). Set explicit keys with "
                    "set_robot_state_keys() for meaningful joint names.",
                    action_dim,
                    action_dim - 1,
                )

        # Load processor pipeline (preprocessor + postprocessor)
        if self.use_processor and self.pretrained_name_or_path:
            try:
                self._processor_bridge = ProcessorBridge.from_pretrained(
                    self.pretrained_name_or_path,
                    device=str(self._device) if self._device else None,
                    overrides=self.processor_overrides or {},
                )
                if self._processor_bridge.is_active:
                    logger.info("Processor bridge loaded: %s", self._processor_bridge)
                else:
                    self._processor_bridge = None
                    logger.debug("No processor configs found, using raw obs/action flow")
            except Exception as exc:
                logger.debug("Processor bridge not loaded: %s", exc)
                self._processor_bridge = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Get actions from policy given observation and instruction."""
        if not self._loaded:
            if self.pretrained_name_or_path:
                self._load_model()
            else:
                return self._zero_actions()

        import torch

        try:
            observation = dict(observation_dict)
            if instruction and "task" not in observation:
                observation["task"] = instruction

            if self._processor_bridge and self._processor_bridge.has_preprocessor:
                observation = self._processor_bridge.preprocess(observation)

            batch = self._build_observation_batch(observation, instruction)
            with torch.no_grad():
                action_tensor = self._policy.select_action(batch)

            if self._processor_bridge and self._processor_bridge.has_postprocessor:
                action_tensor = self._processor_bridge.postprocess(action_tensor)

            self._consecutive_failures = 0
            return self._tensor_to_action_dicts(action_tensor)
        except Exception as exc:
            self._consecutive_failures += 1
            logger.error("Inference error (%d/%d): %s", self._consecutive_failures, self._max_consecutive_failures, exc)
            if self._consecutive_failures >= self._max_consecutive_failures:
                raise RuntimeError(
                    f"LeRobot policy failed {self._consecutive_failures} consecutive times, last error: {exc}"
                ) from exc
            return self._zero_actions()

    def select_action_sync(self, observation_dict: Dict[str, Any], instruction: str = "") -> np.ndarray:
        """Synchronous inference — returns raw action numpy array.

        Convenience for simulation loops that don't need async.
        Applies processor pipeline if available.

        For VLA models (xvla, smolvla, etc.) that require language tokens:
        Pass ``instruction`` or include a ``'task'`` key in observation_dict.

        Args:
            observation_dict: Observation dict (state + images).
            instruction: Natural language instruction for VLA models.
        """
        import torch

        if not self._loaded:
            self._load_model()

        if not instruction:
            instruction = observation_dict.get("task", "")

        observation = dict(observation_dict)
        if instruction and "task" not in observation:
            observation["task"] = instruction

        if self._processor_bridge and self._processor_bridge.has_preprocessor:
            observation = self._processor_bridge.preprocess(observation)

        batch = self._build_observation_batch(observation, instruction)
        with torch.no_grad():
            action_tensor = self._policy.select_action(batch)

        if self._processor_bridge and self._processor_bridge.has_postprocessor:
            action_tensor = self._processor_bridge.postprocess(action_tensor)

        if hasattr(action_tensor, "cpu"):
            return action_tensor.cpu().numpy()
        return np.asarray(action_tensor)

    # ------------------------------------------------------------------
    # Observation batch building
    # ------------------------------------------------------------------

    def _build_observation_batch(self, observation_dict: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """Convert observation dict to LeRobot-compatible batch tensors."""
        import torch

        batch = {}

        has_lerobot_keys = any(key.startswith("observation.") for key in observation_dict)
        if has_lerobot_keys:
            batch = self._build_batch_from_lerobot_format(observation_dict, batch)
        else:
            batch = self._build_batch_from_strands_format(observation_dict, batch)

        # VLA language token injection
        if instruction and "observation.language.tokens" not in batch and self._needs_language_tokens():
            try:
                result = self._tokenize_instruction(instruction)
                if result is not None:
                    tokens, mask = result
                    batch["observation.language.tokens"] = tokens
                    if mask is not None:
                        batch["observation.language.attention_mask"] = mask
                    logger.debug("VLA tokenized instruction: '%s...' -> %s", instruction[:50], tokens.shape)
            except Exception as exc:
                logger.debug("VLA tokenization fallback failed: %s", exc)

        # Fill task key for models that expect it
        if instruction and has_lerobot_keys:
            needs_task = any("task" in key for key in self._input_features) and "task" not in batch
            if needs_task:
                for feat_name in self._input_features:
                    if "task" in feat_name and feat_name not in batch:
                        batch[feat_name] = instruction

        # Fill missing image features with zeros
        for feat_name, feat_info in self._input_features.items():
            if feat_name not in batch and "image" in feat_name:
                shape = feat_info.shape if hasattr(feat_info, "shape") else (3, 480, 640)
                batch[feat_name] = torch.zeros(1, *shape, device=self._device)

        return batch

    def _build_batch_from_lerobot_format(self, observation_dict: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        """Build batch from observation dict already in LeRobot format (observation.* keys)."""
        import torch

        for key, value in observation_dict.items():
            is_image = "image" in key or (
                key in self._input_features
                and hasattr(self._input_features.get(key), "shape")
                and len(getattr(self._input_features.get(key), "shape", ())) >= 2
            )

            if isinstance(value, torch.Tensor):
                tensor = value
                if not is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                    is_image = True
                if is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                    tensor = tensor.permute(2, 0, 1)
                if is_image and tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                elif tensor.dim() < 2 and not is_image:
                    tensor = tensor.unsqueeze(0)
                batch[key] = tensor.to(self._device)

            elif isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value.copy()).float()
                if not is_image and value.ndim == 3 and value.shape[-1] in (1, 3, 4):
                    is_image = True
                if is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                    tensor = tensor.permute(2, 0, 1)
                if is_image and value.dtype == np.uint8:
                    tensor = tensor / 255.0
                if is_image and tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                elif value.ndim < 2 and not is_image:
                    tensor = tensor.unsqueeze(0)
                batch[key] = tensor.to(self._device)

            elif isinstance(value, (int, float)):
                batch[key] = torch.tensor([value], dtype=torch.float32).unsqueeze(0).to(self._device)

            elif isinstance(value, (list, tuple)):
                try:
                    array = np.array(value, dtype=np.float32)
                except (ValueError, TypeError):
                    continue
                tensor = torch.from_numpy(array).float()
                if array.ndim >= 2:
                    if is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                        tensor = tensor.permute(2, 0, 1)
                    if is_image and array.dtype == np.uint8:
                        tensor = tensor / 255.0
                    if is_image and tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    batch[key] = tensor.to(self._device)
                else:
                    batch[key] = tensor.unsqueeze(0).to(self._device)
            # Pass through non-tensor types (e.g. already-batched int64 tokens)

        return batch

    def _build_batch_from_strands_format(self, observation_dict: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        """Build batch from strands-robots native observation format."""
        import torch

        if not self.robot_state_keys:
            logger.warning(
                "robot_state_keys is empty — observation.state will be skipped. "
                "Call set_robot_state_keys() with the robot's motor names for proper state handling."
            )

        # State values
        state_values = []
        for key in self.robot_state_keys:
            if key in observation_dict:
                value = observation_dict[key]
                if isinstance(value, (int, float)):
                    state_values.append(float(value))
                elif isinstance(value, (np.floating, np.integer)):
                    state_values.append(float(value))
                elif isinstance(value, np.ndarray) and value.ndim == 0:
                    state_values.append(float(value))

        if state_values:
            state_feature = self._input_features.get("observation.state")
            if state_feature:
                expected_dim = state_feature.shape[0] if hasattr(state_feature, "shape") else len(state_values)
                while len(state_values) < expected_dim:
                    state_values.append(0.0)
                state_values = state_values[:expected_dim]
            batch["observation.state"] = torch.tensor(state_values, dtype=torch.float32).unsqueeze(0).to(self._device)

        # Camera images
        for key, value in observation_dict.items():
            if key in self.robot_state_keys:
                continue
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                image_tensor = torch.from_numpy(value.copy()).float()
                if image_tensor.dim() == 3 and image_tensor.shape[-1] in (1, 3, 4):
                    image_tensor = image_tensor.permute(2, 0, 1)
                if value.dtype == np.uint8:
                    image_tensor = image_tensor / 255.0
                for feat_name in self._input_features:
                    if "image" in feat_name and feat_name not in batch:
                        batch[feat_name] = image_tensor.unsqueeze(0).to(self._device)
                        break

        return batch

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    def _tensor_to_action_dicts(self, action_tensor) -> List[Dict[str, Any]]:
        """Convert action tensor to list of robot action dicts."""
        action_array = action_tensor.cpu().numpy()

        if action_array.ndim == 1:
            actions_list = [action_array]
        elif action_array.ndim == 2:
            actions_list = [action_array[i] for i in range(min(len(action_array), self.actions_per_step))]
        elif action_array.ndim == 3:
            actions_list = [action_array[0, i] for i in range(min(action_array.shape[1], self.actions_per_step))]
        else:
            actions_list = [action_array.flatten()]

        result = []
        for action_values in actions_list:
            action_dict = {}
            for index, key in enumerate(self.robot_state_keys):
                action_dict[key] = float(action_values[index]) if index < len(action_values) else 0.0
            result.append(action_dict)

        return result if result else self._zero_actions()

    def _zero_actions(self) -> List[Dict[str, Any]]:
        """Return zero-valued action dicts as fallback."""
        return [{key: 0.0 for key in self.robot_state_keys}]

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        info = {
            "provider": "lerobot_local",
            "model_id": self.pretrained_name_or_path,
            "policy_type": self.policy_type,
            "loaded": self._loaded,
            "device": str(self._device) if self._device else None,
        }
        if self._loaded:
            info["input_features"] = {
                key: str(val.shape) if hasattr(val, "shape") else str(val)
                for key, val in self._input_features.items()
            }
            info["output_features"] = {
                key: str(val.shape) if hasattr(val, "shape") else str(val)
                for key, val in self._output_features.items()
            }
            info["policy_class"] = type(self._policy).__name__
            info["n_parameters"] = sum(param.numel() for param in self._policy.parameters())
        if self._processor_bridge:
            info["processor"] = self._processor_bridge.get_info()
        return info
