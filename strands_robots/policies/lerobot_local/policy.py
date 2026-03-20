"""LeRobot Local Policy — Direct HuggingFace model inference (no server needed).

Uses LeRobot's own factory for auto-detection. Any model LeRobot supports,
this policy supports.

Architecture:
    Observation (dict)
        → ProcessorBridge.preprocess (normalize, device, crop, ...)
        → LeRobot PreTrainedPolicy.select_action
        → ProcessorBridge.postprocess (unnormalize, delta-action, ...)
        → Action dict
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .. import Policy
from .processor import ProcessorBridge
from .resolution import resolve_policy_class_by_name, resolve_policy_class_from_hub

logger = logging.getLogger(__name__)


class LerobotLocalPolicy(Policy):
    """Policy that loads and runs LeRobot models directly (no server).

    Auto-detects policy type from HF config.json → delegates to LeRobot's
    own class registry. Supports ACT, Diffusion, Pi0, SmolVLA, XVLA, etc.

    Optionally loads the model's processor pipeline (preprocessor.json /
    postprocessor.json) for automatic normalization, device transfer,
    observation formatting, and action unnormalization.

    Args:
        pretrained_name_or_path: HF model ID or local path. If empty, model
            is not loaded until first inference call.
        policy_type: Explicit LeRobot policy type (e.g. "act", "diffusion").
            Auto-detected from config.json if not provided.
        device: Target device (e.g. "cuda", "cpu"). Auto-detected if None.
        actions_per_step: Number of action steps to return per inference call.
        use_processor: Whether to load the model's processor pipeline.
        processor_overrides: Dict of overrides for processor pipeline steps.
        max_consecutive_failures: Number of consecutive inference failures
            before raising instead of returning zero actions.
    """

    def __init__(
        self,
        pretrained_name_or_path: str = "",
        policy_type: Optional[str] = None,
        device: Optional[str] = None,
        actions_per_step: int = 1,
        use_processor: bool = True,
        processor_overrides: Optional[dict] = None,
        max_consecutive_failures: int = 5,
        **kwargs,
    ):
        self.pretrained_name_or_path = pretrained_name_or_path
        self.policy_type = policy_type
        self.requested_device = device
        self.actions_per_step = actions_per_step
        self.use_processor = use_processor
        self.processor_overrides = processor_overrides
        self.robot_state_keys: List[str] = []

        self._policy: Optional[Any] = None
        self._device: Optional[torch.device] = None
        self._input_features: Dict[str, Any] = {}
        self._output_features: Dict[str, Any] = {}
        self._loaded = False
        self._processor_bridge: Optional[ProcessorBridge] = None
        self._tokenizer = None
        self._tokenizer_max_length: int = 48
        self._tokenizer_padding_side: str = "right"
        self._consecutive_failures = 0
        self._max_consecutive_failures = max_consecutive_failures

        if pretrained_name_or_path:
            self._load_model()

    @property
    def provider_name(self) -> str:
        return "lerobot_local"

    def reset(self) -> None:
        """Reset policy state between episodes.

        **MUST** be called whenever the environment or task episode resets.
        LeRobot policies (ACT, Diffusion, etc.) cache internal state such as
        action queues and temporal ensemble buffers. Without resetting, stale
        actions from the previous episode leak into the next one.
        """
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()
            logger.debug("Policy internal state reset")
        if self._processor_bridge is not None:
            self._processor_bridge.reset()
        self._consecutive_failures = 0

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        """Set robot state keys for observation→tensor mapping.

        Args:
            robot_state_keys: List of joint/motor names. If empty, auto-detects
                from model output_features (action dim) or input_features (state dim).
                Auto-detected keys are generic (joint_0, joint_1, ...).
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

        # Auto-detect from model's action output dimension
        if self._loaded and self._output_features:
            action_feat = self._output_features.get("action")
            if action_feat and hasattr(action_feat, "shape") and action_feat.shape:
                action_dim = action_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(action_dim)]
                logger.info(
                    "Auto-detected %d state keys from output_features.action.shape=%s. "
                    "For meaningful names, pass the robot's actual joint names.",
                    action_dim,
                    action_feat.shape,
                )
                return

        # Fallback: try input state dimension
        if self._loaded and self._input_features:
            state_feat = self._input_features.get("observation.state")
            if state_feat and hasattr(state_feat, "shape") and state_feat.shape:
                state_dim = state_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(state_dim)]
                logger.info(
                    "Auto-detected %d state keys from input_features.observation.state.shape=%s.",
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

        Returns:
            The tokenizer instance, or None if not available.
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

        # 1. tokenizer_name (explicit config field)
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
            except (ImportError, OSError, ValueError) as exc:
                logger.warning("Failed to load tokenizer from '%s': %s", tokenizer_id, exc)

        # 3. policy.processor.tokenizer (built-in)
        processor = getattr(self._policy, "processor", None)
        if processor and hasattr(processor, "tokenizer"):
            self._tokenizer = processor.tokenizer
            self._tokenizer.padding_side = self._tokenizer_padding_side
            logger.info("Using policy's built-in processor tokenizer (%s)", type(self._tokenizer).__name__)
            return self._tokenizer

        return None

    def _tokenize_instruction(self, instruction: str):
        """Tokenize an instruction into (input_ids, attention_mask) tensors.

        Args:
            instruction: Natural language instruction string.

        Returns:
            Tuple of (input_ids, attention_mask) tensors, or None if no tokenizer.
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
        """Check whether this policy requires observation.language.tokens.

        Returns True if the model config indicates VLA language input is needed
        (tokenizer_name, vlm_model_name, or language-related input features).
        """
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

    def _load_model(self) -> None:
        """Load the LeRobot model from pretrained path.

        Raises:
            ImportError: If required dependencies are missing.
            ValueError: If model path is invalid or config cannot be parsed.
            RuntimeError: If model loading fails.
        """
        warnings.filterwarnings("ignore", message=".*Device.*")

        # XVLA compat: Florence2LanguageConfig.forced_bos_token_id missing in transformers 5.x.
        # Florence2 was originally built with an older transformers that had this attribute.
        # Without this patch, XVLA models fail to load with AttributeError.
        try:
            from transformers.models.florence2.configuration_florence2 import Florence2LanguageConfig

            if not hasattr(Florence2LanguageConfig, "forced_bos_token_id"):
                Florence2LanguageConfig.forced_bos_token_id = None
                logger.debug("Patched Florence2LanguageConfig.forced_bos_token_id for XVLA compat")
        except ImportError:
            pass

        logger.info("Loading %s...", self.pretrained_name_or_path)
        start = time.time()

        # Resolve the correct policy class
        if self.policy_type:
            PolicyClass = resolve_policy_class_by_name(self.policy_type)
        else:
            PolicyClass, self.policy_type = resolve_policy_class_from_hub(self.pretrained_name_or_path)

        self._policy = PolicyClass.from_pretrained(self.pretrained_name_or_path)
        assert self._policy is not None

        self._policy.eval()

        # Resolve device: prefer config.device, fallback to first parameter's device
        if hasattr(self._policy, "config") and hasattr(self._policy.config, "device"):
            self._device = torch.device(self._policy.config.device)
        else:
            self._device = next(self._policy.parameters()).device

        if hasattr(self._policy, "config"):
            config = self._policy.config
            if hasattr(config, "input_features"):
                self._input_features = config.input_features
            if hasattr(config, "output_features"):
                self._output_features = config.output_features

        elapsed = time.time() - start
        logger.info(
            "Loaded %s (type='%s') in %.1fs on %s",
            type(self._policy).__name__,
            self.policy_type,
            elapsed,
            self._device,
        )
        self._loaded = True

        # Auto-detect robot_state_keys from model config if not set
        if not self.robot_state_keys and self._output_features:
            action_feat = self._output_features.get("action")
            if action_feat and hasattr(action_feat, "shape") and action_feat.shape:
                action_dim = action_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(action_dim)]
                logger.info(
                    "Auto-generated %d generic state keys (joint_0..joint_%d). "
                    "Set explicit keys with set_robot_state_keys() for meaningful joint names.",
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
        """Get actions from policy given observation and instruction.

        Args:
            observation_dict: Robot observation (cameras + state).
            instruction: Natural language instruction.

        Returns:
            List of action dicts for robot execution.

        Raises:
            RuntimeError: If model is not loaded and no path is set, or
                if inference fails more than max_consecutive_failures times.
        """
        if not self._loaded:
            if self.pretrained_name_or_path:
                self._load_model()
            else:
                raise RuntimeError(
                    "No model loaded and no pretrained_name_or_path set. "
                    "Create the policy with a model path or call _load_model() first."
                )

        try:
            observation = dict(observation_dict)
            if instruction and "task" not in observation:
                observation["task"] = instruction

            # When the processor bridge has a preprocessor, it handles all
            # observation conversion (HWC→CHW, uint8→float, normalization,
            # device transfer, batching).  Feeding its output through
            # _build_observation_batch would double-normalize images.
            if self._processor_bridge and self._processor_bridge.has_preprocessor:
                batch = self._processor_bridge.preprocess(observation)
                # Ensure the batch is a dict (processor returns observation dict)
                if not isinstance(batch, dict):
                    batch = {"observation.state": batch}
            else:
                batch = self._build_observation_batch(observation, instruction)

            with torch.inference_mode():
                assert self._policy is not None
                self._policy.eval()
                action_tensor = self._policy.select_action(batch)

            if self._processor_bridge and self._processor_bridge.has_postprocessor:
                action_tensor = self._processor_bridge.postprocess(action_tensor)

            self._consecutive_failures = 0
            return self._tensor_to_action_dicts(action_tensor)
        except Exception as exc:
            self._consecutive_failures += 1
            logger.error(
                "Inference error (%d/%d): %s",
                self._consecutive_failures,
                self._max_consecutive_failures,
                exc,
            )
            if self._consecutive_failures >= self._max_consecutive_failures:
                raise RuntimeError(
                    f"LeRobot policy failed {self._consecutive_failures} consecutive times, " f"last error: {exc}"
                ) from exc
            raise

    def select_action_sync(self, observation_dict: Dict[str, Any], instruction: str = "") -> np.ndarray:
        """Synchronous inference — returns raw action numpy array.

        Convenience for simulation loops that don't need async.
        Applies processor pipeline if available.

        For VLA models (xvla, smolvla, etc.) that require language tokens:
        pass ``instruction`` or include a ``'task'`` key in observation_dict.

        Args:
            observation_dict: Observation dict (state + images).
            instruction: Natural language instruction for VLA models.

        Returns:
            Action numpy array with shape (action_dim,).
        """
        if not self._loaded:
            self._load_model()

        if not instruction:
            instruction = observation_dict.get("task", "")

        observation = dict(observation_dict)
        if instruction and "task" not in observation:
            observation["task"] = instruction

        if self._processor_bridge and self._processor_bridge.has_preprocessor:
            batch = self._processor_bridge.preprocess(observation)
            if not isinstance(batch, dict):
                batch = {"observation.state": batch}
        else:
            batch = self._build_observation_batch(observation, instruction)

        with torch.inference_mode():
            assert self._policy is not None
            self._policy.eval()
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
        """Convert observation dict to LeRobot-compatible batch tensors.

        Handles two observation formats:
        1. LeRobot native: keys prefixed with "observation." (e.g. "observation.state")
        2. strands-robots native: individual joint keys (e.g. "shoulder", "elbow")

        For VLA models, injects tokenized language instructions into the batch
        as "observation.language.tokens" and "observation.language.attention_mask".

        Args:
            observation_dict: Raw observation dict from robot/sim.
            instruction: Natural language instruction for VLA models.

        Returns:
            Dict of tensors ready for LeRobot policy.select_action().
        """
        batch: Dict[str, Any] = {}

        has_lerobot_keys = any(key.startswith("observation.") for key in observation_dict)
        if has_lerobot_keys:
            batch = self._build_batch_from_lerobot_format(observation_dict, batch)
        else:
            batch = self._build_batch_from_strands_format(observation_dict, batch)

        # Inject tokenized language instruction for VLA models.
        # VLA models (SmolVLA, XVLA, etc.) expect language tokens as part of the
        # observation batch. We tokenize the instruction and add it to the batch
        # only if the model declares language-related input features.
        if instruction and "observation.language.tokens" not in batch and self._needs_language_tokens():
            try:
                result = self._tokenize_instruction(instruction)
                if result is not None:
                    tokens, mask = result
                    batch["observation.language.tokens"] = tokens
                    if mask is not None:
                        batch["observation.language.attention_mask"] = mask
                    logger.debug("VLA tokenized instruction: '%s...' -> %s", instruction[:50], tokens.shape)
            except (ImportError, OSError, ValueError) as exc:
                logger.debug("VLA tokenization failed: %s", exc)

        # Fill task key for models that read it directly from the batch
        if instruction and has_lerobot_keys:
            needs_task = any("task" in key for key in self._input_features) and "task" not in batch
            if needs_task:
                for feat_name in self._input_features:
                    if "task" in feat_name and feat_name not in batch:
                        batch[feat_name] = instruction

        # Fill missing image features with zero tensors so the model doesn't crash
        # on missing expected inputs. This handles cases where the robot doesn't have
        # all cameras the model was trained with.
        for feat_name, feat_info in self._input_features.items():
            if feat_name not in batch and "image" in feat_name:
                shape = feat_info.shape if hasattr(feat_info, "shape") else (3, 480, 640)
                batch[feat_name] = torch.zeros(1, *shape, device=self._device)

        return batch

    def _build_batch_from_lerobot_format(
        self, observation_dict: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build batch from observation dict already in LeRobot format (observation.* keys).

        Converts each value to the appropriate tensor format:
        - Images (HWC uint8) → CHW float32 [0, 1] with batch dim
        - State vectors → float32 with batch dim
        - Scalars → float32 tensor with batch dim

        Args:
            observation_dict: Dict with "observation.*" prefixed keys.
            batch: Existing batch dict to extend.

        Returns:
            Updated batch dict with tensors on target device.
        """
        for key, value in observation_dict.items():
            # Determine if this key represents an image based on key name or feature metadata
            is_image = "image" in key or (
                key in self._input_features
                and hasattr(self._input_features.get(key), "shape")
                and len(getattr(self._input_features.get(key), "shape", ())) >= 2
            )

            if isinstance(value, torch.Tensor):
                tensor = value
                if not is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                    is_image = True
                # HWC → CHW conversion for images
                if is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                    tensor = tensor.permute(2, 0, 1)
                # Add batch dimension
                if is_image and tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                elif tensor.dim() < 2 and not is_image:
                    tensor = tensor.unsqueeze(0)
                batch[key] = tensor.to(self._device)

            elif isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value.copy()).float()
                if not is_image and value.ndim == 3 and value.shape[-1] in (1, 3, 4):
                    is_image = True
                # HWC → CHW conversion for images
                if is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                    tensor = tensor.permute(2, 0, 1)
                # Normalize uint8 images to [0, 1] float range
                if is_image and value.dtype == np.uint8:
                    tensor = tensor / 255.0
                # Add batch dimension
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
            # Non-numeric types (strings, pre-batched int64 tokens) pass through unchanged

        return batch

    def _build_batch_from_strands_format(
        self, observation_dict: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build batch from strands-robots native observation format.

        Maps individual joint keys (e.g. {"shoulder": 0.5, "elbow": -0.3}) to
        LeRobot's "observation.state" tensor using robot_state_keys ordering.
        Camera images are matched to the model's image input features.

        Args:
            observation_dict: Dict with individual joint/image keys.
            batch: Existing batch dict to extend.

        Returns:
            Updated batch dict with "observation.state" and image tensors.
        """
        if not self.robot_state_keys:
            logger.warning(
                "robot_state_keys is empty — observation.state will be skipped. "
                "Call set_robot_state_keys() with the robot's motor names for proper state handling."
            )

        # Collect state values in robot_state_keys order
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
            # Pad or truncate to match model's expected state dimension
            state_feature = self._input_features.get("observation.state")
            if state_feature:
                expected_dim = state_feature.shape[0] if hasattr(state_feature, "shape") else len(state_values)
                while len(state_values) < expected_dim:
                    state_values.append(0.0)
                state_values = state_values[:expected_dim]
            batch["observation.state"] = torch.tensor(state_values, dtype=torch.float32).unsqueeze(0).to(self._device)

        # Map camera images to model's image input features.
        # Non-state ndarray values with ndim >= 2 are assumed to be images.
        # Each image is matched to the first unoccupied image feature slot.
        for key, value in observation_dict.items():
            if key in self.robot_state_keys:
                continue
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                image_tensor = torch.from_numpy(value.copy()).float()
                # HWC → CHW conversion
                if image_tensor.dim() == 3 and image_tensor.shape[-1] in (1, 3, 4):
                    image_tensor = image_tensor.permute(2, 0, 1)
                # Normalize uint8 to [0, 1]
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

    def _tensor_to_action_dicts(self, action_tensor: torch.Tensor) -> List[Dict[str, Any]]:
        """Convert action tensor to list of robot action dicts.

        Maps tensor values to robot_state_keys by index. Handles 1D (single action),
        2D (action sequence), and 3D (batched action sequence) tensors.

        Args:
            action_tensor: Raw action tensor from policy.select_action().

        Returns:
            List of action dicts, length capped by actions_per_step.
        """
        action_array = action_tensor.cpu().numpy()

        if action_array.ndim == 1:
            actions_list = [action_array]
        elif action_array.ndim == 2:
            actions_list = [action_array[i] for i in range(min(len(action_array), self.actions_per_step))]
        elif action_array.ndim == 3:
            actions_list = [action_array[0, i] for i in range(min(action_array.shape[1], self.actions_per_step))]
        else:
            actions_list = [action_array.flatten()]

        if not self.robot_state_keys:
            raise RuntimeError(
                "Cannot convert action tensor to dicts: robot_state_keys is empty. "
                "Call set_robot_state_keys() before inference."
            )

        result = []
        for action_values in actions_list:
            action_dict = {}
            for index, key in enumerate(self.robot_state_keys):
                action_dict[key] = float(action_values[index]) if index < len(action_values) else 0.0
            result.append(action_dict)

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata.

        Returns:
            Dict with provider, model_id, policy_type, device, features, etc.
        """
        info: Dict[str, Any] = {
            "provider": "lerobot_local",
            "model_id": self.pretrained_name_or_path,
            "policy_type": self.policy_type,
            "loaded": self._loaded,
            "device": str(self._device) if self._device else None,
        }
        if self._loaded:
            info["input_features"] = {
                key: str(val.shape) if hasattr(val, "shape") else str(val) for key, val in self._input_features.items()
            }
            info["output_features"] = {
                key: str(val.shape) if hasattr(val, "shape") else str(val) for key, val in self._output_features.items()
            }
            info["policy_class"] = type(self._policy).__name__ if self._policy else "None"
            info["n_parameters"] = sum(param.numel() for param in self._policy.parameters()) if self._policy else 0
        if self._processor_bridge:
            info["processor"] = self._processor_bridge.get_info()
        return info
