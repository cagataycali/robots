"""GR00T policy — N1.5/N1.6 service and local inference.

This module contains the main :class:`Gr00tPolicy` class that implements the
:class:`~strands_robots.policies.base.Policy` interface for NVIDIA GR00T models.
"""

import importlib.util
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from strands_robots.policies.base import Policy

from .client import Gr00tInferenceClient
from .data_config import Gr00tDataConfig, load_data_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Isaac-GR00T version detection
# ---------------------------------------------------------------------------

_GROOT_VERSION: Optional[str] = None  # "n1.5", "n1.6", or None


def _detect_groot_version() -> Optional[str]:
    """Auto-detect which Isaac-GR00T version (if any) is installed.

    Uses lightweight probes (``importlib.util.find_spec``) to avoid importing
    heavy CUDA / torch dependencies at detection time.
    """
    global _GROOT_VERSION
    if _GROOT_VERSION is not None:
        return _GROOT_VERSION

    # Try N1.6 first (newer)
    try:
        if importlib.util.find_spec("gr00t.policy.gr00t_policy") is not None:
            _GROOT_VERSION = "n1.6"
            logger.info("Detected Isaac-GR00T N1.6 (gr00t.policy.gr00t_policy)")
            return _GROOT_VERSION
    except (ModuleNotFoundError, ValueError):
        pass

    # Try N1.5
    try:
        if importlib.util.find_spec("gr00t.model.policy") is not None:
            _GROOT_VERSION = "n1.5"
            logger.info("Detected Isaac-GR00T N1.5 (gr00t.model.policy)")
            return _GROOT_VERSION
    except (ModuleNotFoundError, ValueError):
        pass

    logger.debug("No local Isaac-GR00T installation found — service mode only")
    return None


# ---------------------------------------------------------------------------
# Camera alias mapping
# ---------------------------------------------------------------------------

_CAMERA_ALIASES: Dict[str, List[str]] = {
    "webcam": ["webcam", "front", "wrist", "main"],
    "front": ["front", "webcam", "top", "ego_view", "main"],
    "wrist": ["wrist", "hand", "end_effector", "gripper"],
    "ego_view": ["front", "ego_view", "webcam", "main"],
    "top": ["top", "overhead", "front"],
    "side": ["side", "lateral", "left", "right"],
}

# ---------------------------------------------------------------------------
# State layout definitions — keyed by data config name patterns
# ---------------------------------------------------------------------------

# Maps config-name substrings to ordered (state_key, dimension) tuples.
# Used by _map_robot_state_to_groot_state to avoid hard-coded if/elif chains.
_STATE_LAYOUTS: Dict[str, List[tuple]] = {
    "so100": [("state.single_arm", 5), ("state.gripper", 1)],
    "so101": [("state.single_arm", 5), ("state.gripper", 1)],
    "fourier_gr1": [("state.left_arm", 7), ("state.right_arm", 7)],
    "unitree_g1": [
        ("state.left_leg", 6),
        ("state.right_leg", 6),
        ("state.waist", 3),
        ("state.left_arm", 7),
        ("state.right_arm", 7),
        ("state.left_hand", 7),
        ("state.right_hand", 7),
    ],
}


class Gr00tPolicy(Policy):
    """GR00T policy supporting N1.5, N1.6, service mode, and local inference.

    Args:
        data_config: Config name (e.g. ``"so100_dualcam"``) or :class:`Gr00tDataConfig`.
        host: Inference-service host (service mode).
        port: Inference-service port (service mode).
        model_path: Path or HF model ID for local inference. When set,
            the policy loads the model directly on GPU (requires Isaac-GR00T).
        embodiment_tag: Embodiment tag string (mapped to enum for N1.6).
            Uppercased by convention to match ``EmbodimentTag`` enum values.
        denoising_steps: Diffusion denoising steps (default 4).
        device: ``"cuda"`` or ``"cpu"`` (local mode).
        groot_version: Force ``"n1.5"`` or ``"n1.6"`` (auto-detected if *None*).
        strict: Enable strict input/output validation (default *False*).
        api_token: Optional API token for authenticated ZMQ servers.

    Examples::

        # Service mode (no Isaac-GR00T needed)
        policy = Gr00tPolicy(data_config="so100_dualcam", host="localhost", port=5555)

        # Local N1.5
        policy = Gr00tPolicy(
            data_config="so100_dualcam",
            model_path="/data/checkpoints/gr00t-wave/checkpoint-300000",
        )

        # Local N1.6 (base model)
        policy = Gr00tPolicy(
            data_config="so100_dualcam",
            model_path="nvidia/GR00T-N1-2B",
            groot_version="n1.6",
        )
    """

    def __init__(
        self,
        data_config: Union[str, Gr00tDataConfig] = "so100_dualcam",
        host: str = "localhost",
        port: int = 5555,
        model_path: Optional[str] = None,
        embodiment_tag: str = "NEW_EMBODIMENT",
        denoising_steps: int = 4,
        device: str = "cuda",
        groot_version: Optional[str] = None,
        strict: bool = False,
        api_token: Optional[str] = None,
        **kwargs,
    ):
        self.data_config = load_data_config(data_config)
        self.data_config_name = data_config if isinstance(data_config, str) else type(data_config).__name__

        self.camera_keys = self.data_config.video_keys
        self.state_keys = self.data_config.state_keys
        self.action_keys = self.data_config.action_keys
        self.language_keys = self.data_config.language_keys
        self.robot_state_keys: List[str] = []

        self._local_policy = None
        self._client = None
        self._groot_version = groot_version or _detect_groot_version()
        self._strict = strict

        if model_path is not None:
            self._mode = "local"
            logger.info("Initializing GR00T in local mode with model_path=%s", model_path)
            self._load_local_policy(model_path, embodiment_tag, denoising_steps, device)
        else:
            self._mode = "service"
            logger.info("Initializing GR00T in service mode connecting to %s:%s", host, port)
            self._client = Gr00tInferenceClient(host=host, port=port, api_token=api_token)

        logger.info(
            "GR00T Policy ready [mode=%s, version=%s, config=%s, cameras=%s, strict=%s]",
            self._mode,
            self._groot_version or "service-only",
            self.data_config_name,
            self.camera_keys,
            self._strict,
        )

    # ------------------------------------------------------------------
    # Local model loading
    # ------------------------------------------------------------------

    def _load_local_policy(self, model_path: str, embodiment_tag: str, denoising_steps: int, device: str):
        """Dispatch to N1.5 or N1.6 loader based on detected version."""
        if self._groot_version == "n1.6":
            self._load_n16(model_path, embodiment_tag, denoising_steps, device)
        elif self._groot_version == "n1.5":
            self._load_n15(model_path, embodiment_tag, denoising_steps, device)
        else:
            raise ImportError(
                "Isaac-GR00T not installed. Cannot use local mode. "
                "Either install Isaac-GR00T or use service mode (host/port)."
            )

    def _load_n15(self, model_path: str, embodiment_tag: str, denoising_steps: int, device: str):
        """Load N1.5 model via ``gr00t.model.policy.Gr00tPolicy``."""
        from gr00t.experiment.data_config import DATA_CONFIG_MAP as N15_CONFIGS
        from gr00t.model.policy import Gr00tPolicy as N15Policy

        config_name = self.data_config_name if isinstance(self.data_config_name, str) else "so100_dualcam"
        native_config = N15_CONFIGS.get(config_name)

        if native_config:
            modality_config = native_config.modality_config()
            modality_transform = native_config.transform()
        else:
            modality_config = self.data_config.modality_config()
            modality_transform = None
            logger.warning(
                "No native N1.5 config for '%s' — using generated modality_config, transforms may be missing",
                config_name,
            )

        init_kwargs = {
            "model_path": model_path,
            "embodiment_tag": embodiment_tag,
            "modality_config": modality_config,
            "modality_transform": modality_transform,
            "denoising_steps": denoising_steps,
            "device": device,
        }
        init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}

        self._local_policy = N15Policy(**init_kwargs)
        logger.info("GR00T N1.5 model loaded from %s", model_path)

    def _load_n16(self, model_path: str, embodiment_tag: str, denoising_steps: int, device: str):
        """Load N1.6 model via ``gr00t.policy.gr00t_policy.Gr00tPolicy``.

        Wraps the policy in ``Gr00tSimPolicyWrapper`` which handles the
        flat→nested observation conversion natively.
        """
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy as N16Policy
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        tag_enum = getattr(EmbodimentTag, embodiment_tag.upper(), EmbodimentTag.NEW_EMBODIMENT)
        logger.debug("Resolved embodiment tag '%s' to enum %s", embodiment_tag, tag_enum)

        base_policy = N16Policy(
            embodiment_tag=tag_enum,
            model_path=model_path,
            device=device,
            strict=self._strict,
        )
        self._local_policy = Gr00tSimPolicyWrapper(base_policy, strict=self._strict)
        logger.info("GR00T N1.6 model loaded from %s (wrapped with Gr00tSimPolicyWrapper)", model_path)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "groot"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self.robot_state_keys = robot_state_keys
        logger.debug("Robot state keys set: %s", robot_state_keys)

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Get actions from GR00T (service or local mode).

        Args:
            observation_dict: Robot observations (cameras + state).
            instruction: Natural-language instruction.

        Returns:
            List of per-timestep action dicts keyed by :attr:`robot_state_keys`.
        """
        observation = self._build_observation(observation_dict, instruction)

        if self._mode == "local":
            logger.debug("Running local inference (version=%s)", self._groot_version)
            action_chunk = self._local_inference(observation)
        else:
            logger.debug("Requesting action from service at %s:%s", self._client.host, self._client.port)
            action_chunk = self._client.get_action(observation)

        actions = self._convert_to_robot_actions(action_chunk)
        logger.debug("Got %d action timesteps from GR00T", len(actions))
        return actions

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _build_observation(self, observation_dict: Dict[str, Any], instruction: str) -> dict:
        """Build GR00T-formatted observation dict from raw robot data."""
        observation: dict = {}

        for video_key in self.camera_keys:
            camera_key = self._map_video_key_to_camera(video_key, observation_dict)
            if camera_key and camera_key in observation_dict:
                observation[video_key] = observation_dict[camera_key]

        robot_state = np.array([observation_dict.get(key, 0.0) for key in self.robot_state_keys])
        self._map_robot_state_to_groot_state(observation, robot_state)

        if self.language_keys:
            observation[self.language_keys[0]] = instruction

        # Service mode needs a batch dimension
        if self._mode == "service":
            for key in observation:
                if isinstance(observation[key], np.ndarray):
                    observation[key] = observation[key][np.newaxis, ...]
                else:
                    observation[key] = [observation[key]]

        return observation

    def _local_inference(self, observation_dict: dict) -> dict:
        """Run local inference (N1.5 or N1.6)."""
        if self._groot_version == "n1.5":
            return self._local_inference_n15(observation_dict)
        if self._groot_version == "n1.6":
            return self._local_inference_n16(observation_dict)
        raise RuntimeError(f"Unknown GR00T version: {self._groot_version}")

    def _local_inference_n15(self, observation_dict: dict) -> dict:
        """N1.5 inference: flat dict with batch dim."""
        batched = {}
        for key, value in observation_dict.items():
            if isinstance(value, np.ndarray):
                batched[key] = value[np.newaxis, ...]
            else:
                batched[key] = [value] if not isinstance(value, list) else value
        return self._local_policy.get_action(batched)

    def _local_inference_n16(self, observation_dict: dict) -> dict:
        """N1.6 inference via ``Gr00tSimPolicyWrapper``.

        The wrapper handles flat→nested observation conversion natively.
        Vendor expects flat keys with shapes: video ``(B, T, H, W, C)``,
        state ``(B, T, D)``, language/task ``[str]``.
        """
        batched: dict = {}
        for key, value in observation_dict.items():
            if isinstance(value, np.ndarray):
                if "video" in key and value.ndim == 3:  # (H, W, C) → (1, 1, H, W, C)
                    batched[key] = value[np.newaxis, np.newaxis, ...]
                elif "video" in key and value.ndim == 4:  # (T, H, W, C) → (1, T, H, W, C)
                    batched[key] = value[np.newaxis, ...]
                elif "state" in key and value.ndim == 1:  # (D,) → (1, 1, D)
                    batched[key] = value.astype(np.float32)[np.newaxis, np.newaxis, ...]
                elif "state" in key and value.ndim == 2:  # (T, D) → (1, T, D)
                    batched[key] = value.astype(np.float32)[np.newaxis, ...]
                else:
                    batched[key] = value[np.newaxis, ...] if value.ndim < 3 else value
            elif isinstance(value, str):
                batched[key] = [value]
            elif isinstance(value, list):
                batched[key] = value
            else:
                batched[key] = [value]

        actions, _ = self._local_policy.get_action(batched)
        # Wrapper returns flat keys like "single_arm" — prefix with "action."
        return {f"action.{key}" if not key.startswith("action.") else key: value for key, value in actions.items()}

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_obs_key(observation_dict: dict, *candidates: str) -> str | None:
        """Return the first candidate key present in *observation_dict*, or *None*."""
        for candidate in candidates:
            if candidate in observation_dict:
                return candidate
        return None

    def _map_video_key_to_camera(self, video_key: str, observation_dict: dict) -> str | None:
        """Map a GR00T ``video.*`` key to an available camera key in observations."""
        camera_name = video_key.replace("video.", "")
        if camera_name in observation_dict:
            return camera_name

        for alias_name in _CAMERA_ALIASES.get(camera_name, [camera_name]):
            if alias_name in observation_dict:
                logger.debug("Camera '%s' resolved via alias to '%s'", camera_name, alias_name)
                return alias_name

        # Fallback: first non-state key — warn because this may be wrong
        non_state_keys = [key for key in observation_dict if not key.startswith("state")]
        if non_state_keys:
            fallback_key = non_state_keys[0]
            logger.warning(
                "No camera match for '%s' — falling back to first non-state key '%s'. "
                "This may produce incorrect results. Set camera names to match the data config.",
                video_key,
                fallback_key,
            )
            return fallback_key

        logger.warning("No camera found for '%s' in observation keys: %s", video_key, list(observation_dict.keys()))
        return None

    def _map_robot_state_to_groot_state(self, observation_dict: dict, robot_state: np.ndarray):
        """Map a flat robot-state array into per-group GR00T state keys.

        State layouts are defined in :data:`_STATE_LAYOUTS` keyed by config
        name patterns, avoiding hard-coded if/elif chains.
        """
        config_name = self.data_config_name.lower() if isinstance(self.data_config_name, str) else ""
        dtype = np.float32

        # Find matching state layout by config name prefix
        matched_layout = None
        for pattern, layout in _STATE_LAYOUTS.items():
            if pattern in config_name:
                # Match if the robot state has enough data for at least the first group
                first_group_dim = layout[0][1] if layout else 0
                if len(robot_state) >= first_group_dim:
                    matched_layout = layout
                    break

        if matched_layout is not None:
            index = 0
            for state_key, dimension in matched_layout:
                if index + dimension <= len(robot_state):
                    observation_dict[state_key] = robot_state[index : index + dimension].astype(dtype)
                else:
                    observation_dict[state_key] = np.zeros(dimension, dtype=dtype)
                    logger.debug(
                        "Robot state too short for '%s' (need %d more values) — zero-filling",
                        state_key,
                        index + dimension - len(robot_state),
                    )
                index += dimension
        elif self.state_keys and len(robot_state) > 0:
            observation_dict[self.state_keys[0]] = robot_state.astype(dtype)
            logger.debug(
                "No specific state layout for config '%s' — mapped %d values to '%s'",
                config_name,
                len(robot_state),
                self.state_keys[0],
            )

    def _convert_to_robot_actions(self, action_chunk: dict) -> List[Dict[str, Any]]:
        """Convert raw GR00T action output to a list of per-timestep robot action dicts.

        Returns an empty list and logs a warning if *action_chunk* is empty.
        """
        # Normalize keys: strip "action." prefix, squeeze batch dim → (H, D)
        normalized: dict = {}
        for key, value in action_chunk.items():
            clean_key = key.replace("action.", "") if key.startswith("action.") else key
            if hasattr(value, "shape"):
                array = value
                while array.ndim > 2:
                    array = array[0]
                normalized[clean_key] = array
            else:
                normalized[clean_key] = np.atleast_2d(value)

        if not normalized:
            logger.warning("GR00T returned empty action chunk — no actions to execute")
            return []

        horizon = next(iter(normalized.values())).shape[0]

        robot_actions = []
        for timestep in range(horizon):
            parts = [np.atleast_1d(normalized[action_key][timestep]) for action_key in normalized]
            concatenated = np.concatenate(parts, axis=0) if parts else np.zeros(6)

            if self.robot_state_keys:
                action_dict = {
                    key: float(concatenated[idx]) if idx < len(concatenated) else 0.0
                    for idx, key in enumerate(self.robot_state_keys)
                }
            else:
                # No robot_state_keys set — return per-group dict
                action_dict = {}
                for action_key in normalized:
                    row = normalized[action_key][timestep]
                    action_dict[action_key] = row.tolist() if hasattr(row, "tolist") else list(row)

            robot_actions.append(action_dict)

        return robot_actions
