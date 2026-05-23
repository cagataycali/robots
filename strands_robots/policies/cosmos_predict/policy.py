"""CosmosPredictPolicy - NVIDIA Cosmos-Predict2.5 as a strands-robots Policy.

This module implements the Policy interface using NVIDIA's Cosmos-Predict2.5
world foundation model in its robot/policy variant. The model uses rectified
flow diffusion in latent space to predict action chunks (16-step, 7-DoF)
from camera observations + proprioception + language instruction.

Supports two inference modes:
    1. Local: loads the model directly (requires cosmos-predict2, CUDA GPU)
    2. Server: connects to a remote inference server via HTTP (for env isolation)

The server mode follows the same pattern as Gr00tPolicy (ZMQ/HTTP), enabling
Python version isolation (cosmos-predict2 requires Python 3.10, strands-robots
uses 3.12+).
"""

import logging
import time
import types
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy

logger = logging.getLogger(__name__)

# Default action dimension (7-DoF: x, y, z, roll, pitch, yaw, gripper)
_ACTION_DIM = 7
# Standard image size for Cosmos policy models


class CosmosPredictPolicy(Policy):
    """Cosmos Predict 2.5 robot policy - action prediction via latent diffusion.

    The policy predicts action chunks (16 timesteps x 7 DoF) from:
    - Camera observations (wrist + third-person)
    - Proprioceptive state
    - Language instruction (via T5 or Reason1 text embeddings)

    Supported evaluation suites:
    - libero: 1 wrist + 1 third-person camera
    - robocasa: 1 wrist + 2 third-person cameras
    - aloha: 2 wrist + 1 third-person camera

    Thread safety:
        This class is NOT thread-safe. The underlying model maintains GPU
        state that must not be accessed concurrently. Use one instance per
        thread, or serialize access externally.
    """

    # Suite configurations define camera layout and latent sequence structure
    _SUITE_CONFIGS: dict[str, dict[str, Any]] = {
        "libero": {
            "cameras": ["wrist", "primary"],
            "num_wrist_images": 1,
            "num_third_person_images": 1,
            "state_t": 9,
            "min_conditional_frames": 4,
        },
        "robocasa": {
            "cameras": ["wrist", "primary", "secondary"],
            "num_wrist_images": 1,
            "num_third_person_images": 2,
            "state_t": 11,
            "min_conditional_frames": 5,
        },
        "aloha": {
            "cameras": ["left_wrist", "right_wrist", "primary"],
            "num_wrist_images": 2,
            "num_third_person_images": 1,
            "state_t": 11,
            "min_conditional_frames": 5,
        },
    }

    def __init__(
        self,
        model_id: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        suite: str = "libero",
        device: str | None = None,
        chunk_size: int = 16,
        num_denoising_steps: int = 5,
        dataset_stats_path: str | None = None,
        t5_embeddings_path: str | None = None,
        text_embeddings_kind: str = "t5",
        config_name: str | None = None,
        use_wrist_image: bool = True,
        use_proprio: bool = True,
        normalize_proprio: bool = True,
        unnormalize_actions: bool = True,
        action_dim: int = _ACTION_DIM,
        server_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Cosmos Predict 2.5 policy.

        Args:
            model_id: HuggingFace model ID or local path to checkpoint.
            suite: Evaluation suite - "libero", "robocasa", or "aloha".
            device: CUDA device string (auto-detected if None).
            chunk_size: Number of actions per predicted chunk.
            num_denoising_steps: Denoising steps for action sampling.
            dataset_stats_path: Path to dataset statistics JSON for action
                un-normalization. Auto-resolved from HF checkpoint if None.
            t5_embeddings_path: Path to pre-computed T5 text embeddings.
                Auto-resolved from HF checkpoint if None.
            text_embeddings_kind: Type of text embeddings - "t5" or "reason1".
            config_name: Cosmos experiment config name (auto-inferred if None).
            use_wrist_image: Whether to include wrist camera in observations.
            use_proprio: Whether to include proprioceptive state.
            normalize_proprio: Whether to normalize proprioception values.
            unnormalize_actions: Whether to un-normalize predicted actions.
            action_dim: Action dimension (default 7 for manipulation).
            server_url: URL for remote inference server. When set, bypasses
                local model loading entirely.
            **kwargs: Additional cosmos-predict2 configuration overrides.

        Raises:
            ValueError: If suite is not one of "libero", "robocasa", "aloha".
        """
        if suite not in self._SUITE_CONFIGS:
            valid = ", ".join(sorted(self._SUITE_CONFIGS))
            raise ValueError(f"Unknown suite '{suite}'. Valid: {valid}")

        self._model_id = model_id
        self._suite = suite
        self._requested_device = device
        self._chunk_size = chunk_size
        self._num_denoising_steps = num_denoising_steps
        self._dataset_stats_path = dataset_stats_path
        self._t5_embeddings_path = t5_embeddings_path
        self._text_embeddings_kind = text_embeddings_kind
        self._config_name = config_name
        self._use_wrist_image = use_wrist_image
        self._use_proprio = use_proprio
        self._normalize_proprio = normalize_proprio
        self._unnormalize_actions = unnormalize_actions
        self._action_dim = action_dim
        self._server_url = server_url
        self._extra_kwargs = kwargs
        self._robot_state_keys: list[str] = []

        # Lazy-loaded state
        self._model: Any = None
        self._config: Any = None
        self._dataset_stats: dict[str, Any] | None = None
        self._device: str | None = None
        self._loaded = False
        self._step = 0

        mode_str = f"server={server_url}" if server_url else f"local ({model_id})"
        logger.info(
            "CosmosPredictPolicy: suite=%s, %s",
            suite,
            mode_str,
        )

    @property
    def provider_name(self) -> str:
        """Provider name for identification."""
        return "cosmos_predict"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Configure the policy with robot state keys for action mapping."""
        self._robot_state_keys = list(robot_state_keys)
        logger.info("CosmosPredictPolicy robot_state_keys: %s", self._robot_state_keys)

    def _ensure_loaded(self) -> None:
        """Lazy-load model, dataset stats, and text embeddings on first use."""
        if self._loaded:
            return

        if self._server_url:
            self._verify_server()
            self._loaded = True
            return

        self._load_local_model()
        self._loaded = True

    def _verify_server(self) -> None:
        """Verify the remote inference server is reachable."""
        import requests  # noqa: I001 - local import for optional dep

        try:
            resp = requests.get(f"{self._server_url}/health", timeout=5)
            resp.raise_for_status()
            logger.info("Cosmos server connected: %s", self._server_url)
        except Exception as e:
            logger.warning(
                "Cosmos server not reachable at %s: %s. Will retry on first inference call.",
                self._server_url,
                e,
            )

    def _load_local_model(self) -> None:
        """Load the Cosmos policy model locally (requires cosmos-predict2 + CUDA)."""
        logger.info("Loading Cosmos Predict 2.5 from %s...", self._model_id)
        start = time.time()

        try:
            import torch
        except ImportError as e:
            raise ImportError("CosmosPredictPolicy local mode requires PyTorch. Install: pip install torch") from e

        self._device = self._requested_device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        try:
            from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
                get_model as cosmos_get_model,
            )
            from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
                init_t5_text_embeddings_cache,
            )
            from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
                load_dataset_stats as cosmos_load_dataset_stats,
            )
        except ImportError as e:
            raise ImportError(
                "CosmosPredictPolicy requires the cosmos-predict2 package.\n"
                "Install from source:\n"
                "  git clone https://github.com/nvidia-cosmos/cosmos-predict2.5\n"
                "  cd cosmos-predict2.5\n"
                "  pip install -e packages/cosmos-oss -e packages/cosmos-cuda -e .\n\n"
                "Note: Requires CUDA toolkit, cuDNN, and NVIDIA GPU (16GB+ VRAM).\n"
                f"Error: {e}"
            ) from e

        # Build config namespace for cosmos_get_model
        config_file = self._extra_kwargs.get(
            "config_file",
            "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py",
        )
        cfg = types.SimpleNamespace(
            ckpt_path=self._model_id,
            config=self._config_name or self._infer_config_name(),
            config_file=config_file,
        )

        self._model, self._config = cosmos_get_model(cfg)

        # Load dataset statistics for action un-normalization
        self._dataset_stats = self._resolve_dataset_stats(cosmos_load_dataset_stats)

        # Initialize text embeddings cache
        self._resolve_text_embeddings(init_t5_text_embeddings_cache)

        elapsed = time.time() - start
        logger.info(
            "Cosmos loaded in %.1fs on %s (config=%s)",
            elapsed,
            self._device,
            cfg.config,
        )

    def _resolve_dataset_stats(self, loader_fn: Any) -> dict[str, Any] | None:
        """Resolve dataset statistics from explicit path or HF checkpoint."""
        if self._dataset_stats_path:
            stats = loader_fn(self._dataset_stats_path)
            logger.info("Dataset stats loaded: %s", self._dataset_stats_path)
            return stats  # type: ignore[no-any-return]

        # Auto-resolve from HuggingFace checkpoint
        try:
            import os

            from huggingface_hub import snapshot_download

            ckpt_dir = snapshot_download(self._model_id, allow_patterns=["*.json", "*.pkl"])
            candidates = [
                f"{self._suite}_dataset_statistics.json",
                "dataset_statistics.json",
            ]
            for fname in candidates:
                path = os.path.join(ckpt_dir, fname)
                if os.path.exists(path):
                    self._dataset_stats_path = path
                    stats = loader_fn(path)
                    logger.info("Dataset stats auto-resolved: %s", path)
                    return stats  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("Could not auto-resolve dataset stats: %s", e)

        logger.warning("No dataset statistics found - actions will not be un-normalized")
        return None

    def _resolve_text_embeddings(self, init_fn: Any) -> None:
        """Resolve and initialize text embeddings cache."""
        import os

        if not self._t5_embeddings_path and self._dataset_stats_path:
            ckpt_dir = os.path.dirname(self._dataset_stats_path)
            candidates = [
                f"{self._suite}_t5_embeddings.pkl",
                "t5_embeddings.pkl",
            ]
            for fname in candidates:
                path = os.path.join(ckpt_dir, fname)
                if os.path.exists(path):
                    self._t5_embeddings_path = path
                    logger.info("T5 embeddings auto-resolved: %s", path)
                    break

        if self._t5_embeddings_path:
            init_fn(
                self._t5_embeddings_path,
                worker_id=0,
                embeddings_kind=self._text_embeddings_kind,
            )
            logger.info("Text embeddings loaded (%s)", self._text_embeddings_kind)

    def _infer_config_name(self) -> str:
        """Infer cosmos experiment config name from model_id and suite."""
        model_lower = self._model_id.lower()
        if "libero" in model_lower or self._suite == "libero":
            return "cosmos_predict2_2b_480p_libero"
        elif "robocasa" in model_lower or self._suite == "robocasa":
            return "cosmos_predict2_2b_480p_robocasa"
        elif "aloha" in model_lower or self._suite == "aloha":
            return "cosmos_predict2_2b_480p_aloha"
        return "cosmos_predict2_2b_480p_libero"

    async def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Get action chunk from Cosmos Predict 2.5.

        Predicts a chunk of 16 actions (7-DoF each) via latent diffusion
        denoising, conditioned on camera images, proprioception, and a
        language instruction.

        Args:
            observation_dict: Robot observation containing:
                - Camera images as numpy arrays (H, W, 3) uint8
                - Proprioceptive state as "proprio" or "observation.state"
            instruction: Natural language task description.
            **kwargs: Overrides for seed, num_denoising_steps, etc.

        Returns:
            List of action dicts. Each dict maps robot_state_keys to floats,
            plus a "gripper" key.
        """
        self._ensure_loaded()

        if self._server_url:
            return await self._infer_server(observation_dict, instruction, **kwargs)

        return self._infer_local(observation_dict, instruction, **kwargs)

    def _infer_local(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run local inference using cosmos-predict2 get_action()."""
        from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
            get_action as cosmos_get_action,
        )

        obs = self._build_observation(observation_dict)
        suite_cfg = self._SUITE_CONFIGS[self._suite]

        # Build the config namespace expected by cosmos_get_action
        cfg = types.SimpleNamespace(
            suite=self._suite,
            use_wrist_image=self._use_wrist_image,
            use_third_person_image=True,
            num_wrist_images=suite_cfg["num_wrist_images"],
            num_third_person_images=suite_cfg["num_third_person_images"],
            use_proprio=self._use_proprio,
            normalize_proprio=self._normalize_proprio,
            unnormalize_actions=self._unnormalize_actions,
            use_jpeg_compression=kwargs.get("use_jpeg_compression", True),
            trained_with_image_aug=kwargs.get("trained_with_image_aug", True),
            chunk_size=self._chunk_size,
            model_family="predict2",
            scale_multiplier=kwargs.get("scale_multiplier", 1.0),
            num_denoising_steps_action=kwargs.get("num_denoising_steps", self._num_denoising_steps),
            seed=kwargs.get("seed", 1),
            randomize_seed=kwargs.get("randomize_seed", False),
            shift=kwargs.get("shift", 1.0),
            t=suite_cfg["state_t"],
            use_variance_scale=kwargs.get("use_variance_scale", False),
            # Future/value prediction (disabled by default for speed)
            ar_future_prediction=kwargs.get("ar_future_prediction", False),
            ar_value_prediction=kwargs.get("ar_value_prediction", False),
            ar_qvalue_prediction=kwargs.get("ar_qvalue_prediction", False),
            use_ensemble_future_state_predictions=False,
            use_ensemble_value_predictions=False,
            num_future_state_predictions_in_ensemble=1,
            num_value_predictions_in_ensemble=1,
            future_state_ensemble_aggregation_scheme="mean",
            value_ensemble_aggregation_scheme="mean",
            mask_current_state_action_for_value_prediction=False,
            mask_future_state_for_qvalue_prediction=False,
            num_denoising_steps_future_state=5,
            num_denoising_steps_value=5,
            num_queries_best_of_n=kwargs.get("best_of_n", 1),
            parallel_timeout=30,
            search_depth=1,
            planning_model_ckpt_path=None,
            planning_model_config_name=None,
        )

        seed = kwargs.get("seed", 1)
        num_steps = kwargs.get("num_denoising_steps", self._num_denoising_steps)

        result = cosmos_get_action(
            cfg=cfg,
            model=self._model,
            dataset_stats=self._dataset_stats or {},
            obs=obs,
            task_label_or_embedding=instruction,
            seed=seed,
            num_denoising_steps_action=num_steps,
            generate_future_state_and_value_in_parallel=True,
        )

        actions = self._decode_actions(result)
        self._step += 1
        logger.debug(
            "Cosmos step %d: %d actions predicted",
            self._step,
            len(actions),
        )
        return actions

    async def _infer_server(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run inference via remote HTTP server.

        The server protocol matches cosmos-predict2's evaluation server:
        POST /act with JSON payload containing observation + instruction.
        """
        import requests

        payload: dict[str, Any] = {
            "instruction": instruction,
            "suite": self._suite,
        }

        for key, val in observation_dict.items():
            if isinstance(val, np.ndarray):
                payload[key] = val.tolist()
            else:
                payload[key] = val

        endpoint = kwargs.get("endpoint", "/act")
        resp = requests.post(
            f"{self._server_url}{endpoint}",
            json=payload,
            timeout=kwargs.get("timeout", 120),
        )
        resp.raise_for_status()
        result = resp.json()

        actions: list[dict[str, Any]] = []
        for action_data in result.get("actions", []):
            if isinstance(action_data, list):
                actions.append(self._vec_to_action_dict(np.array(action_data)))
            elif isinstance(action_data, dict):
                actions.append(action_data)

        self._step += 1
        return actions

    def _build_observation(self, observation_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert strands-robots observation to cosmos-predict2 format.

        Maps camera image keys to the naming convention expected by
        cosmos_get_action() depending on the suite:
        - libero: wrist_image, primary_image, proprio
        - robocasa: wrist_image, primary_image, secondary_image, proprio
        - aloha: left_wrist_image, right_wrist_image, primary_image, proprio
        """
        obs: dict[str, Any] = {}

        # Camera key mapping - search for images by pattern
        camera_mappings: dict[str, list[str]] = {
            "primary_image": [
                "primary",
                "camera_0",
                "cam_high",
                "front",
                "exterior",
                "third_person",
            ],
            "wrist_image": ["wrist", "hand", "cam_low", "gripper"],
            "secondary_image": ["secondary", "camera_1", "cam_side", "side"],
            "left_wrist_image": ["left_wrist", "cam_left_wrist"],
            "right_wrist_image": ["right_wrist", "cam_right_wrist"],
        }

        for cosmos_key, patterns in camera_mappings.items():
            # Direct match
            if cosmos_key in observation_dict:
                val = observation_dict[cosmos_key]
                if isinstance(val, np.ndarray) and val.ndim == 3:
                    obs[cosmos_key] = val[:, :, :3].astype(np.uint8)
                    continue

            # Pattern search
            for pattern in patterns:
                found = False
                for obs_key, val in observation_dict.items():
                    if pattern in obs_key.lower() and isinstance(val, np.ndarray) and val.ndim == 3:
                        obs[cosmos_key] = val[:, :, :3].astype(np.uint8)
                        found = True
                        break
                if found:
                    break

        # Proprioceptive state
        proprio = None
        for key in ("proprio", "observation.state", "state", "joint_positions"):
            if key in observation_dict:
                val = observation_dict[key]
                if isinstance(val, np.ndarray):
                    proprio = val.astype(np.float32)
                elif isinstance(val, (list, tuple)):
                    proprio = np.array(val, dtype=np.float32)
                break

        # Build from individual state keys if needed
        if proprio is None and self._robot_state_keys:
            values = []
            for key in self._robot_state_keys:
                if key in observation_dict:
                    values.append(float(observation_dict[key]))
            if values:
                proprio = np.array(values, dtype=np.float32)

        if proprio is not None:
            obs["proprio"] = proprio

        return obs

    def _decode_actions(self, result: Any) -> list[dict[str, Any]]:
        """Convert cosmos_get_action result to list of action dicts."""
        actions: list[dict[str, Any]] = []
        raw_actions = result.get("actions", []) if isinstance(result, dict) else result

        for action_vec in raw_actions:
            actions.append(self._vec_to_action_dict(np.asarray(action_vec, dtype=np.float32)))

        return actions

    def _vec_to_action_dict(self, action_vec: np.ndarray) -> dict[str, Any]:
        """Map a flat action vector to a named action dict."""
        action_dict: dict[str, Any] = {}

        if self._robot_state_keys:
            for j, key in enumerate(self._robot_state_keys):
                if j < len(action_vec) - 1:
                    action_dict[key] = float(action_vec[j])
            if len(action_vec) > 0:
                action_dict["gripper"] = float(action_vec[-1])
        else:
            # Default 7-DoF labels
            labels = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
            for j, label in enumerate(labels):
                if j < len(action_vec):
                    action_dict[label] = float(action_vec[j])

        return action_dict

    def reset(self, seed: int | None = None) -> None:
        """Reset internal step counter."""
        self._step = 0
