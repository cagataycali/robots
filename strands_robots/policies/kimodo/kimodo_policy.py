"""KimodoPolicy - text-to-motion via NVlabs/kimodo diffusion model.

Wraps the kimodo Python API to generate kinematic qpos trajectories from
natural language prompts. The generated frames are cached locally so
repeated identical prompts skip inference.

The policy operates in two modes:

1. **Standalone (kinematic replay)**: Generated qpos frames are applied
   directly to the robot via position control. Smooth and expressive but
   not physics-stable for contact-rich scenarios.

2. **Composite (kimodo + WBC)**: Kimodo generates the reference trajectory;
   a downstream WBC policy tracks it with torque control for physics
   stability. This mirrors the NVlabs ProtoMotions composition pattern.

Requires: ``pip install kimodo`` (Apache 2.0, NVlabs).
Model weights: ``nvidia/Kimodo-G1-RP-v1`` on HuggingFace (auto-downloaded).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy

logger = logging.getLogger(__name__)

# Default generation parameters matching kimodo CLI defaults.
_DEFAULTS = {
    "model": "nvidia/Kimodo-G1-RP-v1",
    "duration": 5.0,
    "diffusion_steps": 50,
    "cfg_weight": 7.5,
    "seed": 42,
    "text_encoder_device": "cuda",
    "fps": 30,
}

# Cache directory for generated trajectories.
_CACHE_DIR = Path(
    os.environ.get(
        "KIMODO_CACHE_DIR",
        os.path.expanduser("~/.cache/strands_robots/kimodo"),
    )
)


def _cache_key(prompt: str, model: str, duration: float, seed: int) -> str:
    """Deterministic hash key for a generation config."""
    blob = f"{model}|{prompt}|{duration}|{seed}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _load_cached(key: str) -> np.ndarray | None:
    """Load cached qpos trajectory if it exists."""
    path = _CACHE_DIR / f"{key}.npy"
    if path.exists():
        try:
            arr = np.load(path)
            logger.info("Kimodo cache hit: %s (%d frames)", key, arr.shape[0])
            return arr
        except Exception as e:
            logger.warning("Kimodo cache read failed for %s: %s", key, e)
    return None


def _save_cache(key: str, qpos: np.ndarray) -> None:
    """Persist generated qpos to disk cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.npy"
    try:
        np.save(path, qpos)
        logger.info("Kimodo cached: %s -> %s", key, path)
    except Exception as e:
        logger.warning("Kimodo cache write failed: %s", e)


def _detect_text_encoder_device(requested: str) -> str:
    """Auto-select text encoder device based on available VRAM.

    If requested device is "cuda" but free VRAM is under 12GB, fall back
    to "cpu" to avoid OOM (slower but functional on smaller GPUs).
    """
    if requested != "cuda":
        return requested
    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("CUDA unavailable, kimodo text encoder -> cpu")
            return "cpu"
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        if free_mem < 12.0:
            logger.info(
                "Free VRAM %.1fGB < 12GB, kimodo text encoder -> cpu",
                free_mem,
            )
            return "cpu"
    except ImportError:
        return "cpu"
    return "cuda"


def _generate_qpos(
    prompt: str,
    model: str,
    duration: float,
    diffusion_steps: int,
    cfg_weight: float,
    seed: int,
    text_encoder_device: str,
    fps: int,
) -> np.ndarray:
    """Run kimodo inference to produce a qpos trajectory.

    Returns:
        np.ndarray of shape (T, n_joints) where T = duration * fps.

    Raises:
        ImportError: If kimodo is not installed.
        RuntimeError: If generation fails.
    """
    try:
        from kimodo.scripts.generate import generate  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "kimodo is not installed. Install with:\n"
            "  pip install kimodo\n"
            "Or from source:\n"
            "  git clone https://github.com/nv-tlabs/kimodo && pip install -e kimodo\n"
            "Note: on aarch64, remove 'scenepic' from pyproject.toml before install."
        ) from e

    resolved_device = _detect_text_encoder_device(text_encoder_device)

    # Set env var that kimodo reads for text encoder placement.
    os.environ["TEXT_ENCODER_DEVICE"] = resolved_device

    logger.info(
        "Kimodo generating: prompt=%r, model=%s, duration=%.1fs, steps=%d, cfg=%.1f, seed=%d, text_enc=%s",
        prompt,
        model,
        duration,
        diffusion_steps,
        cfg_weight,
        seed,
        resolved_device,
    )

    try:
        result = generate(
            text=prompt,
            model_path=model,
            duration=duration,
            diffusion_steps=diffusion_steps,
            cfg_weight=cfg_weight,
            seed=seed,
            fps=fps,
        )
    except TypeError:
        # Fallback: older kimodo versions may have different signature.
        # Try CLI-style invocation via subprocess.
        logger.warning("kimodo.scripts.generate.generate() signature mismatch; falling back to subprocess invocation.")
        result = _generate_via_cli(prompt, model, duration, seed, fps)

    if isinstance(result, np.ndarray):
        return result

    # kimodo may return a dict with various output keys.
    if isinstance(result, dict):
        # Prefer the qpos CSV output (direct MuJoCo compat).
        for key in ("qpos", "posed_joints", "joint_positions"):
            if key in result and isinstance(result[key], np.ndarray):
                return result[key]
        # NPZ-style output: try to extract the most useful array.
        if "local_rot_mats" in result:
            logger.info("Kimodo returned rotation matrices; qpos extraction needed")
            # For G1, the model may output rotation matrices that need
            # conversion to joint angles. This path will be implemented
            # in PR-K2 with proper FK/IK mapping.
            raise RuntimeError(
                "Kimodo returned rotation matrices but qpos extraction "
                "is not yet implemented. Use a G1-specific model that "
                "outputs qpos directly (e.g. Kimodo-G1-RP-v1 with --output csv)."
            )

    raise RuntimeError(f"Unexpected kimodo output type: {type(result)}. Expected np.ndarray or dict with 'qpos' key.")


def _generate_via_cli(prompt: str, model: str, duration: float, seed: int, fps: int) -> np.ndarray:
    """Fallback: invoke kimodo_gen CLI and load the output CSV."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        out_path = tmp.name

    try:
        cmd = [
            "kimodo_gen",
            prompt,
            "--model",
            model.split("/")[-1] if "/" in model else model,
            "--duration",
            str(duration),
            "--seed",
            str(seed),
            "--output",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(
                f"kimodo_gen failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[:500]}\n"
                f"stderr: {result.stderr[:500]}"
            )
        qpos = np.loadtxt(out_path, delimiter=",")
        logger.info("Kimodo CLI generated %d frames", qpos.shape[0])
        return qpos
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


class KimodoPolicy(Policy):
    """Text-to-motion policy using NVlabs/kimodo diffusion model.

    Generates a full kinematic trajectory from a text prompt, then replays
    the frames as position-control actions at the configured fps. The
    trajectory is generated once at construction (or on first get_actions
    call) and cached for repeated use.

    Args:
        prompt: Natural language motion description (required).
        model: HuggingFace model ID or local path. Default: nvidia/Kimodo-G1-RP-v1.
        duration: Motion duration in seconds. Default: 5.0.
        diffusion_steps: Number of denoising steps. Default: 50.
        cfg_weight: Classifier-free guidance weight. Default: 7.5.
        seed: RNG seed for reproducibility. Default: 42.
        text_encoder_device: Device for text encoder ("cuda" or "cpu").
            Auto-detects: falls back to "cpu" if free VRAM < 12GB.
        fps: Output frame rate. Default: 30.
        lazy: If True (default), defer generation until first get_actions call.
            If False, generate immediately at construction.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._prompt: str = kwargs.get("prompt", "")
        self._model: str = kwargs.get("model", _DEFAULTS["model"])
        self._duration: float = float(kwargs.get("duration", _DEFAULTS["duration"]))
        self._diffusion_steps: int = int(kwargs.get("diffusion_steps", _DEFAULTS["diffusion_steps"]))
        self._cfg_weight: float = float(kwargs.get("cfg_weight", _DEFAULTS["cfg_weight"]))
        self._seed: int = int(kwargs.get("seed", _DEFAULTS["seed"]))
        self._text_encoder_device: str = kwargs.get("text_encoder_device", _DEFAULTS["text_encoder_device"])
        self._fps: int = int(kwargs.get("fps", _DEFAULTS["fps"]))
        self._lazy: bool = kwargs.get("lazy", True)

        self._robot_state_keys: list[str] = []
        self._qpos: np.ndarray | None = None
        self._frame_idx: int = 0

        if not self._lazy and self._prompt:
            self._ensure_trajectory()

        logger.info(
            "KimodoPolicy initialized: model=%s, prompt=%r, duration=%.1fs",
            self._model,
            self._prompt[:60],
            self._duration,
        )

    @property
    def provider_name(self) -> str:
        return "kimodo"

    @property
    def requires_images(self) -> bool:
        """Kimodo is text-conditioned, no camera input needed."""
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Reset playback to frame 0. Optionally re-seed for new generation."""
        self._frame_idx = 0
        if seed is not None:
            self._seed = seed
            # Invalidate cached trajectory so next get_actions regenerates.
            self._qpos = None

    def _ensure_trajectory(self) -> None:
        """Generate or load cached trajectory."""
        if self._qpos is not None:
            return

        if not self._prompt:
            raise ValueError(
                "KimodoPolicy requires a 'prompt' in policy_config. "
                "Example: policy_config={'prompt': 'walk forward then wave'}"
            )

        cache_key = _cache_key(self._prompt, self._model, self._duration, self._seed)
        cached = _load_cached(cache_key)
        if cached is not None:
            self._qpos = cached
            return

        qpos = _generate_qpos(
            prompt=self._prompt,
            model=self._model,
            duration=self._duration,
            diffusion_steps=self._diffusion_steps,
            cfg_weight=self._cfg_weight,
            seed=self._seed,
            text_encoder_device=self._text_encoder_device,
            fps=self._fps,
        )
        self._qpos = qpos
        _save_cache(cache_key, qpos)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Return the next chunk of qpos frames as joint-position actions.

        On first call, generates (or loads cached) the full trajectory.
        Subsequent calls advance through the trajectory, returning up to
        8 frames per call (matching the action_horizon convention).

        When the trajectory is exhausted, returns the final frame
        repeatedly (hold-last-pose behavior).
        """
        # Allow runtime prompt override via instruction or kwargs.
        runtime_prompt = kwargs.get("prompt") or instruction
        if runtime_prompt and runtime_prompt != self._prompt:
            self._prompt = runtime_prompt
            self._qpos = None
            self._frame_idx = 0

        self._ensure_trajectory()
        assert self._qpos is not None

        n_frames = self._qpos.shape[0]
        n_joints = self._qpos.shape[1]

        # Determine joint key mapping.
        if not self._robot_state_keys:
            # Use observation state dimension to infer joint count.
            if "observation.state" in observation_dict:
                state = observation_dict["observation.state"]
                dim = len(state) if hasattr(state, "__len__") else n_joints
                self._robot_state_keys = [f"joint_{i}" for i in range(dim)]
            else:
                self._robot_state_keys = [f"joint_{i}" for i in range(n_joints)]

        # Produce up to 8 action frames (standard action_horizon).
        actions: list[dict[str, Any]] = []
        for _ in range(8):
            # Clamp to last frame if trajectory exhausted.
            idx = min(self._frame_idx, n_frames - 1)
            frame = self._qpos[idx]

            action_dict: dict[str, Any] = {}
            for j, key in enumerate(self._robot_state_keys):
                if j < len(frame):
                    action_dict[key] = float(frame[j])

            actions.append(action_dict)
            if self._frame_idx < n_frames:
                self._frame_idx += 1

        return actions

    @property
    def trajectory_length(self) -> int:
        """Number of frames in the generated trajectory (0 if not yet generated)."""
        return self._qpos.shape[0] if self._qpos is not None else 0

    @property
    def is_exhausted(self) -> bool:
        """True if playback has reached the end of the trajectory."""
        if self._qpos is None:
            return False
        return self._frame_idx >= self._qpos.shape[0]
