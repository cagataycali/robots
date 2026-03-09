#!/usr/bin/env python3
"""
InternVLA Policy Provider — Shanghai AI Lab's Vision-Language-Action models.

InternVLA family (A1, M1, N1, F1) built on Qwen3-VL / InternVL3 backbones.
State-of-the-art on RoboTwin 2.0 benchmark (89.4% avg success).

InternVLA-A1 uses Mixture-of-Transformers (MoT) to unify understanding,
generation, and action. Uses openpi-style server protocol.

Variants:
    - InternVLA-A1-3B: Pretrained on InternData-A1 + Agibot-World
    - InternVLA-A1-2B: Lighter variant
    - InternVLA-M1: Manipulation specialist (LIBERO benchmarks)
    - InternVLA-N1: Navigation specialist

Usage:
    # Via openpi server (recommended)
    policy = create_policy("internvla", server_url="http://localhost:8000")

    # Direct HF loading
    policy = create_policy("internvla", model_id="InternRobotics/InternVLA-A1-3B")

Reference: Cai et al., "InternVLA-A1", arXiv:2601.02456, 2026
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from strands_robots.policies import Policy

logger = logging.getLogger(__name__)


class InternvlaPolicy(Policy):
    """InternVLA — Qwen3-VL backbone VLA for manipulation.

    Supports two modes:
    1. Server mode (openpi protocol): connects to running InternVLA server
    2. Local mode: loads model directly from HuggingFace (requires GPU)
    """

    def __init__(
        self,
        model_id: str = "InternRobotics/InternVLA-A1-3B",
        server_url: Optional[str] = None,
        device: Optional[str] = None,
        action_dim: int = 7,
        **kwargs,
    ):
        self._model_id = model_id
        self._server_url = server_url
        self._device = device
        self._action_dim = action_dim
        self._robot_state_keys: List[str] = []

        self._model = None
        self._processor = None
        self._loaded = False
        self._step = 0

        mode = "server" if server_url else "local"
        logger.info(f"🧠 InternVLA policy: {model_id} (mode={mode})")

    @property
    def provider_name(self) -> str:
        return "internvla"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._robot_state_keys = robot_state_keys

    def _ensure_loaded(self):
        if self._loaded:
            return

        if self._server_url:
            # Server mode — just verify connectivity
            try:
                import requests

                requests.get(f"{self._server_url}/health", timeout=5)
                logger.info(f"🧠 InternVLA server connected: {self._server_url}")
            except Exception as e:
                logger.warning(f"InternVLA server not reachable: {e}")
            self._loaded = True
            return

        # Local mode — load from HF
        import torch

        logger.info(f"🧠 Loading InternVLA from {self._model_id}...")
        start = time.time()

        device = self._device
        if not device:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = device

        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self._model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(device)
            self._model.eval()

            elapsed = time.time() - start
            n_params = sum(p.numel() for p in self._model.parameters())
            logger.info(f"🧠 InternVLA loaded: {n_params/1e9:.1f}B in {elapsed:.1f}s on {device}")
        except Exception as e:
            raise ImportError(
                f"Failed to load InternVLA. Ensure the model repo is accessible.\n"
                f"For server mode: policy = create_policy('internvla', server_url='http://...')\n"
                f"Repo: https://github.com/InternRobotics/InternVLA-A1\nError: {e}"
            )

        self._loaded = True

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        self._ensure_loaded()

        if self._server_url:
            return await self._infer_server(observation_dict, instruction)
        return self._infer_local(observation_dict, instruction)

    async def _infer_server(self, observation_dict: Dict[str, Any], instruction: str) -> List[Dict[str, Any]]:
        """Inference via openpi-style HTTP server."""
        import io

        import requests
        from PIL import Image  # noqa: F401

        image = self._extract_image(observation_dict)
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        # openpi protocol
        payload = {
            "instruction": instruction,
            "image": buf.getvalue().hex(),
        }
        # Add state if available
        state = [float(observation_dict.get(k, 0.0)) for k in self._robot_state_keys]
        if state:
            payload["state"] = state

        resp = requests.post(f"{self._server_url}/act", json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        actions = np.array(result.get("actions", result.get("action", [0.0] * self._action_dim)))
        return [self._array_to_dict(actions.flatten())]

    def _infer_local(self, observation_dict: Dict[str, Any], instruction: str) -> List[Dict[str, Any]]:
        """Direct local inference."""
        import torch
        from PIL import Image  # noqa: F401

        image = self._extract_image(observation_dict)

        # InternVLA uses Qwen3-VL style chat format
        prompt = f"What action should the robot take to {instruction}?"

        inputs = self._processor(prompt, image, return_tensors="pt").to(self._device, dtype=torch.bfloat16)

        with torch.no_grad():
            if hasattr(self._model, "predict_action"):
                action = self._model.predict_action(**inputs)
            else:
                outputs = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
                action_text = self._processor.decode(outputs[0], skip_special_tokens=True)
                action = self._parse_action_text(action_text)

        action_np = np.asarray(action).flatten()
        self._step += 1
        return [self._array_to_dict(action_np)]

    def _extract_image(self, observation_dict):
        from PIL import Image

        for key in sorted(observation_dict.keys()):
            val = observation_dict[key]
            if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[-1] in (3, 4):
                return Image.fromarray(val[:, :, :3].astype(np.uint8))
        return Image.new("RGB", (224, 224))

    def _parse_action_text(self, text: str) -> np.ndarray:
        """Parse action values from generated text."""
        import re

        numbers = re.findall(r"[-+]?\d*\.?\d+", text.split("Out:")[-1] if "Out:" in text else text)
        values = [float(n) for n in numbers[: self._action_dim]]
        while len(values) < self._action_dim:
            values.append(0.0)
        return np.array(values, dtype=np.float32)

    def _array_to_dict(self, arr: np.ndarray) -> Dict[str, Any]:
        action_dict = {}
        eef_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        keys = self._robot_state_keys if len(self._robot_state_keys) >= len(arr) else eef_keys
        for i in range(min(len(keys), len(arr))):
            action_dict[keys[i]] = float(arr[i])
        return action_dict


__all__ = ["InternvlaPolicy"]
