"""Magma Policy Provider — Microsoft's Multi-modal Agentic Model.

8B-param model that bridges VLM reasoning with robot action prediction.
Built on the insight that agentic capabilities (planning, tool use, spatial
reasoning) transfer to robot control.

Unique: Magma is both a VLM (can answer questions, plan) AND a VLA (can
predict actions). This means the agent can first reason about a scene,
then predict actions — in the same model.

Usage:
    policy = create_policy("magma", model_id="microsoft/Magma-8B")
    actions = await policy.get_actions(obs, "pick up the cup")

Reference: Microsoft Research, "Magma: A Foundation Model for Multimodal AI Agents", 2025
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from strands_robots.policies import Policy

logger = logging.getLogger(__name__)


class MagmaPolicy(Policy):
    """Magma — Microsoft's Multi-modal Agentic VLA.

    Uses HF transformers interface. The model can:
    1. Understand scenes (VLM mode)
    2. Generate action plans (agentic mode)
    3. Predict robot actions (VLA mode)
    """

    def __init__(
        self,
        model_id: str = "microsoft/Magma-8B",
        device: Optional[str] = None,
        action_dim: int = 7,
        do_sample: bool = False,
        max_new_tokens: int = 256,
        **kwargs,
    ):
        self._model_id = model_id
        self._requested_device = device
        self._action_dim = action_dim
        self._do_sample = do_sample
        self._max_new_tokens = max_new_tokens
        self._robot_state_keys: List[str] = []

        self._model = None
        self._processor = None
        self._loaded = False
        self._step = 0

        logger.info(f"🔷 Magma policy: {model_id}")

    @property
    def provider_name(self) -> str:
        return "magma"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._robot_state_keys = robot_state_keys
        self._action_dim = len(robot_state_keys) or self._action_dim

    def _ensure_loaded(self):
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info(f"🔷 Loading Magma from {self._model_id}...")
        start = time.time()

        device = self._requested_device
        if not device:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._processor = AutoProcessor.from_pretrained(
            self._model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self._model.eval()

        elapsed = time.time() - start
        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            f"🔷 Magma loaded: {n_params/1e9:.1f}B in {elapsed:.1f}s on {device}"
        )
        self._loaded = True

    async def get_actions(
        self, observation_dict: Dict[str, Any], instruction: str, **kwargs
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        import torch
        from PIL import Image  # noqa: F401

        image = self._extract_image(observation_dict)

        # Magma prompt for action prediction
        prompt = f"<image>\nWhat action should the robot take to {instruction}?"

        inputs = self._processor(prompt, image, return_tensors="pt").to(
            self._device, dtype=torch.bfloat16
        )

        with torch.no_grad():
            # Check for direct action prediction API
            if hasattr(self._model, "predict_action"):
                action = self._model.predict_action(**inputs)
                action_np = np.asarray(action).flatten()
            else:
                # Generate text and parse action
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=self._do_sample,
                )
                text = self._processor.decode(outputs[0], skip_special_tokens=True)
                action_np = self._parse_action_text(text)

        action_dict = {}
        eef_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        keys = self._robot_state_keys if self._robot_state_keys else eef_keys
        for i in range(min(len(keys), len(action_np))):
            action_dict[keys[i]] = float(action_np[i])

        self._step += 1
        return [action_dict]

    def _extract_image(self, observation_dict):
        from PIL import Image

        for key in sorted(observation_dict.keys()):
            val = observation_dict[key]
            if (
                isinstance(val, np.ndarray)
                and val.ndim == 3
                and val.shape[-1] in (3, 4)
            ):
                return Image.fromarray(val[:, :, :3].astype(np.uint8))
        return Image.new("RGB", (224, 224))

    def _parse_action_text(self, text: str) -> np.ndarray:
        import re

        # Extract numbers from generated text
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        values = [float(n) for n in numbers[-self._action_dim :]]  # Take last N numbers
        while len(values) < self._action_dim:
            values.append(0.0)
        return np.array(values[: self._action_dim], dtype=np.float32)

    def reason_about_scene(
        self, observation_dict: Dict[str, Any], question: str
    ) -> str:
        """Use Magma as a VLM to reason about the scene (bonus capability)."""
        self._ensure_loaded()
        import torch
        from PIL import Image  # noqa: F401

        image = self._extract_image(observation_dict)
        prompt = f"<image>\n{question}"

        inputs = self._processor(prompt, image, return_tensors="pt").to(
            self._device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=512, do_sample=False
            )
        return self._processor.decode(outputs[0], skip_special_tokens=True)


__all__ = ["MagmaPolicy"]
