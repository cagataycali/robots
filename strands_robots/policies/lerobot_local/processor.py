"""Processor Pipeline Bridge for LeRobot Local policy.

Integrates LeRobot's DataProcessorPipeline into the strands-robots policy flow.
Handles observation preprocessing and action postprocessing using the model's
own saved pipeline configs (preprocessor.json / postprocessor.json).

Architecture:
    Robot observation (dict)
        → ProcessorBridge.preprocess(obs)
            → LeRobot DataProcessorPipeline (normalize, device, batch, ...)
        → Policy.select_action(processed_obs)
        → ProcessorBridge.postprocess(action)
            → LeRobot DataProcessorPipeline (unnormalize, delta-action, ...)
        → Robot action (dict)
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Standard pipeline config filenames used by LeRobot
PREPROCESSOR_CONFIG = "policy_preprocessor.json"
POSTPROCESSOR_CONFIG = "policy_postprocessor.json"


def _try_import_processor():
    """Import LeRobot processor module.

    Returns:
        Dict of processor classes/functions, or None if not available.
    """
    try:
        from lerobot.processor.converters import (
            batch_to_transition,
            observation_to_transition,
            transition_to_batch,
            transition_to_observation,
        )
        from lerobot.processor.core import EnvTransition, TransitionKey
        from lerobot.processor.pipeline import DataProcessorPipeline

        logger.debug("LeRobot processor module loaded successfully")
        return {
            "DataProcessorPipeline": DataProcessorPipeline,
            "batch_to_transition": batch_to_transition,
            "transition_to_batch": transition_to_batch,
            "observation_to_transition": observation_to_transition,
            "transition_to_observation": transition_to_observation,
            "EnvTransition": EnvTransition,
            "TransitionKey": TransitionKey,
        }
    except ImportError:
        logger.debug(
            "LeRobot processor module not available. "
            "ProcessorBridge will pass data through unchanged. "
            "Install lerobot >= 0.5.0 for full processor support."
        )
        return None


class ProcessorBridge:
    """Bridge between strands-robots observation/action format and LeRobot's processor pipeline.

    Handles:
    - Loading preprocessor + postprocessor from pretrained model dirs / HF Hub
    - Running the pipeline steps (normalize, device transfer, observation processing, etc.)
    - Converting processed output back to strands-robots format

    Thread-safe: each bridge instance holds its own pipeline state.
    """

    def __init__(
        self,
        preprocessor=None,
        postprocessor=None,
        device: Optional[str] = None,
    ):
        """Initialize with optional pre/post processor pipelines.

        Args:
            preprocessor: LeRobot DataProcessorPipeline for observation preprocessing.
            postprocessor: LeRobot DataProcessorPipeline for action postprocessing.
            device: Target device for tensor operations (auto-detected if None).
        """
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._device = device
        self._modules = _try_import_processor()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        device: Optional[str] = None,
        preprocessor_config: str = PREPROCESSOR_CONFIG,
        postprocessor_config: str = POSTPROCESSOR_CONFIG,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "ProcessorBridge":
        """Load processor pipelines from a pretrained model.

        Tries to load preprocessor.json and postprocessor.json from the model
        directory or HuggingFace Hub. If either doesn't exist, that pipeline
        is skipped (passthrough).

        Args:
            pretrained_name_or_path: HF model ID or local path.
            device: Target device (auto-detected if None).
            preprocessor_config: Filename for preprocessor config.
            postprocessor_config: Filename for postprocessor config.
            overrides: Dict of step overrides (passed to both pipelines).

        Returns:
            ProcessorBridge instance with loaded pipelines.
        """
        modules = _try_import_processor()
        if modules is None:
            logger.info("LeRobot processor not available, creating passthrough bridge")
            return cls(device=device)

        DataProcessorPipeline = modules["DataProcessorPipeline"]

        preprocessor = None
        postprocessor = None

        # Load preprocessor
        try:
            preprocessor = DataProcessorPipeline.from_pretrained(
                pretrained_name_or_path,
                config_filename=preprocessor_config,
                overrides=overrides or {},
            )
            logger.info("Loaded preprocessor from %s: %d steps", pretrained_name_or_path, len(preprocessor))
        except (FileNotFoundError, ValueError) as exc:
            logger.debug("No preprocessor found: %s", exc)
        except OSError as exc:
            logger.warning("Failed to load preprocessor: %s", exc)

        # Load postprocessor
        try:
            postprocessor = DataProcessorPipeline.from_pretrained(
                pretrained_name_or_path,
                config_filename=postprocessor_config,
                overrides=overrides or {},
            )
            logger.info("Loaded postprocessor from %s: %d steps", pretrained_name_or_path, len(postprocessor))
        except (FileNotFoundError, ValueError) as exc:
            logger.debug("No postprocessor found: %s", exc)
        except OSError as exc:
            logger.warning("Failed to load postprocessor: %s", exc)

        return cls(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
        )

    @property
    def has_preprocessor(self) -> bool:
        """Whether a preprocessor pipeline is loaded."""
        return self._preprocessor is not None

    @property
    def has_postprocessor(self) -> bool:
        """Whether a postprocessor pipeline is loaded."""
        return self._postprocessor is not None

    @property
    def is_active(self) -> bool:
        """Whether any processing pipeline is active."""
        return self.has_preprocessor or self.has_postprocessor

    def preprocess(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a raw observation dict through the pipeline.

        If no preprocessor is loaded, returns observation unchanged.

        Args:
            observation: Raw observation dict from robot/sim.

        Returns:
            Processed observation dict (tensors on target device, normalized, etc.).

        Raises:
            RuntimeError: If the preprocessor pipeline fails.
        """
        if self._preprocessor is None or self._modules is None:
            return observation

        try:
            return self._preprocessor.process_observation(observation)
        except Exception as exc:
            raise RuntimeError(f"Preprocessor pipeline failed: {exc}") from exc

    def postprocess(self, action: Any) -> Any:
        """Postprocess a policy action through the pipeline.

        If no postprocessor is loaded, returns action unchanged.

        Args:
            action: Raw action from policy (tensor or dict).

        Returns:
            Processed action (unnormalized, converted to robot format, etc.).

        Raises:
            RuntimeError: If the postprocessor pipeline fails.
        """
        if self._postprocessor is None or self._modules is None:
            return action

        try:
            return self._postprocessor.process_action(action)
        except Exception as exc:
            raise RuntimeError(f"Postprocessor pipeline failed: {exc}") from exc

    def reset(self):
        """Reset pipeline state (e.g., clear running stats in stateful steps)."""
        if self._preprocessor is not None:
            self._preprocessor.reset()
        if self._postprocessor is not None:
            self._postprocessor.reset()

    def get_info(self) -> Dict[str, Any]:
        """Get information about loaded pipelines."""
        info: Dict[str, Any] = {
            "has_preprocessor": self.has_preprocessor,
            "has_postprocessor": self.has_postprocessor,
            "is_active": self.is_active,
            "device": self._device,
        }
        if self._preprocessor is not None:
            info["preprocessor_steps"] = len(self._preprocessor)
            info["preprocessor_step_names"] = [type(step).__name__ for step in self._preprocessor.steps]
        if self._postprocessor is not None:
            info["postprocessor_steps"] = len(self._postprocessor)
            info["postprocessor_step_names"] = [type(step).__name__ for step in self._postprocessor.steps]
        return info

    def __repr__(self) -> str:
        pre = f"pre={len(self._preprocessor)}steps" if self._preprocessor else "pre=None"
        post = f"post={len(self._postprocessor)}steps" if self._postprocessor else "post=None"
        return f"ProcessorBridge({pre}, {post})"


__all__ = [
    "ProcessorBridge",
    "PREPROCESSOR_CONFIG",
    "POSTPROCESSOR_CONFIG",
]
