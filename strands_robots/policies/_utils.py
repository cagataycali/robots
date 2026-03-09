"""Shared utility functions for policy providers.

Extracted to reduce code duplication across providers. Each function handles
common patterns that were previously copy-pasted across 6+ policy modules.

Import conventions across providers:
- ``numpy`` is imported at module level (lightweight, always available)
- ``torch``, ``transformers``, ``PIL`` are lazy-imported inside methods
  (heavy dependencies, not required until actual inference)
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_image(
    observation_dict: Dict[str, Any],
    default_size: Tuple[int, int] = (224, 224),
):
    """Extract a single PIL Image from an observation dictionary.

    Searches observation keys (sorted) for a 3D numpy array with 3 or 4 channels
    (HWC format). Returns the first match as an RGB PIL Image.

    Args:
        observation_dict: Observation dictionary from the robot.
        default_size: Fallback image size if no image is found.

    Returns:
        PIL.Image.Image in RGB format.
    """
    from PIL import Image

    for key in sorted(observation_dict.keys()):
        val = observation_dict[key]
        if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[-1] in (3, 4):
            return Image.fromarray(val[:, :, :3].astype(np.uint8))

    return Image.new("RGB", default_size)


def detect_device(requested: Optional[str] = None) -> str:
    """Detect the best available torch device.

    Priority: requested > CUDA > MPS (Apple Silicon) > CPU.

    Args:
        requested: Explicit device string (e.g. "cuda:0", "cpu"). If provided
            and valid, it is returned as-is.

    Returns:
        Device string suitable for ``torch.device()``.
    """
    if requested and requested != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def parse_numbers_from_text(text: str, n: int, fallback: float = 0.0) -> np.ndarray:
    """Parse floating-point numbers from model-generated text.

    Extracts up to *n* numbers from *text* using regex. If fewer than *n*
    numbers are found, pads with *fallback*.

    Args:
        text: Raw text output from a VLA model.
        n: Expected number of action dimensions.
        fallback: Value used to pad missing dimensions.

    Returns:
        numpy array of shape ``(n,)`` with dtype float32.
    """
    # Strip common prefixes like "Out:" that some models emit
    cleaned = text.split("Out:")[-1] if "Out:" in text else text
    numbers = re.findall(r"[-+]?\d*\.?\d+", cleaned)
    values = [float(v) for v in numbers[:n]]

    while len(values) < n:
        values.append(fallback)

    return np.array(values[:n], dtype=np.float32)
