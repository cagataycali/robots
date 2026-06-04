"""Qwen-VLA Policy - unified vision-language-action model support.

Qwen-VLA (arXiv:2605.30280v2) pairs a Qwen3.5-4B VLM backbone with a 1.15B-param
DiT flow-matching action expert, conditioned on an embodiment-aware text prompt
that is the sole platform-specific interface.

Two inference modes:

1. **Service mode**: connect to a running Qwen-VLA inference server via ZMQ.
   Works without the Qwen-VLA package installed on the client.
2. **Local mode**: load the model directly on GPU. Requires the upstream
   ``qwen-vla`` package.

Observations and actions flow through explicit mappings between robot
sensor/actuator names and the model's modality keys.
"""

from strands_robots.policies.qwen_vla.client import MsgSerializer, QwenVlaInferenceClient
from strands_robots.policies.qwen_vla.data_config import (
    DATA_CONFIG_MAP,
    QwenVlaDataConfig,
    create_custom_data_config,
    load_data_config,
)
from strands_robots.policies.qwen_vla.normalize import (
    build_channel_mask,
    compute_quantile_stats,
    normalize,
    pad_to_width,
    unnormalize,
    unpad_from_width,
)
from strands_robots.policies.qwen_vla.policy import ActionMapping, ObservationMapping, QwenVlaPolicy
from strands_robots.policies.qwen_vla.prompt import build_embodiment_prompt

__all__ = [
    "QwenVlaPolicy",
    "QwenVlaDataConfig",
    "QwenVlaInferenceClient",
    "MsgSerializer",
    "ObservationMapping",
    "ActionMapping",
    "load_data_config",
    "create_custom_data_config",
    "DATA_CONFIG_MAP",
    "build_embodiment_prompt",
    "compute_quantile_stats",
    "normalize",
    "unnormalize",
    "build_channel_mask",
    "pad_to_width",
    "unpad_from_width",
]
