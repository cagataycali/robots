"""Qwen-VLA training pipeline - 4-stage recipe (T2A -> CPT -> SFT -> RL).

Heavy training dependencies (torch, transformers, accelerate) are gated behind
the ``qwen-vla-train`` extra and loaded lazily, so importing this package's
config / embodiment-tag helpers never pulls them in.
"""

from strands_robots.training.qwen_vla.config import (
    CPTConfig,
    RLConfig,
    SFTConfig,
    T2AConfig,
    TimestepDist,
)
from strands_robots.training.qwen_vla.data.embodiment_tags import (
    EMBODIMENT_TAGS,
    EmbodimentTag,
    get_embodiment_tag,
)

__all__ = [
    "T2AConfig",
    "CPTConfig",
    "SFTConfig",
    "RLConfig",
    "TimestepDist",
    "EmbodimentTag",
    "EMBODIMENT_TAGS",
    "get_embodiment_tag",
]
