"""GR00T Policy - NVIDIA Isaac-GR00T inference.

Two inference modes:

1. **Service mode**: Connect to a running GR00T inference service via ZMQ.
   Works without Isaac-GR00T installed on the client, and is what
   ``strands-robots[groot-service]`` installs for.
2. **Local mode**: Load the model in this process. Requires NVIDIA's
   Isaac-GR00T package, which no extra declares - it installs from
   ``github.com/NVIDIA/Isaac-GR00T`` and pins ``transformers==4.57.3``, so it
   cannot share an interpreter with lerobot (``transformers>=5``). Where
   lerobot is installed, ``create_policy("lerobot_local",
   policy_type="groot", pretrained_name_or_path=...)`` is the in-process route:
   lerobot ships its own GR00T N1.7, parity-tested against NVIDIA's.

Observations and actions flow through explicit mappings between robot
sensor/actuator names and the model's modality keys.
"""

from strands_robots.policies.groot.client import Gr00tInferenceClient, MsgSerializer
from strands_robots.policies.groot.data_config import (
    DATA_CONFIG_MAP,
    Gr00tDataConfig,
    ModalityConfig,
    create_custom_data_config,
    load_data_config,
)
from strands_robots.policies.groot.policy import ActionMapping, Gr00tPolicy, ObservationMapping

__all__ = [
    "Gr00tPolicy",
    "Gr00tDataConfig",
    "Gr00tInferenceClient",
    "MsgSerializer",
    "ModalityConfig",
    "ObservationMapping",
    "ActionMapping",
    "load_data_config",
    "DATA_CONFIG_MAP",
    "create_custom_data_config",
]
