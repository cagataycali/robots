"""GR00T Policy — NVIDIA GR00T N1.5 and N1.6 support.

Two inference modes:

1. **Service mode**: Connect to a running GR00T inference service via ZMQ.
   Works without Isaac-GR00T installed on the client.
2. **Local mode**: Load model directly on GPU.  Requires Isaac-GR00T.

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
from strands_robots.policies.groot.policy import (
    ActionMapping,
    Gr00tPolicy,
    ObservationMapping,
)

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
