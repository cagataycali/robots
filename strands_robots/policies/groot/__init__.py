"""GR00T Policy — NVIDIA GR00T N1.5 and N1.6 support.

Two inference modes, both supporting N1.5 and N1.6 model generations:

1. **Service mode**: Connect to a running GR00T inference service via ZMQ.
   Works without Isaac-GR00T installed on the client.
2. **Local mode**: Load model directly on GPU. Requires the Isaac-GR00T
   package to be installed.
"""

from strands_robots.policies.groot.client import Gr00tInferenceClient, MsgSerializer
from strands_robots.policies.groot.data_config import (
    DATA_CONFIG_MAP,
    Gr00tDataConfig,
    ModalityConfig,
    create_custom_data_config,
    load_data_config,
)
from strands_robots.policies.groot.policy import Gr00tPolicy

__all__ = [
    "Gr00tPolicy",
    "Gr00tDataConfig",
    "Gr00tInferenceClient",
    "MsgSerializer",
    "ModalityConfig",
    "load_data_config",
    "DATA_CONFIG_MAP",
    "create_custom_data_config",
]
