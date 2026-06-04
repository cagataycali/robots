"""Policy Abstraction for Universal VLA Support.

Plugin-based registry - all provider definitions live in registry/policies.json.
No hardcoded if/elif chains. New providers are auto-discovered or registered at runtime.

Built-in providers (see policies.json for full list):
    - mock: Sinusoidal test actions
    - groot: NVIDIA GR00T via ZMQ
    - lerobot_local: Direct HuggingFace inference (ACT, Pi0, SmolVLA, Diffusion, ...)

Usage::

    from strands_robots.policies import create_policy, Policy

    # By provider name
    policy = create_policy("groot", port=5555)
    policy = create_policy("lerobot_local",
        pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human")

    # By smart string (auto-resolves provider)
    policy = create_policy("lerobot/act_aloha_sim")
    policy = create_policy("zmq://localhost:5555")
    policy = create_policy("mock")

    # Custom provider
    register_policy("my_provider", lambda: MyPolicy, aliases=["my"])
"""

from strands_robots.policies.base import Policy
from strands_robots.policies.factory import (
    UntrustedRemoteCodeError,
    create_policy,
    list_providers,
    register_policy,
)
from strands_robots.policies.mock import MockPolicy

# Cosmos3Policy is import-safe: it depends only on numpy; the optional
# ``openpi-client`` dependency is imported lazily inside the WebSocket client.
# Imported unconditionally, exactly like MockPolicy above.
from strands_robots.policies.cosmos3 import Cosmos3Policy

__all__ = [
    "Policy",
    "MockPolicy",
    "Cosmos3Policy",
    "create_policy",
    "register_policy",
    "list_providers",
    "UntrustedRemoteCodeError",
]
