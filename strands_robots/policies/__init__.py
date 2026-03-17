"""Policy Abstraction for Universal VLA Support.

Plugin-based registry — all provider definitions live in registry/policies.json.
No hardcoded if/elif chains. New providers are auto-discovered or registered at runtime.

Built-in providers (see policies.json for full list):
    - mock: Sinusoidal test actions
    - groot: NVIDIA GR00T via ZMQ

Usage::

    from strands_robots.policies import create_policy, Policy

    # By provider name
    policy = create_policy("groot", port=5555)

    # By smart string (auto-resolves provider)
    policy = create_policy("zmq://localhost:5555")
    policy = create_policy("mock")

    # Custom provider
    register_policy("my_provider", lambda: MyPolicy, aliases=["my"])
"""

from strands_robots.policies.base import Policy
from strands_robots.policies.factory import create_policy, list_providers, register_policy
from strands_robots.policies.mock import MockPolicy

__all__ = [
    "Policy",
    "MockPolicy",
    "create_policy",
    "register_policy",
    "list_providers",
]
