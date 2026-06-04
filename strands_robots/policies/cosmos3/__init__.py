"""Cosmos 3 Policy — NVIDIA omnimodal VLA policy for strands-robots.

Wraps the Cosmos 3 Generator *action* surface (e.g.
``nvidia/Cosmos3-Nano-Policy-DROID``) as a robots :class:`Policy`. Service mode
speaks to the Cosmos Framework RoboLab WebSocket policy server over OpenPI's
msgpack+NumPy protocol.

Usage::

    from strands_robots.policies import create_policy

    policy = create_policy("cosmos3", embodiment="droid", port=8000)
    chunk = policy.get_actions_sync(observation, "pick up the cube")

Available embodiments: droid, umi, av, bridge (see ``embodiments.py``).
"""

from .client import Cosmos3WebsocketClient
from .embodiments import (
    EMBODIMENTS,
    Cosmos3Embodiment,
    get_embodiment,
    list_embodiments,
)
from .policy import Cosmos3Policy

__all__ = [
    "Cosmos3Policy",
    "Cosmos3WebsocketClient",
    "Cosmos3Embodiment",
    "EMBODIMENTS",
    "get_embodiment",
    "list_embodiments",
]
