"""Shared helpers for the rogue-fleet pentest kit.

Nothing in this package is fleet-specific -- it just keeps the rogues
and the victim from duplicating PKI + zenoh-config boilerplate.
"""

from rogue_fleet._lib.pki import EphemeralCA
from rogue_fleet._lib.report import RogueResult, write_result
from rogue_fleet._lib.zenoh_helpers import (
    build_mtls_peer_config,
    build_plain_peer_config,
    free_port,
    settle,
)

__all__ = [
    "EphemeralCA",
    "RogueResult",
    "write_result",
    "build_mtls_peer_config",
    "build_plain_peer_config",
    "free_port",
    "settle",
]
