"""What a zenoh link implies for LeRobot's client settings: the LAN keeps LeRobot's defaults, the internet does not."""

from __future__ import annotations

import ipaddress
from typing import Any

WAN_TIMEOUT_S = (
    30.0  # a first call across the internet wakes the model and carries two frames: LeRobot's 5-10 s are too tight
)


def is_local(endpoint: str) -> bool:
    """``<link>/host:port`` names this machine or its LAN (loopback, private ranges, link-local, ``.local``)."""
    host = endpoint.split("/", 1)[-1].rpartition(":")[0].strip("[]")
    if host in ("localhost", "") or host.endswith(".local"):
        return True
    try:
        return ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def wan_defaults(endpoint: str, settings: dict[str, Any], descriptor: dict[str, Any] | None = None) -> dict[str, Any]:
    """LeRobot client settings a link implies, under anything the caller set by name.

    Off the LAN: handshake/action/startup timeouts >= 30 s and JPEG frames. A ``tls``/``quic``/``ws`` link is a
    router, so ``zenoh_mode="client"``. Once the server has spoken, the action timeout also covers its own
    ``action_deadline_s`` (the hello's ``limits``) — a server allowed 5 s to answer is not a client fault at 5 s.
    """
    out: dict[str, Any] = {}
    link = endpoint.split("/", 1)[0]
    if link in ("tls", "quic", "ws", "wss"):
        out["zenoh_mode"] = "client"
    if not is_local(endpoint):
        out.update(handshake_timeout_s=WAN_TIMEOUT_S, action_timeout_s=WAN_TIMEOUT_S, startup_timeout_s=WAN_TIMEOUT_S)
        out["encoding"] = "jpeg"
    if descriptor is not None:
        deadline = float((descriptor.get("limits") or {}).get("action_deadline_s") or 0.0)
        if deadline:
            out["action_timeout_s"] = max(
                out.get("action_timeout_s", 5.0), deadline + (0.0 if is_local(endpoint) else 5.0)
            )
    out.update(settings)  # the caller's word wins
    return out
