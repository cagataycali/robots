"""Tiny Zenoh-config helpers reused across rogues + victim.

We build configs by hand here (rather than calling
``strands_robots.mesh._zenoh_config``) because most rogues want to
*deviate* from the defended posture: bypass the namespace, drop the
low_pass_filter, flip ACL ``enabled`` to ``false``, etc. Building
from scratch makes those deviations explicit and reviewable.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from time import sleep
from typing import Iterable

import zenoh


def free_port() -> int:
    """Pick a free localhost port (race-y but fine for tests)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def settle(seconds: float = 0.4) -> None:
    """Wait for the gossip layer to converge."""
    sleep(seconds)


def build_mtls_peer_config(
    *,
    ca_path: Path,
    cert_path: Path,
    key_path: Path,
    listen: Iterable[str] = (),
    connect: Iterable[str] = (),
    namespace: str = "strands",
) -> zenoh.Config:
    """Build a peer config that participates in the defended fleet.

    Mirrors what :mod:`strands_robots.mesh._zenoh_config` produces in
    its happy path: tls-only links, mTLS verify-name-on-connect, the
    fleet namespace, multicast off.
    """
    cfg = zenoh.Config()
    cfg.insert_json5("mode", '"peer"')
    cfg.insert_json5("namespace", json.dumps(namespace))
    cfg.insert_json5("scouting/multicast/enabled", "false")
    cfg.insert_json5("scouting/gossip/enabled", "true")
    cfg.insert_json5(
        "transport/link/protocols",
        json.dumps(["tls"]),
    )
    cfg.insert_json5(
        "transport/link/tls",
        json.dumps(
            {
                "enable_mtls": True,
                "verify_name_on_connect": True,
                "root_ca_certificate": str(ca_path),
                "listen_private_key": str(key_path),
                "listen_certificate": str(cert_path),
                "connect_private_key": str(key_path),
                "connect_certificate": str(cert_path),
            }
        ),
    )
    if listen:
        cfg.insert_json5("listen/endpoints", json.dumps([f"tls/{e}" for e in listen]))
    if connect:
        cfg.insert_json5("connect/endpoints", json.dumps([f"tls/{e}" for e in connect]))
    return cfg


def build_plain_peer_config(
    *,
    listen: Iterable[str] = (),
    connect: Iterable[str] = (),
    namespace: str | None = "strands",
) -> zenoh.Config:
    """Build a no-cert peer config (the outsider posture).

    Used by rogues that simulate an attacker without PKI material.
    A defended target on tls-only listen will refuse the link.
    """
    cfg = zenoh.Config()
    cfg.insert_json5("mode", '"peer"')
    if namespace is not None:
        cfg.insert_json5("namespace", json.dumps(namespace))
    cfg.insert_json5("scouting/multicast/enabled", "false")
    cfg.insert_json5("scouting/gossip/enabled", "false")
    if listen:
        cfg.insert_json5("listen/endpoints", json.dumps([f"tcp/{e}" for e in listen]))
    if connect:
        cfg.insert_json5("connect/endpoints", json.dumps([f"tcp/{e}" for e in connect]))
    return cfg
