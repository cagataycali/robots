"""An attributable wire source for tests that dispatch motion on a hardware peer.

A hardware ``Mesh`` attributes a motion command to its sender before it spends
any approval on it: the sample's publisher session id must be the one the sender
announced its presence from. :func:`bound_source` performs that announcement on
the mesh under test and returns the matching ``WireSource`` to pass to
``Mesh._dispatch(cmd, source=...)``.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

from strands_robots.mesh.core import Mesh, WireSource

LEADER_ZID = "a1b2c3d4e5f60718"


class _Zid:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


def presence_sample(peer: str, *, zid: str | None, robot_type: str = "operator") -> Any:
    """A zenoh-shaped presence sample; ``zid=None`` carries no ``source_info``."""
    payload = {"robot_id": peer, "robot_type": robot_type, "timestamp": time.time()}
    body = SimpleNamespace(to_bytes=lambda: json.dumps(payload).encode())
    source_info = None if zid is None else SimpleNamespace(source_id=SimpleNamespace(zid=_Zid(zid)))
    return SimpleNamespace(payload=body, source_info=source_info, key_expr="strands/presence")


def bound_source(mesh: Mesh, sender: str = "leader-1", zid: str = LEADER_ZID) -> WireSource:
    """Announce *sender* from session *zid* on *mesh* and return the source its commands arrive from."""
    mesh._on_presence(presence_sample(sender, zid=zid))
    return WireSource(sender_id=sender, wire_zid=zid, leg="lan")
