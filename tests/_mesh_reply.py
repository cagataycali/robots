"""One sample shape for a command reply as an honest peer sends it on Zenoh.

:meth:`strands_robots.mesh.core.Mesh._on_response` accepts a reply only when it
arrives on the responder's own five-segment response topic and carries the wire
zid that responder announced itself from on the presence topic. A unit test that
drives the handler directly needs a sample with both, and the receiver must
already have learned the binding, so :func:`reply_sample` does all three.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any


class _Zid:
    """Stringifies as a Zenoh session id, the way ``ZenohId`` does."""

    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


def zid_for(peer_id: str) -> str:
    """A stable 32-hex session id for *peer_id* (one session per peer)."""
    return hashlib.md5(peer_id.encode(), usedforsecurity=False).hexdigest()


def reply_sample(receiver: Any, payload: dict[str, Any], *, zid: str | None = None) -> SimpleNamespace:
    """The sample *receiver* gets when ``payload["responder_id"]`` answers it.

    Binds the responder's session on *receiver* as its presence would, then
    returns a sample on ``strands/<receiver>/response/<responder>/<turn>``
    carrying that session's zid. ``zid`` overrides the wire zid without moving
    the binding, which is what a reply from another session looks like.
    """
    responder = str(payload.get("responder_id"))
    bound = zid_for(responder)
    receiver._bind_peer_wire_zid(responder, bound)
    wire = bound if zid is None else zid
    return SimpleNamespace(
        payload=SimpleNamespace(to_bytes=lambda: json.dumps(payload).encode()),
        source_info=SimpleNamespace(source_id=SimpleNamespace(zid=_Zid(wire))),
        key_expr=f"strands/{receiver.peer_id}/response/{responder}/{payload.get('turn_id')}",
    )
