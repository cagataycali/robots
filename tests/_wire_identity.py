"""Signed wire samples for tests that drive the mesh handlers directly.

A receiver that requires signed identity (:func:`require_signatures`) admits a
presence, a reply or a command only when it carries a ``sig`` envelope by a
certificate chained to its trust root whose common name speaks for the claimed
peer id. These helpers mint such certificates from
:class:`tests.mesh._pki.EphemeralCA` and build the zenoh-shaped samples the
handlers read, so a test can say "the victim's genuine reply" and "the
attacker's reply in the victim's name" in one line each.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.mesh import wire_identity as wi
from tests.mesh._pki import EphemeralCA, make_test_ca


def identity_for(ca: EphemeralCA, cn: str, out_dir: Path) -> wi.WireIdentity:
    """A signing identity for *cn*, issued by *ca*."""
    cert_path, key_path = ca.issue(cn, out_dir / cn)
    return wi.WireIdentity._from_files(cert_path, key_path, f"test:{cn}")


def roots_for(ca: EphemeralCA) -> wi.TrustRoots:
    """The trust roots a receiver verifies *ca*'s leaves against."""
    roots = wi.TrustRoots.from_pem(ca.cert_path.read_bytes())
    assert roots is not None
    return roots


class _Zid:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


def sample(payload: dict[str, Any], *, key: str, zid: str | None = None) -> Any:
    """A zenoh-shaped sample carrying *payload*; ``zid`` attaches a SourceInfo (the legacy label)."""
    body = SimpleNamespace(to_bytes=lambda: json.dumps(payload).encode())
    source_info = None if zid is None else SimpleNamespace(source_id=SimpleNamespace(zid=_Zid(zid)))
    return SimpleNamespace(payload=body, source_info=source_info, key_expr=key)


def signed_presence_sample(
    identity: wi.WireIdentity | None, peer_id: str, *, zid: str | None = None, **extra: Any
) -> Any:
    """A presence for *peer_id*, signed by *identity* (unsigned when ``None``)."""
    payload: dict[str, Any] = {"robot_id": peer_id, "robot_type": "robot", "hostname": "h", "timestamp": time.time()}
    payload.update(extra)
    if identity is not None:
        payload = wi.sign(identity, payload)
    return sample(payload, key=f"strands/{peer_id}/presence", zid=zid)


def signed_reply_sample(
    identity: wi.WireIdentity | None,
    receiver: str,
    responder: str,
    turn: str,
    result: dict[str, Any],
    *,
    zid: str | None = None,
    nonce: str | None = None,
) -> Any:
    """*responder*'s reply to *receiver* on *turn*, signed by *identity* (unsigned when ``None``)."""
    payload: dict[str, Any] = {
        "type": "response",
        "responder_id": responder,
        "turn_id": turn,
        "result": result,
        "timestamp": time.time(),
    }
    if identity is not None:
        payload = wi.sign(identity, payload, nonce=nonce)
    return sample(payload, key=f"strands/{receiver}/response/{responder}/{turn}", zid=zid)


def signed_cmd_sample(
    identity: wi.WireIdentity | None,
    sender: str,
    target: str,
    cmd: dict[str, Any],
    *,
    turn: str | None = None,
    zid: str | None = None,
    signed_for: str | None = None,
    name_target: bool = True,
) -> Any:
    """A command envelope from *sender* to *target*, signed by *identity* (unsigned when ``None``).

    The signed body names its target the way ``Mesh.send`` does. ``signed_for``
    makes it a command captured from *another* robot's topic and republished
    on *target*'s unchanged; ``name_target=False`` leaves the field out, the
    shape of a signed command that predates target binding.
    """
    payload: dict[str, Any] = {
        "sender_id": sender,
        "turn_id": turn or uuid.uuid4().hex,
        "command": cmd,
        "timestamp": time.time(),
    }
    if name_target:
        payload["target_id"] = signed_for or target
    if identity is not None:
        payload = wi.sign(identity, payload)
    return sample(payload, key=f"strands/{target}/cmd", zid=zid)


def require_signed_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> EphemeralCA:
    """Put the process on a mesh that requires signed identity (the ``require_signatures`` fixture).

    Writes a CA and a leaf for ``this-peer`` under ``tmp_path``, points the mTLS
    variables at them, and sets ``STRANDS_MESH_REQUIRE_SIGNED_IDENTITY=1``. The
    ``tests/mesh`` conftest defaults the auth mode to ``none`` only when the
    variable is unset, so setting ``mtls`` here is the documented opt-out.
    Returns the CA so the test can issue the peers it needs.
    """
    ca = make_test_ca(tmp_path / "ca")
    cert_path, key_path = ca.issue("this-peer", tmp_path / "this-peer")
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.delenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", raising=False)
    monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca.cert_path))
    monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert_path))
    monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key_path))
    monkeypatch.setenv(wi.REQUIRE_ENV, "1")
    return ca


def arm_receiver(mesh: Any, ca: EphemeralCA) -> None:
    """Give an un-started ``Mesh`` the trust roots a started one loads in ``start()``."""
    mesh._trust_roots = roots_for(ca)
