"""Operator resume keys for the mesh tests: lock a peer, sign its resume."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from strands_robots.mesh import resume_authority


def trust_new_key(monkeypatch: Any) -> Ed25519PrivateKey:
    """A fresh operator key whose public half every peer in the test trusts."""
    key = Ed25519PrivateKey.generate()
    monkeypatch.setenv(resume_authority.PUBLIC_KEY_ENV, resume_authority.public_key_text(key.public_key()))
    return key


def lock(mesh: Any, epoch: str | None = None) -> str:
    """Engage *mesh*'s lockout the way an e-stop does; return its epoch."""
    mesh._estop_lockout.set()
    mesh._lockout_epoch = epoch or resume_authority.new_epoch()
    return mesh._lockout_epoch


def sign_for(key: Ed25519PrivateKey, *meshes: Any, **kw: Any) -> dict[str, Any]:
    """An assertion clearing the lockout the first of *meshes* holds, for all of them."""
    return resume_authority.sign_assertion(
        key, epoch=kw.pop("epoch", meshes[0]._lockout_epoch), targets=[m.peer_id for m in meshes], **kw
    )


def resume_sample(assertion: Any, *, peer_id: str = "relay-1", t: float | None = None) -> MagicMock:
    """A ``strands/safety/resume`` sample relaying *assertion*."""
    import time

    sample = MagicMock()
    sample.payload.to_bytes.return_value = json.dumps(
        {"peer_id": peer_id, "t": time.time() if t is None else t, "lockout_elapsed_s": 1.0, "assertion": assertion}
    ).encode()
    sample.source_info = None
    sample.retain = False
    return sample
