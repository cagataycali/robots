"""Regression: a remote lockout resume must clear peers on the SourceInfo-less
fallback publish path.

When a Zenoh build lacks ``SourceInfo`` (or no session/publisher is available)
the safety envelope is published on the fallback ``put()`` path, which strips
``source_zid`` from the body. The relayed assertion must still clear a receiver
on that transport, or the fleet stays e-stopped forever.
"""

import json
import types
from unittest.mock import MagicMock

from strands_robots.mesh import core

from ._resume import lock, trust_new_key


def _fallback_sample(payload: dict) -> object:
    """A wire sample with NO source_info (the fallback / bridge transport):
    _extract_sample_source_zid() returns None for it."""
    sample = types.SimpleNamespace()
    sample.payload = types.SimpleNamespace(to_bytes=lambda: json.dumps(payload).encode())
    # deliberately no ``source_info`` attribute -> wire_zid resolves to None
    return sample


def test_resume_proof_verifies_when_published_on_fallback_path(monkeypatch):
    key = trust_new_key(monkeypatch)

    # --- Issuer: an open session (so _local_session_zid resolves a real zid)
    # but the native SourceInfo path is unavailable, so the envelope is
    # published on the fallback path that strips source_zid. ---
    issuer = core.Mesh(robot=object(), peer_id="issuer")
    issuer.publish_safety_event = MagicMock()
    monkeypatch.setattr(issuer, "_local_session_zid", lambda: "deadbeefdeadbeef")
    # No native publisher available -> fallback path (and _safety_wire_zid None).
    monkeypatch.setattr(issuer, "_safety_publisher_for", lambda key: None)

    published: dict = {}

    def capture_put(key, payload):
        published["key"] = key
        published["payload"] = payload

    monkeypatch.setattr(core, "put", capture_put)

    # Engage the local lockout, then resume with the operator's key.
    epoch = lock(issuer)
    issuer._last_estop_ts = core.time.time()
    issuer._last_estop_mono = core.time.monotonic()
    result = issuer.resume(key, targets=["receiver"])
    assert result == {"status": "ok"}

    assert published["key"] == "strands/safety/resume"
    envelope = published["payload"]
    # Fallback path stripped source_zid from the body...
    assert "source_zid" not in envelope
    # ...and still relays the signed assertion.
    assert envelope["assertion"]["epoch"] == epoch

    # --- Receiver on the same fallback transport (no wire source_zid). ---
    receiver = core.Mesh(robot=object(), peer_id="receiver")
    receiver.publish_safety_event = MagicMock()
    lock(receiver, epoch)

    receiver._on_safety_resume(_fallback_sample(envelope))

    # The assertion verified on the zid-less transport -> lockout cleared.
    assert receiver._estop_lockout.is_set() is False


def test_safety_wire_zid_none_when_source_info_unavailable(monkeypatch):
    """_safety_wire_zid returns None on a zenoh build lacking SourceInfo, so
    the resume body carries no zid the fallback path would strip."""
    import sys

    m = core.Mesh(robot=object(), peer_id="t1")
    monkeypatch.setattr(m, "_local_session_zid", lambda: "abc123")
    monkeypatch.setattr(m, "_safety_publisher_for", lambda key: object())
    fake_zenoh = types.ModuleType("zenoh")  # no SourceInfo attribute
    monkeypatch.setitem(sys.modules, "zenoh", fake_zenoh)
    assert m._safety_wire_zid("strands/safety/resume") is None


def test_safety_wire_zid_returns_zid_on_native_path(monkeypatch):
    import sys

    m = core.Mesh(robot=object(), peer_id="t1")
    monkeypatch.setattr(m, "_local_session_zid", lambda: "abc123")
    monkeypatch.setattr(m, "_safety_publisher_for", lambda key: object())
    fake_zenoh = types.ModuleType("zenoh")
    fake_zenoh.SourceInfo = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zenoh", fake_zenoh)
    assert m._safety_wire_zid("strands/safety/resume") == "abc123"
