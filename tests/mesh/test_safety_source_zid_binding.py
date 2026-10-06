"""Regression tests for the wire-level ``source_zid`` binding on
safety estop / resume envelopes.

A safety envelope carries the publisher's TLS-authenticated Zenoh session ID
(``sample.source_info.source_id.zid``) in its body as ``source_zid``. The
receiver:

- extracts ``sample.source_info.source_id.zid`` (set by Zenoh during
  the mTLS-bootstrapped session handshake; ``ZenohId`` has no public
  Python constructor),
- requires body ``source_zid`` to equal wire ``source_zid`` when both
  are present,
- requires both-present-or-both-absent (no silent downgrade).

A resume is additionally decided by the operator's signed assertion it
relays (see ``test_resume_needs_the_operator_signature.py``).
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh import core as core_module
from strands_robots.mesh.core import Mesh, _extract_sample_source_zid

from ._resume import lock, sign_for, trust_new_key

# Real-format zid: 32 lowercase hex chars
_LEGIT_ZID = "0123456789abcdef0123456789abcdef"
_ATTACKER_ZID = "fedcba9876543210fedcba9876543210"


def _zid_obj(zid_str: str) -> Any:
    """Stand-in for ``zenoh.ZenohId`` whose ``str()`` returns the hex digest."""

    class _Zid:
        def __init__(self, s: str) -> None:
            self._s = s

        def __str__(self) -> str:
            return self._s

    return _Zid(zid_str)


def _make_sample(payload: dict, source_zid: str | None = None) -> SimpleNamespace:
    """Construct a minimal Zenoh-sample fake.

    When *source_zid* is provided the sample's
    ``source_info.source_id.zid`` returns it; otherwise ``source_info``
    is None (mirroring a transport that did not attach SourceInfo).
    """
    body = json.dumps(payload).encode("utf-8")
    if source_zid is None:
        return SimpleNamespace(
            payload=SimpleNamespace(to_bytes=lambda: body),
            source_info=None,
        )
    return SimpleNamespace(
        payload=SimpleNamespace(to_bytes=lambda: body),
        source_info=SimpleNamespace(
            source_id=SimpleNamespace(zid=_zid_obj(source_zid)),
            source_sn=1,
        ),
    )


@pytest.fixture
def receiver():
    """Bare ``Mesh`` instance with the bits ``_on_safety_*`` touch."""
    m = Mesh.__new__(Mesh)
    Mesh.__init__(m, MagicMock(), peer_id="receiver-1")
    m.publish_safety_event = MagicMock()
    return m


# Extractor ---------------------------------------------------------------


def test_extract_zid_from_well_formed_sample():
    """A sample carrying a 32-char hex zid returns that string."""
    sample = _make_sample({"hello": "world"}, source_zid=_LEGIT_ZID)
    assert _extract_sample_source_zid(sample) == _LEGIT_ZID


def test_extract_zid_returns_none_when_source_info_absent():
    """Bridge / IoT transports do not propagate source_info."""
    sample = _make_sample({"hello": "world"}, source_zid=None)
    assert _extract_sample_source_zid(sample) is None


def test_extract_zid_rejects_non_hex_stand_ins():
    """A ``MagicMock`` whose source_id.zid stringifies to a Mock repr
    must NOT be accepted as a real wire-bound zid."""
    sample = MagicMock()
    # MagicMock auto-creates source_info / source_id / zid; the str()
    # of a MagicMock is the well-known ``<MagicMock id=...>`` form,
    # which does not match the strict hex pattern.
    assert _extract_sample_source_zid(sample) is None


def test_extract_zid_rejects_uppercase_hex():
    """ZenohId stringifies to lowercase hex; uppercase is rejected.

    Defence-in-depth: the format pin is part of the binding contract.
    A future zenoh-python that emitted uppercase would force a code
    review (this test fails) rather than silently changing the wire
    invariant.
    """
    sample = _make_sample({}, source_zid=_LEGIT_ZID.upper())
    assert _extract_sample_source_zid(sample) is None


def test_extract_zid_rejects_overlong_string():
    """Strings longer than 32 hex chars are rejected as malformed."""
    sample = _make_sample({}, source_zid="0" * 33)
    assert _extract_sample_source_zid(sample) is None


def test_extract_zid_returns_none_when_source_id_missing():
    """A sample that attaches ``source_info`` but no ``source_id`` (a partial
    SourceInfo from a transport shim) degrades to None rather than crashing
    the safety handler."""
    sample = SimpleNamespace(
        payload=SimpleNamespace(to_bytes=lambda: b"{}"),
        source_info=SimpleNamespace(source_id=None),
    )
    assert _extract_sample_source_zid(sample) is None


def test_extract_zid_returns_none_when_zid_missing():
    """A ``source_id`` present but carrying no ``zid`` (an incomplete
    EntityGlobalId) yields None, not an empty/garbage identity."""
    sample = SimpleNamespace(
        payload=SimpleNamespace(to_bytes=lambda: b"{}"),
        source_info=SimpleNamespace(source_id=SimpleNamespace(zid=None)),
    )
    assert _extract_sample_source_zid(sample) is None


def test_extract_zid_returns_none_when_extraction_raises():
    """Defence in depth: if stringifying the zid raises (a hostile or broken
    sample object), the extractor treats it as \"no zid available\" and
    returns None instead of propagating into the safety handler."""

    class _ExplodingZid:
        def __str__(self) -> str:
            raise TypeError("zid str() blew up")

    sample = SimpleNamespace(
        payload=SimpleNamespace(to_bytes=lambda: b"{}"),
        source_info=SimpleNamespace(source_id=SimpleNamespace(zid=_ExplodingZid())),
    )
    assert _extract_sample_source_zid(sample) is None


# Receiver-side: estop ----------------------------------------------------


def test_estop_body_zid_matches_wire_zid_accepted(receiver):
    """When wire and body source_zid agree, the envelope is accepted."""
    now = time.time()
    payload = {
        "peer_id": "op-1",
        "t": now,
        "source_zid": _LEGIT_ZID,
        "trigger": "remote",
    }
    receiver._on_safety_estop(_make_sample(payload, source_zid=_LEGIT_ZID))
    assert receiver._estop_lockout.is_set()


def test_estop_body_zid_disagreeing_with_wire_rejected(receiver, caplog):
    """An attacker on a different session whose body claims a peer's
    session zid is rejected: the wire zid (set by Zenoh, attacker
    cannot choose) does not match the body claim."""
    now = time.time()
    payload = {
        "peer_id": "op-1",
        "t": now,
        "source_zid": _LEGIT_ZID,  # attacker's body claims legit zid
        "trigger": "remote",
    }
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        # ...but wire carries the attacker's actual zid.
        receiver._on_safety_estop(_make_sample(payload, source_zid=_ATTACKER_ZID))
    assert not receiver._estop_lockout.is_set()
    assert any("cross-session forgery rejected" in rec.message for rec in caplog.records), (
        "expected an explicit cross-session forgery rejection log"
    )


def test_estop_body_zid_present_wire_zid_absent_rejected(receiver, caplog):
    """A publisher that advertises body source_zid but failed to attach
    SourceInfo on the wire is rejected (no silent downgrade)."""
    now = time.time()
    payload = {
        "peer_id": "op-1",
        "t": now,
        "source_zid": _LEGIT_ZID,
    }
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        receiver._on_safety_estop(_make_sample(payload, source_zid=None))
    assert not receiver._estop_lockout.is_set()
    assert any("body source_zid present but wire" in rec.message for rec in caplog.records)


def test_estop_wire_zid_present_body_zid_absent_rejected(receiver, caplog):
    """A pre-binding publisher (no body source_zid) on a Zenoh session
    that DOES propagate source_info is rejected: operators must
    upgrade all peers together so the binding is never silently
    downgraded."""
    now = time.time()
    payload = {"peer_id": "op-1", "t": now}
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        receiver._on_safety_estop(_make_sample(payload, source_zid=_LEGIT_ZID))
    assert not receiver._estop_lockout.is_set()
    assert any("wire source_zid present but body" in rec.message for rec in caplog.records)


def test_estop_neither_wire_nor_body_zid_accepted(receiver):
    """Bridge / IoT transports where neither wire nor body carry a zid:
    the envelope is accepted via the body-level HMAC defences alone."""
    now = time.time()
    payload = {"peer_id": "op-1", "t": now, "trigger": "remote"}
    receiver._on_safety_estop(_make_sample(payload, source_zid=None))
    assert receiver._estop_lockout.is_set()


# Receiver-side: resume ----------------------------------------------------


def _make_resume_envelope(receiver, monkeypatch, *, source_zid: str | None = None) -> dict:
    """Lock *receiver* and relay an operator-signed resume for it, optionally
    carrying ``source_zid`` in the body."""
    key = trust_new_key(monkeypatch)
    lock(receiver)
    env = {"peer_id": "op-1", "t": time.time(), "lockout_elapsed_s": 1.0, "assertion": sign_for(key, receiver)}
    if source_zid is not None:
        env["source_zid"] = source_zid
    return env


def test_resume_body_zid_matches_wire_zid_clears_lockout(receiver, monkeypatch):
    """Happy path: wire and body source_zid agree, the signature verifies."""
    env = _make_resume_envelope(receiver, monkeypatch, source_zid=_LEGIT_ZID)
    receiver._on_safety_resume(_make_sample(env, source_zid=_LEGIT_ZID))

    assert not receiver._estop_lockout.is_set()


def test_resume_cross_session_forgery_rejected(receiver, monkeypatch, caplog):
    """A body claiming another session's ``source_zid`` is refused on the
    body != wire mismatch before the assertion is looked at."""
    env = _make_resume_envelope(receiver, monkeypatch, source_zid=_LEGIT_ZID)
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        receiver._on_safety_resume(_make_sample(env, source_zid=_ATTACKER_ZID))

    assert receiver._estop_lockout.is_set(), "cross-session forgery must NOT clear lockout"
    assert any("cross-session forgery rejected" in rec.message for rec in caplog.records)


def test_resume_pre_binding_publisher_rejected_when_wire_has_zid(receiver, monkeypatch, caplog):
    """A pre-binding peer (no body source_zid) communicating over a
    Zenoh session that DOES propagate source_info is rejected so the
    fleet upgrade is atomic."""
    env = _make_resume_envelope(receiver, monkeypatch, source_zid=None)
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        receiver._on_safety_resume(_make_sample(env, source_zid=_LEGIT_ZID))

    assert receiver._estop_lockout.is_set()
    assert any("publisher predates source_zid binding" in rec.message for rec in caplog.records)


def test_resume_bridge_transport_no_zid_either_side_accepted(receiver, monkeypatch):
    """Bridge / IoT transport: neither wire nor body carries a zid. The
    signed assertion alone decides; cross-session-forgery defence is
    Zenoh-specific."""
    env = _make_resume_envelope(receiver, monkeypatch, source_zid=None)
    receiver._on_safety_resume(_make_sample(env, source_zid=None))

    assert not receiver._estop_lockout.is_set()


# Publisher-side helpers --------------------------------------------------


def test_local_session_zid_returns_none_without_zenoh_session(receiver):
    """When no Zenoh session is open, ``_local_session_zid`` returns
    ``None`` and the safety publisher path falls back to body-only
    binding."""
    # Default fixture has no live session; the helper must return None
    # rather than raising.
    assert receiver._local_session_zid() is None


def test_safety_publisher_for_returns_none_without_session(receiver):
    """``_safety_publisher_for`` returns None when no session is open;
    callers fall back to ``put()``."""
    assert receiver._safety_publisher_for("strands/safety/estop") is None


def test_next_safety_sn_is_monotonic_per_topic(receiver):
    """Sequence numbers increment per-topic and never repeat."""
    a1 = receiver._next_safety_sn("strands/safety/estop")
    a2 = receiver._next_safety_sn("strands/safety/estop")
    a3 = receiver._next_safety_sn("strands/safety/estop")
    assert a1 < a2 < a3

    b1 = receiver._next_safety_sn("strands/safety/resume")
    b2 = receiver._next_safety_sn("strands/safety/resume")
    assert b1 < b2

    # Per-topic counters are independent.
    assert b1 == 1, "resume topic counter starts at 1 independently of the estop topic"


def test_publish_safety_envelope_falls_back_to_put_without_session(receiver, monkeypatch):
    """Without a Zenoh session ``_publish_safety_envelope`` MUST call
    ``put()`` so the bridge / IoT transport path still delivers the
    envelope (just without TLS-bound source attribution)."""
    calls = []

    def fake_put(key, payload):
        calls.append((key, dict(payload)))

    monkeypatch.setattr(core_module, "put", fake_put)
    receiver._publish_safety_envelope("strands/safety/estop", {"hello": "world"})

    assert calls == [("strands/safety/estop", {"hello": "world"})]
