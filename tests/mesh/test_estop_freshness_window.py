"""Freshness-window defenses on inbound remote estop envelopes.

``Mesh._on_safety_estop`` is the handler for the fleet-wide
``strands/safety/estop`` broadcast. Before it engages the local lockout it
applies three timestamp checks so that a malformed or replayed envelope can
never lock the fleet:

1. A missing or non-numeric ``t`` field is rejected (the canonical
   :meth:`emergency_stop` issuer always stamps ``t``).
2. A ``t`` beyond ``now + forward_skew_s`` in the future is rejected
   (defeats clock-rollback attacks against the freshness check).
3. A ``t`` older than ``freshness_window_s`` is rejected (a captured
   envelope replayed long after capture is stale).

Each rejection must leave the lockout *un-engaged* and the replay cache
untouched. These tests pin that contract behaviorally (observe the lockout
event, not internal flow).
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from strands_robots.mesh import core


def _stub_mesh() -> core.Mesh:
    """Minimal Mesh stub exercising only the safety estop handler."""
    m = core.Mesh.__new__(core.Mesh)
    m.peer_id = "test-peer"
    m._estop_replay_cache = {}
    m._resume_replay_cache = {}
    m._estop_replay_lock = threading.Lock()
    m._resume_replay_lock = threading.Lock()
    m._estop_lockout = threading.Event()
    m._last_estop_ts = 0.0
    m._last_estop_mono = 0.0
    m.publish_safety_event = lambda **kwargs: None  # type: ignore[method-assign]
    return m


def _envelope(**body):
    raw = json.dumps(body).encode()
    return SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda r=raw: r))


def test_estop_with_fresh_t_engages_lockout():
    """Control: a well-formed, fresh envelope DOES engage the lockout so
    the negative assertions below are meaningful (not vacuously passing
    because the handler rejects everything)."""
    mesh = _stub_mesh()
    mesh._on_safety_estop(_envelope(peer_id="issuer", t=time.time()))
    assert mesh._estop_lockout.is_set()


def test_estop_missing_t_rejected():
    """An envelope with no ``t`` is malformed -> rejected, no lockout."""
    mesh = _stub_mesh()
    mesh._on_safety_estop(_envelope(peer_id="issuer"))
    assert not mesh._estop_lockout.is_set()
    assert mesh._estop_replay_cache == {}


def test_estop_non_numeric_t_rejected():
    """A non-numeric ``t`` (string) is invalid -> rejected, no lockout."""
    mesh = _stub_mesh()
    mesh._on_safety_estop(_envelope(peer_id="issuer", t="not-a-number"))
    assert not mesh._estop_lockout.is_set()
    assert mesh._estop_replay_cache == {}


def test_estop_future_t_beyond_skew_rejected(monkeypatch):
    """A ``t`` past ``now + forward_skew_s`` defeats clock-rollback replay
    and must be rejected without engaging the lockout."""
    monkeypatch.setenv("STRANDS_MESH_RESUME_FORWARD_SKEW_S", "5")
    mesh = _stub_mesh()
    # 1 hour in the future -- far beyond the 5s skew tolerance.
    mesh._on_safety_estop(_envelope(peer_id="issuer", t=time.time() + 3600.0))
    assert not mesh._estop_lockout.is_set()
    assert mesh._estop_replay_cache == {}


def test_estop_stale_t_beyond_freshness_window_rejected(monkeypatch):
    """A ``t`` older than ``freshness_window_s`` is a stale/replayed
    envelope and must be rejected without engaging the lockout."""
    monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", "60")
    mesh = _stub_mesh()
    # 1 hour in the past -- far beyond the 60s freshness window.
    mesh._on_safety_estop(_envelope(peer_id="issuer", t=time.time() - 3600.0))
    assert not mesh._estop_lockout.is_set()
    assert mesh._estop_replay_cache == {}
