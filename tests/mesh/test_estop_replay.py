"""Regression tests for estop-replay defenses in Mesh._on_safety_estop.

PR #195 R9: addresses reviewer feedback that ``_on_safety_estop`` previously
had no freshness or replay defenses (asymmetric with ``_on_safety_resume``).
The reviewer's threat: an attacker with brief ACL access on ``safety/**``
captures one legitimate estop envelope, then replays it indefinitely on a
new TLS session to keep the fleet locked out. Combined with the default
permissive ACL (``default_acl()``), any CA-signed peer can originate and
replay estops.

The fix in core.py adds (mirroring _on_safety_resume):
1. Freshness window (``RESUME_FRESHNESS_WINDOW_S``, default 60s).
2. Forward-skew bound (``RESUME_FORWARD_SKEW_S``, default 5s).
3. Per-receiver replay cache keyed on ``(issuer_peer_id, t)`` --
   bounded LRU at ``RESUME_REPLAY_CACHE_MAX``.

Each test below would fail on the pre-R9 _on_safety_estop (which engaged
the lockout for any decode-able JSON object).
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh import core as core_module
from strands_robots.mesh.core import Mesh


def _make_mesh(peer_id: str = "r-test") -> Mesh:
    """Construct a minimally-instantiated Mesh without calling init_mesh."""
    robot = MagicMock()
    m = Mesh.__new__(Mesh)
    Mesh.__init__(m, robot, peer_id)
    return m


def _sample(payload_dict: dict) -> MagicMock:
    s = MagicMock()
    s.payload.to_bytes.return_value = json.dumps(payload_dict).encode()
    return s


def _envelope(*, peer_id: str = "op-1", t: float | None = None) -> dict:
    return {
        "peer_id": peer_id,
        "t": t if t is not None else time.time(),
        "responses_received": 0,
        "lockout_engaged": True,
    }


# --------------------------------------------------------------------------
# Happy path -- one fresh envelope engages the lockout.
# --------------------------------------------------------------------------


def test_fresh_envelope_engages_lockout() -> None:
    m = _make_mesh()
    assert not m._estop_lockout.is_set()
    m._on_safety_estop(_sample(_envelope()))
    assert m._estop_lockout.is_set()


# --------------------------------------------------------------------------
# Freshness window -- envelope older than RESUME_FRESHNESS_WINDOW_S rejected.
# --------------------------------------------------------------------------


def test_stale_envelope_rejected(caplog: pytest.LogCaptureFixture) -> None:
    m = _make_mesh()
    stale_t = time.time() - core_module.RESUME_FRESHNESS_WINDOW_S - 10
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        m._on_safety_estop(_sample(_envelope(t=stale_t)))
    assert not m._estop_lockout.is_set(), "stale envelope should not engage lockout"
    assert any("too old" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# Forward-skew bound -- future-dated envelope rejected.
# --------------------------------------------------------------------------


def test_future_envelope_rejected(caplog: pytest.LogCaptureFixture) -> None:
    m = _make_mesh()
    future_t = time.time() + core_module.RESUME_FORWARD_SKEW_S + 10
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        m._on_safety_estop(_sample(_envelope(t=future_t)))
    assert not m._estop_lockout.is_set()
    assert any("in future" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# Missing ``t`` -- malformed envelope rejected (also closes the strip-t
# bypass attack against the freshness check).
# --------------------------------------------------------------------------


def test_missing_t_rejected(caplog: pytest.LogCaptureFixture) -> None:
    m = _make_mesh()
    env = _envelope()
    env.pop("t")
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        m._on_safety_estop(_sample(env))
    assert not m._estop_lockout.is_set()
    assert any("missing/invalid ``t``" in r.message for r in caplog.records)


def test_non_numeric_t_rejected(caplog: pytest.LogCaptureFixture) -> None:
    m = _make_mesh()
    env = _envelope()
    env["t"] = "not-a-number"
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        m._on_safety_estop(_sample(env))
    assert not m._estop_lockout.is_set()


# --------------------------------------------------------------------------
# Replay cache -- captured envelope cannot re-engage after lockout cleared.
# This is the core attack the reviewer flagged.
# --------------------------------------------------------------------------


def test_captured_envelope_replay_after_clear_rejected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    m = _make_mesh()
    env = _envelope(peer_id="op-attacker", t=time.time())

    # Step 1: initial estop engages lockout.
    m._on_safety_estop(_sample(env))
    assert m._estop_lockout.is_set()

    # Step 2: operator clears lockout out-of-band (e.g. resume override).
    m._estop_lockout.clear()
    assert not m._estop_lockout.is_set()

    # Step 3: attacker replays the captured envelope -- must be rejected.
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        m._on_safety_estop(_sample(env))
    assert not m._estop_lockout.is_set(), "replayed estop envelope must NOT re-engage the lockout"
    assert any("REJECTED remote estop -- replay" in r.message for r in caplog.records)


def test_distinct_t_from_same_issuer_accepted() -> None:
    """Legitimate re-estop from the same issuer at a new time succeeds."""
    m = _make_mesh()
    issuer = "op-1"

    env1 = _envelope(peer_id=issuer, t=time.time())
    m._on_safety_estop(_sample(env1))
    assert m._estop_lockout.is_set()

    # operator resumes
    m._estop_lockout.clear()

    # New estop envelope (different t) -- must be accepted.
    env2 = _envelope(peer_id=issuer, t=time.time() + 0.001)
    assert env1["t"] != env2["t"]
    m._on_safety_estop(_sample(env2))
    assert m._estop_lockout.is_set()


def test_same_t_distinct_issuers_both_accepted() -> None:
    """Two issuers minting envelopes at the same instant both succeed --
    keying is per-(issuer, t), not just t."""
    m = _make_mesh()
    t = time.time()

    env1 = _envelope(peer_id="op-1", t=t)
    m._on_safety_estop(_sample(env1))
    assert m._estop_lockout.is_set()
    m._estop_lockout.clear()

    env2 = _envelope(peer_id="op-2", t=t)
    m._on_safety_estop(_sample(env2))
    assert m._estop_lockout.is_set()


# --------------------------------------------------------------------------
# Replay cache eviction -- bounded growth is enforced.
# --------------------------------------------------------------------------


def test_replay_cache_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay cache must not grow without bound under high estop volume."""
    monkeypatch.setattr(core_module, "RESUME_REPLAY_CACHE_MAX", 8)
    m = _make_mesh()

    base_t = time.time()
    for i in range(50):
        # each (issuer, t) is unique; lockout already engaged so handler
        # only updates cache and short-circuits the lockout branch
        m._on_safety_estop(_sample(_envelope(peer_id=f"op-{i}", t=base_t + i * 0.001)))

    assert len(m._estop_replay_cache) <= core_module.RESUME_REPLAY_CACHE_MAX, (
        f"cache exceeded bound: {len(m._estop_replay_cache)} > {core_module.RESUME_REPLAY_CACHE_MAX}"
    )


# --------------------------------------------------------------------------
# Malformed payload -- non-dict / non-JSON is rejected silently.
# (Verifies the narrow except clause from R9 docstring fix didn't regress
# the existing tolerance for malformed wire data.)
# --------------------------------------------------------------------------


def test_non_dict_rejected() -> None:
    m = _make_mesh()
    s = MagicMock()
    s.payload.to_bytes.return_value = b'"not-a-dict"'
    m._on_safety_estop(s)
    assert not m._estop_lockout.is_set()


def test_invalid_json_rejected() -> None:
    m = _make_mesh()
    s = MagicMock()
    s.payload.to_bytes.return_value = b"{{not valid json"
    m._on_safety_estop(s)
    assert not m._estop_lockout.is_set()
