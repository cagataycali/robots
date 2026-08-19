"""Pin tests for BUGS.md #9 - a dead hub must not leave a dead session.

The first process on ``STRANDS_MESH_PORT`` becomes the listener ("hub"); every
later process fails the bind and falls back. The pre-fix fallback opened a plain
CLIENT session **with no reconnect**: when the hub died, every fallen-back peer
kept a dead session with no surfaced error and stayed dark until its process was
restarted (observed live twice - orphaned ``Robot()`` children after a dashboard
restart).

WHAT CHANGED 2026-08-19 (Q28, commit 8ece663f). The #9 fix chose PEER mode on an
ephemeral listener, on the belief that peers "can also route to each other
directly" while the hub is down. That belief was false, and it cost every
child-to-child topic in the fleet: a zenoh 1.x peer assumes a full mesh and
refuses traffic relayed by an intermediary - even by a router. Measured with
three sessions (hub, publisher, then a LATE subscriber, hub as the only
configured endpoint): peer child 0 of 62 frames, ROUTER hub + peer child 0 of 62,
client child 42 of 62. Teleop was the first child-to-child topic anyone tried and
it silently delivered nothing.

So the fallback is CLIENT mode again - but with the retry that was missing the
first time, which is what made client mode a dead end. #9's own property was
re-measured on the fix: 30 frames delivered, 0 during a 6s hub outage, 63 after
the hub came back, no process restart.

These tests therefore assert the resulting CONFIG, not the source text of the
function: the old versions grepped the fallback block for ``connect/retry`` and
``:0`` and broke the moment the two call sites were refactored into one helper,
which is a test coupled to layout rather than to behaviour. Delivery itself is
pinned by tests/test_mesh_fallback_topology.py, which runs real sessions.
"""

from __future__ import annotations

import inspect
import json

import pytest

from strands_robots.mesh import session as session_mod

zenoh = pytest.importorskip("zenoh")

HUB_EP = "tcp/127.0.0.1:7447"


def _fallback_cfg(monkeypatch, mode: str | None = None) -> dict:
    """The config a non-hub session actually opens with."""
    if mode is None:
        monkeypatch.delenv("STRANDS_MESH_FALLBACK_MODE", raising=False)
    else:
        monkeypatch.setenv("STRANDS_MESH_FALLBACK_MODE", mode)
    cfg = zenoh.Config()
    session_mod._apply_fallback_topology(cfg, HUB_EP, "tcp")
    return json.loads(str(cfg))


class TestFallbackReconnects:
    """#9: the fallback must never stop trying to reach the hub."""

    def test_fallback_sets_connect_retry(self, monkeypatch) -> None:
        cfg = _fallback_cfg(monkeypatch)
        assert cfg["connect"]["retry"] is not None, (
            "#9 regression - fallback session no longer retries the hub endpoint"
        )

    def test_fallback_never_gives_up(self, monkeypatch) -> None:
        cfg = _fallback_cfg(monkeypatch)
        assert cfg["connect"]["exit_on_failure"] is False, (
            "#9 regression - fallback session exits on connect failure instead of retrying"
        )

    def test_fallback_dials_the_hub(self, monkeypatch) -> None:
        cfg = _fallback_cfg(monkeypatch)
        assert cfg["connect"]["endpoints"] == [HUB_EP]

    def test_retry_backoff_is_bounded(self, monkeypatch) -> None:
        retry = _fallback_cfg(monkeypatch)["connect"]["retry"]
        assert retry["period_init_ms"] > 0
        assert retry["period_max_ms"] >= retry["period_init_ms"], (
            "fallback retry must bound its backoff period"
        )

    def test_both_call_sites_use_the_one_helper(self) -> None:
        """The bridge twin must not drift away from get_session's fallback.

        ``_get_zenoh_session_directly`` carried the pre-fix client-mode fallback
        for months after ``get_session`` was fixed, so a peer on the bridge
        transport stayed dark forever. One helper, asserted at both sites.
        """
        for fn in (session_mod.get_session, session_mod._get_zenoh_session_directly):
            src = inspect.getsource(fn)
            assert "_apply_fallback_topology(" in src, (
                f"{fn.__name__} must configure its fallback through the shared helper"
            )


class TestFallbackCanReceiveRelayedFrames:
    """Q28: the reconnect property is worthless if nothing is delivered."""

    def test_default_is_client_mode(self, monkeypatch) -> None:
        cfg = _fallback_cfg(monkeypatch)
        assert cfg["mode"] == "client", (
            "Q28 regression - a peer-mode child receives NOTHING a sibling child "
            "publishes (measured 0 of 62 frames, even through a router)"
        )

    def test_client_mode_declares_no_listener(self, monkeypatch) -> None:
        cfg = _fallback_cfg(monkeypatch)
        # zenoh's default listen/endpoints is a per-mode map with no "client"
        # entry, so a client listens on nothing without us overriding anything.
        assert isinstance(cfg["listen"]["endpoints"], dict)
        assert "client" not in cfg["listen"]["endpoints"]

    def test_peer_mode_remains_available_for_an_operator(self, monkeypatch) -> None:
        """An operator who wants direct peer links can still have them.

        They then have to arrange those links themselves: a peer only hears
        publishers it is directly connected to.
        """
        cfg = _fallback_cfg(monkeypatch, "peer")
        assert cfg["listen"]["endpoints"] == ["tcp/127.0.0.1:0"]
        assert cfg["connect"]["exit_on_failure"] is False, (
            "the peer path must keep #9's retry too"
        )
