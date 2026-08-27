"""Dead peers must age OUT of the fleet snapshot, not linger as ghost cards.

A finished replay/collect peer (replay-14005 and its child replay-14005__so101)
stops heartbeating and never returns, so "stale forever" is a permanent lie on
the dashboard. Aged out means gone -- unless a live managed local process is
behind it, in which case a quiet state stream must not erase a running robot.
"""

from __future__ import annotations

import time

from strands_robots.dashboard.mesh_bridge import (
    PEER_STALE_S,
    MeshBridge,
    prune_peers,
)

NOW = 1_000_000.0
TTL = 300.0


def _peers(now: float = NOW) -> dict[str, dict]:
    return {
        "fresh": {"last_seen": now - 1.0},
        "quiet": {"last_seen": now - (PEER_STALE_S + 5.0)},
        "dead": {"last_seen": now - (TTL + 1.0)},
    }


def _live_peers() -> dict[str, dict]:
    """Same shape, but anchored to the wall clock snapshot() actually reads."""
    return _peers(time.time())


def test_fresh_peer_stays_and_is_not_stale():
    out = prune_peers(_peers(), NOW, TTL)
    assert "fresh" in out
    assert out["fresh"]["stale"] is False


def test_quiet_but_recent_peer_stays_marked_stale():
    out = prune_peers(_peers(), NOW, TTL)
    assert out["quiet"]["stale"] is True


def test_peer_older_than_ttl_disappears():
    out = prune_peers(_peers(), NOW, TTL)
    assert "dead" not in out
    assert set(out) == {"fresh", "quiet"}


def test_dead_peer_with_live_managed_process_stays():
    out = prune_peers(_peers(), NOW, TTL, protected_ids={"dead"})
    assert "dead" in out
    assert out["dead"]["stale"] is True  # visible, but honestly quiet


def test_child_sim_peer_is_protected_by_its_parent():
    peers = {"replay-1__so101": {"last_seen": NOW - (TTL + 60.0)}}
    assert prune_peers(peers, NOW, TTL) == {}
    kept = prune_peers(peers, NOW, TTL, protected_ids={"replay-1"})
    assert "replay-1__so101" in kept


def test_ttl_zero_disables_ageing_out():
    out = prune_peers(_peers(), NOW, 0.0)
    assert set(out) == {"fresh", "quiet", "dead"}


def test_missing_last_seen_counts_as_ancient():
    out = prune_peers({"never": {}}, NOW, TTL)
    assert out == {}


def test_original_mapping_is_not_mutated():
    peers = _peers()
    prune_peers(peers, NOW, TTL)
    assert set(peers) == {"fresh", "quiet", "dead"}
    assert "stale" not in peers["fresh"]


def test_bridge_snapshot_drops_dead_peers_and_forgets_them():
    bridge = MeshBridge(peer_id="dashboard-test")
    bridge.peers = _live_peers()
    snap = bridge.snapshot()
    assert "dead" not in snap["peers"]
    assert snap["peers"]["quiet"]["stale"] is True
    # Forgotten for good: the ghost cannot come back on the next snapshot.
    assert "dead" not in bridge.peers


def test_bridge_snapshot_keeps_protected_peer_and_survives_bad_hook():
    bridge = MeshBridge(peer_id="dashboard-test")
    bridge.peers = _live_peers()
    bridge.protected_peer_ids = lambda: {"dead"}
    assert "dead" in bridge.snapshot()["peers"]

    bridge.peers = _live_peers()

    def boom():
        raise RuntimeError("device manager exploded")

    bridge.protected_peer_ids = boom
    snap = bridge.snapshot()  # must not raise
    assert "dead" not in snap["peers"]
    assert "fresh" in snap["peers"]
