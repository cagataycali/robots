"""Out of contact is not gone, and a reader can state the freshness it needs.

Two related behaviors of the peer registry:

1. **Retention.** :func:`~strands_robots.mesh.session.prune_peers` deletes a
   silent peer at ``max(PEER_TIMEOUT, STRANDS_MESH_PEER_RETENTION_S)``. With
   retention unset the maximum is the timeout and behavior is byte-identical
   to before retention existed - that back-compat is pinned first, because it
   is the promise every existing fleet relies on. With retention set, a peer
   whose silence is *planned* (a satellite between ground-station passes, a
   rover in an RF shadow) stays in the registry as a row whose ``reachable``
   verdict is ``False``, instead of being erased as if it never existed -
   erasure is what turns a scheduled contact gap into a failover.

2. **Freshness bounds.** ``get_peer(peer_id, max_age_s=...)`` answers ``None``
   for a record older than the caller's bound, because for that caller it is
   unknown: a dispatcher must not assign work on a forty-minute-old sighting
   without saying so. The bound's domain is the shared positive-finite one -
   ``nan`` would make the age comparison answer ``False`` for every record,
   a bound failing open on exactly the stale record it exists to refuse, and
   ``True`` would be a silent one-second bound.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from strands_robots.mesh import session as mesh_session
from strands_robots.mesh.session import PEER_TIMEOUT, get_peer, prune_peers, update_peer


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The process-wide peer registry, emptied, with retention unset."""
    monkeypatch.delenv("STRANDS_MESH_PEER_RETENTION_S", raising=False)
    mesh_session._PEERS.clear()
    yield mesh_session
    mesh_session._PEERS.clear()


def _age_peer(peer_id: str, age_s: float) -> None:
    """Backdate a registered peer's heartbeat by *age_s* seconds."""
    mesh_session._PEERS[peer_id].last_seen_mono = time.monotonic() - age_s


class TestRetentionOffIsTheHistoricBehavior:
    def test_a_silent_peer_is_pruned_at_the_timeout(self, registry: Any) -> None:
        update_peer("sat-1", "robot", "h", {})
        _age_peer("sat-1", PEER_TIMEOUT + 1.0)

        assert prune_peers() == ["sat-1"]
        assert get_peer("sat-1") is None

    def test_an_unusable_retention_spelling_stays_off_and_off_means_timeout(
        self, registry: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``nan``/``inf``/negative retention would mean *no peer is ever
        pruned* (nan compares False against every age; inf is above all of
        them) - the registry's only bound left would be the eviction cap,
        reached silently. Such a spelling is a misconfiguration, not a long
        retention, so pruning behaves as if retention were unset."""
        for bad in ("nan", "inf", "-5", "soon"):
            monkeypatch.setenv("STRANDS_MESH_PEER_RETENTION_S", bad)
            update_peer("sat-1", "robot", "h", {})
            _age_peer("sat-1", PEER_TIMEOUT + 1.0)
            assert prune_peers() == ["sat-1"], f"retention {bad!r} must fall back to off"


class TestOutOfContactIsRetainedNotErased:
    def test_a_peer_in_a_contact_gap_is_kept_and_reported_unreachable(
        self, registry: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_MESH_PEER_RETENTION_S", "3600")
        update_peer("sat-1", "robot", "h", {"robot_id": "sat-1"})
        _age_peer("sat-1", PEER_TIMEOUT + 50.0)

        assert prune_peers() == [], "a peer inside retention must not be deleted"
        row = get_peer("sat-1")
        assert row is not None
        assert row["reachable"] is False, "presence stops meaning alive under retention; the row must say so"
        assert row["age"] > PEER_TIMEOUT

    def test_a_peer_past_retention_is_gone(self, registry: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_PEER_RETENTION_S", "60")
        update_peer("sat-1", "robot", "h", {})
        _age_peer("sat-1", 61.0)

        assert prune_peers() == ["sat-1"], "retention is a window, not immortality"

    def test_contact_restored_flips_reachable_back(self, registry: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_PEER_RETENTION_S", "3600")
        update_peer("sat-1", "robot", "h", {})
        _age_peer("sat-1", PEER_TIMEOUT + 50.0)
        assert get_peer("sat-1")["reachable"] is False

        update_peer("sat-1", "robot", "h", {})  # the next pass: heartbeats resume

        assert get_peer("sat-1")["reachable"] is True

    def test_a_fresh_peer_is_reachable(self, registry: Any) -> None:
        update_peer("arm-1", "robot", "h", {})
        assert get_peer("arm-1")["reachable"] is True


class TestTheCallerStatesTheFreshnessItNeeds:
    def test_a_record_older_than_the_bound_answers_unknown(self, registry: Any) -> None:
        update_peer("sat-1", "robot", "h", {})
        _age_peer("sat-1", 40.0)

        assert get_peer("sat-1", max_age_s=30.0) is None
        assert get_peer("sat-1") is not None, "no bound accepts any age (historic behavior)"

    def test_a_fresh_record_passes_the_bound(self, registry: Any) -> None:
        update_peer("sat-1", "robot", "h", {})
        assert get_peer("sat-1", max_age_s=30.0) is not None

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0, True])
    def test_an_unusable_bound_is_refused_not_read_permissively(self, registry: Any, bad: Any) -> None:
        """Each of these has a silent wrong meaning if read permissively:
        ``nan`` never trips (fails open on the stale record), ``inf`` is a
        spelled-out no-op, ``0``/negative refuse every record including a
        fresh one, and ``True`` is a one-second bound nobody wrote."""
        update_peer("sat-1", "robot", "h", {})
        with pytest.raises(ValueError, match="max_age_s"):
            get_peer("sat-1", max_age_s=bad)

    def test_an_unknown_peer_is_none_before_the_bound_is_consulted(self, registry: Any) -> None:
        assert get_peer("ghost", max_age_s=30.0) is None
