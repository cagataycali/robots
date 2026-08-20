"""The server's own defence against a viewer that reopens a camera forever (Q46).

Measured twice on the live rig: 1.53 opens/s for twelve hours, unchanged after both
client-side cures landed, because the tab never reloaded to receive them.
"""

from __future__ import annotations

from strands_robots.dashboard.churn_guard import (
    CHURN_CAP_FPS,
    CHURN_OPENS_PER_MIN,
    ChurnGuard,
    effective_cap,
    viewer_identity,
)


class TestWhoIsWatching:
    def test_the_auth_subject_outranks_the_address(self) -> None:
        """Behind the tunnel every viewer is 127.0.0.1, so keying on the address would
        throttle the operator's laptop for a storming phone's behaviour."""
        a = viewer_identity(subject="passkey-abc", host="127.0.0.1", peer_id="p", cam="top")
        b = viewer_identity(subject="passkey-xyz", host="127.0.0.1", peer_id="p", cam="top")
        assert a != b

    def test_it_falls_back_to_the_host_when_auth_is_off(self) -> None:
        i = viewer_identity(subject=None, host="192.168.1.50", peer_id="p", cam="top")
        assert "192.168.1.50" in i

    def test_cameras_are_counted_separately(self) -> None:
        top = viewer_identity(subject="s", host=None, peer_id="p", cam="top")
        wrist = viewer_identity(subject="s", host=None, peer_id="p", cam="wrist")
        assert top != wrist


class TestTheVerdict:
    def test_a_human_reloading_is_never_throttled(self) -> None:
        """A person opens a tile a handful of times a minute. The measured storm was 92."""
        g = ChurnGuard()
        for i in range(CHURN_OPENS_PER_MIN):
            v = g.note_open("me", now=1000.0 + i)
            assert not v.throttled, f"throttled a human at open {i + 1}"

    def test_a_storm_is_capped_and_told_why(self) -> None:
        g = ChurnGuard()
        v = None
        for i in range(CHURN_OPENS_PER_MIN + 5):
            v = g.note_open("storm", now=1000.0 + i * 0.65)  # the measured 1.53/s
        assert v is not None and v.cap_fps == CHURN_CAP_FPS
        assert str(v.opens_in_window) in (v.reason or ""), v.reason
        assert "fps" in (v.reason or "") and "Reload" in (v.reason or "")

    def test_the_cap_lifts_by_itself_when_the_churn_stops(self) -> None:
        """A sliding window, so there is no state an operator must know how to clear."""
        g = ChurnGuard()
        for i in range(CHURN_OPENS_PER_MIN + 5):
            g.note_open("storm", now=1000.0 + i * 0.65)
        assert g.note_open("storm", now=1000.0 + 400.0).cap_fps is None

    def test_it_never_refuses_the_connection(self) -> None:
        """A refusal would blank the tile and hide the robot — and a storming client
        would simply reconnect. The verdict only ever carries a RATE."""
        g = ChurnGuard()
        for i in range(200):
            v = g.note_open("storm", now=1000.0 + i * 0.1)
            assert v.cap_fps in (None, CHURN_CAP_FPS)


class TestItCannotBecomeTheLeakItPrevents:
    def test_tracked_identities_are_bounded(self) -> None:
        g = ChurnGuard(max_tracked=8)
        for i in range(200):
            g.note_open(f"viewer-{i}", now=1000.0 + i)
        assert len(g._seen) <= 9  # the cap, plus the entry being noted right now

    def test_a_flooder_cannot_evict_the_operator_to_get_a_clean_slate(self) -> None:
        """Eviction drops the QUIETEST identities, and never the one being judged — so a
        flood cannot buy itself a fresh window, which is Q11's lesson one layer down."""
        g = ChurnGuard(max_tracked=4)
        for i in range(CHURN_OPENS_PER_MIN + 3):
            g.note_open("flooder", now=1000.0 + i * 0.5)
        for i in range(50):  # a wave of one-shot viewers
            g.note_open(f"noise-{i}", now=1000.0 + 20 + i * 0.1)
        assert g.note_open("flooder", now=1000.0 + 25).throttled


class TestEffectiveCap:
    def test_the_lower_rate_wins(self) -> None:
        assert effective_cap(10.0, 2.0) == 2.0
        assert effective_cap(1.0, 2.0) == 1.0

    def test_asking_for_more_cannot_escape_the_throttle(self) -> None:
        assert effective_cap(30.0, CHURN_CAP_FPS) == CHURN_CAP_FPS

    def test_no_cap_anywhere_means_no_cap(self) -> None:
        assert effective_cap(None, None) is None

    def test_either_one_alone_applies(self) -> None:
        assert effective_cap(None, 2.0) == 2.0
        assert effective_cap(5.0, None) == 5.0
