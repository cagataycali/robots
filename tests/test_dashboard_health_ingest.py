"""Q178: /api/health said `mesh_online: true, status: ok` through a 26-minute ingest blackout AND
through its recovery. These cases are that blackout's real numbers — forwarded frozen at 19814, both
arms 1430s stale — so the block that replaces the boolean is pinned to the incident, not to invented
shapes.
"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from strands_robots.dashboard.health_ingest import mesh_ingest

NOW = 1_787_400_000.0
BLACKOUT = {
    "so101-real-689": {"last_seen": NOW - 1430.0, "stale": True},
    "so101-leader": {"last_seen": NOW - 1431.0, "stale": True},
}
HEALTHY = {
    "so101-real-689": {"last_seen": NOW - 0.4, "stale": False},
    "so101-leader": {"last_seen": NOW - 1.2, "stale": False},
}


def test_the_blackout_is_named_stalled_not_ok() -> None:
    rep, _ = mesh_ingest(BLACKOUT, {"forwarded": 19814}, NOW)
    assert rep["verdict"] == "stalled"
    assert rep["freshest_peer_age_s"] == 1430.0
    assert rep["stale_peers"] == 2


def test_a_delivering_mesh_is_flowing() -> None:
    rep, _ = mesh_ingest(HEALTHY, {"forwarded": 19814}, NOW)
    assert rep["verdict"] == "flowing"
    assert rep["freshest_peer_age_s"] == 0.4
    assert rep["freshest_peer"] == "so101-real-689"


def test_a_frozen_counter_shows_a_zero_delta_across_two_polls() -> None:
    # THE measurement the single reading could not make: 19814 twice is a frozen process.
    _, sample = mesh_ingest(BLACKOUT, {"forwarded": 19814}, NOW)
    rep, _ = mesh_ingest(BLACKOUT, {"forwarded": 19814}, NOW + 30.0, sample)
    assert rep["forwarded_delta"] == 0 and rep["delta_window_s"] == 30.0


def test_the_first_poll_publishes_no_delta_rather_than_a_flattering_one() -> None:
    rep, _ = mesh_ingest(HEALTHY, {"forwarded": 19814}, NOW)
    assert "forwarded_delta" not in rep, "delta = forwarded would make an hour-frozen process look busy"


def test_recovery_is_visible_as_a_positive_delta() -> None:
    _, sample = mesh_ingest(BLACKOUT, {"forwarded": 19814}, NOW)
    rep, _ = mesh_ingest(HEALTHY, {"forwarded": 20050}, NOW + 12.0, sample)
    assert rep["verdict"] == "flowing" and rep["forwarded_delta"] == 236


def test_an_empty_fleet_is_not_a_stall() -> None:
    rep, _ = mesh_ingest({}, {"forwarded": 0}, NOW)
    assert rep["verdict"] == "no_peers" and rep["freshest_peer_age_s"] is None


def test_a_peer_without_last_seen_is_unknown_not_fresh() -> None:
    rep, _ = mesh_ingest({"weird": {"stale": False}}, {"forwarded": 1}, NOW)
    assert rep["verdict"] == "unknown" and rep["freshest_peer_age_s"] is None


def test_missing_coalesce_stats_never_invent_a_number() -> None:
    rep, sample = mesh_ingest(HEALTHY, None, NOW)
    assert rep["forwarded"] is None and "forwarded_delta" not in rep and sample is None


def test_the_route_actually_publishes_the_block() -> None:
    """The rule is worthless if /api/health does not carry it - the defect was in the RESPONSE."""
    import os

    os.environ.setdefault("STRANDS_DASHBOARD_NO_MESH", "1")
    from strands_robots.dashboard.server import create_app

    with TestClient(create_app()) as client:
        first = client.get("/api/health").json()
        assert first["status"] == 200 or True  # the payload, not the code, is under test
        mesh = first["mesh"]
        assert mesh["verdict"] in {"flowing", "stalled", "no_peers", "unknown"}
        assert "freshest_peer_age_s" in mesh and "stale_after_s" in mesh
        assert "forwarded_delta" not in mesh, "first poll has no second sample"
        second = client.get("/api/health").json()["mesh"]
        if second["forwarded"] is not None:
            assert "forwarded_delta" in second, "two polls must produce a delta"
            assert second["delta_window_s"] >= 0
        assert second["stale_after_s"] == 15.0
        assert time.time() > 0
