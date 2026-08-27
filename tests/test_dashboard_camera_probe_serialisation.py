"""Two /api/devices requests may not probe the cameras at the same time.

Every request is handed to a worker thread, so the operator's rescan, a browser
tab's 5s poll and a phone all probe concurrently by default. Concurrent probes
open the SAME camera indices, each sees the other's camera as busy, and the loser
writes a false "unavailable" over a good answer with a fresh timestamp on it.
"""

from __future__ import annotations

import threading
import time

from strands_robots.dashboard import cameras as camera_facts
from strands_robots.dashboard.device_manager import DeviceManager


# ---------------------------------------------------------------- pure decision
def test_probe_needed_refresh_probes_when_cache_predates_the_request() -> None:
    # The operator pressed rescan to learn about a cable they just plugged in;
    # an answer measured BEFORE they asked cannot contain it.
    assert camera_facts.probe_needed(
        refresh=True, requested_at=100.0, cache_t=99.0, ttl_s=30.0, now=100.0
    )


def test_probe_needed_answer_that_landed_after_the_request_is_enough() -> None:
    # A probe finished while we waited for the lock: its result is at least as new
    # as the question, so re-probing would only fight it for the devices.
    assert not camera_facts.probe_needed(
        refresh=True, requested_at=100.0, cache_t=100.5, ttl_s=30.0, now=101.0
    )
    assert not camera_facts.probe_needed(
        refresh=False, requested_at=100.0, cache_t=100.0, ttl_s=30.0, now=101.0
    )


def test_probe_needed_plain_poll_still_honours_the_ttl() -> None:
    assert not camera_facts.probe_needed(
        refresh=False, requested_at=100.0, cache_t=80.0, ttl_s=30.0, now=100.0
    )
    assert camera_facts.probe_needed(
        refresh=False, requested_at=100.0, cache_t=60.0, ttl_s=30.0, now=100.0
    )


# ------------------------------------------------------------- the real manager
def test_concurrent_refresh_requests_probe_the_hardware_once(monkeypatch) -> None:
    mgr = DeviceManager()
    overlap = []
    running = threading.Event()
    calls = []

    def slow_scan(skip=None):
        calls.append(time.time())
        # If another thread is already inside the probe, record it: this is the
        # double-open that makes a healthy camera report "unavailable".
        if running.is_set():
            overlap.append(True)
        running.set()
        time.sleep(0.25)
        running.clear()
        return ([{"index": 0, "width": 640, "height": 480}], {})

    monkeypatch.setattr(
        "strands_robots.dashboard.device_manager.scan_cameras_with_failures", slow_scan
    )
    monkeypatch.setattr(mgr, "_camera_names", lambda refresh=False: [])
    monkeypatch.setattr(mgr, "_claimed_camera_indices", lambda: {})
    monkeypatch.setattr(mgr, "_streaming_indices", lambda live: set())

    threads = [threading.Thread(target=lambda: mgr._cameras(refresh=True)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert overlap == [], "two probes held the cameras open at the same time"
    assert len(calls) == 1, f"the hardware was probed {len(calls)} times for 4 concurrent requests"


def test_a_later_refresh_still_gets_a_fresh_probe(monkeypatch) -> None:
    """Serialising must not turn rescan into a no-op for the NEXT press."""
    mgr = DeviceManager()
    calls = []

    def scan(skip=None):
        calls.append(1)
        return ([{"index": len(calls) - 1, "width": 640, "height": 480}], {})

    monkeypatch.setattr(
        "strands_robots.dashboard.device_manager.scan_cameras_with_failures", scan
    )
    monkeypatch.setattr(mgr, "_camera_names", lambda refresh=False: [])
    monkeypatch.setattr(mgr, "_claimed_camera_indices", lambda: {})
    monkeypatch.setattr(mgr, "_streaming_indices", lambda live: set())

    mgr._cameras(refresh=True)
    time.sleep(0.01)
    mgr._cameras(refresh=True)
    assert len(calls) == 2
