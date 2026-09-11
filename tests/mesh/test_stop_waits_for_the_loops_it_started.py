"""``Mesh.stop`` waits for the sensor loops before releasing what they publish through.

:meth:`~strands_robots.mesh.core.Mesh.start` launches nine background loops
(heartbeat, state, and the seven :class:`~strands_robots.mesh.sensors.SensorLoopsMixin`
loops), collecting them on a roster; ten with camera publishing enabled. Each is
paced by a :class:`~strands_robots.mesh.pacing.Ticker` on a shared stop event, so
each notices a stop within 10ms of the *next* tick boundary - but a tick already
inside ``publish()`` when the flag flipped is not interrupted by it.

:meth:`~strands_robots.mesh.core.Mesh.stop` used to flip the flag and immediately
undeclare the subscribers, drop the session reference and log ``off mesh``. So an
in-flight tick landed on the wire after the peer announced it had left, through a
session reference the peer no longer held - and ``stop()`` returned with every loop
it started still running. The roster was collected at four sites and read at none.

Pinned here: every loop is joined before the teardown that follows, the join is
bounded so one wedged sensor read cannot wedge ``stop()``, the budget is spent
across the loops rather than per loop, and a loop that outlasts it is named at
WARNING rather than being announced as a stop that happened.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import strands_robots.mesh.core as core_mod
import strands_robots.mesh.session as session_mod
from strands_robots.mesh.core import Mesh

#: Loops ``start()`` launches unconditionally, by roster thread-name suffix.
ALWAYS_ON_LOOPS = (
    "heartbeat",
    "state",
    "pose",
    "health",
    "imu",
    "odom",
    "lidar",
    "hand",
    "map-info",
)

PEER = "peer-a"


class _FakeRobot:
    """The smallest robot the loops can read without a driver or a transport."""

    tool_name_str = "fakebot"

    def get_task_status(self) -> dict[str, Any]:
        return {"status": "idle"}

    def get_features(self) -> dict[str, Any]:
        return {"foo": "bar"}


@contextmanager
def running_mesh(
    on_put: Callable[[str, dict[str, Any]], None] | None = None,
    on_release: Callable[[], None] | None = None,
) -> Iterator[Mesh]:
    """Yield a started :class:`Mesh` whose transport is a mock.

    Args:
        on_put: Called for every publish, so a test can hold one tick open or
            record the order publishes land in.
        on_release: Called when ``stop()`` drops the session reference.

    Yields:
        The started mesh. Always stopped again on the way out, so a test that
        leaves a loop wedged still cannot leak it into a later test.
    """
    session = MagicMock()
    with (
        patch.object(session_mod, "get_session", return_value=session),
        patch.object(session_mod, "current_session", return_value=session),
        patch.object(core_mod, "get_session", return_value=session),
        patch.object(core_mod, "current_session", return_value=session),
        patch.object(core_mod, "release_session", side_effect=on_release),
        patch.object(core_mod, "put", side_effect=on_put),
    ):
        mesh = Mesh(_FakeRobot(), peer_id=PEER, peer_type="robot")
        mesh.start()
        try:
            yield mesh
        finally:
            if mesh.alive:
                mesh.stop()


@pytest.fixture(autouse=True)
def _startable_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let ``start()`` past the ACL posture gate, and pace the loops fast."""
    monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "true")
    monkeypatch.setenv("STRANDS_MESH_HEALTH_HZ", "50")


def _tick_holder(topic_suffix: str) -> tuple[Any, Any, list[str], threading.Event, threading.Event]:
    """Build the put/release hooks that hold one tick of ``topic_suffix`` open."""
    entered, release = threading.Event(), threading.Event()
    order: list[str] = []
    lock = threading.Lock()

    def on_put(key: str, _data: dict[str, Any]) -> None:
        if key.endswith(f"/{topic_suffix}") and not entered.is_set():
            entered.set()
            release.wait(10)
            with lock:
                order.append("publish-landed")

    def on_release() -> None:
        with lock:
            order.append("session-released")

    return on_put, on_release, order, entered, release


def test_the_roster_holds_every_loop_start_launched() -> None:
    """The join can only cover the loops ``start()`` actually recorded."""
    with running_mesh() as mesh:
        names = {t.name for t in mesh._threads}
    assert names == {f"mesh-{loop}-{PEER}" for loop in ALWAYS_ON_LOOPS}


def test_no_loop_is_still_running_when_stop_returns() -> None:
    """``stop()`` does not announce a stop it has not waited for."""
    with running_mesh() as mesh:
        mesh.stop()
        still_running = [t.name for t in mesh._threads if t.is_alive()]
    assert still_running == []


def test_an_in_flight_publish_lands_before_stop_returns() -> None:
    """A tick inside ``publish()`` finishes while ``stop()`` is still waiting."""
    on_put, on_release, order, entered, release = _tick_holder("health")
    with running_mesh(on_put=on_put, on_release=on_release) as mesh:
        mesh._read_health = lambda: {"peer_id": PEER, "battery_pct": 42.0}  # type: ignore[method-assign]
        assert entered.wait(5), "the health loop never reached publish()"
        threading.Timer(0.2, release.set).start()
        mesh.stop()
        assert "publish-landed" in order, "stop() returned while a publish was in flight"


def test_an_in_flight_publish_lands_before_the_session_reference_is_dropped() -> None:
    """No loop publishes through a session reference the peer has given back."""
    on_put, on_release, order, entered, release = _tick_holder("health")
    with running_mesh(on_put=on_put, on_release=on_release) as mesh:
        mesh._read_health = lambda: {"peer_id": PEER, "battery_pct": 42.0}  # type: ignore[method-assign]
        assert entered.wait(5), "the health loop never reached publish()"
        threading.Timer(0.2, release.set).start()
        mesh.stop()
    assert order == ["publish-landed", "session-released"]


def test_the_camera_loop_is_joined_like_the_always_on_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in loop is on the roster, so waiting covers it too."""
    monkeypatch.setenv("STRANDS_MESH_CAMERA_HZ", "20")
    with running_mesh() as mesh:
        assert f"mesh-camera-{PEER}" in {t.name for t in mesh._threads}
        mesh.stop()
        assert [t.name for t in mesh._threads if t.is_alive()] == []


def test_a_wedged_loop_is_named_rather_than_wedging_stop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A sensor read that never returns costs the budget, not the process."""
    monkeypatch.setattr(core_mod, "LOOP_JOIN_TIMEOUT_S", 0.3)
    on_put, _on_release, _order, entered, release = _tick_holder("health")
    with caplog.at_level("WARNING"), running_mesh(on_put=on_put) as mesh:
        mesh._read_health = lambda: {"peer_id": PEER, "battery_pct": 42.0}  # type: ignore[method-assign]
        assert entered.wait(5), "the health loop never reached publish()"
        start = time.monotonic()
        mesh.stop()
        elapsed = time.monotonic() - start
        release.set()
    assert elapsed < 3.0, f"stop() waited {elapsed:.1f}s on a wedged loop"
    assert f"mesh-health-{PEER}" in caplog.text
    assert "0.3s" in caplog.text


def test_the_budget_is_spent_across_the_loops_not_once_per_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three wedged loops cost one budget: they wind down in parallel."""
    monkeypatch.setattr(core_mod, "LOOP_JOIN_TIMEOUT_S", 0.4)
    wedged = threading.Event()
    entered = threading.Semaphore(0)
    held = {"health", "imu", "odom"}

    def on_put(key: str, _data: dict[str, Any]) -> None:
        if key.rsplit("/", 1)[-1] in held:
            entered.release()
            wedged.wait(10)

    with running_mesh(on_put=on_put) as mesh:
        mesh._read_health = lambda: {"peer_id": PEER, "battery_pct": 1.0}  # type: ignore[method-assign]
        mesh._read_imu = lambda: {"peer_id": PEER, "accel": [0.0, 0.0, 9.8]}  # type: ignore[method-assign]
        mesh._read_odom = lambda: {"peer_id": PEER, "x": 0.0}  # type: ignore[method-assign]
        for _ in range(len(held)):
            assert entered.acquire(timeout=5), "not every held loop reached publish()"
        start = time.monotonic()
        mesh.stop()
        elapsed = time.monotonic() - start
        wedged.set()
    assert elapsed < 3 * 0.4, f"{elapsed:.2f}s looks like one budget per loop, not one shared"


def test_a_peer_whose_loops_are_idle_stops_promptly() -> None:
    """The ordinary path still returns at once - waiting is not a new delay."""
    with running_mesh() as mesh:
        start = time.monotonic()
        mesh.stop()
        elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"an idle peer took {elapsed:.2f}s to stop"
