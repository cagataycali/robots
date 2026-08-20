"""The state loop must publish at STATE_HZ on THIS machine, not at 40% of it.

BUGS.md Q69: the mesh's publish loops paced themselves with
``self._stop_event.wait(period)``, and in a process tree descended from a daemon
(or launchd agent, or a supervising loop) every such wait is inflated by ~145ms
on macOS. STATE_HZ is 10, so the state stream ran at ~4Hz and every counter
reported 4Hz as the rate the robot managed.

This test measures the loop's ACHIEVED rate through the real ``Mesh._state_loop``
with the transport mocked, and calibrates its floor against the machine it is
running on (:func:`sleep_penalty_s`) instead of asserting a number that happens
to pass in a terminal. That calibration is the whole point: the regression it
guards against is invisible in an interactive shell and severe under a daemon.

The session is mocked exactly the way tests/mesh/test_mesh.py mocks it, so no
real zenoh session is ever built here - Q30's law (a test that cannot prove
transport isolation must refuse to run) applies to anything that touches a Mesh.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh.core import Mesh
from strands_robots.mesh.pacing import sleep_penalty_s
from strands_robots.mesh.session import STATE_HZ


class _StatefulRobot:
    """Duck-typed robot whose state read is cheap and always succeeds."""

    tool_name_str = "pacingbot"

    def __init__(self) -> None:
        self._world = MagicMock()
        self._world._data.time = 1.0
        self._world.robots = {"arm0": object()}


def _run_state_loop_for(seconds: float) -> tuple[int, float]:
    """Drive the real _state_loop for a wall-clock window; count state publishes."""
    mesh = Mesh(_StatefulRobot(), peer_id="pace-1", peer_type="sim")
    published: list[str] = []
    mesh.publish = lambda topic, _payload=None, **_kw: published.append(topic)  # type: ignore[method-assign]
    mesh._running = True
    stop = mesh._stop_event
    started = threading.Event()

    def loop() -> None:
        started.set()
        mesh._state_loop()

    thread = threading.Thread(target=loop, daemon=True)
    with patch("strands_robots.mesh.core.put"):
        thread.start()
        started.wait(2.0)
        start = time.perf_counter()
        # Busy-wait the window: time.sleep here would itself be taxed by the very
        # penalty under test, making the window longer than asked and the
        # measured rate look better than it is.
        while time.perf_counter() - start < seconds:
            pass
        elapsed = time.perf_counter() - start
        ticks = len([t for t in published if t.endswith("/state")])
        mesh._running = False
        stop.set()
        thread.join(timeout=5.0)
    assert not thread.is_alive(), "the state loop did not stop within 5s of the stop event"
    return ticks, elapsed


class TestTheStateLoopHitsItsNominalRate:
    def test_achieved_rate_is_close_to_state_hz_even_where_sleeps_are_taxed(self) -> None:
        penalty = sleep_penalty_s()
        window = 1.2
        ticks, elapsed = _run_state_loop_for(window)
        achieved = ticks / elapsed

        floor = STATE_HZ * 0.7
        assert achieved >= floor, (
            f"state loop achieved {achieved:.1f}Hz against STATE_HZ={STATE_HZ} "
            f"(sleep penalty on this machine: {penalty * 1000:.0f}ms). Below {floor:.1f}Hz means the loop is "
            "paced by an inflated blocking wait again - see BUGS.md Q69."
        )

        if penalty >= 0.01:
            # On a taxed machine the OLD pacing could not have passed: 1/(0.1 +
            # penalty) is at most ~5Hz. Pin that gap so the test is known to be
            # measuring the thing it claims to measure.
            old_style_ceiling = 1.0 / (1.0 / STATE_HZ + penalty)
            assert achieved > old_style_ceiling * 1.4, (
                f"achieved {achieved:.1f}Hz is within noise of what Event.wait pacing would give "
                f"({old_style_ceiling:.1f}Hz) - the conversion may not be in effect"
            )

    def test_the_loop_stops_promptly_rather_than_at_the_end_of_a_tick(self) -> None:
        """A stop must not wait out the period, whatever the pacing.

        _run_state_loop_for joins with a 5s timeout, so a loop that only checked
        its stop event once per period would still pass that; this asserts the
        stop is fast in absolute terms with a slow nominal rate in play.
        """
        mesh = Mesh(_StatefulRobot(), peer_id="pace-2", peer_type="sim")
        mesh.publish = lambda *_a, **_kw: None  # type: ignore[method-assign]
        mesh._running = True
        with patch("strands_robots.mesh.core.put"), patch.object(mesh, "_read_state", return_value=None):
            thread = threading.Thread(target=mesh._state_loop, daemon=True)
            thread.start()
            time.sleep(0.05)  # let it enter the wait
            start = time.perf_counter()
            mesh._running = False
            mesh._stop_event.set()
            thread.join(timeout=3.0)
            took = time.perf_counter() - start
        assert not thread.is_alive()
        budget = 1.0 / STATE_HZ + sleep_penalty_s() + 0.3
        assert took < budget, f"stop took {took:.3f}s (budget {budget:.3f}s)"

    def test_a_slow_state_read_is_subtracted_from_the_period_not_added(self) -> None:
        """20ms of bus time inside a 100ms period must still tick at ~10Hz."""
        mesh = Mesh(_StatefulRobot(), peer_id="pace-3", peer_type="sim")
        published: list[float] = []

        def slow_read() -> dict[str, float]:
            deadline = time.perf_counter() + 0.02
            while time.perf_counter() < deadline:
                pass
            return {"t": time.perf_counter()}

        mesh.publish = lambda *_a, **_kw: published.append(time.perf_counter())  # type: ignore[method-assign]
        mesh._running = True
        with patch("strands_robots.mesh.core.put"), patch.object(mesh, "_read_state", side_effect=slow_read):
            thread = threading.Thread(target=mesh._state_loop, daemon=True)
            thread.start()
            start = time.perf_counter()
            while time.perf_counter() - start < 1.0:
                pass
            mesh._running = False
            mesh._stop_event.set()
            thread.join(timeout=5.0)
        gaps = [b - a for a, b in zip(published, published[1:], strict=False)]
        assert gaps, "no state publishes to measure"
        median_gap = sorted(gaps)[len(gaps) // 2]
        nominal = 1.0 / STATE_HZ
        assert median_gap < nominal * 1.5, (
            f"median gap {median_gap * 1000:.0f}ms against a {nominal * 1000:.0f}ms period - "
            "the 20ms read is being added to the period instead of subtracted from it"
        )


@pytest.mark.parametrize("attr", ["_state_loop"])
def test_the_converted_loop_no_longer_paces_on_the_stop_event(attr: str) -> None:
    """Pin the conversion in source, so a later edit cannot quietly revert it.

    The rate tests above are the real proof, but they are timing tests: on a
    heavily loaded machine their floors could in principle be met by luck. This
    one cannot be satisfied by luck.
    """
    import inspect

    func = getattr(Mesh, attr)
    source = inspect.getsource(func)
    # Scan the CODE, not the prose: the docstring of the converted loop explains
    # what it stopped doing and therefore contains the very string this test
    # bans. My first version failed on its own explanation - a source-scanning
    # test that reads comments is a test that punishes documentation.
    doc = func.__doc__
    if doc:
        source = source.replace(doc, "")
    assert "_stop_event.wait(" not in source, (
        f"Mesh.{attr} is pacing on _stop_event.wait again - that wait is inflated ~145ms in a "
        "daemon-descended process tree (BUGS.md Q69); use mesh.pacing.Ticker"
    )
    assert "Ticker(" in source, f"Mesh.{attr} should pace on a Ticker"
