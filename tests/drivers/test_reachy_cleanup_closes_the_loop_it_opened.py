"""A Reachy teardown closes the loop it opened, and waits so it can.

:meth:`~strands_robots.drivers.reachy.ReachyDriver._start_link` gets its loop
from ``asyncio.new_event_loop``, whose documented counterpart is
``loop.close()``. ``loop.stop()`` is not that counterpart: it only asks
``run_forever`` to return, leaving the loop's selector and self-pipe open. So a
teardown that stopped without closing abandoned one open loop per
connect/teardown cycle - on the success path through
:meth:`ReachyDriver.cleanup` and on both give-up paths through
:meth:`ReachyDriver._release_link` - and Python raised one
``ResourceWarning: unclosed event loop`` for each, reported wherever the
collector happened to reclaim it rather than at the teardown responsible.

Closing it means waiting for the thread first, because closing a running loop
raises ``RuntimeError`` and ``stop()`` is asynchronous: it schedules the stop
and returns while the thread is still inside ``run_forever``. That wait is what
``_loop_thread`` is for, and it is bounded by
:data:`~strands_robots.drivers.reachy._LOOP_JOIN_TIMEOUT_S` so a wedged link
callback cannot hold teardown open for as long as its socket read takes.

Every test runs the driver's real loop and thread, with no Reachy attached.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
import warnings
from typing import Any

import pytest

import strands_robots.drivers.reachy as reachy_mod
from strands_robots.drivers.reachy import ReachyDriver

# A Lite: the variant that needs no Zenoh transport, so the cheapest to bring up.
_LITE_STATUS: dict[str, Any] = {"wireless_version": False, "motors": "on"}

# Short enough to keep the give-up paths quick, long enough that a loaded machine
# does not trip them on the healthy control.
_BUDGET = 0.5


class _Link:
    """A link that comes up cleanly and records the stop asked of it."""

    def __init__(self) -> None:
        self.stopped = False

    async def start(self, on_joints: Any, on_imu: Any) -> None:
        """Come up cleanly, keeping the callbacks this double never invokes."""

    async def stop(self) -> None:
        """Record the stop the driver asked for."""
        self.stopped = True

    async def send_cmd(self, command: dict[str, Any]) -> None:
        """Accept a command; the wire format is not this module's subject."""


class _RaisingLink(_Link):
    """Fails the handshake - a daemon that rejects the SDK."""

    async def start(self, on_joints: Any, on_imu: Any) -> None:
        """Raise instead of handing back."""
        del on_joints, on_imu
        raise RuntimeError("handshake rejected")


class _SlowLink(_Link):
    """Takes far longer than any budget to hand back."""

    async def start(self, on_joints: Any, on_imu: Any) -> None:
        """Wait past the driver's handshake budget."""
        del on_joints, on_imu
        await asyncio.sleep(30)


class _WedgingLink(_RaisingLink):
    """A failed bring-up whose ``stop`` leaves the loop behind a blocked callback.

    A link callback wedged on a socket read is what this stands for: the loop
    thread cannot return from ``run_forever`` until the callback does, so the
    join budget is the only thing that ends the wait.

    Wedged on the give-up path rather than on :meth:`ReachyDriver.cleanup`
    because both waits ahead of the join are then :data:`_BUDGET`, which these
    tests set. The join it is holding is the same
    :meth:`ReachyDriver._stop_loop` both paths call.
    """

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.wedged = threading.Event()

    async def stop(self) -> None:
        """Record the stop, then wedge the loop behind a blocking callback."""
        self.stopped = True
        asyncio.get_running_loop().call_soon(self._block)

    def _block(self) -> None:
        """Hold the loop thread until the test lets it go."""
        self.wedged.set()
        self.release.wait(timeout=30)


def _install(monkeypatch: pytest.MonkeyPatch, link: _Link, *, budget: float = _BUDGET) -> ReachyDriver:
    """Build a driver whose daemon probe succeeds and whose link is ``link``.

    Args:
        monkeypatch: pytest's patcher.
        link: The link double :meth:`ReachyDriver._build_link` will return.
        budget: Handshake budget to install on the module.

    Returns:
        An unconnected driver.
    """

    def _api(host: str, port: int, path: str, method: str = "GET", data: Any = None) -> dict[str, Any]:
        del host, port, method, data
        return dict(_LITE_STATUS) if path == reachy_mod._PATH_STATUS else {"ok": True}

    monkeypatch.setattr("strands_robots.device_connect.reachy_transport.api", _api)
    monkeypatch.setattr(reachy_mod, "_LINK_START_TIMEOUT_S", budget)
    monkeypatch.setattr(ReachyDriver, "_build_link", lambda self, *, is_lite: link)
    return ReachyDriver(tool_name="reachy_mini", port="reachy-a.local:8000")


def _spy_threads(monkeypatch: pytest.MonkeyPatch, out: list[threading.Thread]) -> None:
    """Record every thread the driver starts, so an unadopted one can be released.

    A failed bring-up never publishes ``_loop_thread``, so a test that has to let
    a wedged loop go has no other handle on it.

    Args:
        monkeypatch: pytest's patcher.
        out: Collected threads, in the order they were started.
    """

    class _Recording(threading.Thread):
        def start(self) -> None:
            out.append(self)
            super().start()

    monkeypatch.setattr(reachy_mod.threading, "Thread", _Recording)


class TestATornDownDriverLeavesNoOpenLoop:
    """The success path: the loop is closed, and the thread is gone first."""

    def test_cleanup_closes_the_loop_it_opened(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _install(monkeypatch, _Link())
        assert driver.connect_eagerly() is None
        loop = driver._loop
        assert loop is not None and not loop.is_closed(), "the double must really bring a loop up"
        driver.cleanup()
        assert loop.is_closed()

    def test_the_loop_thread_has_returned_when_cleanup_does(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ``stop()`` only schedules the stop, so without the wait the thread was
        # still inside ``run_forever`` here - and a running loop cannot be closed.
        driver = _install(monkeypatch, _Link())
        assert driver.connect_eagerly() is None
        thread = driver._loop_thread
        assert thread is not None and thread.is_alive()
        driver.cleanup()
        assert not thread.is_alive()

    def test_the_link_is_still_stopped_before_its_loop_is(self, monkeypatch: pytest.MonkeyPatch) -> None:
        link = _Link()
        driver = _install(monkeypatch, link)
        assert driver.connect_eagerly() is None
        driver.cleanup()
        assert link.stopped, "the link stop must not be skipped by the loop teardown"
        assert driver._connected is False

    def test_a_second_cleanup_is_still_a_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Closing an already-closed loop is the one way the new close could turn
        # an idempotent teardown into a raise.
        driver = _install(monkeypatch, _Link())
        assert driver.connect_eagerly() is None
        driver.cleanup()
        driver.cleanup()
        assert driver._loop is None and driver._loop_thread is None

    def test_python_no_longer_complains_about_an_abandoned_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The consequence Python itself reports. Nothing may outlive the cycle or
        # the collector has a live reference and cannot reclaim the loop at all.
        def cycle() -> None:
            driver = _install(monkeypatch, _Link())
            assert driver.connect_eagerly() is None
            driver.cleanup()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cycle()
            cycle()
            gc.collect()
        unclosed = [w for w in caught if "unclosed event loop" in str(w.message)]
        assert unclosed == [], f"teardown abandoned {len(unclosed)} open loop(s)"


class TestAnUnadoptedBringUpLeavesNoOpenLoopEither:
    """Both give-up paths close the loop the failed bring-up opened."""

    @pytest.mark.parametrize("link", [_RaisingLink(), _SlowLink()], ids=["handshake-raised", "budget-expired"])
    def test_the_loop_a_failed_bring_up_opened_is_closed(self, monkeypatch: pytest.MonkeyPatch, link: _Link) -> None:
        driver = _install(monkeypatch, link)
        loops: list[asyncio.AbstractEventLoop] = []
        real = ReachyDriver._stop_loop

        def _spy(self: ReachyDriver, loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
            loops.append(loop)
            real(self, loop, thread)

        monkeypatch.setattr(ReachyDriver, "_stop_loop", _spy)
        assert driver.connect_eagerly() is not None, "the double must fail, or it pins nothing"
        assert driver._loop is None, "an unadopted bring-up must not publish its loop"
        assert len(loops) == 1 and loops[0].is_closed()

    def test_repeated_failed_bring_ups_abandon_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Every ``connect_eagerly`` builds a fresh loop, so an unclosed one here
        # accumulated once per retry.
        def cycle() -> None:
            driver = _install(monkeypatch, _RaisingLink())
            for _ in range(3):
                assert driver.connect_eagerly() is not None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cycle()
            gc.collect()
        unclosed = [w for w in caught if "unclosed event loop" in str(w.message)]
        assert unclosed == [], f"three failed bring-ups abandoned {len(unclosed)} open loop(s)"


class TestAWedgedLoopDoesNotHoldTeardownOpen:
    """The budget bounds the wait, and a loop still running is not closed."""

    def test_teardown_returns_on_the_budget_rather_than_the_callback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        link = _WedgingLink()
        driver = _install(monkeypatch, link)
        monkeypatch.setattr(reachy_mod, "_LOOP_JOIN_TIMEOUT_S", 0.3)
        threads: list[threading.Thread] = []
        _spy_threads(monkeypatch, threads)
        started = time.monotonic()
        assert driver.connect_eagerly() is not None
        elapsed = time.monotonic() - started
        thread = threads[0]
        try:
            assert link.wedged.is_set(), "the callback must really hold the loop, or this pins nothing"
            assert thread.is_alive()
            # The link-stop wait plus the join, not the callback's own 30s wait.
            assert elapsed < _BUDGET + 0.3 + 2.0
        finally:
            link.release.set()
            thread.join(timeout=5)

    def test_a_running_loop_is_left_open_and_said_so(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        link = _WedgingLink()
        driver = _install(monkeypatch, link)
        monkeypatch.setattr(reachy_mod, "_LOOP_JOIN_TIMEOUT_S", 0.3)
        loops: list[asyncio.AbstractEventLoop] = []
        threads: list[threading.Thread] = []
        _spy_threads(monkeypatch, threads)
        real = ReachyDriver._stop_loop

        def _spy(self: ReachyDriver, loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
            loops.append(loop)
            real(self, loop, thread)

        monkeypatch.setattr(ReachyDriver, "_stop_loop", _spy)
        with caplog.at_level(logging.WARNING, logger=reachy_mod.__name__):
            assert driver.connect_eagerly() is not None
        try:
            # Closing a loop that is still running raises, so it keeps its loop -
            # and the teardown that could not finish is reported rather than
            # passed off as done.
            assert len(loops) == 1 and not loops[0].is_closed()
            assert "did not stop within 0.3s" in caplog.text
            assert "left open" in caplog.text
        finally:
            link.release.set()
            threads[0].join(timeout=5)
