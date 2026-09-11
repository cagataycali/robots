"""A Reachy link bring-up that will not be adopted is released, not abandoned.

:meth:`~strands_robots.drivers.reachy.ReachyDriver._start_link` submits the
link's ``start`` coroutine to a background loop and waits
:data:`~strands_robots.drivers.reachy._LINK_START_TIMEOUT_S` for it. Both ways
that wait can end badly - the handshake raises, or it outruns the budget - leave
the link unadopted: ``_link`` stays ``None``, so :meth:`ReachyDriver.cleanup`
has nothing to stop and no later verb can reach the link again. A ``start`` that
got far enough to open something therefore has to be stopped here or never:
:meth:`WebSocketLink.start` assigns its connected socket before spawning the read
task, and the Zenoh link subscribes to its first topic before its second, so
"far enough" is the ordinary case rather than a narrow race.

The link doubles are faithful in the one respect these tests turn on: each
records whether it was subscribed, whether it was stopped, and whether its
handshake was cancelled, so "the driver released the link" is distinguishable
from "the driver returned a reason and walked away".

Every test runs the driver's real loop and thread, with no Reachy attached.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import strands_robots.drivers.reachy as reachy_mod
from strands_robots.drivers.reachy import ReachyDriver

# A Lite: the variant that needs no Zenoh transport, so the cheapest to bring up.
_LITE_STATUS: dict[str, Any] = {"wireless_version": False, "motors": "on"}

# Short enough to keep the give-up path quick, long enough that a loaded machine
# does not trip it on the healthy control.
_BUDGET = 0.5

# The module's shipped budget, read at import so no test's patch can reach it. The
# reference page quotes this one.
_DEFAULT_BUDGET: float = reachy_mod._LINK_START_TIMEOUT_S


class _Link:
    """A link that records what the driver did to it.

    Attributes:
        subscribed: Whether ``start`` got as far as putting the link on the wire.
        stopped: Whether the driver asked the link to stop.
        cancelled: Whether the handshake was cancelled rather than abandoned.
        build_calls: How many times the driver built this link.
    """

    def __init__(self) -> None:
        self.subscribed = False
        self.stopped = False
        self.cancelled = False
        self.build_calls = 0

    async def start(self, on_joints: Any, on_imu: Any) -> None:
        """Come up cleanly."""
        del on_joints, on_imu
        self.subscribed = True

    async def stop(self) -> None:
        """Record the stop the driver asked for."""
        self.stopped = True

    async def send_cmd(self, command: dict[str, Any]) -> None:
        """Accept a command; the wire format is not this module's subject."""


class _RaisingLink(_Link):
    """Subscribes, then fails the handshake - a daemon that rejects the SDK."""

    async def start(self, on_joints: Any, on_imu: Any) -> None:
        """Subscribe, then raise."""
        del on_joints, on_imu
        self.subscribed = True
        raise RuntimeError("handshake rejected")


class _SlowLink(_Link):
    """Subscribes, then takes far longer than any budget to hand back."""

    async def start(self, on_joints: Any, on_imu: Any) -> None:
        """Subscribe, then wait past the driver's budget."""
        del on_joints, on_imu
        self.subscribed = True
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _UnstoppableLink(_SlowLink):
    """A link whose own ``stop`` fails, as a dropped socket's close can."""

    async def stop(self) -> None:
        """Fail the stop."""
        raise OSError("socket already gone")


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

    def _build(self: ReachyDriver, *, is_lite: bool) -> Any:
        del is_lite
        link.build_calls += 1
        return link

    monkeypatch.setattr(ReachyDriver, "_build_link", _build)
    return ReachyDriver(tool_name="reachy_mini", port="reachy-a.local:8000")


class TestAHandshakeThatOutrunsTheBudgetIsReleased:
    """The give-up path: the budget is named, and the link is not left running."""

    def test_the_reason_names_the_budget_that_expired(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _install(monkeypatch, _SlowLink())
        reason = driver.connect_eagerly()
        assert reason is not None
        assert "did not finish its handshake" in reason
        assert "reachy-a.local:8000" in reason
        assert f"{_BUDGET:g}s" in reason

    def test_the_reason_is_not_an_empty_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ``str(TimeoutError())`` is the empty string, so reporting the timeout
        # as a cause produced a reason that trailed off after the colon.
        driver = _install(monkeypatch, _SlowLink())
        reason = driver.connect_eagerly()
        assert reason is not None
        assert not reason.rstrip().endswith(":")
        assert "failed to start: " not in reason

    def test_the_link_it_gave_up_on_is_stopped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        link = _SlowLink()
        driver = _install(monkeypatch, link)
        assert driver.connect_eagerly() is not None
        assert link.subscribed, "the double must reach the wire, or it pins nothing"
        assert link.stopped

    def test_the_abandoned_handshake_is_cancelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        link = _SlowLink()
        driver = _install(monkeypatch, link)
        assert driver.connect_eagerly() is not None
        assert link.cancelled

    def test_the_budget_is_read_off_the_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _install(monkeypatch, _SlowLink(), budget=0.25)
        reason = driver.connect_eagerly()
        assert reason is not None and "0.25s" in reason


class TestAHandshakeThatRaisesIsReleasedToo:
    """The other unadopted path releases the same way, and still names its cause."""

    def test_the_link_that_raised_is_stopped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        link = _RaisingLink()
        driver = _install(monkeypatch, link)
        assert driver.connect_eagerly() is not None
        assert link.subscribed and link.stopped

    def test_the_reason_still_names_the_raised_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _install(monkeypatch, _RaisingLink())
        reason = driver.connect_eagerly()
        assert reason is not None
        assert "failed to start" in reason and "handshake rejected" in reason


class TestReleasingAFailedBringUpNeverRaises:
    """A teardown that fails must not turn a named connect failure into a raise."""

    def test_a_stop_that_fails_still_yields_the_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _install(monkeypatch, _UnstoppableLink())
        reason = driver.connect_eagerly()
        assert reason is not None and "did not finish its handshake" in reason
        assert driver._connected is False


class TestAFailedBringUpLeavesTheDriverUsable:
    """The documented posture: disconnected, with the reason readable, reusable."""

    @pytest.mark.parametrize("link_cls", [_SlowLink, _RaisingLink])
    def test_the_driver_is_disconnected_and_remembers_why(
        self, monkeypatch: pytest.MonkeyPatch, link_cls: type[_Link]
    ) -> None:
        driver = _install(monkeypatch, link_cls())
        reason = driver.connect_eagerly()
        assert driver._connected is False
        assert driver._link is None
        assert driver._connect_error == reason

    def test_a_later_connect_adopts_exactly_one_fresh_link(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The stranded-reader hazard in reverse: because the failed link was
        # released, the link this connect adopts is the only one on the wire.
        failed = _SlowLink()
        driver = _install(monkeypatch, failed)
        assert driver.connect_eagerly() is not None

        healthy = _Link()
        monkeypatch.setattr(ReachyDriver, "_build_link", lambda self, *, is_lite: healthy)
        assert driver.connect_eagerly() is None
        assert driver._link is healthy
        assert healthy.subscribed and not healthy.stopped
        assert failed.stopped


class TestTheHealthyBringUpIsUnchanged:
    """Controls: releasing a failure must not touch the path that succeeds."""

    def test_a_link_that_comes_up_is_adopted_and_left_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        link = _Link()
        driver = _install(monkeypatch, link)
        assert driver.connect_eagerly() is None
        assert driver._connected is True
        assert driver._link is link
        assert link.subscribed and not link.stopped

    def test_a_link_that_comes_up_publishes_its_loop_and_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _install(monkeypatch, _Link())
        assert driver.connect_eagerly() is None
        assert driver._loop is not None
        assert driver._loop_thread is not None and driver._loop_thread.is_alive()
        driver.cleanup()


class TestTheDocumentedReasonIsTheRealOne:
    """The reference page quotes this reason, so the quote is part of the contract.

    ``docs/getting-started/robot-factory.md`` shows the give-up reason as the output
    of ``Robot("reachy_mini", mode="real").connect_eagerly()``. A quoted output rots
    the moment the surface it quotes changes, so it is derived from the driver here
    rather than repeated - the same relation the page's transport-import quote is
    held to.
    """

    @staticmethod
    def _quoted_reason() -> str:
        """The give-up reason the reference page quotes, unwrapped to one line."""
        page = Path(reachy_mod.__file__).parents[2] / "docs" / "getting-started" / "robot-factory.md"
        text = page.read_text(encoding="utf-8")
        blocks = [c for c in text.split("```") if "did not finish its handshake" in c]
        assert len(blocks) == 1, f"expected exactly one quoted handshake reason, found {len(blocks)}"
        after = blocks[0].split("connect_eagerly()", 1)[1]
        return " ".join(after.replace('"', " ").split())

    def test_the_page_quotes_the_reason_the_driver_returns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Word for word, with only the budget swapped for the shipped default."""
        driver = _install(monkeypatch, _SlowLink())
        produced = driver.connect_eagerly()
        assert produced is not None
        expected = produced.replace(f"{_BUDGET:g}s", f"{_DEFAULT_BUDGET:g}s")
        assert self._quoted_reason() == expected

    def test_the_page_quotes_the_budget_the_module_ships(self) -> None:
        """A page showing a budget nobody runs with would misdirect an operator."""
        assert f"within {_DEFAULT_BUDGET:g}s" in self._quoted_reason()
