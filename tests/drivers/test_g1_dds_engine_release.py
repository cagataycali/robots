"""G1 DDS teardown: closing an engine releases readers and writers, not just references.

``DDSSubscriberSet.close`` is the driver's only release path -
:meth:`~strands_robots.drivers.g1.G1Driver.cleanup` calls it and then drops the
set - and ``DDSPublisher.close`` is its sibling on the write path. Both used to
drop their references and rely on garbage collection, which is measurably wrong
for a subscriber: :meth:`DDSSubscriberSet.subscribe` builds every subscriber
with a non-zero queue length, and at any queue length above zero
``unitree_sdk2py`` starts a ``ch_reader`` daemon thread whose target is a bound
method of the channel's reader. The running thread keeps the channel reachable,
so CycloneDDS' ``__del__`` never runs, and ``ChannelSubscriber.Close()`` - which
both docstrings claimed did not exist - is the only thing that stops the thread
and drops the ``DataReader``.

A subscription that outlives ``cleanup()`` matters twice over: the decoder
callback keeps writing caches for a driver that reports itself disconnected,
and the next ``connect_eagerly()`` subscribes the same topics again - the
duplicate that the engine's shared construction lock exists to prevent.

The SDK is absent on a headless runner, so the behavioural cells drive
recording stand-ins that model ``Init``/``Close``. The facts a stand-in cannot
establish - that the real classes *have* ``Close`` - are pinned separately, and
skip where the SDK is missing.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

import pytest

from strands_robots.drivers.g1 import G1Driver
from strands_robots.tools.g1._dds_engine import DDSPublisher, DDSSubscriberSet

_ENGINE_LOGGER = "strands_robots.tools.g1._dds_engine"


class _RecordingEndpoint:
    """Stands in for a ``ChannelSubscriber``/``ChannelPublisher``, counting closes."""

    def __init__(self, topic: str, message_class: type) -> None:
        self.topic = topic
        self.message_class = message_class
        self.queue_len: int | None = None
        self.closes = 0

    def Init(self, handler: Any = None, queue_len: int | None = None) -> None:  # noqa: N802 - SDK spelling
        self.queue_len = queue_len

    def Close(self) -> None:  # noqa: N802 - SDK spelling
        self.closes += 1


class _RaisingEndpoint(_RecordingEndpoint):
    """An endpoint whose ``Close()`` fails the way the SDK's can."""

    def Close(self) -> None:  # noqa: N802 - SDK spelling
        self.closes += 1
        raise RuntimeError("dds entity already gone")


def _install_channel(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    endpoint_class: type[_RecordingEndpoint],
) -> list[_RecordingEndpoint]:
    """Point ``unitree_sdk2py.core.channel.<attribute>`` at a recording factory.

    ``monkeypatch.setitem`` restores whatever was in ``sys.modules`` before,
    so a machine that really has the SDK gets it back.
    """
    built: list[_RecordingEndpoint] = []

    def _factory(topic: str, message_class: type) -> _RecordingEndpoint:
        endpoint = endpoint_class(topic, message_class)
        built.append(endpoint)
        return endpoint

    channel = types.ModuleType("unitree_sdk2py.core.channel")
    setattr(channel, attribute, _factory)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", channel)
    return built


def _subscribed(
    monkeypatch: pytest.MonkeyPatch,
    topics: tuple[str, ...],
    endpoint_class: type[_RecordingEndpoint] = _RecordingEndpoint,
) -> tuple[DDSSubscriberSet, list[_RecordingEndpoint]]:
    """Return a set that really ran :meth:`DDSSubscriberSet.subscribe`.

    The endpoints graded below are the ones ``subscribe`` built, so these cells
    measure the objects the production path creates rather than a list a test
    handed over. ``start()`` is skipped because it would call
    ``ChannelFactoryInitialize`` against a real bus.
    """
    built = _install_channel(monkeypatch, "ChannelSubscriber", endpoint_class)
    subs = DDSSubscriberSet("eth0")
    subs._started = True
    for topic in topics:
        assert subs.subscribe(topic, dict, lambda _msg: None) is None
    assert len(built) == len(topics)
    return subs, built


def _published(
    monkeypatch: pytest.MonkeyPatch,
    topics: tuple[str, ...],
    endpoint_class: type[_RecordingEndpoint] = _RecordingEndpoint,
) -> tuple[DDSPublisher, list[_RecordingEndpoint]]:
    """Return a publisher cache that really ran :meth:`DDSPublisher.get_publisher`."""
    built = _install_channel(monkeypatch, "ChannelPublisher", endpoint_class)
    pubs = DDSPublisher("eth0")
    pubs._started = True
    for topic in topics:
        endpoint, err = pubs.get_publisher(topic, dict)
        assert err is None
        assert endpoint is not None
    assert len(built) == len(topics)
    return pubs, built


def _close_with_failing_endpoints(
    caplog: pytest.LogCaptureFixture,
    engine: DDSSubscriberSet | DDSPublisher,
) -> list[logging.LogRecord]:
    """Close ``engine`` while capturing what it reported."""
    with caplog.at_level(logging.WARNING, logger=_ENGINE_LOGGER):
        engine.close()
    return list(caplog.records)


# =========================================================================
# Premise - the SDK really does expose an explicit close.                  #
# =========================================================================


class TestTheSdkHasAnExplicitClose:
    """The facts a recording stand-in cannot establish.

    Both ``close`` methods used to justify dropping references by asserting
    that the SDK class "has no explicit close" and "relies on garbage
    collection". A stand-in that defines ``Close`` would happily agree with a
    fix built on the same wrong belief, so these cells ask the installed SDK
    instead and skip where it is absent.
    """

    @staticmethod
    def _channel() -> Any:
        return pytest.importorskip(
            "unitree_sdk2py.core.channel",
            reason="needs unitree_sdk2py (office bring-up, not a headless runner)",
        )

    def test_channel_subscriber_exposes_close(self) -> None:
        assert callable(getattr(self._channel().ChannelSubscriber, "Close", None))

    def test_channel_publisher_exposes_close(self) -> None:
        assert callable(getattr(self._channel().ChannelPublisher, "Close", None))

    def test_the_channel_closes_a_reader_and_a_writer_separately(self) -> None:
        """``Close`` routes to these, which is why closing releases anything."""
        channel = self._channel().Channel
        assert callable(getattr(channel, "CloseReader", None))
        assert callable(getattr(channel, "CloseWriter", None))

    def test_no_finaliser_releases_a_dropped_subscriber(self) -> None:
        """No ``__del__``, so dropping the last reference runs no release code."""
        assert getattr(self._channel().ChannelSubscriber, "__del__", None) is None


class TestSubscribeAsksForAQueuedReader:
    """The queue length is the whole reason a dropped subscriber leaks.

    ``unitree_sdk2py`` only starts the ``ch_reader`` thread when the queue
    length is above zero, and that thread is what keeps a dropped channel
    reachable. This needs no SDK: the argument is one the engine chooses.
    """

    def test_every_subscriber_is_built_with_a_non_zero_queue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _subs, built = _subscribed(monkeypatch, ("rt/lowstate", "rt/lf/bmsstate"))
        assert [endpoint.queue_len for endpoint in built] == [10, 10]


# =========================================================================
# Regression - the release actually happens.                               #
# =========================================================================


class TestClosingASubscriberSetReleasesEverySubscriber:
    """Every subscriber the set built is closed, not merely forgotten."""

    def test_close_closes_every_subscriber_it_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        subs, built = _subscribed(monkeypatch, ("rt/lowstate", "rt/lf/bmsstate"))
        subs.close()
        assert [endpoint.closes for endpoint in built] == [1, 1]

    def test_the_driver_s_cleanup_closes_the_subscribers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``cleanup()`` is the route a caller takes; it must reach ``Close()``."""
        subs, built = _subscribed(monkeypatch, ("rt/lowstate",))
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._subs = subs
        driver._connected = True

        driver.cleanup()

        assert [endpoint.closes for endpoint in built] == [1]
        assert driver._subs is None
        assert driver._connected is False

    def test_a_failing_close_does_not_strand_the_subscribers_behind_it(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One SDK failure must not leak the rest - that is the whole job."""
        subs, built = _subscribed(
            monkeypatch, ("rt/lowstate", "rt/lf/bmsstate", "rt/utlidar/lidar_state"), _RaisingEndpoint
        )
        _close_with_failing_endpoints(caplog, subs)
        assert [endpoint.closes for endpoint in built] == [1, 1, 1]

    def test_every_close_failure_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A swallowed teardown failure is a leak nobody can see."""
        subs, _built = _subscribed(
            monkeypatch, ("rt/lowstate", "rt/lf/bmsstate", "rt/utlidar/lidar_state"), _RaisingEndpoint
        )
        assert len(_close_with_failing_endpoints(caplog, subs)) == 3

    def test_the_report_names_the_reason_the_sdk_gave(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ "a subscriber failed to close" alone would not help anyone."""
        subs, _built = _subscribed(monkeypatch, ("rt/lowstate",), _RaisingEndpoint)
        _close_with_failing_endpoints(caplog, subs)
        assert "dds entity already gone" in caplog.text

    def test_a_second_close_does_not_close_anything_twice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Idempotent means the SDK is asked exactly once per subscriber."""
        subs, built = _subscribed(monkeypatch, ("rt/lowstate",))
        subs.close()
        subs.close()
        assert [endpoint.closes for endpoint in built] == [1]


class TestClosingAPublisherReleasesEveryPublisher:
    """The write path's cache gets the same teardown as the read path's list.

    A writer starts no reader thread, so a dropped publisher does eventually
    reach CycloneDDS' ``__del__``. Closing it is about *when*: the release
    happens at the call rather than whenever the last reference happens to go.
    """

    def test_close_closes_every_publisher_it_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pubs, built = _published(monkeypatch, ("rt/lowcmd", "rt/armsdk"))
        pubs.close()
        assert [endpoint.closes for endpoint in built] == [1, 1]

    def test_a_failing_close_does_not_strand_the_publishers_behind_it(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        pubs, built = _published(monkeypatch, ("rt/lowcmd", "rt/armsdk", "rt/api/sport"), _RaisingEndpoint)
        _close_with_failing_endpoints(caplog, pubs)
        assert [endpoint.closes for endpoint in built] == [1, 1, 1]

    def test_every_close_failure_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        pubs, _built = _published(monkeypatch, ("rt/lowcmd", "rt/armsdk"), _RaisingEndpoint)
        records = _close_with_failing_endpoints(caplog, pubs)
        assert len(records) == 2
        assert "publisher" in caplog.text

    def test_a_second_close_does_not_close_anything_twice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pubs, built = _published(monkeypatch, ("rt/lowcmd",))
        pubs.close()
        pubs.close()
        assert [endpoint.closes for endpoint in built] == [1]


# =========================================================================
# Boundary - what the old contract promised is still true.                 #
# =========================================================================


class TestCloseStillForgetsWhatItClosed:
    """Closing must not cost the properties the shipped ``close`` already had.

    Every expectation here is one the pre-fix code also met, so a release
    bolted on at the cost of the old contract fails here rather than passing
    quietly. Idempotence sits with the regression cells instead: "the SDK is
    asked exactly once" is not a claim the pre-fix code could satisfy, because
    it never asked at all.
    """

    def test_a_closed_subscriber_set_holds_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        subs, _built = _subscribed(monkeypatch, ("rt/lowstate", "rt/lf/bmsstate"))
        subs.close()
        assert subs._subs == []

    def test_a_closed_publisher_cache_holds_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pubs, _built = _published(monkeypatch, ("rt/lowcmd", "rt/armsdk"))
        pubs.close()
        assert pubs._pubs == {}

    def test_closing_a_subscriber_set_that_never_subscribed_is_a_no_op(self) -> None:
        DDSSubscriberSet("eth0").close()

    def test_closing_a_publisher_that_never_published_is_a_no_op(self) -> None:
        DDSPublisher("eth0").close()
