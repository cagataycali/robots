"""A DDS endpoint whose construction failed part way is closed, not dropped.

:meth:`~strands_robots.tools.g1._dds_engine.DDSSubscriberSet.subscribe` and
:meth:`~strands_robots.tools.g1._dds_engine.DDSPublisher.get_publisher` each
build an SDK endpoint in two steps - construct, then ``Init`` - and report a
named reason if either step raises. Neither step is atomic, and the endpoint
that failed holds real DDS state: ``ChannelSubscriber.__init__`` creates a live
``Channel`` before ``Init`` runs, and ``unitree_sdk2py`` starts the
``ch_reader`` daemon thread *before* it constructs the ``DataReader`` that can
raise.

Returning the reason and dropping the endpoint therefore leaks exactly what
``close`` exists to release, and it leaks it where ``close`` can never reach:
the failing call never appended to ``_subs`` or wrote to ``_pubs``, so the
driver's teardown walks a collection the half-built endpoint is not in. The
engine's own ``close`` docstrings spell out why a dropped reference is not a
release - the reader thread keeps the channel reachable, so no finaliser runs.

The behavioural cells drive recording stand-ins, so they grade the production
path on a box with no SDK. The facts a stand-in cannot establish - that a
partial ``Init`` really leaves a live thread, and that ``Close()`` really
releases it - are driven against the installed SDK and skip where it is
absent. They need no DDS bus: the reader is constructed directly and the
``DataReader`` it would build is the injected fault, so nothing here
initialises the process-wide channel factory.
"""

from __future__ import annotations

import ast
import inspect
import logging
import sys
import threading
import types
from collections.abc import Iterator
from typing import Any

import pytest

from strands_robots.tools.g1 import _dds_engine, reset_dds_state
from strands_robots.tools.g1._dds_engine import DDSPublisher, DDSSubscriberSet

_ENGINE_LOGGER = "strands_robots.tools.g1._dds_engine"

# The queue length ``subscribe`` asks the SDK for. Stated here rather than read
# off the engine so a change to either side has to be a deliberate one.
_QUEUE_LEN = 10


# =========================================================================
# Stand-ins - endpoints that fail the way the SDK's can.                   #
# =========================================================================


class _Endpoint:
    """Stands in for a ``ChannelSubscriber``/``ChannelPublisher``.

    Counts ``Close()`` calls, because the question this file asks is whether
    the engine closed an endpoint it is about to stop referring to.
    """

    def __init__(self, topic: str, message_class: type) -> None:
        self.topic = topic
        self.message_class = message_class
        self.queue_len: int | None = None
        self.closes = 0
        self.init_should_raise: Exception | None = None
        self.close_should_raise: Exception | None = None

    def Init(self, handler: Any = None, queue_len: int | None = None) -> None:  # noqa: N802 - SDK spelling
        self.queue_len = queue_len
        if self.init_should_raise is not None:
            raise self.init_should_raise

    def Close(self) -> None:  # noqa: N802 - SDK spelling
        self.closes += 1
        if self.close_should_raise is not None:
            raise self.close_should_raise


class _Factory:
    """Builds :class:`_Endpoint` objects and records every one it built.

    ``init_raises`` makes the *second* construction step fail, which is the
    partial init this file is about. ``construct_raises`` makes the *first*
    step fail, where there is no endpoint to release at all.
    """

    def __init__(
        self,
        init_raises: Exception | None = None,
        close_raises: Exception | None = None,
        construct_raises: Exception | None = None,
    ) -> None:
        self.built: list[_Endpoint] = []
        self.init_raises = init_raises
        self.close_raises = close_raises
        self.construct_raises = construct_raises

    def __call__(self, topic: str, message_class: type) -> _Endpoint:
        if self.construct_raises is not None:
            raise self.construct_raises
        endpoint = _Endpoint(topic, message_class)
        endpoint.init_should_raise = self.init_raises
        endpoint.close_should_raise = self.close_raises
        self.built.append(endpoint)
        return endpoint


def _install(monkeypatch: pytest.MonkeyPatch, attribute: str, factory: _Factory) -> None:
    """Make the engine's lazy ``from ... import <attribute>`` resolve to ``factory``.

    ``monkeypatch.setitem`` restores whatever was in ``sys.modules``, so a box
    that really has the SDK gets it back.
    """
    channel = types.ModuleType("unitree_sdk2py.core.channel")
    setattr(channel, attribute, factory)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py", types.ModuleType("unitree_sdk2py"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core", types.ModuleType("unitree_sdk2py.core"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", channel)


@pytest.fixture
def dds_ready(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Let ``start()`` succeed without touching a real interface.

    The engine's ``start`` calls ``ensure_dds``, which initialises the
    process-wide channel factory. Short-circuiting it means these cells drive
    the real ``start`` success path rather than assigning ``_started``.
    """
    monkeypatch.setattr(f"{_ENGINE_LOGGER}.ensure_dds", lambda interface: None)
    yield
    reset_dds_state()


def _started_subscriber(monkeypatch: pytest.MonkeyPatch, factory: _Factory) -> DDSSubscriberSet:
    """A subscriber set that really ran ``start()``, with ``factory`` on the bus."""
    _install(monkeypatch, "ChannelSubscriber", factory)
    subs = DDSSubscriberSet("eth0")
    assert subs.start() is None
    return subs


def _started_publisher(monkeypatch: pytest.MonkeyPatch, factory: _Factory) -> DDSPublisher:
    """A publisher cache that really ran ``start()``, with ``factory`` on the bus."""
    _install(monkeypatch, "ChannelPublisher", factory)
    pubs = DDSPublisher("eth0")
    assert pubs.start() is None
    return pubs


# =========================================================================
# Premise - the SDK really leaves live state behind a partial init.        #
# =========================================================================


class TestTheSdkLeavesRealStateBehindAPartialInit:
    """What a recording stand-in cannot establish, asked of the installed SDK.

    A stand-in models ``Init`` raising, but it cannot show that the real
    ``Init`` has already started a thread by the time it raises - and that is
    the whole reason a dropped endpoint is a leak rather than an untidy
    reference. These cells drive the SDK's own reader directly with a raising
    ``DataReader``, so no DDS bus and no channel factory are involved.
    """

    @staticmethod
    def _channel() -> Any:
        return pytest.importorskip(
            "unitree_sdk2py.core.channel",
            reason="needs unitree_sdk2py (office bring-up, not a headless runner)",
        )

    @staticmethod
    def _reader_threads() -> list[threading.Thread]:
        return [thread for thread in threading.enumerate() if thread.name == "ch_reader"]

    def _partially_init(self, monkeypatch: pytest.MonkeyPatch, queue_len: int) -> Any:
        """Run the SDK reader's ``Init`` with a ``DataReader`` that raises."""
        channel = self._channel()

        class _Boom:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("DDS resource limit reached")

        monkeypatch.setattr(channel, "DataReader", _Boom)
        reader = channel.Channel._Channel__Reader()
        with pytest.raises(RuntimeError, match="DDS resource limit reached"):
            reader.Init(object(), object(), None, lambda _msg: None, queue_len)
        return reader

    def test_a_partial_init_leaves_the_reader_thread_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The thread starts before the ``DataReader`` that can raise."""
        before = len(self._reader_threads())
        reader = self._partially_init(monkeypatch, _QUEUE_LEN)
        try:
            assert len(self._reader_threads()) == before + 1
        finally:
            reader.Close()

    def test_closing_a_half_built_reader_stops_the_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``Close()`` is what releases it - and it works on a partial init."""
        before = len(self._reader_threads())
        reader = self._partially_init(monkeypatch, _QUEUE_LEN)
        reader.Close()
        assert len(self._reader_threads()) == before

    def test_a_zero_queue_length_starts_no_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """So the queue length ``subscribe`` asks for is what makes this a leak."""
        before = len(self._reader_threads())
        reader = self._partially_init(monkeypatch, 0)
        try:
            assert len(self._reader_threads()) == before
        finally:
            reader.Close()

    def test_subscribe_asks_for_the_queue_length_that_starts_one(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """The engine's own choice, so the two premises above meet."""
        factory = _Factory()
        subs = _started_subscriber(monkeypatch, factory)
        assert subs.subscribe("rt/lowstate", dict, lambda _msg: None) is None
        assert [endpoint.queue_len for endpoint in factory.built] == [_QUEUE_LEN]

    def test_closing_an_endpoint_whose_init_never_ran_is_safe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Releasing is safe even when the first step is the one that failed."""
        channel = self._channel()
        monkeypatch.setattr(channel, "DataReader", lambda *a, **k: object())
        reader = channel.Channel._Channel__Reader()
        reader.Close()


# =========================================================================
# Premise - close() cannot reach a half-built endpoint, either way.        #
# =========================================================================


class TestCloseCannotReachAHalfBuiltEndpoint:
    """Why the release has to happen where the construction failed.

    Both cells hold before and after the release was added: a failed call
    records nothing, so the driver's teardown walks a collection the half-built
    endpoint is not in. That is the argument for closing at the failure site
    rather than leaving it to ``close``.
    """

    def test_a_failed_subscribe_records_no_subscriber_for_close_to_find(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        factory = _Factory(init_raises=RuntimeError("DDS resource limit reached"))
        subs = _started_subscriber(monkeypatch, factory)
        assert subs.subscribe("rt/lowstate", dict, lambda _msg: None) is not None
        assert subs._subs == []
        closes_before_teardown = [endpoint.closes for endpoint in factory.built]
        subs.close()
        assert [endpoint.closes for endpoint in factory.built] == closes_before_teardown

    def test_a_failed_publisher_build_caches_nothing_for_close_to_find(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        factory = _Factory(init_raises=RuntimeError("DDS resource limit reached"))
        pubs = _started_publisher(monkeypatch, factory)
        assert pubs.get_publisher("rt/lowcmd", dict)[1] is not None
        assert pubs._pubs == {}
        closes_before_teardown = [built.closes for built in factory.built]
        pubs.close()
        assert [built.closes for built in factory.built] == closes_before_teardown


# =========================================================================
# Regression - the half-built endpoint is released.                        #
# =========================================================================


class TestAFailedSubscribeReleasesTheHalfBuiltSubscriber:
    """A subscriber whose ``Init`` raised is closed before the reason returns."""

    def test_the_half_built_subscriber_is_closed(self, monkeypatch: pytest.MonkeyPatch, dds_ready: None) -> None:
        factory = _Factory(init_raises=RuntimeError("DDS resource limit reached"))
        subs = _started_subscriber(monkeypatch, factory)
        assert subs.subscribe("rt/lowstate", dict, lambda _msg: None) is not None
        assert [endpoint.closes for endpoint in factory.built] == [1]

    def test_each_failure_releases_its_own_subscriber_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """Two failed subscribes release two subscribers, one close each."""
        factory = _Factory(init_raises=RuntimeError("DDS resource limit reached"))
        subs = _started_subscriber(monkeypatch, factory)
        for topic in ("rt/lowstate", "rt/lf/bmsstate"):
            assert subs.subscribe(topic, dict, lambda _msg: None) is not None
        assert [endpoint.closes for endpoint in factory.built] == [1, 1]


class TestAFailedPublisherBuildReleasesTheHalfBuiltPublisher:
    """The write path's half-built publisher gets the same release.

    ``DDSPublisher.close``'s own docstring says a dropped writer reaches
    CycloneDDS' finaliser only "whenever the last reference happens to go" and
    that closing "says when" - the same argument, so the same rule.
    """

    def test_the_half_built_publisher_is_closed(self, monkeypatch: pytest.MonkeyPatch, dds_ready: None) -> None:
        factory = _Factory(init_raises=RuntimeError("DDS resource limit reached"))
        pubs = _started_publisher(monkeypatch, factory)
        endpoint, reason = pubs.get_publisher("rt/lowcmd", dict)
        assert endpoint is None
        assert reason is not None
        assert [built.closes for built in factory.built] == [1]

    def test_each_failure_releases_its_own_publisher_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        factory = _Factory(init_raises=RuntimeError("DDS resource limit reached"))
        pubs = _started_publisher(monkeypatch, factory)
        for topic in ("rt/lowcmd", "rt/arm_sdk"):
            assert pubs.get_publisher(topic, dict)[1] is not None
        assert [built.closes for built in factory.built] == [1, 1]


class TestAFailingReleaseIsReported:
    """A swallowed cleanup failure is a leak nobody can see."""

    def test_the_report_names_the_reason_and_the_endpoint(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        factory = _Factory(
            init_raises=RuntimeError("DDS resource limit reached"),
            close_raises=RuntimeError("dds entity already gone"),
        )
        subs = _started_subscriber(monkeypatch, factory)
        with caplog.at_level(logging.WARNING, logger=_ENGINE_LOGGER):
            subs.subscribe("rt/lowstate", dict, lambda _msg: None)
        messages = [record.getMessage() for record in caplog.records]
        assert any("dds entity already gone" in message for message in messages), messages
        assert any("rt/lowstate" in message for message in messages), messages


# =========================================================================
# Over-reach - what the release must not change.                           #
# =========================================================================


class TestWhatTheReleaseMustNotChange:
    """Everything asserted here held before the release was added.

    These are the cells that keep the fix from becoming "close every
    endpoint", and that keep the caller's reason byte-identical.
    """

    def test_the_subscribe_reason_names_the_topic_and_the_cause(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        factory = _Factory(init_raises=RuntimeError("DDS resource limit reached"))
        subs = _started_subscriber(monkeypatch, factory)
        assert (
            subs.subscribe("rt/lowstate", dict, lambda _msg: None)
            == "failed to subscribe to 'rt/lowstate': DDS resource limit reached"
        )

    def test_the_publisher_reason_names_the_topic_and_the_cause(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        factory = _Factory(init_raises=RuntimeError("DDS resource limit reached"))
        pubs = _started_publisher(monkeypatch, factory)
        assert pubs.get_publisher("rt/lowcmd", dict)[1] == (
            "failed to build publisher for 'rt/lowcmd': DDS resource limit reached"
        )

    def test_a_release_that_also_fails_does_not_mask_the_reason(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """The construction failure is what the caller needs, not the cleanup one."""
        factory = _Factory(
            init_raises=RuntimeError("DDS resource limit reached"),
            close_raises=RuntimeError("dds entity already gone"),
        )
        subs = _started_subscriber(monkeypatch, factory)
        assert (
            subs.subscribe("rt/lowstate", dict, lambda _msg: None)
            == "failed to subscribe to 'rt/lowstate': DDS resource limit reached"
        )

    def test_a_subscriber_constructor_that_raised_leaves_nothing_to_release(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """No endpoint exists yet, so the reason is reported and nothing is closed."""
        factory = _Factory(construct_raises=RuntimeError("no such topic"))
        subs = _started_subscriber(monkeypatch, factory)
        assert subs.subscribe("rt/lowstate", dict, lambda _msg: None) == (
            "failed to subscribe to 'rt/lowstate': no such topic"
        )
        assert factory.built == []

    def test_a_constructor_failure_says_nothing_about_closing(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No endpoint was built, so there is nothing to report about releasing one."""
        factory = _Factory(construct_raises=RuntimeError("no such topic"))
        subs = _started_subscriber(monkeypatch, factory)
        with caplog.at_level(logging.WARNING, logger=_ENGINE_LOGGER):
            subs.subscribe("rt/lowstate", dict, lambda _msg: None)
        assert [record.getMessage() for record in caplog.records] == []

    def test_a_publisher_constructor_that_raised_leaves_nothing_to_release(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        factory = _Factory(construct_raises=RuntimeError("no such topic"))
        pubs = _started_publisher(monkeypatch, factory)
        endpoint, reason = pubs.get_publisher("rt/lowcmd", dict)
        assert endpoint is None
        assert reason == "failed to build publisher for 'rt/lowcmd': no such topic"
        assert factory.built == []

    def test_a_successful_subscribe_closes_nothing(self, monkeypatch: pytest.MonkeyPatch, dds_ready: None) -> None:
        factory = _Factory()
        subs = _started_subscriber(monkeypatch, factory)
        assert subs.subscribe("rt/lowstate", dict, lambda _msg: None) is None
        assert [endpoint.closes for endpoint in factory.built] == [0]

    def test_a_successful_subscriber_is_still_recorded_and_still_closed(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        factory = _Factory()
        subs = _started_subscriber(monkeypatch, factory)
        assert subs.subscribe("rt/lowstate", dict, lambda _msg: None) is None
        assert len(subs._subs) == 1
        subs.close()
        assert [endpoint.closes for endpoint in factory.built] == [1]

    def test_a_successful_publisher_build_closes_nothing(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        factory = _Factory()
        pubs = _started_publisher(monkeypatch, factory)
        endpoint, reason = pubs.get_publisher("rt/lowcmd", dict)
        assert reason is None
        assert endpoint is not None
        assert [built.closes for built in factory.built] == [0]

    def test_a_successful_publisher_is_still_cached_and_still_closed(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        factory = _Factory()
        pubs = _started_publisher(monkeypatch, factory)
        assert pubs.get_publisher("rt/lowcmd", dict)[1] is None
        assert len(pubs._pubs) == 1
        pubs.close()
        assert [built.closes for built in factory.built] == [1]

    def test_a_refusal_before_the_bus_is_reached_builds_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``start()`` never ran, so there is nothing to construct or release."""
        factory = _Factory(init_raises=RuntimeError("unreachable"))
        _install(monkeypatch, "ChannelSubscriber", factory)
        subs = DDSSubscriberSet("eth0")
        assert subs.subscribe("rt/lowstate", dict, lambda _msg: None) == "DDS not initialised - call start() first"
        assert factory.built == []


# =========================================================================
# Structural - one release, reached by every endpoint construction.        #
# =========================================================================


def _endpoint_class_names(module_source: str) -> set[str]:
    """Names the module lazily imports from the SDK's channel module.

    Derived rather than listed, so an endpoint kind added later is graded on
    arrival instead of being silently outside the rule.
    """
    return {
        alias.name
        for node in ast.walk(ast.parse(module_source))
        if isinstance(node, ast.ImportFrom) and node.module == "unitree_sdk2py.core.channel"
        for alias in node.names
    }


def _constructions_that_do_not_release(module_source: str) -> list[str]:
    """``try`` blocks building an SDK endpoint whose handlers do not release it."""
    endpoint_classes = _endpoint_class_names(module_source)
    offenders = []
    for node in ast.walk(ast.parse(module_source)):
        if not isinstance(node, ast.Try):
            continue
        builds = [
            call
            for statement in node.body
            for call in ast.walk(statement)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in endpoint_classes
        ]
        if not builds:
            continue
        releases = any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_release_partial"
            for handler in node.handlers
            for statement in handler.body
            for call in ast.walk(statement)
        )
        if not releases:
            offenders.append(ast.unparse(builds[0]))
    return offenders


class TestEveryEndpointConstructionRoutesThroughOneRelease:
    """A third endpoint kind cannot reintroduce the drop.

    The rule is derived from the SDK classes the module imports, so it covers
    a construction added later without anyone remembering to list it.
    """

    def test_no_endpoint_construction_drops_a_half_built_endpoint(self) -> None:
        source = inspect.getsource(_dds_engine)
        offenders = _constructions_that_do_not_release(source)
        assert offenders == [], f"release these through _release_partial(): {offenders}"

    def test_the_rule_has_something_to_grade(self) -> None:
        """Both endpoint kinds are found, so the check above is not vacuous."""
        source = inspect.getsource(_dds_engine)
        assert _endpoint_class_names(source) == {"ChannelSubscriber", "ChannelPublisher"}

    def test_the_release_is_named_once_for_both_paths(self) -> None:
        """One helper, so the two paths cannot drift apart."""
        source = inspect.getsource(_dds_engine)
        assert source.count("def _release_partial(") == 1
        assert source.count("_release_partial(") == 3

    @pytest.mark.parametrize(
        ("label", "handler_body", "expected_offenders"),
        [
            ("drops-it", "return 'failed'", 1),
            ("releases-it", "_release_partial(endpoint, 'x')\n            return 'failed'", 0),
        ],
    )
    def test_the_rule_separates_a_dropping_construction_from_a_releasing_one(
        self, label: str, handler_body: str, expected_offenders: int
    ) -> None:
        """Graded on constructed exemplars, since the module itself is now clean."""
        source = (
            "def build():\n"
            "    from unitree_sdk2py.core.channel import ChannelSubscriber\n"
            "    endpoint = None\n"
            "    try:\n"
            "        endpoint = ChannelSubscriber('t', dict)\n"
            "        endpoint.Init(None, 10)\n"
            "    except Exception:\n"
            f"            {handler_body}\n"
            "    return endpoint\n"
        )
        assert len(_constructions_that_do_not_release(source)) == expected_offenders, label

    def test_the_exemplars_reach_both_verdicts(self) -> None:
        """Neither exemplar row can be passing for the same reason as the other."""

        def _built(handler_body: str) -> str:
            return (
                "def build():\n"
                "    from unitree_sdk2py.core.channel import ChannelSubscriber\n"
                "    endpoint = None\n"
                "    try:\n"
                "        endpoint = ChannelSubscriber('t', dict)\n"
                "    except Exception:\n"
                f"            {handler_body}\n"
            )

        verdicts = {
            not _constructions_that_do_not_release(_built("return 'failed'")),
            not _constructions_that_do_not_release(_built("_release_partial(endpoint, 'x')")),
        }
        assert verdicts == {True, False}
