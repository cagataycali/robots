"""A teardown racing a subscribe/publish leaves no live endpoint behind.

:meth:`~strands_robots.tools.g1._dds_engine.DDSSubscriberSet.close` and
:meth:`~strands_robots.tools.g1._dds_engine.DDSPublisher.close` swap their
collection out under ``_lock`` and then release what they took.
:meth:`~strands_robots.tools.g1._dds_engine.DDSSubscriberSet.subscribe` and
:meth:`~strands_robots.tools.g1._dds_engine.DDSPublisher.get_publisher` build
their endpoint under ``_DDS_INIT_LOCK`` - a different lock, held for the whole
construction because the CycloneDDS bindings segfault on concurrent
construction.

Those two locks do not exclude each other, so a teardown can complete between
the moment a construction finishes and the moment it records what it built.
Recording into the fresh collection then puts a live endpoint somewhere
``close`` has already walked past. Nothing reaches it afterwards:
:meth:`~strands_robots.drivers.g1.G1Driver.cleanup` drops the set right after
closing it, so the second ``close`` that would collect the orphan never
happens.

What is left behind is the state the engine's own ``close`` docstring describes
at length: ``subscribe`` asks for ``queueLen=10``, and at any queue length
above zero ``unitree_sdk2py`` starts a ``ch_reader`` daemon thread whose target
is a bound method of the channel's reader, so the reader stays matched and the
decoder callback keeps filling caches for a driver that believes it is
disconnected. :func:`~strands_robots.tools.g1._dds_engine._release_partial`
already exists to uphold exactly this invariant on the path where construction
*fails*; these cells grade it on the path where construction succeeds and the
recording is what could not happen.

The interleaving is driven deterministically: the stand-in's ``Init`` blocks
after signalling, which parks the caller inside ``_DDS_INIT_LOCK`` holding a
fully built endpoint, and the teardown then runs to completion. No sleeps, no
DDS bus, and no SDK - the stand-ins are what the production path constructs.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import threading
import types
from collections.abc import Iterator
from typing import Any

import pytest

from strands_robots.tools.g1 import _dds_engine, reset_dds_state
from strands_robots.tools.g1._dds_engine import DDSPublisher, DDSSubscriberSet

_ENGINE_LOGGER = "strands_robots.tools.g1._dds_engine"

# How long a cell waits for the other thread to reach its barrier. Generous:
# it bounds a hang, it does not pace anything.
_BARRIER_TIMEOUT_S = 10.0

# The queue length ``subscribe`` asks the SDK for. Stated here rather than read
# off the engine, so a change to either side has to be a deliberate one. Any
# value above zero is what starts the ``ch_reader`` thread that makes a
# forgotten subscriber a live one.
_QUEUE_LEN = 10

# The topics these cells drive. The read path parks ``_SUB_TOPIC`` and keeps
# ``_SETTLED_TOPIC`` unparked; the write path parks ``_PUB_TOPIC``.
_SUB_TOPIC = "rt/lowstate"
_SETTLED_TOPIC = "rt/lf/bmsstate"
_PUB_TOPIC = "rt/lowcmd"


# =========================================================================
# Stand-ins - an endpoint that can be parked mid-construction.             #
# =========================================================================


class _Endpoint:
    """Stands in for a ``ChannelSubscriber``/``ChannelPublisher``.

    Counts ``Close()`` calls, because the question here is whether the engine
    released an endpoint it is about to stop referring to. ``Init`` optionally
    parks, which is what makes the interleaving deterministic rather than timed.
    """

    def __init__(self, topic: str, message_class: type) -> None:
        self.topic = topic
        self.message_class = message_class
        self.queue_len: int | None = None
        self.closes = 0
        self.entered_init = threading.Event()
        self.may_finish_init: threading.Event | None = None

    def Init(self, handler: Any = None, queue_len: int | None = None) -> None:  # noqa: N802 - SDK spelling
        self.queue_len = queue_len
        self.entered_init.set()
        if self.may_finish_init is not None:
            assert self.may_finish_init.wait(_BARRIER_TIMEOUT_S), "never released"

    def Close(self) -> None:  # noqa: N802 - SDK spelling
        self.closes += 1


class _Factory:
    """Builds :class:`_Endpoint` objects and records every one it built.

    ``park`` makes the endpoint for ``park_topic`` block inside ``Init``, so
    its caller sits inside ``_DDS_INIT_LOCK`` holding something fully
    constructed. Keyed on the topic rather than on build order: ``Init`` runs
    while ``_DDS_INIT_LOCK`` is held, so a cell that needs a *settled*
    endpoint as well must be able to build that one without parking - a park
    that caught it too would block the whole construction lane.
    """

    def __init__(self, park: threading.Event | None = None, park_topic: str | None = None) -> None:
        self.built: list[_Endpoint] = []
        self.park = park
        self.park_topic = park_topic

    def __call__(self, topic: str, message_class: type) -> _Endpoint:
        endpoint = _Endpoint(topic, message_class)
        if self.park is not None and topic == self.park_topic:
            endpoint.may_finish_init = self.park
        self.built.append(endpoint)
        return endpoint


def _parking(topic: str) -> tuple[threading.Event, _Factory]:
    """A release event and a factory that parks only ``topic``'s endpoint."""
    park = threading.Event()
    return park, _Factory(park=park, park_topic=topic)


def _the(factory: _Factory, topic: str) -> _Endpoint:
    """The single endpoint ``factory`` built for ``topic``."""
    matches = [endpoint for endpoint in factory.built if endpoint.topic == topic]
    assert len(matches) == 1, f"expected one endpoint for {topic!r}, got {len(matches)}"
    return matches[0]


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
    """Let ``start()`` succeed without touching a real interface."""
    monkeypatch.setattr(f"{_ENGINE_LOGGER}.ensure_dds", lambda interface: None)
    yield
    reset_dds_state()


def _run_and_park(work: Any) -> tuple[threading.Thread, dict[str, Any]]:
    """Run ``work`` on a thread, collecting its return value."""
    out: dict[str, Any] = {}

    def _target() -> None:
        out["result"] = work()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread, out


# =========================================================================
# Regression - a teardown mid-construction releases what it cannot record. #
# =========================================================================


class TestASubscribeRacingCloseLeavesNoLiveSubscriber:
    """The subscriber built across a ``close`` is released, not recorded."""

    def test_the_subscriber_is_closed_rather_than_left_live(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """``close`` returning must mean every built subscriber is released."""
        park, factory = _parking(_SUB_TOPIC)
        _install(monkeypatch, "ChannelSubscriber", factory)
        subs = DDSSubscriberSet("eth0")
        assert subs.start() is None

        thread, _ = _run_and_park(lambda: subs.subscribe(_SUB_TOPIC, dict, lambda _m: None))
        assert factory.built, "the factory was never asked for a subscriber"
        endpoint = factory.built[0]
        assert endpoint.entered_init.wait(_BARRIER_TIMEOUT_S), "Init never ran"

        subs.close()
        park.set()
        thread.join(_BARRIER_TIMEOUT_S)
        assert not thread.is_alive()

        assert endpoint.closes == 1

    def test_the_subscriber_is_not_left_in_a_list_close_has_walked_past(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """Recording it into the fresh list is what nothing would ever reach."""
        park, factory = _parking(_SUB_TOPIC)
        _install(monkeypatch, "ChannelSubscriber", factory)
        subs = DDSSubscriberSet("eth0")
        assert subs.start() is None

        thread, _ = _run_and_park(lambda: subs.subscribe(_SUB_TOPIC, dict, lambda _m: None))
        assert _the(factory, _SUB_TOPIC).entered_init.wait(_BARRIER_TIMEOUT_S)
        subs.close()
        park.set()
        thread.join(_BARRIER_TIMEOUT_S)

        assert subs._subs == []

    def test_the_caller_is_told_the_subscription_did_not_take(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """A released subscriber is not a subscription, so it reports a reason."""
        park, factory = _parking(_SUB_TOPIC)
        _install(monkeypatch, "ChannelSubscriber", factory)
        subs = DDSSubscriberSet("eth0")
        assert subs.start() is None

        thread, out = _run_and_park(lambda: subs.subscribe(_SUB_TOPIC, dict, lambda _m: None))
        assert _the(factory, _SUB_TOPIC).entered_init.wait(_BARRIER_TIMEOUT_S)
        subs.close()
        park.set()
        thread.join(_BARRIER_TIMEOUT_S)

        reason = out["result"]
        assert reason is not None

    def test_the_reason_names_the_topic_and_the_teardown(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """An operator reading it should not have to guess which call lost."""
        park, factory = _parking(_SUB_TOPIC)
        _install(monkeypatch, "ChannelSubscriber", factory)
        subs = DDSSubscriberSet("eth0")
        assert subs.start() is None

        thread, out = _run_and_park(lambda: subs.subscribe(_SUB_TOPIC, dict, lambda _m: None))
        assert _the(factory, _SUB_TOPIC).entered_init.wait(_BARRIER_TIMEOUT_S)
        subs.close()
        park.set()
        thread.join(_BARRIER_TIMEOUT_S)

        reason = out["result"]
        assert _SUB_TOPIC in reason
        assert "closed" in reason

    def test_the_subscriber_that_was_already_recorded_is_still_closed(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """Releasing the racing one must not cost the ones already recorded."""
        park, factory = _parking(_SUB_TOPIC)
        _install(monkeypatch, "ChannelSubscriber", factory)
        subs = DDSSubscriberSet("eth0")
        assert subs.start() is None
        # Settled first, and unparked: the parked thread holds the construction
        # lane, so a second subscribe taken while it is parked would block on
        # that lock rather than on anything this cell is about.
        assert subs.subscribe(_SETTLED_TOPIC, dict, lambda _m: None) is None
        settled = _the(factory, _SETTLED_TOPIC)

        thread, _ = _run_and_park(lambda: subs.subscribe(_SUB_TOPIC, dict, lambda _m: None))
        racing = _the(factory, _SUB_TOPIC)
        assert racing.entered_init.wait(_BARRIER_TIMEOUT_S)

        subs.close()
        park.set()
        thread.join(_BARRIER_TIMEOUT_S)

        assert (racing.closes, settled.closes) == (1, 1)


class TestAGetPublisherRacingCloseLeavesNoLivePublisher:
    """The same rule on the write path, which advertises parity with the read one."""

    def test_the_publisher_is_closed_rather_than_left_live(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """``close`` returning must mean every built publisher is released."""
        park, factory = _parking(_PUB_TOPIC)
        _install(monkeypatch, "ChannelPublisher", factory)
        pubs = DDSPublisher("eth0")
        assert pubs.start() is None

        thread, _ = _run_and_park(lambda: pubs.get_publisher(_PUB_TOPIC, dict))
        assert _the(factory, _PUB_TOPIC).entered_init.wait(_BARRIER_TIMEOUT_S)
        pubs.close()
        park.set()
        thread.join(_BARRIER_TIMEOUT_S)

        assert _the(factory, _PUB_TOPIC).closes == 1

    def test_the_publisher_is_not_left_in_a_cache_close_has_emptied(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """Caching it after the swap is what nothing would ever reach."""
        park, factory = _parking(_PUB_TOPIC)
        _install(monkeypatch, "ChannelPublisher", factory)
        pubs = DDSPublisher("eth0")
        assert pubs.start() is None

        thread, _ = _run_and_park(lambda: pubs.get_publisher(_PUB_TOPIC, dict))
        assert _the(factory, _PUB_TOPIC).entered_init.wait(_BARRIER_TIMEOUT_S)
        pubs.close()
        park.set()
        thread.join(_BARRIER_TIMEOUT_S)

        assert pubs._pubs == {}

    def test_the_caller_gets_no_publisher_and_a_reason(self, monkeypatch: pytest.MonkeyPatch, dds_ready: None) -> None:
        """Handing back a released publisher would be worse than refusing."""
        park, factory = _parking(_PUB_TOPIC)
        _install(monkeypatch, "ChannelPublisher", factory)
        pubs = DDSPublisher("eth0")
        assert pubs.start() is None

        thread, out = _run_and_park(lambda: pubs.get_publisher(_PUB_TOPIC, dict))
        assert _the(factory, _PUB_TOPIC).entered_init.wait(_BARRIER_TIMEOUT_S)
        pubs.close()
        park.set()
        thread.join(_BARRIER_TIMEOUT_S)

        publisher, reason = out["result"]
        assert publisher is None
        assert reason is not None and _PUB_TOPIC in reason


# =========================================================================
# Controls - what the ordinary, unraced paths already did.                 #
# =========================================================================


class TestTheUnracedPathsAreUnchanged:
    """Every expectation here is one the pre-fix engine also met."""

    def test_a_settled_subscription_is_recorded_and_reports_success(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """Nothing tore down, so the subscriber belongs in the list."""
        factory = _Factory()
        _install(monkeypatch, "ChannelSubscriber", factory)
        subs = DDSSubscriberSet("eth0")
        assert subs.start() is None

        assert subs.subscribe(_SUB_TOPIC, dict, lambda _m: None) is None
        assert subs._subs == [_the(factory, _SUB_TOPIC)]

    def test_a_settled_subscription_is_closed_by_close(self, monkeypatch: pytest.MonkeyPatch, dds_ready: None) -> None:
        """The release path this file is about must not cost the normal one."""
        factory = _Factory()
        _install(monkeypatch, "ChannelSubscriber", factory)
        subs = DDSSubscriberSet("eth0")
        assert subs.start() is None
        assert subs.subscribe(_SUB_TOPIC, dict, lambda _m: None) is None

        subs.close()
        assert _the(factory, _SUB_TOPIC).closes == 1

    def test_subscribing_after_a_close_still_succeeds(self, monkeypatch: pytest.MonkeyPatch, dds_ready: None) -> None:
        """A sequential re-subscribe reads the post-teardown collection.

        The generation it captures is the one ``close`` left behind, so it is
        current when it records. A driver going through disconnect/reconnect
        must not be refused for a teardown that already finished.
        """
        factory = _Factory()
        _install(monkeypatch, "ChannelSubscriber", factory)
        subs = DDSSubscriberSet("eth0")
        assert subs.start() is None
        subs.close()

        assert subs.subscribe(_SUB_TOPIC, dict, lambda _m: None) is None
        assert subs._subs == [_the(factory, _SUB_TOPIC)]

    def test_getting_a_publisher_after_a_close_still_reconstructs(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """The write path's own disconnect/reconnect contract, unchanged."""
        factory = _Factory()
        _install(monkeypatch, "ChannelPublisher", factory)
        pubs = DDSPublisher("eth0")
        assert pubs.start() is None
        first, _ = pubs.get_publisher(_PUB_TOPIC, dict)
        pubs.close()

        second, reason = pubs.get_publisher(_PUB_TOPIC, dict)
        assert reason is None
        assert second is not None and second is not first

    def test_close_is_still_idempotent(self, monkeypatch: pytest.MonkeyPatch, dds_ready: None) -> None:
        """A second teardown finds nothing, and closes nothing twice."""
        factory = _Factory()
        _install(monkeypatch, "ChannelSubscriber", factory)
        subs = DDSSubscriberSet("eth0")
        assert subs.start() is None
        assert subs.subscribe(_SUB_TOPIC, dict, lambda _m: None) is None

        subs.close()
        subs.close()
        assert _the(factory, _SUB_TOPIC).closes == 1


# =========================================================================
# Premises - the facts the race depends on.                                #
# =========================================================================


class TestThePremisesTheRaceRestsOn:
    """Without these the interleaving above would not be reachable at all."""

    def test_the_construction_lane_and_the_collection_lock_are_different_locks(self, dds_ready: None) -> None:
        """One lock would exclude the other; two do not."""
        subs = DDSSubscriberSet("eth0")
        pubs = DDSPublisher("eth0")
        assert subs._lock is not _dds_engine._DDS_INIT_LOCK
        assert pubs._lock is not _dds_engine._DDS_INIT_LOCK

    def test_close_rebinds_the_collection_rather_than_emptying_it(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """Rebinding is what makes the collection's identity a generation.

        Clearing in place would leave the identity unchanged, and a racing
        recording would have no way to tell that a teardown had run.
        """
        factory = _Factory()
        _install(monkeypatch, "ChannelSubscriber", factory)
        subs = DDSSubscriberSet("eth0")
        assert subs.start() is None
        before = subs._subs

        subs.close()
        assert subs._subs is not before

    def test_a_subscriber_is_asked_for_a_queue_that_starts_a_reader_thread(
        self, monkeypatch: pytest.MonkeyPatch, dds_ready: None
    ) -> None:
        """A forgotten subscriber is a *live* one, which is why this matters.

        ``unitree_sdk2py`` starts the ``ch_reader`` daemon thread at any queue
        length above zero, and that thread keeps the channel reachable - so a
        dropped reference is not a release.
        """
        factory = _Factory()
        _install(monkeypatch, "ChannelSubscriber", factory)
        subs = DDSSubscriberSet("eth0")
        assert subs.start() is None
        assert subs.subscribe(_SUB_TOPIC, dict, lambda _m: None) is None

        assert _the(factory, _SUB_TOPIC).queue_len == _QUEUE_LEN
        assert _QUEUE_LEN > 0


# =========================================================================
# Structural - the rule has one shape, and a third endpoint kind inherits. #
# =========================================================================


def _methods_that_release_an_endpoint() -> dict[str, Any]:
    """Every engine method that routes a built endpoint to ``_release_partial``.

    Derived rather than listed, so an endpoint kind added later is held to the
    same rule instead of inheriting an exemption by being absent from a tuple.
    """
    found: dict[str, Any] = {}
    for owner in (DDSSubscriberSet, DDSPublisher):
        for name, member in vars(owner).items():
            if not callable(member) or name.startswith("__"):
                continue
            if "_release_partial(" in inspect.getsource(member):
                found[f"{owner.__name__}.{name}"] = member
    return found


def _compares_the_collection_against_a_captured_one(source: str) -> bool:
    """True iff ``source`` tests ``self.<collection> is <captured>``.

    Takes source rather than a callable so the same predicate grades the
    shipped methods and the constructed exemplars below.
    """
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], ast.Is):
            continue
        left = node.left
        if isinstance(left, ast.Attribute) and isinstance(left.value, ast.Name) and left.value.id == "self":
            return True
    return False


class TestTheRuleHasOneShapeOnEveryEndpointKind:
    """Both recording paths compare a generation; neither re-derives the rule."""

    def test_the_rule_has_something_to_grade(self) -> None:
        """Both endpoint kinds are found, so the check below is not vacuous."""
        assert set(_methods_that_release_an_endpoint()) == {
            "DDSSubscriberSet.subscribe",
            "DDSPublisher.get_publisher",
        }

    def test_every_recording_path_compares_the_collection_it_captured(self) -> None:
        """A path that records without comparing can still orphan an endpoint."""
        offenders = [
            name
            for name, method in _methods_that_release_an_endpoint().items()
            if not _compares_the_collection_against_a_captured_one(inspect.getsource(method))
        ]
        assert offenders == [], f"these can record into a collection close() has passed: {offenders}"

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            pytest.param("if self._subs is target:\n        self._subs.append(x)\n", True, id="compares-it"),
            pytest.param("self._subs.append(x)\n", False, id="records-blind"),
        ],
    )
    def test_the_rule_separates_a_comparing_record_from_a_blind_one(self, body: str, expected: bool) -> None:
        """Graded on constructed exemplars, since the module itself is now clean."""
        source = "def record(self, target, x):\n    " + body
        assert _compares_the_collection_against_a_captured_one(source) is expected

    def test_the_exemplars_reach_both_verdicts(self) -> None:
        """Neither exemplar row can be passing for the same reason as the other."""
        comparing = "def record(self, target, x):\n    if self._subs is target:\n        self._subs.append(x)\n"
        blind = "def record(self, target, x):\n    self._subs.append(x)\n"
        verdicts = {
            _compares_the_collection_against_a_captured_one(comparing),
            _compares_the_collection_against_a_captured_one(blind),
        }
        assert verdicts == {True, False}
