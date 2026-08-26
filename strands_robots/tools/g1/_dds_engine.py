"""Subscription and publication helpers for the G1 driver's read and write paths.

The driver holds one :class:`DDSSubscriberSet` and asks it for background
subscribers whose callbacks fill in-memory caches (the newest message wins).
The mesh publishes those caches at its own cadence, so the DDS callback stays
fast: parse, drop into a slot, return.

The driver also holds one :class:`DDSPublisher` for the ``rt/lowcmd`` write
path (and any other topic the control loop grows in issue #361). Publishers
are cached per ``(topic, message_class)`` and constructed under the shared
:data:`_DDS_INIT_LOCK`; that is the same lock the subscriber set holds, so a
reader and a writer never construct concurrently on the CycloneDDS bindings
that segfault under it.

Neither class imports ``unitree_sdk2py`` at module load: the SDK is loaded
lazily inside :meth:`DDSSubscriberSet.subscribe` and
:meth:`DDSPublisher.get_publisher`. That is what lets every test in this repo
mock the bus and skips the SDK entirely on headless machines.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from strands_robots.tools.g1._g1_common import _DDS_INIT_LOCK, ensure_dds

logger = logging.getLogger(__name__)


def _release_partial(endpoint: Any | None, what: str) -> None:
    """Close an endpoint whose construction failed, rather than dropping it.

    ``ChannelSubscriber.__init__`` and ``ChannelPublisher.__init__`` create a
    live ``Channel`` before ``Init`` runs, and ``Init`` is not atomic either:
    ``unitree_sdk2py`` starts the ``ch_reader`` daemon thread *before* it
    constructs the ``DataReader`` that can raise. A construction that fails
    part way therefore leaves an endpoint holding real DDS state, and for a
    subscriber that state includes a running thread whose target is a bound
    method of the channel's reader - the same thread
    :meth:`DDSSubscriberSet.close` documents, which keeps the channel
    reachable so no finaliser releases it.

    Such an endpoint is unreachable from :meth:`DDSSubscriberSet.close` and
    :meth:`DDSPublisher.close`, because the call that failed never recorded
    it - so releasing it here is the only opportunity. ``Close()`` is safe on
    a half-built endpoint: it routes to ``Channel.CloseReader`` /
    ``CloseWriter``, which skip an entity that was never created and still
    stop and join the reader thread.

    Args:
        endpoint: The half-built subscriber or publisher, or ``None`` when the
            constructor itself raised and there is nothing to release.
        what: How to name the endpoint if closing it also fails.
    """
    if endpoint is None:
        return
    try:
        endpoint.Close()
    except Exception as exc:  # noqa: BLE001 - SDK raises bare Exception
        logger.warning("failed to close a partially built %s: %s", what, exc)


class DDSSubscriberSet:
    """A bag of :class:`unitree_sdk2py.core.channel.ChannelSubscriber` objects.

    Constructed by :meth:`~strands_robots.drivers.g1.G1Driver.connect_eagerly`
    when the SDK is present. On a headless machine the driver never builds one
    - the subscribers are stubbed by the test so nothing here needs SDK
    imports.
    """

    def __init__(self, network_interface: str) -> None:
        """Record the interface; :meth:`start` does the DDS work.

        Args:
            network_interface: The interface to bind subscribers to. Passed
                through to :func:`~strands_robots.tools.g1.ensure_dds`.
        """
        self._interface = network_interface
        self._subs: list[Any] = []
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> str | None:
        """Initialise DDS if it is not already. No subscribers are created here.

        Returns:
            ``None`` on success, or the reason the DDS init failed. A caller
            that gets a reason should not proceed to :meth:`subscribe`, whose
            subscribers would attach to nothing.
        """
        with self._lock:
            if self._started:
                return None
            err = ensure_dds(self._interface)
            if err is not None:
                return err
            self._started = True
            return None

    def subscribe(
        self,
        topic: str,
        message_class: type,
        callback: Callable[[Any], None],
    ) -> str | None:
        """Attach ``callback`` to messages arriving on ``topic``.

        The subscriber is constructed under :data:`_DDS_INIT_LOCK` because the
        CycloneDDS bindings segfault under concurrent
        ``ChannelSubscriber(...)``. Same reason the agent tools (issue #358)
        share the lock: two consumers, one bus, one construction lane.

        Args:
            topic: The DDS topic name.
            message_class: The IDL class the bus deserialises into.
            callback: A single-argument function called with each decoded
                message. It runs on the DDS thread - keep it fast.

        Returns:
            ``None`` on success. An error string if the SDK is missing or the
            subscriber constructor raised; never raises. A construction that
            failed part way is closed before the reason is returned, because
            an unrecorded subscriber is one :meth:`close` can never reach -
            see :func:`_release_partial`.
        """
        if not self._started:
            return "DDS not initialised - call start() first"
        try:
            # Lazy import so this module is safe to import without the SDK.
            from unitree_sdk2py.core.channel import ChannelSubscriber
        except ImportError as exc:  # pragma: no cover - exercised on hardware
            return f"unitree_sdk2py is not installed: {exc}"
        with _DDS_INIT_LOCK:
            sub: Any | None = None
            try:
                sub = ChannelSubscriber(topic, message_class)
                sub.Init(callback, 10)
            except Exception as exc:  # noqa: BLE001 - SDK raises bare Exception
                _release_partial(sub, f"subscriber for {topic!r}")
                return f"failed to subscribe to {topic!r}: {exc}"
            self._subs.append(sub)
            logger.debug("subscribed to %s", topic)
            return None

    def close(self) -> None:
        """Close every subscriber and forget it. Idempotent, and never raises.

        ``ChannelSubscriber.Close()`` is what releases a subscription, and
        dropping the reference is not a substitute. :meth:`subscribe` builds
        each subscriber with ``queueLen=10``, and at any queue length above
        zero ``unitree_sdk2py`` starts a ``ch_reader`` daemon thread whose
        target is a bound method of the channel's reader. That running thread
        keeps the whole channel reachable, so the CycloneDDS ``__del__`` that
        would release the ``DataReader`` never runs: the thread keeps looping,
        the reader stays matched, and the decoder callback keeps filling caches
        for a driver that believes it is disconnected. ``Close()`` sets the
        thread's stop event, joins it and drops the reader; nothing else does.

        The list is swapped out under :attr:`_lock` before any ``Close()``
        runs, so a second call finds nothing to close and a concurrent
        :meth:`subscribe` - which appends without that lock - cannot mutate
        the list being walked. Each subscriber is then closed under its own
        ``try``, because a teardown that abandons the subscribers behind a
        failing one leaks exactly what it was called to release.

        Called from :meth:`~strands_robots.drivers.g1.G1Driver.cleanup`.
        """
        with self._lock:
            subs, self._subs = self._subs, []
        for sub in subs:
            try:
                sub.Close()
            except Exception as exc:  # noqa: BLE001 - SDK raises bare Exception
                logger.warning("failed to close a DDS subscriber: %s", exc)


class DDSPublisher:
    """A bag of :class:`unitree_sdk2py.core.channel.ChannelPublisher` objects.

    Sibling of :class:`DDSSubscriberSet` and shares the same construction lane:
    ``ChannelPublisher(...)`` under :data:`_DDS_INIT_LOCK`, lazy SDK import,
    idempotent :meth:`start` (a second call is a no-op if the same interface
    was already initialised).

    A ``ChannelPublisher`` is constructed once per ``(topic, message_class)``
    pair and cached, because re-constructing a publisher on every write costs
    a DDS round-trip and, more importantly, races the shared lock with the
    subscriber set. :meth:`publish` looks the publisher up, and if it is
    missing constructs it - one lane, one place.

    The engine deliberately exposes ``get_publisher`` too so a caller that
    wants to build its own message and call ``Write`` directly (issue #358's
    arm-SDK client is that caller) can share the cache with the driver's
    control loop (issue #361). One process, one publisher per topic.
    """

    def __init__(self, network_interface: str) -> None:
        """Record the interface; :meth:`start` does the DDS work.

        Args:
            network_interface: The interface to bind publishers to. Passed
                through to :func:`~strands_robots.tools.g1.ensure_dds`.
        """
        self._interface = network_interface
        self._pubs: dict[tuple[str, type], Any] = {}
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> str | None:
        """Initialise DDS if it is not already. No publishers are created here.

        Returns:
            ``None`` on success, or the reason the DDS init failed. A caller
            that gets a reason should not proceed to :meth:`publish`, whose
            publisher would attach to nothing.
        """
        with self._lock:
            if self._started:
                return None
            err = ensure_dds(self._interface)
            if err is not None:
                return err
            self._started = True
            return None

    def get_publisher(
        self,
        topic: str,
        message_class: type,
    ) -> tuple[Any | None, str | None]:
        """Return (publisher, None) or (None, reason).

        Constructs the ``ChannelPublisher`` on first ask and caches it keyed
        by ``(topic, message_class)`` so a re-ask returns the same object.
        Under :data:`_DDS_INIT_LOCK` so the subscriber set cannot construct
        concurrently.

        Args:
            topic: The DDS topic name (e.g. ``"rt/lowcmd"``).
            message_class: The IDL class the bus serialises. Must be the same
                object across calls that share a cache entry - two distinct
                ``LowCmd_`` classes from different import paths would be two
                cache entries, which is what a caller wants for isolation and
                what a bug wants for confusion.

        Returns:
            ``(publisher, None)`` on success. ``(None, reason)`` if the SDK
            is missing or the publisher constructor raised; never raises. A
            construction that failed part way is closed before the reason is
            returned, because an uncached publisher is one :meth:`close` can
            never reach - see :func:`_release_partial`.
        """
        if not self._started:
            return None, "DDS not initialised - call start() first"
        key = (topic, message_class)
        cached = self._pubs.get(key)
        if cached is not None:
            return cached, None
        try:
            # Lazy import so this module is safe to import without the SDK.
            from unitree_sdk2py.core.channel import ChannelPublisher
        except ImportError as exc:  # pragma: no cover - exercised on hardware
            return None, f"unitree_sdk2py is not installed: {exc}"
        with _DDS_INIT_LOCK:
            # Double-check under lock: another caller may have created it
            # while we were waiting on the lock.
            cached = self._pubs.get(key)
            if cached is not None:
                return cached, None
            pub: Any | None = None
            try:
                pub = ChannelPublisher(topic, message_class)
                pub.Init()
            except Exception as exc:  # noqa: BLE001 - SDK raises bare Exception
                _release_partial(pub, f"publisher for {topic!r}")
                return None, f"failed to build publisher for {topic!r}: {exc}"
            self._pubs[key] = pub
            logger.debug("built publisher for %s", topic)
            return pub, None

    def publish(
        self,
        topic: str,
        message_class: type,
        message: Any,
    ) -> str | None:
        """Send ``message`` on ``topic``.

        Args:
            topic: The DDS topic name.
            message_class: The IDL class the bus serialises. Also the cache
                key - see :meth:`get_publisher`.
            message: The already-built IDL message. The caller owns the
                message shape because the shape is the caller's contract
                (a LowCmd_ from the driver's control loop is not a LowCmd_
                from a one-shot agent tool - they populate different fields).

        Returns:
            ``None`` on success. An error string if the publisher could not
            be built or if ``Write`` raised; never raises.
        """
        pub, err = self.get_publisher(topic, message_class)
        if err is not None:
            return err
        assert pub is not None  # narrowing for the type checker; err is None means pub is set
        try:
            pub.Write(message)
        except Exception as exc:  # noqa: BLE001 - SDK raises bare Exception
            return f"publish to {topic!r} failed: {exc}"
        return None

    def close(self) -> None:
        """Close every publisher and forget it. Idempotent, and never raises.

        ``ChannelPublisher.Close()`` releases the writer through
        ``Channel.CloseWriter``. A writer starts no reader thread, so unlike
        :meth:`DDSSubscriberSet.close` a dropped reference here does reach
        CycloneDDS' ``__del__`` and the ``DataWriter`` is collected - but only
        whenever the last reference happens to go, which a live traceback or a
        reference cycle can defer indefinitely. Closing says when.

        Mirrors :meth:`DDSSubscriberSet.close`: the cache is swapped out under
        :attr:`_lock` so a second call closes nothing twice, and each publisher
        is closed under its own ``try`` so one failure cannot strand the rest.

        The driver holds no :class:`DDSPublisher` yet - the control loop that
        owns one lands with issue #361 - so today this is reached only by a
        caller that built its own.
        """
        with self._lock:
            pubs, self._pubs = list(self._pubs.values()), {}
        for pub in pubs:
            try:
                pub.Close()
            except Exception as exc:  # noqa: BLE001 - SDK raises bare Exception
                logger.warning("failed to close a DDS publisher: %s", exc)
