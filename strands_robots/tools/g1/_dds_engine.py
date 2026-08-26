"""Subscription helpers for the G1 driver's read path.

The driver holds one :class:`DDSSubscriberSet` and asks it for background
subscribers whose callbacks fill in-memory caches (the newest message wins).
The mesh publishes those caches at its own cadence, so the DDS callback stays
fast: parse, drop into a slot, return.

The engine deliberately does not surface a publisher API - writing to
``rt/armsdk`` and ``rt/lowcmd`` is the arm-SDK client's job (issue #358), and
mixing the two here would tempt callers into ad-hoc writes that skip the FSM
gate. Read-only, and small.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from strands_robots.tools.g1._g1_common import _DDS_INIT_LOCK, ensure_dds

logger = logging.getLogger(__name__)


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
            subscriber constructor raised; never raises.
        """
        if not self._started:
            return "DDS not initialised - call start() first"
        try:
            # Lazy import so this module is safe to import without the SDK.
            from unitree_sdk2py.core.channel import ChannelSubscriber
        except ImportError as exc:  # pragma: no cover - exercised on hardware
            return f"unitree_sdk2py is not installed: {exc}"
        with _DDS_INIT_LOCK:
            try:
                sub = ChannelSubscriber(topic, message_class)
                sub.Init(callback, 10)
            except Exception as exc:  # noqa: BLE001 - SDK raises bare Exception
                return f"failed to subscribe to {topic!r}: {exc}"
            self._subs.append(sub)
            logger.debug("subscribed to %s", topic)
            return None

    def close(self) -> None:
        """Release every subscriber. Idempotent.

        ``ChannelSubscriber`` in ``unitree_sdk2py`` has no explicit close - it
        relies on garbage collection - so this drops the references and lets
        the collector claim them. Called from
        :meth:`~strands_robots.drivers.g1.G1Driver.cleanup`.
        """
        with self._lock:
            self._subs.clear()
