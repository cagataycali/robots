"""Bridge transport — Zenoh LAN + AWS IoT WAN behind one MeshTransport.

This is the **production** transport for fleets that have both LAN peers
(robot + sim + leader arm in the same room) AND cloud connectivity
(operator dashboards, audit, fleet ops in AWS).

How it works
------------
:class:`BridgeTransport` owns one :class:`ZenohTransport` and one
:class:`IotMqttTransport`. Every :meth:`put` fans out to both, but the
**MQTT side is filtered**: high-volume / latency-sensitive topics
(``state``, ``pose``, ``imu``, ``odom``, ``camera``, ``input``, ``hand``)
default to LAN-only. Cloud-relevant topics (``presence``, ``health``,
``cmd``, ``response``, ``broadcast``, ``safety/event``, ``safety/estop``)
default to **both transports**.

The filter is configurable via ``STRANDS_MESH_BRIDGE_TOPICS`` (a
comma-separated list of suffixes that bridge to MQTT). The default value
is conservative and matches the cost analysis in the design doc.

Subscriptions are **fanned out**: a single ``declare_subscriber("X", h)``
subscribes ``h`` on both Zenoh and MQTT. Inbound deduplication happens at
the :class:`Mesh` layer (existing :meth:`_on_presence` / :meth:`_on_cmd` /
:meth:`_on_response` already handle duplicate / self-loop dropouts via
``sender_id`` and ``turn_id`` correlation), so we don't try to be clever
here — the fact that a presence might arrive twice is harmless and the
existing peer registry deduplicates by ``peer_id``.

Failure isolation
-----------------
If either side's :meth:`connect` fails, the bridge degrades gracefully:

- Zenoh failed, IoT OK → behaves as a pure :class:`IotMqttTransport`.
- IoT failed, Zenoh OK → behaves as a pure :class:`ZenohTransport`.
- Both failed → :meth:`is_alive` returns False, all puts are no-ops.

This matches the "mesh is enrichment, never crash the robot" contract
that :class:`Mesh` already follows for failed Zenoh sessions.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands_robots.mesh.transport.iot_transport import IotMqttTransport
    from strands_robots.mesh.transport.zenoh_transport import ZenohTransport

logger = logging.getLogger(__name__)


def _get_zenoh_transport_class() -> type[ZenohTransport]:
    """Lazily import ZenohTransport to avoid circular dependency."""
    from strands_robots.mesh.transport.zenoh_transport import ZenohTransport as _ZT

    return _ZT


def _get_iot_transport_class() -> type[IotMqttTransport]:
    """Lazily import IotMqttTransport to avoid circular dependency."""
    from strands_robots.mesh.transport.iot_transport import IotMqttTransport as _IT

    return _IT


# Default bridge filter — derived from cost / latency analysis.
#
# Topics in this set bridge from Zenoh to MQTT. Everything else stays LAN.
# Suffixes are matched against the part of the topic AFTER ``strands/``.
#
# Why this default:
# - presence       — rare, retained on cloud, late operators need it
# - health         — rare, retained, threshold alerts via Rules
# - safety/event   — must hit cloud audit
# - safety/estop   — defence-in-depth E-stop
# - cmd            — operator-to-robot RPC (cloud → robot direction)
# - response       — robot-to-operator RPC reply
# - broadcast      — fan-out RPC
#
# Explicitly NOT bridged by default (opt in via STRANDS_MESH_BRIDGE_TOPICS):
# - state, pose, imu, odom, lidar — high volume, route via Basic Ingest if
#   cloud needs them. See AWS_IOT_MESH_INTEGRATION.md §7.2 for the cost math.
# - camera, input, hand — LAN-only by definition (size / latency).
DEFAULT_BRIDGE_SUFFIXES: frozenset[str] = frozenset(
    {
        "presence",
        "health",
        "safety/event",
        "safety/estop",
        "cmd",
        "response",
        "broadcast",
    }
)


def _resolve_bridge_filter() -> frozenset[str]:
    """Read ``STRANDS_MESH_BRIDGE_TOPICS`` or fall back to the default."""
    env = os.getenv("STRANDS_MESH_BRIDGE_TOPICS")
    if not env:
        return DEFAULT_BRIDGE_SUFFIXES
    parts = [p.strip() for p in env.split(",") if p.strip()]
    if not parts:
        return DEFAULT_BRIDGE_SUFFIXES
    return frozenset(parts)


def _topic_suffix(topic: str) -> str:
    """Return the suffix following ``strands/`` from a Mesh topic.

    Handles three layouts:

    - ``strands/broadcast``               -> ``broadcast``
    - ``strands/safety/estop``            -> ``safety/estop``
    - ``strands/{peer}/{kind}/...``       -> ``{kind}/...``
    """
    if not topic.startswith("strands/"):
        return ""
    rest = topic[len("strands/") :]
    parts = rest.split("/", 1)
    if len(parts) == 1:
        return rest  # e.g. "broadcast"
    head, tail = parts
    # If head is a recognised top-level kind ("safety", "broadcast"), keep it.
    if head in ("safety", "broadcast"):
        return rest
    # Otherwise head is a peer_id and the kind starts at tail.
    return tail


def _should_bridge(topic: str, allowed_suffixes: frozenset[str]) -> bool:
    """True if *topic* should be republished to MQTT.

    Match policy: a topic suffix matches an allowed entry if either is a
    prefix of the other up to a ``/`` boundary. So ``response/abc123``
    matches the allowed suffix ``response``, and ``safety/event`` matches
    itself exactly.
    """
    suffix = _topic_suffix(topic)
    if not suffix:
        return False
    suffix_parts = suffix.split("/")
    for n in range(len(suffix_parts), 0, -1):
        candidate = "/".join(suffix_parts[:n])
        if candidate in allowed_suffixes:
            return True
    return False


class _BridgeSubHandle:
    """Subscription handle that calls ``undeclare`` on whichever side(s)
    actually subscribed.

    Both sides may be present (typical) or only one (the other's connect
    failed). Tearing down only what's there.
    """

    __slots__ = ("_zenoh_sub", "_iot_sub", "_undeclared")

    def __init__(self, zenoh_sub: Any | None, iot_sub: Any | None) -> None:
        self._zenoh_sub = zenoh_sub
        self._iot_sub = iot_sub
        self._undeclared = False

    def undeclare(self) -> None:
        if self._undeclared:
            return
        self._undeclared = True
        for sub in (self._zenoh_sub, self._iot_sub):
            if sub is None:
                continue
            try:
                sub.undeclare()
            except Exception as exc:
                logger.debug("[bridge] sub.undeclare() failed: %s", exc)


class BridgeTransport:
    """:class:`MeshTransport` that fans out to both Zenoh (LAN) and AWS IoT (WAN).

    Construct with no arguments — the underlying :class:`ZenohTransport` and
    :class:`IotMqttTransport` read their config from env vars exactly as
    they would on their own.

    Lifecycle:
        :meth:`connect` succeeds if **either** side connects. The other
        side becomes a silent no-op for that path.
    """

    def __init__(
        self,
        zenoh: ZenohTransport | None = None,
        iot: IotMqttTransport | None = None,
        bridge_suffixes: frozenset[str] | None = None,
    ) -> None:
        ZenohTransportCls = _get_zenoh_transport_class()
        IotMqttTransportCls = _get_iot_transport_class()
        self._zenoh = zenoh or ZenohTransportCls()
        self._iot = iot or IotMqttTransportCls()
        self._bridge_suffixes = bridge_suffixes if bridge_suffixes is not None else _resolve_bridge_filter()
        self._zenoh_alive = False
        self._iot_alive = False
        self._lock = threading.Lock()

    # Lifecycle

    def connect(self) -> bool:
        """Connect both backends. Succeeds if at least one works."""
        with self._lock:
            self._zenoh_alive = self._zenoh.connect()
            self._iot_alive = self._iot.connect()

            if not self._zenoh_alive and not self._iot_alive:
                logger.warning("[bridge] both Zenoh and IoT failed to connect")
                return False

            if self._zenoh_alive and self._iot_alive:
                logger.info(
                    "[bridge] both transports up — bridging %d topic suffix(es): %s",
                    len(self._bridge_suffixes),
                    sorted(self._bridge_suffixes),
                )
            elif self._zenoh_alive:
                logger.info("[bridge] only Zenoh up — running LAN-only")
            else:
                logger.info("[bridge] only IoT up — running cloud-only")
            return True

    def close(self) -> None:
        """Close both backends. Idempotent."""
        with self._lock:
            try:
                self._zenoh.close()
            except Exception as exc:
                logger.debug("[bridge] zenoh.close() failed: %s", exc)
            try:
                self._iot.close()
            except Exception as exc:
                logger.debug("[bridge] iot.close() failed: %s", exc)
            self._zenoh_alive = False
            self._iot_alive = False

    # Inspection

    def is_alive(self) -> bool:
        return self._zenoh.is_alive() or self._iot.is_alive()

    @property
    def zenoh(self) -> ZenohTransport:
        return self._zenoh

    @property
    def iot(self) -> IotMqttTransport:
        return self._iot

    @property
    def bridge_suffixes(self) -> frozenset[str]:
        return self._bridge_suffixes

    @property
    def raw_session(self) -> Any | None:
        """The underlying ``zenoh.Session`` for backwards compatibility.

        Mesh.subscribe() historically called ``current_session().declare_subscriber(...)``
        directly. Bridge mode delegates that to the Zenoh side because user
        subscriptions to arbitrary Zenoh keys aren't necessarily MQTT-safe.
        """
        return self._zenoh.raw_session

    # Pub/Sub

    def put(self, key: str, data: dict[str, Any]) -> None:
        """Publish to Zenoh always; publish to IoT only if the topic bridges.

        Failure of one side does not affect the other.
        """
        # Always Zenoh (LAN is cheap; preserves existing behaviour).
        if self._zenoh.is_alive():
            try:
                self._zenoh.put(key, data)
            except Exception as exc:
                logger.debug("[bridge] zenoh.put error on %s: %s", key, exc)

        # Filtered IoT.
        if self._iot.is_alive() and _should_bridge(key, self._bridge_suffixes):
            try:
                self._iot.put(key, data)
            except Exception as exc:
                logger.debug("[bridge] iot.put error on %s: %s", key, exc)

    def declare_subscriber(self, key_expr: str, handler: Callable[[Any], None]) -> _BridgeSubHandle:
        """Subscribe on both sides. Inbound deduplication is the Mesh layer's job."""
        zenoh_sub: Any | None = None
        iot_sub: Any | None = None

        if self._zenoh.is_alive():
            try:
                zenoh_sub = self._zenoh.declare_subscriber(key_expr, handler)
            except Exception as exc:
                logger.debug("[bridge] zenoh.declare_subscriber(%s) failed: %s", key_expr, exc)

        if self._iot.is_alive():
            try:
                iot_sub = self._iot.declare_subscriber(key_expr, handler)
            except Exception as exc:
                logger.debug("[bridge] iot.declare_subscriber(%s) failed: %s", key_expr, exc)

        if zenoh_sub is None and iot_sub is None:
            raise RuntimeError(f"BridgeTransport.declare_subscriber({key_expr!r}) failed on both sides")

        return _BridgeSubHandle(zenoh_sub, iot_sub)
