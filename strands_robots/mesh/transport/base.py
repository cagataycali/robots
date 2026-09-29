"""Pluggable transport protocol for the strands-robots mesh.

Defines the Protocol that :class:`~strands_robots.mesh.core.Mesh` uses to
publish and subscribe. Every concrete backend (Zenoh, AWS IoT MQTT, Bridge)
implements this protocol.

The protocol is deliberately tiny - exactly what ``mesh.session`` already
exposed. Every behavioural enrichment (peer registry, RPC correlation, audit)
lives at the Mesh layer and is transport-agnostic.

Why ``Sample`` is duck-typed
----------------------------
Zenoh callbacks receive a ``zenoh.Sample`` with ``.key_expr`` and
``.payload.to_bytes()``. AWS IoT MQTT5 callbacks receive a topic string and
``bytes`` payload. Rather than pick a concrete type and force one transport
to adapt, we declare the **structural protocol** all callers actually use:

    sample.key_expr  # str  - the topic / key the message arrived on
    sample.payload.to_bytes()  # bytes - the raw payload

Concrete backends produce objects matching this shape. The MQTT backend ships
a tiny ``_MqttSample`` wrapper; the Zenoh backend passes ``zenoh.Sample``
through unchanged.

Two OPTIONAL attributes ride on the MQTT wrapper only::

    sample.response_topic    # str | None - MQTT5 Response Topic property
    sample.correlation_data  # str | None - MQTT5 Correlation Data, decoded

A ``zenoh.Sample`` carries neither, so a handler that wants them reads
``getattr(sample, "response_topic", None)`` and treats ``None`` as "reply the
way you always did". They exist for :class:`DirectSender`: a peer that
receives a command as a direct message finds the sender's reply address in
the Response Topic instead of computing it.

Why ``DirectSender`` is a second, optional protocol
--------------------------------------------------
Zenoh has no notion of "deliver to one connected client without a
subscription"; AWS IoT Core Direct Messaging is exactly that. Folding an
addressed send into :class:`MeshTransport` would force the Zenoh and Bridge
backends to implement a method they cannot honour. Instead a backend that CAN
address one peer also satisfies :class:`DirectSender`, and
:class:`~strands_robots.mesh.core.Mesh` asks ``isinstance(transport,
DirectSender)`` once at start. A backend that does not is left exactly as it
was.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

#: The vocabulary of :attr:`DirectResult.reason`. ``""`` means delivered. The
#: other seven each name ONE HTTP outcome of the Direct Messaging API so a
#: caller can branch on the string without knowing the status code behind it:
#: 404 offline, 403 forbidden, 429 throttled, 413 too_large, 504 unconfirmed,
#: any other 5xx (or a transport level failure) error, and ``unavailable``
#: when the sender itself cannot be used (no connection, no credentials).
DIRECT_REASONS: tuple[str, ...] = (
    "",
    "offline",
    "forbidden",
    "throttled",
    "too_large",
    "unconfirmed",
    "error",
    "unavailable",
)


@dataclass(frozen=True)
class DirectResult:
    """Outcome of one :meth:`DirectSender.send_direct` call.

    Attributes:
        delivered: ``True`` when the broker accepted the message for the
            target client (HTTP 200). With ``confirm=True`` this also means
            the target acknowledged it (PUBACK).
        reason: ``""`` when delivered, otherwise one of
            :data:`DIRECT_REASONS`.
        latency_ms: Wall time of the call, including any retry.
        trace_id: The broker's trace id when it sent one, else ``""``.
        detail: The broker's message text on failure, else ``""``. For a
            log line, never for branching: branch on ``reason``.
    """

    delivered: bool
    reason: str
    latency_ms: float
    trace_id: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if self.reason not in DIRECT_REASONS:
            raise ValueError(f"DirectResult: reason must be one of {DIRECT_REASONS}, got {self.reason!r}")
        if self.delivered != (self.reason == ""):
            raise ValueError(
                f"DirectResult: delivered={self.delivered!r} disagrees with reason={self.reason!r} "
                "(delivered means reason is empty, and only then)"
            )


@runtime_checkable
class DirectSender(Protocol):
    """A transport that can address ONE peer without a subscription on its side.

    Optional capability next to :class:`MeshTransport`. The AWS IoT backend
    implements it over the Direct Messaging API; Zenoh and Bridge do not, and
    :class:`~strands_robots.mesh.core.Mesh` falls back to ``put`` for them.

    Contract
    --------
    - MUST NOT raise, whatever goes wrong: a failure is a
      :class:`DirectResult` with ``delivered=False`` and a ``reason``. This is
      the same tolerance :meth:`MeshTransport.put` has, for the same caller:
      ``Mesh.send`` runs inside agent tool calls and control loops.
    - MUST return within roughly ``timeout`` seconds plus one retry budget
      when ``confirm`` is on, and within a few hundred milliseconds when it is
      off. An offline target is reported as ``offline`` at once, never waited
      for.
    - ``key`` is the full topic the message is delivered on, exactly the
      string :meth:`MeshTransport.put` would take.
    - ``response_key`` and ``correlation`` travel as the MQTT5 Response Topic
      and Correlation Data properties, so the receiver's sample exposes them
      as ``response_topic`` and ``correlation_data``.
    """

    def send_direct(
        self,
        peer_id: str,
        key: str,
        data: dict[str, Any],
        *,
        confirm: bool = False,
        timeout: float = 5.0,
        response_key: str | None = None,
        correlation: str | None = None,
    ) -> DirectResult:
        """Deliver *data* on *key* to the one client identified by *peer_id*.

        Args:
            peer_id: The target's mesh peer id, which the IoT backend maps to
                the MQTT client id (equal to the Thing name).
            key: The topic the target receives the message on.
            data: A JSON-serialisable dictionary.
            confirm: Deliver at QoS 1 and wait for the target's PUBACK. When
                the target does not acknowledge within ``timeout`` the result
                is ``unconfirmed``.
            timeout: Whole seconds the broker waits for the acknowledgement
                when ``confirm`` is on. Implementations clamp to the API's
                accepted range.
            response_key: Topic the target should reply on (MQTT5 Response
                Topic). ``None`` sends none.
            correlation: Opaque text the target echoes back (MQTT5
                Correlation Data). ``None`` sends none.

        Returns:
            A :class:`DirectResult`; never raises.
        """
        ...


@runtime_checkable
class _PayloadLike(Protocol):
    def to_bytes(self) -> bytes: ...


@runtime_checkable
class Sample(Protocol):
    """Structural protocol for messages delivered to subscriber callbacks.

    Concrete shape that every backend's callback delivery must satisfy.
    Mirrors ``zenoh.Sample`` so existing Mesh handlers (``_on_presence``,
    ``_on_cmd``, ``_on_response``) work unchanged regardless of transport.
    """

    key_expr: Any  # zenoh: KeyExpr; mqtt: str - both stringify
    payload: _PayloadLike


@runtime_checkable
class SubHandle(Protocol):
    """Opaque subscription handle - must support ``undeclare()`` for teardown.

    Mirrors ``zenoh.Subscriber.undeclare()``. MQTT-backed implementations wrap
    the broker's ``unsubscribe`` packet behind the same name so Mesh teardown
    code is transport-agnostic.
    """

    def undeclare(self) -> None:
        """Tear down the subscription, releasing its transport resources.

        Called by Mesh teardown; idempotent-safe implementations are
        preferred. Zenoh maps this to ``Subscriber.undeclare()``; MQTT
        backends send the broker ``unsubscribe`` packet.
        """


@runtime_checkable
class MeshTransport(Protocol):
    """Pluggable transport for :class:`~strands_robots.mesh.core.Mesh`.

    Lifetime contract
    -----------------
    Implementations are ref-counted singletons per process, mirroring the
    existing :func:`~strands_robots.mesh.session.get_session` /
    :func:`~strands_robots.mesh.session.release_session` pair.

    - The first :class:`Mesh` to require a transport calls :func:`get_transport`
      which constructs (or returns) the singleton and increments its refcount.
    - Each :class:`Mesh.stop` calls :func:`release_transport` exactly once.
    - When the refcount reaches zero, :meth:`close` is invoked.

    Failure mode
    ------------
    A transport whose connection is dead returns ``False`` from
    :meth:`is_alive`. Callers (notably :func:`put`) treat a dead transport as
    a no-op rather than raising - preserving the current Mesh contract that
    publish failures never propagate up into hot control loops.
    """

    def put(self, key: str, data: dict[str, Any]) -> None:
        """Publish a JSON-serialisable payload to the wire.

        Fire-and-forget: MUST NOT raise, whatever goes wrong.

        The tolerance is scoped by WHETHER A RETRY COULD SUCCEED, because the two
        cases are not equally reportable:

        * A TRANSIENT failure - a closed session, a dropped broker, a
          socket-level write error - is retried by the caller's next tick.
          Implementations log at debug and continue, matching the Zenoh
          behaviour today.
        * A payload the JSON encoder refuses can NEVER reach the wire, and no
          retry changes that. Implementations report it at ERROR once per topic
          via :func:`~strands_robots.mesh.session._report_unencodable_payload`,
          which owns the wording so one grep finds every transport's report.
          Absorbing it into the transient DEBUG line made a permanently
          undeliverable message indistinguishable from a delivered one.

        Args:
            key: The topic / Zenoh key expression. For MQTT-backed transports
                this is the MQTT topic verbatim (no translation needed - our
                topic scheme is already MQTT-safe).
            data: A JSON-serialisable dictionary. Implementations are expected
                to encode it via ``json.dumps(...).encode()``.
        """
        ...

    def declare_subscriber(self, key_expr: str, handler: Callable[[Sample], None]) -> SubHandle:
        """Subscribe to a key expression and route inbound messages to *handler*.

        ``handler`` receives a :class:`Sample`-shaped object. For Zenoh that's
        ``zenoh.Sample`` directly. For MQTT it's a thin wrapper that exposes
        ``.key_expr`` (the topic string) and ``.payload.to_bytes()`` (the
        payload bytes).

        Wildcard translation:
            Zenoh ``*``  matches one segment → MQTT ``+``
            Zenoh ``**`` matches tail  → MQTT ``#``
            MQTT-backed implementations translate these on the fly.

        Args:
            key_expr: Zenoh-style key expression. Concrete patterns we use:
                ``strands/*/presence``, ``strands/{peer}/cmd``,
                ``strands/{peer}/response/**``, ``strands/broadcast``.
            handler: Callback invoked once per received message. Runs on the
                transport's IO thread; must NOT block.

        Returns:
            An opaque :class:`SubHandle` that the caller must keep alive for
            the duration of the subscription, and call ``.undeclare()`` on
            during teardown.
        """
        ...

    def is_alive(self) -> bool:
        """True if the transport's session is open and usable."""
        ...

    def close(self) -> None:
        """Tear down the transport. Idempotent.

        Called when the last :class:`Mesh` referencing this transport stops.
        Implementations should release sockets, drain queues, and close any
        underlying client.
        """
        ...
