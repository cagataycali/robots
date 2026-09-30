"""AWS IoT Core MQTT5 transport - :class:`MeshTransport` over mTLS.

Wraps an ``awscrt.mqtt5`` client behind the :class:`MeshTransport` Protocol so
:class:`~strands_robots.mesh.core.Mesh` can publish presence, state, RPCs, and
safety events to AWS IoT Core with X.509 mutual TLS.

The strands-robots topic scheme is already MQTT-safe - every Zenoh key like
``strands/{peer}/state`` translates verbatim. The only translations that
happen here are wildcard mapping (``*`` → ``+``, ``**`` → ``#``) and the
delivery shape - MQTT5's flat ``(topic_str, bytes)`` callback is wrapped in
a tiny ``_MqttSample`` so existing :class:`Mesh` handlers work unmodified.

Trust model
-----------
Each robot owns an X.509 cert tied to a Thing whose name **equals** the
:class:`Mesh` peer_id. AWS IoT Policy enforces topic-level ACLs via the
``${iot:Connection.Thing.ThingName}`` substitution. See
:doc:`../../research/IOT_SPIKE_FINDINGS` for the full policy templates.

Required environment
--------------------
``STRANDS_IOT_ENDPOINT``
    The AWS IoT Core ATS endpoint, e.g.
    ``a2acz9p1ge6619-ats.iot.us-west-2.amazonaws.com``.
``STRANDS_IOT_THING_NAME``
    The Thing name. MUST equal the cert's CN. The ``client_id`` used at
    connect time is set to this value so policy variable substitution works.
``STRANDS_IOT_CERT_DIR``
    Directory holding ``{thing}.cert.pem``, ``{thing}.private.key``,
    ``AmazonRootCA1.pem``. Defaults to ``~/.strands_robots/iot``.

Optional
--------
``STRANDS_IOT_CA_FILE``
    Path to the root CA file. Defaults to ``$STRANDS_IOT_CERT_DIR/AmazonRootCA1.pem``.

Failure mode
------------
If ``awsiotsdk`` is not installed, :meth:`connect` returns ``False`` and the
transport behaves as a silent no-op (matching :class:`ZenohTransport` when
Zenoh is missing). If the endpoint or cert files are missing, :meth:`connect`
logs at ERROR and returns ``False`` - the mesh stays off rather than crash
the host.
"""

from __future__ import annotations

import base64
import http.client
import json
import logging
import os
import random
import ssl
import threading
import time
import urllib.parse
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from strands_robots.mesh.session import _report_unencodable_payload
from strands_robots.mesh.transport.base import DirectResult
from strands_robots.utils import positive_finite_number_error

logger = logging.getLogger(__name__)

#: Opt-out switch for AWS IoT Core Direct Messaging on the ``iot`` and ``bridge``
#: backends. Same domain as ``STRANDS_MESH_BRIDGE_DEDUP_STRICT``: ``1`` /
#: ``true`` / ``yes`` on, ``0`` / ``false`` / ``no`` off, unset means on. Anything
#: else is reported with a WARNING and treated as on: a typo must not change
#: how commands reach a robot, and the correct value is the one that works.
#: How long after a publish a broker DISCONNECT is attributed to that publish
#: (:meth:`IotMqttTransport._warn_if_publish_ended_the_session`). AWS IoT ends
#: the session within a few milliseconds of an ungranted publish; a second is
#: wide enough for a loaded event loop and narrow enough that a network drop
#: minutes after the last tick is not blamed on it.
DISCONNECT_AFTER_PUBLISH_WINDOW_S = 1.0

DIRECT_ENV_VAR = "STRANDS_MESH_IOT_DIRECT"
_DIRECT_ON = ("1", "true", "yes")
_DIRECT_OFF = ("0", "false", "no")

#: Which credential signs the HTTPS ``SendDirectMessage`` call. ``x509`` uses the
#: robot's own certificate over port 8443 (the same identity as the MQTT
#: session, so the IoT policy's ``${iot:Certificate.Subject.CommonName}``
#: resolves); ``sigv4`` uses the process's IAM credentials through boto3 (an
#: agent or dashboard that has no device certificate). Unset picks ``x509``
#: when the certificate files are present, else ``sigv4``.
DIRECT_AUTH_ENV_VAR = "STRANDS_IOT_DIRECT_AUTH"
DIRECT_AUTH_MODES = ("x509", "sigv4")

#: The port AWS IoT Core serves X.509 authenticated HTTPS on.
_DIRECT_X509_PORT = 8443

#: Payload cap of the Direct Messaging API (and of an MQTT publish): 128 KB
#: exactly, measured on 2026-09-29 (128 KB accepted, 129 KB refused with 413).
DIRECT_PAYLOAD_CAP = 128 * 1024

#: The acknowledgement wait the API accepts, whole seconds. The documented
#: example is 10; a value below 1 cannot be honoured, and the API caps it at 10.
_DIRECT_MIN_TIMEOUT_S = 1
_DIRECT_MAX_TIMEOUT_S = 10

#: MQTT keep-alive interval. The broker declares a client gone about 1.5
#: intervals after its last packet, which bounds how long a crashed robot still
#: reads as connected to a direct message (504 after the confirmation window
#: instead of 404 at once).
_KEEP_ALIVE_S = 30

#: Idle HTTPS connections kept per transport. Four covers a robot's reply leg
#: plus the dashboard's fan-out without one slow peer stalling the others.
_POOL_SIZE = 4

#: One MQTT5 user property that marks a message as a strands mesh envelope.
#: The API wants a JSON array of ONE-KEY objects (a name/value shape is refused).
_DIRECT_USER_PROPERTIES: list[dict[str, str]] = [{"strands-mesh": "1"}]

_DIRECT_STATUS_REASON: dict[int, str] = {
    404: "offline",
    403: "forbidden",
    401: "forbidden",
    429: "throttled",
    413: "too_large",
    504: "unconfirmed",
}

_DIRECT_BOTO_REASON: dict[str, str] = {
    "ResourceNotFoundException": "offline",
    "ForbiddenException": "forbidden",
    "UnauthorizedException": "forbidden",
    "ThrottlingException": "throttled",
    "RequestEntityTooLargeException": "too_large",
    "GatewayTimeoutException": "unconfirmed",
}


def direct_messaging_enabled() -> bool:
    """Resolve :data:`DIRECT_ENV_VAR`. Unset, ``1``, ``true`` and ``yes`` mean on; ``0``, ``false`` and ``no`` off.

    Any other spelling is reported at WARNING and treated as on: the switch
    selects how a command reaches a robot, and the value that works is the one
    a typo must not take away. Read once per connect, like the bridge's
    dedup switch.

    Returns:
        ``True`` when direct messaging may be used.
    """
    raw = os.getenv(DIRECT_ENV_VAR, "1").strip().lower()
    if raw in _DIRECT_ON:
        return True
    if raw in _DIRECT_OFF:
        return False
    logger.warning(
        "%s=%r is not one of %s or %s - direct messaging stays ON (the default). "
        "Set '0' to route every command over publish/subscribe instead.",
        DIRECT_ENV_VAR,
        raw,
        "/".join(_DIRECT_ON),
        "/".join(_DIRECT_OFF),
    )
    return True


def direct_auth_mode(cert_present: bool) -> str:
    """Resolve :data:`DIRECT_AUTH_ENV_VAR` to one of :data:`DIRECT_AUTH_MODES`.

    Args:
        cert_present: Whether this transport has a certificate and key on disk,
            which decides the default (``x509`` with, ``sigv4`` without).

    Returns:
        ``"x509"`` or ``"sigv4"``. An unrecognised value is reported at WARNING
        and the default applies.
    """
    default = "x509" if cert_present else "sigv4"
    raw = os.getenv(DIRECT_AUTH_ENV_VAR, "").strip().lower()
    if not raw:
        return default
    if raw in DIRECT_AUTH_MODES:
        return raw
    logger.warning(
        "%s=%r is not one of %s - using %r",
        DIRECT_AUTH_ENV_VAR,
        raw,
        ", ".join(DIRECT_AUTH_MODES),
        default,
    )
    return default


def _region_from_endpoint(endpoint: str) -> str | None:
    """Pull the region out of an ATS endpoint like ``xxx-ats.iot.us-west-2.amazonaws.com``."""
    parts = endpoint.split(".")
    if len(parts) >= 4 and parts[1] == "iot":
        return parts[2]
    return None


def _confirm_timeout_seconds(timeout: float) -> int:
    """Whole seconds for the API's ``timeout`` query parameter, clamped to ``[1, 10]``."""
    return max(_DIRECT_MIN_TIMEOUT_S, min(_DIRECT_MAX_TIMEOUT_S, int(timeout)))


def _parse_direct_error_body(body: bytes) -> tuple[str, str]:
    """Return ``(message, traceId)`` from a Direct Messaging error body, tolerant of non-JSON."""
    try:
        doc = json.loads(body.decode("utf-8", "replace") or "{}")
    except ValueError:
        return body.decode("utf-8", "replace")[:200], ""
    if not isinstance(doc, dict):
        return str(doc)[:200], ""
    return str(doc.get("message") or "")[:200], str(doc.get("traceId") or "")


class _X509DirectClient:
    """Pool of mTLS HTTPS connections to the Direct Messaging API on :8443.

    A fresh TLS handshake per call costs about 320 ms against the same
    endpoint; a kept connection answers in about 80 ms, so idle connections
    are kept (up to :data:`_POOL_SIZE`) and handed out one per in-flight post.
    The lock guards only the idle list, never a request: with confirmation on,
    a post can block for the target's PUBACK, and one lock across that wait
    serialised every direct send in the process (measured 2, 4, 6 s for three
    concurrent posts), stalling a robot's replies and the dashboard's fan-out
    behind one slow peer.

    Every socket or TLS failure drops the connection it happened on: a socket
    that timed out mid-response is half-open and a later request on it
    returns the previous response. A request whose connection was already
    closed by the broker (stale idle connection) is retried once on a fresh
    one, within the caller's remaining budget. The per-request socket timeout
    is that remaining budget, so a dead network answers inside the caller's
    ``timeout`` rather than after a fixed 15 s.

    The certificate is the same one the MQTT session authenticates with, so
    the IoT policy sees one identity for both legs.
    """

    def __init__(self, endpoint: str, cert_path: str, key_path: str, ca_path: str) -> None:
        self._endpoint = endpoint
        self._cert_path = cert_path
        self._key_path = key_path
        self._ca_path = ca_path
        self._idle: list[http.client.HTTPSConnection] = []
        self._lock = threading.Lock()
        self._ctx: ssl.SSLContext | None = None

    def _context(self) -> ssl.SSLContext:
        with self._lock:
            if self._ctx is None:
                ctx = ssl.create_default_context(cafile=self._ca_path)
                ctx.load_cert_chain(self._cert_path, self._key_path)
                ctx.minimum_version = ssl.TLSVersion.TLSv1_2
                self._ctx = ctx
            return self._ctx

    def _take(self) -> tuple[http.client.HTTPSConnection, bool]:
        """An idle connection (``reused=True``) or a new, not yet connected one."""
        with self._lock:
            if self._idle:
                return self._idle.pop(), True
        return http.client.HTTPSConnection(self._endpoint, _DIRECT_X509_PORT, context=self._context()), False

    def _give_back(self, conn: http.client.HTTPSConnection) -> None:
        with self._lock:
            if len(self._idle) < _POOL_SIZE:
                self._idle.append(conn)
                return
        _close_quietly(conn)

    def close(self) -> None:
        with self._lock:
            conns, self._idle = self._idle, []
        for conn in conns:
            _close_quietly(conn)

    def post(self, path: str, body: bytes, headers: dict[str, str], *, deadline: float) -> tuple[int, bytes]:
        """POST once within *deadline* (a ``time.monotonic`` instant).

        A stale idle connection (closed by the broker since its last use) is
        replaced and the request repeated once, if budget remains. Any other
        socket, TLS or HTTP failure drops the connection and raises for the
        caller to map to ``error``; a socket timeout raises ``TimeoutError``.

        Returns:
            ``(status, body)``.
        """
        for attempt in (0, 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("direct HTTPS: budget exhausted before the request")
            conn, reused = self._take()
            conn.timeout = remaining
            if conn.sock is not None:
                conn.sock.settimeout(remaining)
            try:
                conn.request("POST", path, body=body, headers=headers)
                resp = conn.getresponse()
                status, data = resp.status, resp.read()
            except (ConnectionError, http.client.CannotSendRequest):
                # ConnectionError covers BrokenPipe, ConnectionReset and
                # http.client.RemoteDisconnected (a ConnectionResetError).
                _close_quietly(conn)
                if reused and attempt == 0:
                    continue
                raise
            except Exception:
                # Timeout, TLS failure, malformed response: the connection is
                # in an unknown state and is never handed out again.
                _close_quietly(conn)
                raise
            if resp.will_close:
                _close_quietly(conn)
            else:
                self._give_back(conn)
            return status, data
        raise http.client.HTTPException("direct HTTPS: no attempt made")  # pragma: no cover - loop returns or raises


def _close_quietly(conn: http.client.HTTPSConnection) -> None:
    try:
        conn.close()
    except Exception as exc:  # noqa: BLE001 - closing a dead socket is not news
        logger.debug("direct HTTPS close: %s", exc)


class _SdkTooOld(RuntimeError):
    """The installed AWS SDK does not model the Direct Messaging operation."""


class _SigV4DirectClient:
    """``SendDirectMessage`` through boto3's ``iot-data`` client (IAM credentials).

    For a process with no device certificate: an agent, a dashboard, a
    notebook. boto3 signs the call with whatever credential chain the process
    has; the IAM grant is ``iot:SendDirectMessage`` on ``client/*`` with the
    ``iot:Topic`` condition (``bootstrap_account`` creates it).
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._client: Any | None = None
        self._lock = threading.Lock()

    def _get(self) -> Any:
        with self._lock:
            if self._client is None:
                import boto3
                from botocore.config import Config

                region = (
                    os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or _region_from_endpoint(self._endpoint)
                )
                # botocore's defaults (60 s connect, 60 s read, several
                # attempts) would let one call outlive any mesh budget; the
                # per-call deadline below tightens the read timeout further.
                self._client = boto3.client(
                    "iot-data",
                    region_name=region,
                    endpoint_url=f"https://{self._endpoint}",
                    config=Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 1}),
                )
            return self._client

    def close(self) -> None:
        with self._lock:
            self._client = None

    def send(self, params: dict[str, Any], *, deadline: float) -> tuple[int, bytes, str]:
        """Call the API within *deadline*. Returns ``(status, message_bytes, trace_id)``.

        ``ClientError`` is mapped to the status the broker returned; a call
        whose budget is already spent raises ``TimeoutError`` without a
        request.
        """
        from botocore.exceptions import ClientError

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("direct SigV4: budget exhausted before the request")
        client = self._get()
        if not callable(getattr(client, "send_direct_message", None)):
            # The installed botocore predates 1.43.17, whose iot-data model
            # first carries SendDirectMessage. Reported as unavailable once
            # rather than as an error retried with a sleep on every command.
            raise _SdkTooOld(
                "botocore's iot-data model has no SendDirectMessage (needs botocore>=1.43.17, "
                "the [mesh-iot] floor); direct messaging is unavailable in this environment"
            )
        try:
            out = client.send_direct_message(**params)
        except ClientError as exc:
            err = exc.response.get("Error", {})
            code = str(err.get("Code", ""))
            meta = exc.response.get("ResponseMetadata", {})
            status = int(meta.get("HTTPStatusCode", 0) or 0)
            reason = _DIRECT_BOTO_REASON.get(code)
            if reason and status == 0:
                status = next(k for k, v in _DIRECT_STATUS_REASON.items() if v == reason)
            return status or 500, str(err.get("Message", code)).encode(), str(exc.response.get("traceId", ""))
        return 200, b"", str(out.get("traceId", ""))


# Default per-topic QoS / retain map.
#
# Derived from the empirically-validated policy in the spike (§4.1 of
# AWS_IOT_MESH_INTEGRATION.md). Topics not listed default to QoS 0, no retain.
#
# QoS values: 0 = at most once, 1 = at least once, 2 = exactly once.
_TOPIC_POLICY: dict[str, tuple[int, bool]] = {
    # Pattern (suffix-matched against key after the peer_id segment) -> (qos, retain).
    "presence": (1, True),
    "state": (0, False),
    "cmd": (1, False),
    "broadcast": (1, False),
    "response": (1, False),  # matches strands/{peer}/response/{turn}
    "pose": (0, False),
    "imu": (0, False),
    "odom": (0, False),
    "health": (0, True),
    "lidar/summary": (0, False),
    "lidar/state": (0, True),
    "map/info": (0, True),
    "safety/event": (1, True),
    "safety/estop": (1, True),
    "safety/resume": (1, True),  # paired with safety/estop; closes incident windows
    "stream": (0, False),
    "stream/meta": (0, False),
    # Camera frames are too big for MQTT - IotMqttTransport drops them.
    # See the design doc §4.2 for the S3 offload pattern.
    "camera": ("DROP", False),  # type: ignore[dict-item]
}

# Topics we strictly never publish over MQTT, regardless of caller intent.
# Camera frames hit MQTT's 128 KB cap; teleop input has fatal latency over WAN.
_NEVER_BRIDGE_PREFIXES: tuple[str, ...] = (
    "camera/",  # JPEG frames - use S3 offload (Layer 3)
    "input/",  # 50 Hz teleop - LAN-only
    "hand/",  # 50 Hz hand control - LAN-only
)


class _MqttSample:
    """Zenoh-shaped Sample wrapper around an MQTT message.

    Mesh handlers (``_on_presence``, ``_on_cmd``, ``_on_response``) all access
    ``sample.key_expr`` and ``sample.payload.to_bytes()``. By exposing the
    same shape we avoid touching any handler when the transport changes.

    Two MQTT5 properties ride along as optional attributes, ``None`` when the
    publisher set none: ``response_topic`` (Response Topic) and
    ``correlation_data`` (Correlation Data, decoded to text). A ``zenoh.Sample``
    has neither attribute, so a handler reads them with ``getattr(sample,
    "response_topic", None)`` and takes ``None`` as "reply on the computed
    key". They carry the reply address of a command that arrived as an AWS IoT
    Core direct message (see :class:`~strands_robots.mesh.transport.base.DirectSender`).
    """

    __slots__ = ("correlation_data", "key_expr", "payload", "response_topic")

    def __init__(
        self,
        topic: str,
        payload_bytes: bytes,
        response_topic: str | None = None,
        correlation_data: str | None = None,
    ) -> None:
        self.key_expr = topic
        self.payload = _MqttPayload(payload_bytes)
        self.response_topic = response_topic
        self.correlation_data = correlation_data


class _MqttPayload:
    """``zenoh.Sample.payload``-shaped wrapper exposing ``to_bytes()``."""

    __slots__ = ("_bytes",)

    def __init__(self, b: bytes) -> None:
        self._bytes = b

    def to_bytes(self) -> bytes:
        return self._bytes


class _MqttSubHandle:
    """Subscription handle that calls ``unsubscribe`` on undeclare.

    Mirrors ``zenoh.Subscriber.undeclare()`` so :class:`Mesh` teardown code
    is transport-agnostic.
    """

    def __init__(self, transport: IotMqttTransport, topic_filter: str, handler: Any = None) -> None:
        self._transport = transport
        self._topic_filter = topic_filter
        self._handler = handler
        self._undeclared = False

    def undeclare(self) -> None:
        if self._undeclared:
            return
        self._undeclared = True
        self._transport._unsubscribe(self._topic_filter, self._handler)


def _zenoh_to_mqtt_filter(key_expr: str) -> str:
    """Translate a Zenoh key-expression to an MQTT topic filter.

    Zenoh uses ``*`` for "one or more characters within a segment" (in
    practice we only use it as a single-segment match) and ``**`` for
    "any number of segments". MQTT uses ``+`` and ``#`` respectively.

    Patterns we actually use in :class:`Mesh`::

        strands/*/presence  -> strands/+/presence
        strands/{peer}/response/**  -> strands/{peer}/response/#
        strands/broadcast  -> strands/broadcast (unchanged)
        strands/{peer}/cmd  -> strands/{peer}/cmd (unchanged)

    We do **not** support arbitrary Zenoh key-expression syntax here. Callers
    that pass anything we don't recognise get a faithful pass-through and a
    DEBUG log entry; the broker will then SUBACK-deny it cleanly.
    """
    # Walk segments: each '*' segment becomes '+', a trailing '**' becomes '#'.
    segments = key_expr.split("/")
    out: list[str] = []
    for i, seg in enumerate(segments):
        if seg == "**":
            # Tail wildcard. Must be last segment in MQTT; if it isn't, log at
            # DEBUG (caller is using a Zenoh idiom that doesn't translate) and
            # leave the rest verbatim - broker will reject.
            if i != len(segments) - 1:
                logger.debug("Zenoh '**' is not in tail position in %r - MQTT may reject", key_expr)
            out.append("#")
        elif seg == "*":
            out.append("+")
        else:
            out.append(seg)
    return "/".join(out)


def _qos_and_retain_for(topic: str) -> tuple[int, bool]:
    """Look up the default QoS and retain flag for a topic.

    Resolves the suffix that follows the ``strands/...`` prefix and matches
    it against :data:`_TOPIC_POLICY`. Handles three layouts:

    - ``strands/broadcast``  -> suffix ``broadcast``
    - ``strands/safety/estop``  -> suffix ``safety/estop``
    - ``strands/{peer}/{topic}/...``  -> suffix ``{topic}/...``

    Topics with no entry in the policy get ``(0, False)``. Topics flagged as
    ``"DROP"`` return ``(-1, False)`` so callers can short-circuit.
    """
    if not topic.startswith("strands/"):
        return 0, False

    # Camera S3-reference metadata is cloud-relevant and small: publish it at
    # QoS 0 (no retain) rather than letting the ``camera`` DROP entry swallow
    # it. The heavy JPEG frames themselves never reach ``put`` (they go to S3).
    if _is_camera_ref(topic):
        return 0, False

    rest = topic[len("strands/") :]
    if not rest:
        return 0, False

    rest_segments = rest.split("/")
    first = rest_segments[0]

    # Two distinct topic layouts in the strands-robots scheme. They MUST NOT
    # be tried as a fallback chain - a peer_id that happens to be named
    # "broadcast" or "safety" must NOT pick up the top-level policy entry.
    #
    #  (a) Top-level system topics - first segment IS the kind:
    #  strands/broadcast
    #  strands/safety/estop
    #
    #  (b) Per-peer topics - first segment is the peer_id, topic kind
    #  starts at segment 1:
    #  strands/{peer}/{kind}  (e.g. presence, state, cmd)
    #  strands/{peer}/{kind}/{sub}  (e.g. lidar/summary, response/{turn})
    #
    # We resolve the layout by checking whether *first* is one of the
    # reserved top-level kinds. The set is small and closed - extending
    # the topic scheme means extending this set.
    _TOP_LEVEL_KINDS = {"broadcast", "safety"}

    if first in _TOP_LEVEL_KINDS:
        # Layout (a) - match suffixes that include the first segment.
        for n in range(len(rest_segments), 0, -1):
            candidate = "/".join(rest_segments[:n])
            entry = _TOPIC_POLICY.get(candidate)
            if entry is not None:
                qos_or_drop, retain = entry
                if qos_or_drop == "DROP":
                    return -1, False
                return int(qos_or_drop), retain
        return 0, False

    # Layout (b) - first segment is a peer_id; skip it.
    if len(rest_segments) < 2:
        return 0, False

    # Match suffixes from rest_segments[1:] longest-first.
    for n in range(len(rest_segments), 1, -1):
        candidate = "/".join(rest_segments[1:n])
        entry = _TOPIC_POLICY.get(candidate)
        if entry is not None:
            qos_or_drop, retain = entry
            if qos_or_drop == "DROP":
                return -1, False
            return int(qos_or_drop), retain

    return 0, False


def _is_camera_ref(topic: str) -> bool:
    """True for camera S3-reference metadata topics (``strands/<peer>/camera/<cam>/ref``).

    Unlike raw JPEG frames (which hit MQTT's 128 KB cap and are dropped), a
    ``/ref`` message carries only the presigned S3 key + frame shape -- a few
    hundred bytes. It is exactly the pointer a cloud subscriber needs to fetch
    the offloaded frame, so it MUST traverse MQTT even though its topic sits
    under the otherwise-dropped ``camera/`` prefix.

    The exemption is granted on the topic's SHAPE, not on substrings of it: the
    family segment must itself be ``camera``, and a camera name must sit between
    it and the ``ref`` tail. A camera name is taken from the robot's own
    ``config.cameras`` keys - a bare token at the doors that accept one
    (:func:`~strands_robots.utils.camera_token_error`), and whatever a config
    built by other means carries - so a substring test hands the exemption to
    topics that carry a frame rather than a pointer -- a camera named ``ref``
    publishes its inline base64 JPEG on ``strands/<peer>/camera/ref``, and a
    ``ref`` tail under any other family (``input/camera/ref``) reads as a
    pointer too. Both are the WAN payload the drop rule exists to refuse, and
    letting one through inflates broker billing exactly the way the tail-append
    match in :func:`~strands_robots.mesh.transport.bridge_transport._should_bridge`
    already refuses to.

    A multi-segment camera name keeps its exemption: the tail is matched, not
    the segment count, so ``camera/front/left/ref`` is still a pointer.
    """
    parts = topic.split("/")
    return len(parts) >= 5 and parts[0] == "strands" and parts[2] == "camera" and parts[-1] == "ref"


def _should_drop(topic: str) -> bool:
    """True if the topic's payload should never traverse MQTT (camera/input/hand).

    Camera S3-reference messages (``.../camera/<cam>/ref``) are exempt: they
    carry only the presigned key, not the frame, and must reach cloud
    subscribers. See :func:`_is_camera_ref`.
    """
    if _is_camera_ref(topic):
        return False
    parts = topic.split("/", 2)
    suffix = parts[2] if len(parts) == 3 else topic
    return suffix.startswith(_NEVER_BRIDGE_PREFIXES)


class IotMqttTransport:
    """Concrete :class:`MeshTransport` backed by AWS IoT Core MQTT5/mTLS.

    One instance manages exactly one ``awscrt.mqtt5.Client``. The client's
    ``client_id`` MUST equal the Thing name attached to the cert - the
    constructor enforces this so policy variable substitution works.

    Subscriptions are tracked in a dict keyed by topic_filter so
    :meth:`_unsubscribe` can locate them on undeclare.

    Thread safety
    -------------
    The ``awscrt`` library calls handlers on its own IO thread. We protect
    the subscription dict with a small lock; ``put`` is lock-free.

    Args:
        thing_name: AWS IoT Thing name, which MUST equal the cert's CN and the
            :class:`~strands_robots.mesh.core.Mesh` peer_id. Falls back to
            ``STRANDS_IOT_THING_NAME``.
        endpoint: The AWS IoT Core ATS endpoint. Falls back to
            ``STRANDS_IOT_ENDPOINT``.
        cert_dir: Directory holding the cert, private key and root CA. Falls
            back to ``STRANDS_IOT_CERT_DIR``, then ``~/.strands_robots/iot``.
        ca_file: Path to the root CA file. Falls back to
            ``STRANDS_IOT_CA_FILE``, then ``AmazonRootCA1.pem`` in ``cert_dir``.
        connect_timeout: Seconds :meth:`connect` waits for CONNACK before
            reporting the broker unreachable. Only a positive finite number can
            be honored, and the value is refused here rather than at the wait:
            :meth:`connect` spends it on ``threading.Event.wait``, which returns
            ``False`` at once for ``0``, a negative and ``nan`` - so a broker
            that is connecting normally is reported as having "timed out", and
            its client is torn down. ``inf`` and a numeric string instead raise
            ``OverflowError`` / ``TypeError`` out of a method documented to
            return ``bool``, after the MQTT5 client has been started and with
            nothing left to stop it. ``None`` is not a spelling for an unbounded
            wait: ``Event.wait(None)`` blocks forever while :meth:`connect`
            holds the instance lock, so :meth:`close` can never run. This is the
            domain the two remote-inference clients already apply to the
            parameter of the same name (#1984).

    Raises:
        ValueError: If ``connect_timeout`` is not a positive finite number.
    """

    def __init__(
        self,
        thing_name: str | None = None,
        endpoint: str | None = None,
        cert_dir: str | None = None,
        ca_file: str | None = None,
        connect_timeout: float = 15.0,
    ) -> None:
        # Refuse a wait budget that names no budget while the caller still holds
        # the value. ``connect()`` spends it on ``Event.wait``, whose reaction to
        # an unusable one is indistinguishable from an unreachable broker: it
        # returns ``False`` immediately, and ``connect()`` then logs "timed out
        # after 0.0s" and stops a client that was connecting fine. Placed here,
        # and not beside the wait, so the refusal also precedes the ``awsiot``
        # import inside ``connect()`` - the same mistake then reports identically
        # with and without the [mesh-iot] extra installed.
        if error := positive_finite_number_error(connect_timeout, "connect_timeout", type(self).__name__):
            raise ValueError(error)
        self._thing_name = thing_name or os.getenv("STRANDS_IOT_THING_NAME", "")
        self._endpoint = endpoint or os.getenv("STRANDS_IOT_ENDPOINT", "")
        self._cert_dir = Path(cert_dir or os.getenv("STRANDS_IOT_CERT_DIR") or Path.home() / ".strands_robots" / "iot")
        self._ca_file = ca_file or os.getenv("STRANDS_IOT_CA_FILE") or str(self._cert_dir / "AmazonRootCA1.pem")
        self._connect_timeout = connect_timeout

        self._client: Any | None = None
        self._connected = threading.Event()
        self._lock = threading.Lock()
        # topic_filter -> list of handlers (multiple subs to same topic OK)
        self._handlers: dict[str, list[Callable[[Any], None]]] = {}
        # Direct Messaging state. The HTTPS client is built lazily on the first
        # send_direct so a transport that never addresses a peer opens no
        # second connection. ``_direct_forbidden`` remembers the peers whose
        # first direct attempt came back 403 so Mesh decides the publish
        # fallback once per peer per connection; ``_unmatched_inbound`` counts
        # the messages no filter claimed (see ``_on_publish_received``).
        self._direct_client: _X509DirectClient | _SigV4DirectClient | None = None
        self._direct_lock = threading.Lock()
        self._direct_forbidden: set[str] = set()
        # Bumped on every CONNACK. A policy change reaches a client at its
        # next connect, so anything remembered about "what this identity may
        # do" (the 403 memos here and at the Mesh layer) is scoped to one
        # generation and compared against this number.
        self.connection_generation = 0
        self._unmatched_inbound = 0
        self.direct_stats: dict[str, int] = {"sent": 0, "delivered": 0, "failed": 0}
        self._sdk_too_old_reported = False
        # The topics handed to the client lately and when: a DISCONNECT that
        # follows a publish within ``DISCONNECT_AFTER_PUBLISH_WINDOW_S`` is how
        # AWS IoT answers a publish the policy does not grant, and the topics
        # are the only clue. The broker's DISCONNECT lands 50 to 100 ms after
        # the offending publish, by which time a 10 Hz state loop has published
        # again on a granted topic, so every topic inside the window is named,
        # newest first. Warned once per such set (``_publish_disconnect_warned``).
        self._recent_publishes: deque[tuple[float, str]] = deque(maxlen=16)
        self._publish_disconnect_warned: set[tuple[str, ...]] = set()
        # Set by close() before the client is stopped: the disconnect that
        # follows is this process's own doing, not the broker's verdict.
        self._closing = threading.Event()

    # Lifecycle

    def connect(self) -> bool:
        """Open the MQTT5 client and wait for CONNACK.

        Returns ``True`` once connected, ``False`` if the SDK is missing,
        configuration is invalid, or the broker is unreachable within
        ``connect_timeout`` seconds.

        Every failure is reported through that return value, including one where
        tearing the half-open client down is itself what fails: a ``stop()``
        that raises is recorded at debug and does not replace the ``False``.
        """
        with self._lock:
            if self._client is not None and self._connected.is_set():
                return True

            # Reconnect after a broker drop: ``_connected`` is clear but a
            # stale client object may linger (its IO thread + socket still
            # open). Stop it before building a new one, otherwise every retry
            # leaks a client and duplicates inbound delivery.
            if self._client is not None:
                try:
                    self._client.stop()
                except Exception as exc:
                    logger.debug("stopping stale MQTT client before reconnect: %s", exc)
                self._client = None

            try:
                from awsiot import mqtt5_client_builder
            except ImportError:
                logger.error(
                    "awsiotsdk not installed - IoT transport disabled. "
                    "Install with: pip install 'strands-robots[mesh-iot]'"
                )
                return False

            # Validate config
            if not self._thing_name:
                logger.error(
                    "STRANDS_IOT_THING_NAME is required for IoT transport "
                    "(must match the AWS IoT Thing name attached to the cert)"
                )
                return False
            if not self._endpoint:
                logger.error("STRANDS_IOT_ENDPOINT is required for IoT transport")
                return False

            cert_path = self._cert_dir / f"{self._thing_name}.cert.pem"
            key_path = self._cert_dir / f"{self._thing_name}.private.key"
            ca_path = Path(self._ca_file)
            for p, label in [
                (cert_path, "certificate"),
                (key_path, "private key"),
                (ca_path, "CA file"),
            ]:
                if not p.exists():
                    logger.error("IoT %s not found: %s", label, p)
                    return False

            self._connected.clear()
            self._closing.clear()  # a re-connect after close() judges its disconnects afresh
            # mtls_from_path (corrupt PEM -> AwsCrtError) and start() can raise
            # synchronously. Contain them here and return False: the mesh must
            # stay OFF rather than crash the host (and, in bridge mode, leave
            # the already-acquired Zenoh session dangling).
            try:
                self._client = mqtt5_client_builder.mtls_from_path(
                    endpoint=self._endpoint,
                    cert_filepath=str(cert_path),
                    pri_key_filepath=str(key_path),
                    ca_filepath=str(ca_path),
                    client_id=self._thing_name,  # MUST match Thing name
                    # 30 s instead of the SDK's 1200 s: with direct messaging an
                    # offline robot is meant to answer 404 in one round trip,
                    # and the broker only learns a client is gone at the next
                    # missed keep-alive (about 1.5 intervals). One PINGREQ per
                    # 30 s is well under the metered message rate.
                    keep_alive_interval_sec=_KEEP_ALIVE_S,
                    on_lifecycle_connection_success=self._on_connection_success,
                    on_lifecycle_connection_failure=self._on_connection_failure,
                    on_lifecycle_disconnection=self._on_disconnection,
                    on_publish_received=self._on_publish_received,
                )
                self._client.start()
            except Exception as exc:
                logger.error("IoT MQTT client construction failed: %s", exc)
                if self._client is not None:
                    try:
                        self._client.stop()
                    except Exception as stop_exc:
                        # Best-effort teardown of a half-constructed client on the
                        # construction-failure path. The connect() has already
                        # failed and we return False regardless; a stop() error
                        # here (e.g. client never reached a startable state) must
                        # not mask the original failure. Log at debug and move on.
                        logger.debug("IoT client stop during failure cleanup: %s", stop_exc)
                    self._client = None
                return False
            ok = self._connected.wait(self._connect_timeout)
            if not ok:
                logger.error(
                    "IoT connection to %s timed out after %.1fs",
                    self._endpoint,
                    self._connect_timeout,
                )
                try:
                    self._client.stop()
                except Exception as stop_exc:
                    # Same contract as the construction-failure path above: the
                    # connect() has already failed and we return False
                    # regardless, so a stop() error here must not replace that
                    # report with a raise out of a method documented to return
                    # bool. Log at debug and move on.
                    logger.debug("IoT client stop after connect timeout: %s", stop_exc)
                self._client = None
                return False

            logger.info(
                "IoT mesh session opened (thing=%s, endpoint=%s)",
                self._thing_name,
                self._endpoint,
            )
            return True

    def close(self) -> None:
        """Disconnect and tear down the MQTT5 client. Idempotent.

        A ``stop()`` that raises does not prevent teardown - the client
        reference is dropped either way. That is what makes the failure worth
        recording rather than swallowing: nothing can reach that client
        afterwards to retry, so the WARNING is the only trace an operator gets
        of an IO thread and socket that may still be open.
        """
        # The direct HTTPS client is independent of the MQTT one: a transport
        # that only ever addressed peers (SigV4 agent, never connected) still
        # holds one, so it is released before the early return below.
        self._close_direct()
        with self._lock:
            if self._client is None:
                return
            # Raised before stop(): awscrt reports the stop through the same
            # lifecycle callback as a broker DISCONNECT, sometimes while stop()
            # is still running, and every 10 Hz publisher has a publish inside
            # the blame window at shutdown. Without this, each normal exit told
            # the owner to reprovision a healthy Thing.
            self._closing.set()
            try:
                self._client.stop()
            except Exception as exc:
                # The two connect()-side teardowns log this; close() is the
                # public one, and it is the only path whose visible report is a
                # success, so a silent swallow leaves "session closed" as the
                # sole record of a client that did not stop. Warn rather than
                # debug: the reference is dropped below either way, so nothing
                # can reach that client afterwards to retry.
                logger.warning(
                    "IoT MQTT client stop() failed during close (thing=%s): %s; "
                    "its IO thread and socket may still be open",
                    self._thing_name,
                    exc,
                )
            self._client = None
            self._connected.clear()
            self._handlers.clear()
            logger.info("IoT mesh session closed (thing=%s)", self._thing_name)

    # Inspection

    def is_alive(self) -> bool:
        """True if the MQTT client is connected."""
        return self._client is not None and self._connected.is_set()

    @property
    def thing_name(self) -> str:
        """The AWS IoT thing name this transport authenticates as, or ``""``
        if unset. Used as the MQTT client id and mTLS identity.
        """
        return self._thing_name or ""

    @property
    def unmatched_inbound(self) -> int:
        """Messages that arrived on a topic no subscription or direct key claimed."""
        return self._unmatched_inbound

    # Direct Messaging

    def direct_forbidden(self, peer_id: str) -> bool:
        """True once a direct send to *peer_id* came back 403 on this connection."""
        with self._direct_lock:
            return peer_id in self._direct_forbidden

    def _direct_inbound_keys(self, topic: str) -> tuple[str, ...]:
        """The registered filter keys an unsubscribed direct message on *topic* belongs to."""
        thing = self._thing_name
        if not thing:
            return ()
        if topic == f"strands/{thing}/cmd":
            return (topic,)
        if topic.startswith(f"strands/{thing}/response/"):
            return (f"strands/{thing}/response/#",)
        return ()

    def _cert_files_present(self) -> bool:
        return (self._cert_dir / f"{self._thing_name}.cert.pem").exists() and (
            self._cert_dir / f"{self._thing_name}.private.key"
        ).exists()

    def _direct_sender(self) -> _X509DirectClient | _SigV4DirectClient | None:
        with self._direct_lock:
            if self._direct_client is not None:
                return self._direct_client
            if not self._endpoint:
                return None
            mode = direct_auth_mode(self._cert_files_present())
            if mode == "x509":
                if not self._cert_files_present():
                    logger.warning(
                        "%s=x509 but no certificate for thing %r under %s - direct messaging unavailable",
                        DIRECT_AUTH_ENV_VAR,
                        self._thing_name,
                        self._cert_dir,
                    )
                    return None
                self._direct_client = _X509DirectClient(
                    self._endpoint,
                    str(self._cert_dir / f"{self._thing_name}.cert.pem"),
                    str(self._cert_dir / f"{self._thing_name}.private.key"),
                    self._ca_file,
                )
            else:
                self._direct_client = _SigV4DirectClient(self._endpoint)
            return self._direct_client

    def _close_direct(self) -> None:
        with self._direct_lock:
            client, self._direct_client = self._direct_client, None
            self._direct_forbidden.clear()
        if client is not None:
            client.close()

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
        """Deliver *data* on *key* to the one connected client *peer_id*.

        The :class:`~strands_robots.mesh.transport.base.DirectSender` contract:
        never raises, one :class:`DirectResult` per call. The HTTP outcome maps
        to the result's ``reason`` (404 offline, 403 forbidden, 429 throttled
        with one jittered retry, 413 too_large, 504 unconfirmed, any other 5xx
        or a socket failure ``error`` with one retry). A 403 is remembered per
        peer for this connection so the caller can decide its fallback once.

        The payload is encoded before anything is sent, and an unencodable one
        is reported through the same channel ``put`` uses; a payload over
        :data:`DIRECT_PAYLOAD_CAP` is refused here as ``too_large`` without a
        round trip.
        """
        started = time.monotonic()
        # ``timeout`` is the whole budget of this call: the confirmation
        # window asked of the broker, the socket timeouts and any retry all
        # fit inside it, so the caller's own deadline is never overrun.
        deadline = started + max(0.0, float(timeout))

        def _done(delivered: bool, reason: str, trace_id: str = "", detail: str = "") -> DirectResult:
            self.direct_stats["sent"] += 1
            self.direct_stats["delivered" if delivered else "failed"] += 1
            return DirectResult(
                delivered=delivered,
                reason=reason,
                latency_ms=(time.monotonic() - started) * 1000.0,
                trace_id=trace_id,
                detail=detail,
            )

        if not isinstance(peer_id, str) or not peer_id or peer_id.startswith("$") or len(peer_id) > 128:
            return _done(False, "error", detail="peer_id must be 1 to 128 characters and not start with '$'")
        try:
            encoded = json.dumps(data).encode()
        except Exception as exc:  # noqa: BLE001 - the encoder's raise set is payload-defined
            _report_unencodable_payload("MQTT direct", key, exc)
            return _done(False, "error", detail=f"payload not JSON encodable: {exc}")
        if len(encoded) > DIRECT_PAYLOAD_CAP:
            return _done(False, "too_large", detail=f"payload {len(encoded)} bytes exceeds {DIRECT_PAYLOAD_CAP}")

        sender = self._direct_sender()
        if sender is None:
            return _done(False, "unavailable", detail="no direct sender configured (endpoint or credentials missing)")

        for attempt in (0, 1):
            confirm_window = deadline - time.monotonic()
            try:
                if isinstance(sender, _X509DirectClient):
                    status, body = self._post_x509(
                        sender, peer_id, key, encoded, confirm, confirm_window, response_key, correlation, deadline
                    )
                    message, trace_id = ("", "") if status == 200 else _parse_direct_error_body(body)
                else:
                    status, body, trace_id = self._post_sigv4(
                        sender, peer_id, key, encoded, confirm, confirm_window, response_key, correlation, deadline
                    )
                    message = body.decode("utf-8", "replace")
            except TimeoutError as exc:
                return _done(False, "error", detail=f"TimeoutError: {exc}"[:200])
            except _SdkTooOld as exc:
                if not self._sdk_too_old_reported:
                    self._sdk_too_old_reported = True
                    logger.warning("direct messaging unavailable: %s", exc)
                return _done(False, "unavailable", detail=str(exc)[:200])
            except Exception as exc:  # noqa: BLE001 - socket, TLS, botocore: all map to "error"
                logger.debug("direct send to %s on %s failed (attempt %d): %s", peer_id, key, attempt, exc)
                if attempt == 0 and _budget_allows_retry(deadline):
                    time.sleep(0.05 + random.random() * 0.1)  # noqa: S311 - jitter, not security
                    continue
                return _done(False, "error", detail=f"{type(exc).__name__}: {exc}"[:200])

            reason = _direct_reason_for_status(status)
            if reason == "":
                return _done(True, "", trace_id)
            if reason == "forbidden":
                with self._direct_lock:
                    self._direct_forbidden.add(peer_id)
            retryable = reason == "throttled" or (reason == "error" and status >= 500)
            if retryable and attempt == 0 and _budget_allows_retry(deadline):
                time.sleep(0.1 + random.random() * 0.2)  # noqa: S311 - jitter, not security
                continue
            return _done(False, reason, trace_id, message)
        return _done(False, "error", detail="exhausted retries")  # pragma: no cover - loop returns

    @staticmethod
    def _direct_query(key: str, confirm: bool, timeout: float, response_key: str | None) -> dict[str, str]:
        query = {
            "topic": key,
            "contentType": "application/json",
            "confirmation": "true" if confirm else "false",
        }
        if confirm:
            query["timeout"] = str(_confirm_timeout_seconds(timeout))
        if response_key:
            query["responseTopic"] = response_key
        return query

    def _post_x509(
        self,
        sender: _X509DirectClient,
        peer_id: str,
        key: str,
        encoded: bytes,
        confirm: bool,
        timeout: float,
        response_key: str | None,
        correlation: str | None,
        deadline: float,
    ) -> tuple[int, bytes]:
        path = f"/connections/{urllib.parse.quote(peer_id, safe='')}/messages?" + urllib.parse.urlencode(
            self._direct_query(key, confirm, timeout, response_key), quote_via=urllib.parse.quote
        )
        headers = {
            "Content-Type": "application/json",
            "x-amz-mqtt5-payload-format-indicator": "UTF8_DATA",
            "x-amz-mqtt5-user-properties": base64.b64encode(json.dumps(_DIRECT_USER_PROPERTIES).encode()).decode(),
        }
        if correlation:
            headers["x-amz-mqtt5-correlation-data"] = base64.b64encode(correlation.encode()).decode()
        return sender.post(path, encoded, headers, deadline=deadline)

    def _post_sigv4(
        self,
        sender: _SigV4DirectClient,
        peer_id: str,
        key: str,
        encoded: bytes,
        confirm: bool,
        timeout: float,
        response_key: str | None,
        correlation: str | None,
        deadline: float,
    ) -> tuple[int, bytes, str]:
        params: dict[str, Any] = {
            "clientId": peer_id,
            "topic": key,
            "contentType": "application/json",
            "payloadFormatIndicator": "UTF8_DATA",
            "userProperties": _DIRECT_USER_PROPERTIES,
            "payload": encoded,
            "confirmation": bool(confirm),
        }
        if confirm:
            params["timeout"] = _confirm_timeout_seconds(timeout)
        if response_key:
            params["responseTopic"] = response_key
        if correlation:
            params["correlationData"] = base64.b64encode(correlation.encode()).decode()
        return sender.send(params, deadline=deadline)

    # Pub/Sub

    def put(self, key: str, data: dict[str, Any]) -> None:
        """Publish *data* to *key*. Fire-and-forget.

        Per-topic QoS and retain flags come from :data:`_TOPIC_POLICY`.
        Topics in :data:`_NEVER_BRIDGE_PREFIXES` (camera/input/hand) are
        silently dropped - they belong on Zenoh-LAN, not MQTT-WAN.

        A broker or client failure stays at DEBUG: it is transient and the next
        tick retries it. A payload the JSON encoder refuses is not, so it is
        reported at ERROR once per topic through
        :func:`~strands_robots.mesh.session._report_unencodable_payload` - the
        same report the Zenoh leg emits, so a reader grepping the log for one
        transport's wording finds the other's.
        """
        if self._client is None or not self._connected.is_set():
            return

        if _should_drop(key):
            return

        qos, retain = _qos_and_retain_for(key)
        if qos < 0:
            return  # explicit DROP

        # Encoded BEFORE the publish attempt, and outside its handler: a payload
        # the encoder refuses can never be published, whereas a broker failure is
        # transient and the next tick retries it. Absorbing both in one DEBUG line
        # made a permanently-undeliverable message indistinguishable from a
        # dropped one. Hoisting the encode above the ``awscrt`` import also keeps
        # the two apart: an absent [mesh-iot] extra is not a bad payload.
        try:
            encoded = json.dumps(data).encode()
        except Exception as exc:  # noqa: BLE001 - the encoder's raise set is payload-defined
            _report_unencodable_payload("MQTT", key, exc)
            return

        try:
            from awscrt import mqtt5

            qos_enum = mqtt5.QoS.AT_MOST_ONCE if qos == 0 else mqtt5.QoS.AT_LEAST_ONCE
            self._recent_publishes.append((time.monotonic(), key))
            self._client.publish(
                mqtt5.PublishPacket(
                    topic=key,
                    payload=encoded,
                    qos=qos_enum,
                    retain=retain,
                )
            )
        except Exception as exc:
            logger.debug("MQTT put error on %s: %s", key, exc)

    def declare_subscriber(self, key_expr: str, handler: Callable[[Any], None]) -> Any:
        """Subscribe to *key_expr* (Zenoh form) translated to an MQTT topic filter.

        Multiple subscribers to the same filter are allowed - handlers are
        appended to a per-filter list. Each :class:`_MqttSubHandle` only
        removes its own handler on undeclare.
        """
        if self._client is None or not self._connected.is_set():
            raise RuntimeError("IoT MQTT client not connected")

        from awscrt import mqtt5

        topic_filter = _zenoh_to_mqtt_filter(key_expr)

        with self._lock:
            already_subscribed = topic_filter in self._handlers
            self._handlers.setdefault(topic_filter, []).append(handler)

        if not already_subscribed:
            try:
                self._client.subscribe(
                    mqtt5.SubscribePacket(
                        subscriptions=[
                            mqtt5.Subscription(
                                topic_filter=topic_filter,
                                qos=mqtt5.QoS.AT_LEAST_ONCE,
                            )
                        ]
                    )
                ).result(timeout=5)
            except Exception as exc:
                # Roll back the handler registration so a retry works cleanly.
                with self._lock:
                    self._handlers.get(topic_filter, []).remove(handler)
                    if not self._handlers.get(topic_filter):
                        self._handlers.pop(topic_filter, None)
                raise RuntimeError(f"MQTT subscribe to {topic_filter!r} failed: {exc}") from exc

        return _MqttSubHandle(self, topic_filter, handler)

    # Internal

    def _unsubscribe(self, topic_filter: str, handler: Any = None) -> None:
        """Remove *handler* for *topic_filter*; unsubscribe if last."""
        with self._lock:
            handlers = self._handlers.get(topic_filter)
            if not handlers:
                return
            if handler is not None:
                try:
                    handlers.remove(handler)
                except ValueError:
                    pass  # handler already gone
            else:
                handlers.pop()  # legacy fallback: remove last
            if handlers:
                return  # other subscribers still active
            self._handlers.pop(topic_filter, None)

        # Last handler removed - unsubscribe at the broker.
        if self._client is None:
            return
        try:
            from awscrt import mqtt5

            self._client.unsubscribe(mqtt5.UnsubscribePacket(topic_filters=[topic_filter])).result(timeout=5)
        except Exception as exc:
            logger.debug("MQTT unsubscribe error on %s: %s", topic_filter, exc)

    # Callbacks

    def _on_connection_success(self, data: Any) -> None:
        logger.info("IoT MQTT connected (thing=%s)", self._thing_name)
        self._connected.set()
        # A policy change is picked up at the next connect, so the per-peer 403
        # memo is scoped to one connection.
        with self._direct_lock:
            self._direct_forbidden.clear()
            self.connection_generation += 1

    def _on_connection_failure(self, data: Any) -> None:
        logger.warning("IoT MQTT connection failure: %s", data.exception)
        self._connected.clear()

    def _on_disconnection(self, data: Any) -> None:
        logger.info("IoT MQTT disconnected (thing=%s)", self._thing_name)
        self._connected.clear()
        if self._closing.is_set():
            return
        self._warn_if_publish_ended_the_session(data)

    def _warn_if_publish_ended_the_session(self, data: Any) -> None:
        """WARN once per set of topics when the broker ends the session right after a publish.

        AWS IoT does not refuse a publish the connected Thing's policy does
        not grant: it drops the MQTT session (DISCONNECT reason code 135, not
        authorized) and the client reconnects, so a robot that keeps
        publishing one ungranted topic lives in a connect/disconnect cycle
        with nothing above DEBUG to say why. The topics named here are those
        handed to the client within :data:`DISCONNECT_AFTER_PUBLISH_WINDOW_S`
        of the disconnect, newest first (measured: the DISCONNECT arrives 47
        to 74 ms after the publish, so the newest is not always the culprit);
        the usual cause is a child peer (``<thing>__<robot>``) on a certificate
        from before the child key space grant, which
        ``strands-robots iot reprovision <thing>`` attaches.
        """
        now = time.monotonic()
        recent = [(at, topic) for at, topic in self._recent_publishes if now - at <= DISCONNECT_AFTER_PUBLISH_WINDOW_S]
        if not recent:
            return
        recent.sort(key=lambda item: item[0], reverse=True)
        topics: list[str] = []
        for _at, topic in recent:
            if topic not in topics:
                topics.append(topic)
        key = tuple(topics)
        if key in self._publish_disconnect_warned:
            return
        self._publish_disconnect_warned.add(key)
        elapsed_ms = (now - recent[0][0]) * 1000.0
        packet = getattr(data, "disconnect_packet", None)
        reason = getattr(packet, "reason_code", None)
        reason_text = f", broker reason code {int(reason)}" if isinstance(reason, int) else ""
        logger.warning(
            "IoT MQTT session ended %.0f ms after publishing %s (thing=%s%s): AWS IoT drops the session on a "
            "publish the Thing's policy does not grant. A child peer (%s__<robot>) needs the strands/%s__*/* grant "
            "(policy strands-robot-children); run `strands-robots iot reprovision %s` to attach it, then restart "
            "this robot.",
            elapsed_ms,
            ", ".join(topics),
            self._thing_name,
            reason_text,
            self._thing_name,
            self._thing_name,
            self._thing_name,
        )

    def _on_publish_received(self, data: Any) -> None:
        """Route inbound messages to subscriber handlers via topic-filter match."""
        topic = data.publish_packet.topic
        payload = bytes(data.publish_packet.payload or b"")

        # Match topic against all registered filters. MQTT brokers route by
        # filter, but our handler-dict is keyed by the original filter - so
        # we need to test each registered filter for a topic match.
        with self._lock:
            matching = [(f, list(handlers)) for f, handlers in self._handlers.items() if _mqtt_topic_matches(f, topic)]
            if not matching:
                # A direct message needs no subscription on this side, so it
                # can arrive on a topic no filter claims. The two addressed
                # topics of the mesh scheme, this thing's ``cmd`` and its
                # ``response/#``, are routed to whichever handlers were
                # registered for those keys even when the broker-side
                # subscription is gone; anything else is counted and logged
                # at debug rather than dropped in silence.
                for key in self._direct_inbound_keys(topic):
                    handlers = self._handlers.get(key)
                    if handlers:
                        matching.append((key, list(handlers)))

        if not matching:
            self._unmatched_inbound += 1
            logger.debug("IoT inbound on %s matched no subscription (thing=%s); dropped", topic, self._thing_name)
            return

        sample = _MqttSample(topic, payload, *_mqtt5_reply_properties(data.publish_packet))
        for _filter, handlers in matching:
            for handler in handlers:
                try:
                    handler(sample)
                except Exception as exc:
                    logger.debug("IoT handler error on %s: %s", topic, exc)


#: A retry is attempted only when at least this much of the budget is left:
#: less than that buys a request that cannot finish before the caller's deadline.
_RETRY_MIN_REMAINING_S = 0.25


def _budget_allows_retry(deadline: float) -> bool:
    return (deadline - time.monotonic()) >= _RETRY_MIN_REMAINING_S


def _direct_reason_for_status(status: int) -> str:
    """Map an HTTP status of the Direct Messaging API to a :data:`DIRECT_REASONS` entry."""
    if status == 200:
        return ""
    return _DIRECT_STATUS_REASON.get(status, "error")


def _mqtt5_reply_properties(packet: Any) -> tuple[str | None, str | None]:
    """Read the Response Topic and Correlation Data off an inbound PUBLISH.

    Both are optional MQTT5 properties; ``awscrt`` exposes them as
    ``publish_packet.response_topic`` (``str | None``) and
    ``publish_packet.correlation_data`` (``bytes`` or ``str`` depending on the
    SDK version, ``None`` when unset). The pair is normalised to text so
    :class:`_MqttSample` hands the Mesh handlers one shape. Correlation bytes
    that are not UTF-8 are dropped rather than raised: the field is an opaque
    echo for the sender, and a handler that cannot read it replies on the
    computed key exactly as it would for a message that carried none.

    Args:
        packet: The ``awscrt.mqtt5.PublishPacket`` of the inbound message.

    Returns:
        ``(response_topic, correlation_data)``, each ``None`` when absent.
    """
    response_topic = getattr(packet, "response_topic", None)
    if response_topic is not None and not isinstance(response_topic, str):
        response_topic = None
    correlation: Any = getattr(packet, "correlation_data", None)
    if isinstance(correlation, (bytes, bytearray, memoryview)):
        try:
            correlation = bytes(correlation).decode("utf-8")
        except UnicodeDecodeError:
            correlation = None
    elif correlation is not None and not isinstance(correlation, str):
        correlation = None
    return response_topic, correlation


def _mqtt_topic_matches(filter_: str, topic: str) -> bool:
    """True if MQTT *topic* matches the topic-filter *filter_*.

    Implements the standard MQTT v5 wildcard semantics:

    - ``+`` matches exactly one topic level
    - ``#`` matches zero or more trailing topic levels (must be at end)
    - other segments must match literally
    """
    f_parts = filter_.split("/")
    t_parts = topic.split("/")

    for i, fp in enumerate(f_parts):
        if fp == "#":
            # Tail wildcard - matches everything from here, including zero
            # remaining segments.
            return True
        if i >= len(t_parts):
            return False
        if fp == "+":
            continue
        if fp != t_parts[i]:
            return False

    # Filter exhausted - topic matches iff topic is also exhausted.
    return len(t_parts) == len(f_parts)
