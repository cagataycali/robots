"""The rosbridge transport every caller in this package forwards through.

The WebSocket mechanics behind ``use_rosbridge``: one long-lived
``roslibpy.Ros`` per ``(host, port)``, the graph introspection that reads the
``rosapi`` node's services, and the topic / service / host / port domains this
transport can carry. Two surfaces need them - the agent-facing
:mod:`~strands_robots.tools.use_rosbridge` tool and
:class:`~strands_robots.mesh.rosbridge_robot.RosbridgeRobot`, which drives a
ROS 1 (or remote) mobile base over the same WebSocket - so they live here rather
than inside one of the two callers.

They lived in the tool, and the mesh robot imported the ``@tool`` (and two of
its private names) to reach them. That is an import from the ``drivers|mesh``
layer up into ``tools``: a library class whose transport was only reachable
through an agent entry point, so a programmatic caller paid for the tool
decorator, the tool's argument envelope and the tool's module import to open a
WebSocket.

Requirements:
    ``pip install "strands-robots[rosbridge]"`` (roslibpy). The robot side runs
    ``rosbridge_server`` (with ``rosapi``) - standard in every rosbridge
    install. rosbridge is unauthenticated by default: use on trusted networks.
    No ROS environment is needed on this machine, which is what distinguishes
    this transport from the in-process ``rclpy`` one
    (:mod:`strands_robots.tools.use_ros`): ROS 1 robots are reachable, and so is
    a robot across a network from macOS or CI.

Graph introspection uses the ``rosapi`` node's services. Interface types are
ROS1-style two-segment names (``geometry_msgs/Twist``); field payloads are plain
JSON dicts, exactly as rosbridge transmits them.

**The operator gate is an argument, not a default.** A ``publish`` or a
``service_call`` onto a blocklisted surface has to reach
:func:`strands_robots._command_gate.gate_command` whichever caller asked - that
is the whole point of one shared blocklist - so :func:`rosbridge_action` takes
the gate as a required ``gate`` argument and consults it at one fixed point in
the flow: after the backend probe, so a bridge that cannot be reached never
prompts, and before the WebSocket is dialed, so a human deciding does not hold
the connection lock and no publisher is advertised on a refusal. A caller cannot
forget it, and cannot move it.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from strands_robots.utils import dial_host_error, tcp_port_error

logger = logging.getLogger(__name__)

#: How a caller asks its operator about one command. Called with the topic or
#: service the command is aimed at, and returns a refusal to report or ``None``
#: to let the command through - the shape
#: :func:`strands_robots._command_gate.gate_command` already returns.
CommandGate = Callable[[str, str], str | None]

#: The name every caller keys its operator prompt and audit row with. The gate
#: builds the interrupt id ``<tool>-command-approval`` and the audit source
#: ``<tool>_tool`` from it, so two spellings would split one incident's audit
#: trail in two and an operator would be asked the same question under two
#: names. A publish over rosbridge is the same physical command whether an agent
#: called ``use_rosbridge`` or a :class:`RosbridgeRobot`'s drive tool did.
GATE_TOOL = "use_rosbridge"


def never_gated(kind: str, target: str) -> str | None:
    """The gate for a verb that carries no command: the reads and the probes.

    :func:`rosbridge_action` requires a gate so that no caller can command a
    blocklisted surface by forgetting one, and it consults that gate for the two
    verbs that carry a command. ``status``, ``list_topics``, ``list_services``
    and ``echo`` only read, so none of them can move a robot - an operator asked
    about one would be asked about nothing. This is the one spelling of that, so
    a caller wiring a read-only verb cannot invent a permissive gate of its own.
    """
    del kind, target
    return None


# Graph names: same allowlist posture as use_ros. Types are ROS1 two-segment.
_NAME_RE = re.compile(r"^[A-Za-z0-9_/~]+\Z")
_TYPE_RE = re.compile(r"^[A-Za-z0-9_]+/[A-Za-z0-9_]+\Z")

# The host allowlist is this transport's own narrowing, applied *after* the
# shared ``dial_host_error`` domain - the same two stages the port half of the
# address beside it already has, where ``tcp_port_error`` establishes the 16-bit
# space and ``_transport_port_error`` places the transport's own ceiling in it.
#
# A pattern can only be offered a string. Matched against the caller's value
# directly this raised ``TypeError`` from ``re`` for every non-string host -
# ``9090``, ``True``, ``b"localhost"`` - out of a tool whose every other refusal
# is a result dict, and out of a constructor documented to report a malformed
# host as ``ValueError``: the same defect the port ceiling below records, with
# ``re``'s message in place of a bare ``assert``, naming neither the tool nor the
# parameter. The port half never had it, because a shared domain graded the value
# before anything spent it.
#
# With the shared domain ahead of it the value reaching this pattern is always a
# string a websocket URI can carry, so what is left here is the narrower question
# of which of those hostnames this ROS transport admits - a posture that belongs
# beside the transport, not to the domain every dialled host in the package
# shares.
_HOST_RE = re.compile(r"^[A-Za-z0-9._-]+\Z")

# The top of the 16-bit port space is a legal TCP port that this transport cannot
# address. autobahn builds the WebSocket URL behind roslibpy with
# ``assert port is None or (type(port) == int and port in range(0, 65535))``
# (``autobahn/websocket/util.py``), and ``range(0, 65535)`` stops one short of
# 65535 - so the shared owner accepts the port, the kernel would bind it, and the
# transport then refuses it with a bare ``assert`` carrying an empty message,
# raised out of a function annotated ``-> dict[str, Any]``. An agent driving the
# tool got an exception where every other refusal is a result dict, and the
# exception named neither the tool nor the parameter.
#
# The narrower domain is therefore declared here and refused ahead of the backend
# probe, for the same reason the numeric options are: the caller learns the same
# thing whether or not roslibpy is installed, and no socket is dialed first. It is
# deliberately narrower than ``tcp_port_error``, which keeps the whole port space
# because that is what a port *is* - this bound belongs to one transport, not to
# the domain, and lives beside the transport that has it.
#
# Under ``python -O`` the assert is stripped and the transport carries 65535, but
# refusing it uniformly is the honest contract: the tool cannot promise a port
# whose acceptance depends on an interpreter flag.
_TRANSPORT_MAX_PORT = 65534


def _transport_port_error(port: int, param: str, context: str) -> str | None:
    """Error text when the rosbridge transport cannot address ``port``.

    Applied after :func:`strands_robots.utils.tcp_port_error`, which establishes
    that ``port`` is an ``int`` in the 16-bit space, so this only has to place it
    against the transport's own ceiling. Shared with
    :class:`strands_robots.mesh.rosbridge_robot.RosbridgeRobot`, which reaches
    this transport through this module, so the two cannot disagree about which
    ports it can carry.

    Args:
        port: A port already accepted by the shared 16-bit domain.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it - the
            requested action for an agent tool, or the class name for a
            constructor parameter.

    Returns:
        An error message, or ``None`` when the transport can address the port.
    """
    if port > _TRANSPORT_MAX_PORT:
        return (
            f"{context}: {param} {port!r} is a legal TCP port that the rosbridge WebSocket "
            f"transport cannot address (it addresses 1-{_TRANSPORT_MAX_PORT}; autobahn's "
            "URL builder excludes the top of the range)"
        )
    return None


_INSTALL_HINT = (
    "roslibpy is not importable - install the rosbridge extra: "
    'pip install "strands-robots[rosbridge]". The rosbridge transport is pure '
    "pip (WebSocket); no ROS environment is needed on this machine."
)

_ACTIONS = frozenset({"status", "list_topics", "list_services", "echo", "publish", "service_call"})


class _RosbridgeBackend:
    """Process-wide cache of live roslibpy connections, keyed by (host, port).

    Tool calls are stateless; the WebSocket underneath is reused across calls
    (the rosbridge analogue of use_ros's single long-lived rclpy node). All
    access is serialised through ``lock``.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._connections: dict[tuple[str, int], Any] = {}
        self._available: bool | None = None

    def available(self) -> bool:
        if self._available is None:
            try:
                import roslibpy  # noqa: F401

                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def connect(self, host: str, port: int, timeout: float) -> Any:
        """Return a live connection to host:port, dialing or awaiting reconnect.

        One roslibpy.Ros is created per (host, port) for the process lifetime
        and NEVER discarded: its ReconnectingClientFactory re-dials a dropped
        WebSocket by itself (observed live), while a freshly constructed Ros
        in a process that has seen reconnect churn can fail to connect at all
        (roslibpy/Twisted limitation, observed live). Keeping the one object
        is therefore both the reliable and the cheap choice - a bridge that is
        down costs one retrying factory with exponential backoff, not a storm.
        Calling ros.terminate() is never an option: it stops the process-wide,
        non-restartable Twisted reactor and would break every connection in
        this process.
        """
        import roslibpy

        # The client builds its WebSocket URL before it dials, and that builder
        # gates the port on type IDENTITY rather than isinstance:
        #
        #     assert port is None or (type(port) == int and port in range(0, 65535))
        #         - autobahn/websocket/util.py:85 (autobahn 26.7.1)
        #
        # So every int SUBCLASS is refused at every value, including the default
        # 9090 - an IntEnum read from a settings module is the realistic case -
        # and the refusal is a bare AssertionError with an empty message, raised
        # from the constructor below and therefore outside the try that converts
        # a failed dial. The value is legal and dials exactly as the equal plain
        # int does, so it is normalized here rather than refused: carrying it is
        # a capability this tool already advertises through its ``port: int``.
        # Ahead of the cache read on purpose - the key is then a plain int
        # whatever flavour arrived, so one (host, port) is one connection
        # regardless of which type reached it first, and the annotation on
        # ``_connections`` is true rather than aspirational.
        port = int(port)

        ros = self._connections.get((host, port))
        if ros is None:
            ros = roslibpy.Ros(host=host, port=port)
            self._connections[(host, port)] = ros
            try:
                ros.run(timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - roslibpy raises library-specific errors; all mean "not connected yet"
                raise TimeoutError(
                    f"could not connect to rosbridge at ws://{host}:{port} within {timeout}s "
                    f"- is rosbridge_server running? ({exc})"
                ) from exc
            if not getattr(ros, "is_connected", False):
                raise TimeoutError(
                    f"could not connect to rosbridge at ws://{host}:{port} within {timeout}s "
                    "- is rosbridge_server running?"
                )
            return ros
        if getattr(ros, "is_connected", False):
            return ros
        # Measured on time.monotonic(): a wall-clock step during the wait moves
        # the deadline by the size of the step, reporting a reconnect that is
        # still in progress as a timeout (forward) or waiting far past the
        # caller's budget (backward).
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if getattr(ros, "is_connected", False):
                return ros
            time.sleep(0.05)
        raise TimeoutError(
            f"rosbridge at ws://{host}:{port} did not reconnect within {timeout}s "
            "- is rosbridge_server running? (the connection keeps retrying in the background)"
        )

    @property
    def lock(self) -> threading.RLock:
        return self._lock


_backend = _RosbridgeBackend()


# Rendered in place of a topic's type when rosapi's reply named none for it.
TYPE_NOT_REPORTED = "type not reported by rosapi"


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": f"use_rosbridge: {text}"}]}


def _rosapi_call(ros: Any, service: str, srv_type: str, values: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Call a service over rosbridge and return the response as a plain dict."""
    import roslibpy

    svc = roslibpy.Service(ros, service, srv_type)
    request = roslibpy.ServiceRequest(dict(values))
    try:
        return dict(svc.call(request, timeout=timeout))
    except Exception as exc:  # noqa: BLE001 - roslibpy raises library-specific errors; convert at the boundary
        raise TimeoutError(
            f"service {service} call failed via rosbridge: {exc} (is rosbridge_server running with rosapi?)"
        ) from exc


def _list_topics(ros: Any, timeout: float) -> str:
    """List every topic rosapi reports, with its type where rosapi reports one.

    ``rosapi/Topics`` answers with ``topics`` and ``types`` as index-paired
    arrays, but only ``topics`` is guaranteed: roslibpy's own client asserts
    ``"topics" in result`` and reads that array alone
    (:meth:`roslibpy.Ros.get_topics`). Pairing the two by index would drop
    every topic past the end of a short ``types``, and drop all of them when
    ``types`` is absent - reporting an empty graph to a caller who asked which
    topics exist. The count of topics is the answer, so each one is listed
    either way and a topic rosapi gave no type for says so.
    """
    resp = _rosapi_call(ros, "/rosapi/topics", "rosapi/Topics", {}, timeout)
    names = list(resp.get("topics", []))
    types = list(resp.get("types", []))
    if len(types) != len(names):
        logger.warning(
            "rosapi /rosapi/topics reported %d topic(s) and %d type(s); every topic is listed "
            "and the ones rosapi named no type for read as [%s]",
            len(names),
            len(types),
            TYPE_NOT_REPORTED,
        )
    pairs = sorted((name, types[i] if i < len(types) else TYPE_NOT_REPORTED) for i, name in enumerate(names))
    return "\n".join(f"{name} [{type_}]" for name, type_ in pairs)


def _list_services(ros: Any, timeout: float) -> str:
    resp = _rosapi_call(ros, "/rosapi/services", "rosapi/Services", {}, timeout)
    return "\n".join(sorted(resp.get("services", [])))


def _resolve_topic_type(ros: Any, topic: str, timeout: float) -> str | None:
    resp = _rosapi_call(ros, "/rosapi/topic_type", "rosapi/TopicType", {"topic": topic}, timeout)
    return resp.get("type") or None


def _echo(ros: Any, topic: str, msg_type: str, timeout: float, count: int) -> list[dict[str, Any]]:
    import roslibpy

    received: list[dict[str, Any]] = []
    done = threading.Event()

    def _on_message(message: dict[str, Any]) -> None:
        received.append(dict(message))
        if len(received) >= count:
            done.set()

    sub = roslibpy.Topic(ros, topic, msg_type)
    sub.subscribe(_on_message)
    try:
        done.wait(timeout)
    finally:
        sub.unsubscribe()
    return received[:count]


def _service_call(ros: Any, service: str, srv_type: str, fields: dict[str, Any], timeout: float) -> dict[str, Any]:
    return _rosapi_call(ros, service, srv_type, fields, timeout)


def rosbridge_action(
    action: str,
    *,
    host: str = "localhost",
    port: int = 9090,
    topic: str | None = None,
    service: str | None = None,
    type: str | None = None,
    fields: dict[str, Any] | None = None,
    timeout: float = 5.0,
    count: int = 1,
    rate: float = 10.0,
    gate: CommandGate,
) -> dict[str, Any]:
    """Run one rosbridge action against the shared WebSocket connection.

    Args:
        action: One of ``status``, ``list_topics``, ``list_services``,
            ``echo``, ``publish``, ``service_call``.
        host: rosbridge server hostname or IP. Held to the shared domain every
            dialled host in this package shares, then to this transport's own
            narrower allowlist.
        port: rosbridge WebSocket port (default 9090).
        topic: Topic name (``echo``, ``publish``). Held to the same name rule
            as ``service``.
        service: Service name (``service_call``). Held to the same name rule
            as ``topic``.
        type: ROS1 two-segment interface type, e.g. ``geometry_msgs/Twist``.
            Required for ``publish`` - a message cannot be built without it -
            and auto-resolved for ``echo`` when omitted.
        fields: JSON field dict (``publish`` message / ``service_call`` request).
        timeout: Seconds for the WebSocket dial, sample collection, or a
            service call. A positive finite number of seconds; every action
            dials the bridge, so every action reads it.
        count: Messages to echo or publish. A positive integer; it is consumed
            as a ``range()`` bound, so ``0`` publishes nothing and a float or a
            numeric string cannot be honored.
        rate: Publish rate in Hz. A positive finite number - the inter-message
            period is ``1 / rate``, so ``0``, a negative value, ``nan`` and
            ``inf`` all leave the burst unthrottled rather than paced.
        gate: The operator gate for the two verbs that carry a command,
            consulted with the verb and the topic or service it is aimed at.
            Required: a caller that forgot it would carry a command to a
            blocklisted drive surface with no prompt, which is the defect
            :mod:`strands_robots._command_gate` exists to prevent. The numeric
            domains of ``timeout`` / ``count`` / ``rate`` belong to the caller
            too - an agent tool reports a malformed option, while a mesh bridge
            such as :class:`~strands_robots.mesh.RosbridgeRobot` has already
            refused one at its own seam, naming the verb its caller invoked.

    Returns:
        A Strands tool result dict ``{"status": ..., "content": [{"text": ...}]}``.
    """
    fields = fields or {}

    if (host_error := dial_host_error(host, "host", action)) is not None:
        return _err(host_error)
    if not _HOST_RE.match(host):
        return _err(f"invalid host: {host!r}")
    if (port_error := tcp_port_error(port, "port", action)) is not None:
        return _err(port_error)
    if (transport_error := _transport_port_error(port, "port", action)) is not None:
        return _err(transport_error)
    if topic is not None and not _NAME_RE.match(topic):
        return _err(f"invalid topic name: {topic!r}")
    if service is not None and not _NAME_RE.match(service):
        return _err(f"invalid service name: {service!r}")
    if type is not None and not _TYPE_RE.match(type):
        return _err(f"invalid interface type: {type!r} (expected ROS1 pkg/Name like geometry_msgs/Twist)")

    if action not in _ACTIONS:
        return _err(f"unknown action: {action}")

    # The names an action cannot run without are graded ahead of the backend
    # probe, so the same caller mistake is reported identically whether or not
    # roslibpy is installed: a forgotten ``type`` must not dial the bridge for
    # the full timeout and then be reported as a bridge that did not reconnect.
    if action == "echo" and not topic:
        return _err("echo requires topic")
    if action == "service_call" and (not service or not type):
        return _err("service_call requires service and type")
    if action == "publish" and (not topic or not type):
        return _err("publish requires topic and type")

    if action == "status":
        if not _backend.available():
            return _ok("backend: none - " + _INSTALL_HINT)
        try:
            with _backend.lock:
                _backend.connect(host, port, timeout)
        except TimeoutError as exc:
            return _ok(f"backend: roslibpy; not connected - {exc}")
        return _ok(f"backend: roslibpy; connected to ws://{host}:{port}")

    if not _backend.available():
        return _err(_INSTALL_HINT)

    # The operator gate is consulted here - after the backend probe, so a bridge
    # that cannot be reached never prompts, and before the WebSocket is dialed, so
    # a human deciding does not hold the connection lock and no publisher is
    # advertised on a refusal. Both verbs that carry a command are covered; the
    # read-only actions are never gated. Both branches require a type, so the
    # condition mirrors theirs and an incomplete call is reported without asking
    # an operator about it.
    for kind, target in (("publish", topic), ("service_call", service)):
        if action == kind and target and type:
            refusal = gate(kind, target)
            if refusal is not None:
                return _err(refusal)

    try:
        with _backend.lock:
            ros = _backend.connect(host, port, timeout)

            if action == "list_topics":
                return _ok(_list_topics(ros, timeout))

            if action == "list_services":
                return _ok(_list_services(ros, timeout))

            # Each verb's names are non-empty here (refused above); spelling them
            # on the condition is what narrows them for the call.
            if action == "echo" and topic:
                msg_type = type or _resolve_topic_type(ros, topic, timeout)
                if not msg_type:
                    return _err(f"cannot resolve type for {topic}; pass type=pkg/Name")
                import json

                samples = _echo(ros, topic, msg_type, timeout, count)
                body = json.dumps(samples, indent=2, default=str)
                note = (
                    "" if samples else f"\n(no messages within {timeout}s - topic may be silent or the type mismatched)"
                )
                return _ok(f"echo {topic} ({msg_type}):\n{body}{note}")

            if action == "service_call" and service and type:
                import json

                resp = _service_call(ros, service, type, fields, timeout)
                return _ok(f"response:\n{json.dumps(resp, indent=2, default=str)}")

            if action == "publish" and topic and type:
                _publish(ros, topic, type, fields, count, rate)
                return _ok(f"published {count} message(s) to {topic}")

            # Unreachable: action is validated against _ACTIONS above. Kept as
            # a defensive fallback because mypy cannot prove the if/elif
            # chain above is exhaustive from a runtime frozenset check.
            return _err(f"unknown action: {action}")  # pragma: no cover
    except TimeoutError as exc:
        return _err(str(exc))
    except (ImportError, KeyError, AttributeError, ValueError, TypeError, OSError) as exc:
        return _err(f"{action} failed: {exc}")


def _publish(ros: Any, topic: str, msg_type: str, fields: dict[str, Any], count: int, rate: float) -> None:
    import roslibpy

    pub = roslibpy.Topic(ros, topic, msg_type)
    pub.advertise()
    try:
        time.sleep(0.2)  # settle so rosbridge registers the publisher before the first send
        period = 1.0 / rate if rate > 0 else 0.0
        for _ in range(count):
            pub.publish(roslibpy.Message(dict(fields)))
            if period:
                time.sleep(period)
    finally:
        pub.unadvertise()


__all__ = ["CommandGate", "GATE_TOOL", "never_gated", "rosbridge_action"]
