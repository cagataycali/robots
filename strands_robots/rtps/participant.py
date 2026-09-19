"""The RTPS participant every caller in this package publishes through.

The DDS mechanics behind ``use_rtps``: one process-wide ``DomainParticipant``,
writers and readers cached per (topic, type), and the IDL sample builder that
turns a JSON field dict into a cyclonedds dataclass. Two surfaces need them -
the agent-facing :mod:`~strands_robots.tools.use_rtps` tool and
:class:`~strands_robots.mesh.rtps_robot.RtpsRobot`, which drives a ROS 2 base
over the same wire - so they live here, beside the mangling and the IDL bundle
they are built on, rather than inside one of the two callers.

They lived in the tool, and the mesh robot imported the ``@tool`` to reach them.
That is an import from the ``drivers|mesh`` layer up into ``tools``: a library
class whose DDS transport was only reachable through an agent entry point, so a
programmatic caller paid for the tool decorator, the tool's argument envelope
and the tool's module import to open a DDS writer.

**The operator gate is an argument, not a default.** A ``publish`` onto a
blocklisted surface has to reach :func:`strands_robots._command_gate.gate_command`
whichever caller asked - that is the whole point of one shared blocklist - so
:func:`rtps_action` takes the gate as a required ``gate`` argument and consults
it at one fixed point in the flow: after the backend probe, so a transport that
cannot reach the graph never prompts, and before the lock, so a human deciding
does not hold the process-wide DDS lock and no writer joins the graph on a
refusal. A caller cannot forget it, and cannot move it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
import typing
from collections.abc import Callable
from typing import Any

from strands_robots.rtps.mangling import dds_type_name, ros_topic_error

logger = logging.getLogger(__name__)

#: How a caller asks its operator about one publish. Called with the topic, and
#: returns a refusal to report or None to let the command through - the shape
#: :func:`strands_robots._command_gate.gate_command` already returns.
PublishGate = Callable[[str], str | None]

#: The name every caller keys its operator prompt and audit row with. The gate
#: builds the interrupt id ``<tool>-command-approval`` and the audit source
#: ``<tool>_tool`` from it, so two spellings would split one incident's audit
#: trail in two and an operator would be asked the same question under two
#: names. A publish over RTPS is the same physical command whether an agent
#: called ``use_rtps`` or an :class:`RtpsRobot`'s drive tool did.
GATE_TOOL = "use_rtps"


def never_gated(target: str) -> str | None:
    """The gate for a verb that carries no command: reading, and advertising.

    :func:`rtps_action` requires a gate so that no caller can publish to a
    blocklisted surface by forgetting one, and it consults that gate for
    ``publish`` alone. ``status``, ``types``, ``subscribe`` and ``echo`` only
    read, and ``advertise`` creates a publisher without writing a sample, so
    none of them can move a robot - an operator asked about one would be asked
    about nothing. This is the one spelling of that, so a caller wiring a
    read-only verb cannot invent a permissive gate of its own.
    """
    del target
    return None


class _RtpsBackend:
    """Process-wide cyclonedds participant + per-topic readers/writers.

    A single DomainParticipant is shared (cheap, and keeps one presence on the
    graph). Writers and readers are cached per (topic, type) so repeated
    publish/echo calls reuse the same DDS entities - and a long-lived advertised
    writer keeps "being a robot" between tool calls. All access is serialised.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._participant: Any = None
        self._writers: dict[tuple[str, str], Any] = {}
        self._readers: dict[tuple[str, str], Any] = {}
        self._available: bool | None = None

    def available(self) -> bool:
        try:
            from strands_robots.rtps.idl import have_cyclonedds

            return have_cyclonedds()
        except ImportError:
            return False

    def _participant_obj(self) -> Any:
        if self._participant is None:
            from cyclonedds.domain import DomainParticipant

            self._participant = DomainParticipant()
        return self._participant

    def writer(self, ros_topic: str, ros_type: str) -> Any:
        """Get-or-create a DataWriter for (topic, type), ROS-mangled."""
        key = (ros_topic, ros_type)
        if key not in self._writers:
            from cyclonedds.pub import DataWriter
            from cyclonedds.topic import Topic

            from strands_robots.rtps.idl import get_type
            from strands_robots.rtps.mangling import dds_topic_name

            idl_cls = get_type(ros_type)
            topic = Topic(self._participant_obj(), dds_topic_name(ros_topic), idl_cls)
            self._writers[key] = DataWriter(self._participant_obj(), topic)
        return self._writers[key]

    def reader(self, ros_topic: str, ros_type: str) -> Any:
        """Get-or-create a DataReader for (topic, type), ROS-mangled."""
        key = (ros_topic, ros_type)
        if key not in self._readers:
            from cyclonedds.sub import DataReader
            from cyclonedds.topic import Topic

            from strands_robots.rtps.idl import get_type
            from strands_robots.rtps.mangling import dds_topic_name

            idl_cls = get_type(ros_type)
            topic = Topic(self._participant_obj(), dds_topic_name(ros_topic), idl_cls)
            self._readers[key] = DataReader(self._participant_obj(), topic)
        return self._readers[key]

    @property
    def lock(self) -> threading.RLock:
        return self._lock


_backend = _RtpsBackend()


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": f"use_rtps: {text}"}]}


def _sample_to_dict(sample: Any) -> Any:
    """Recursively convert an IDL dataclass sample to a plain dict."""
    if dataclasses.is_dataclass(sample) and not isinstance(sample, type):
        return {f.name: _sample_to_dict(getattr(sample, f.name)) for f in dataclasses.fields(sample)}
    if isinstance(sample, (list, tuple)):
        return [_sample_to_dict(x) for x in sample]
    return sample


def _resolve_field_types(idl_cls: Any) -> dict[str, Any]:
    """Return {field_name: resolved_type} for a dataclass.

    ``from __future__ import annotations`` (and cyclonedds's own annotations)
    make ``dataclasses.Field.type`` a *string*, so nested-message detection must
    resolve real types via ``typing.get_type_hints``. Falls back to the raw
    ``Field.type`` if hint resolution fails (e.g. exotic cyclonedds aliases).
    """
    try:
        hints = typing.get_type_hints(idl_cls)
    except Exception:
        hints = {}
    return {f.name: hints.get(f.name, f.type) for f in dataclasses.fields(idl_cls)}


def _build_sample(idl_cls: Any, fields: dict[str, Any]) -> Any:
    """Construct an IDL dataclass instance from a (possibly nested) field dict.

    Nested message fields are built recursively from their annotated dataclass
    type, so ``{"linear": {"x": 2.0}}`` becomes ``Twist(linear=Vector3(x=2.0))``.
    Unknown field names raise so a typo fails loudly rather than silently
    publishing zeros.
    """
    kwargs: dict[str, Any] = {}
    field_types = _resolve_field_types(idl_cls)
    for name, value in fields.items():
        if name not in field_types:
            raise ValueError(f"unknown field {name!r} for {idl_cls.__name__} (have: {sorted(field_types)})")
        ftype = field_types[name]
        if isinstance(value, dict) and dataclasses.is_dataclass(ftype):
            kwargs[name] = _build_sample(ftype, value)
        else:
            kwargs[name] = value
    return idl_cls(**kwargs)


def rtps_action(
    action: str,
    *,
    topic: str | None = None,
    type: str | None = None,
    fields: dict[str, Any] | None = None,
    timeout: float = 5.0,
    count: int = 1,
    rate: float = 10.0,
    gate: PublishGate,
) -> dict[str, Any]:
    """Run one RTPS action against the shared participant.

    Args:
        action: One of ``status``, ``types``, ``advertise``, ``publish``,
            ``subscribe``, ``echo``.
        topic: ROS 2 topic name, absolute (e.g. ``/turtle1/cmd_vel``).
        type: ROS 2 interface type in the IDL bundle (e.g.
            ``geometry_msgs/msg/Twist``).
        fields: Field dict for ``publish``; nested message fields are built
            recursively.
        timeout: Seconds to wait for samples (``echo``).
        count: Messages to publish, or samples to echo.
        rate: Publish rate in Hz; the inter-message period is ``1 / rate``.
        gate: The operator gate for a ``publish``, consulted with ``topic``.
            Required: a caller that forgot it would carry a command to a
            blocklisted drive surface with no prompt, which is the defect
            :mod:`strands_robots._command_gate` exists to prevent. The numeric
            domains of ``count`` / ``rate`` / ``timeout`` belong to the caller
            too - an agent tool reports a malformed option, while a
            :class:`~strands_robots.mesh._mobile_base.MobileBaseRobot` has
            already refused one at its own seam.

    Returns:
        A Strands tool result dict ``{"status": ..., "content": [{"text": ...}]}``.
    """
    fields = fields or {}

    # Validate before mangling so a malformed caller-supplied name fails with a
    # clear message. The rule is read from the mangling that will map the name,
    # not restated here: a second spelling could accept a name the mangling
    # refuses (the caller then gets the refusal from a layer they did not call)
    # or accept one it maps anyway, which is worse - the participant joins a DDS
    # topic no ROS 2 node can create, and DDS reports nothing because matching
    # is by name.
    if topic is not None and (clause := ros_topic_error(topic)) is not None:
        return _err(f"invalid topic name: {topic!r} ({clause})")
    if type is not None:
        try:
            dds_type_name(type)
        except ValueError as exc:
            return _err(f"invalid interface type: {exc}")

    if action == "status":
        if _backend.available():
            return _ok("backend: cyclonedds (RTPS participant) - no rclpy / ROS 2 distro needed")
        from strands_robots.rtps.idl import _INSTALL_HINT

        return _ok("backend: none - " + _INSTALL_HINT)

    if not _backend.available():
        from strands_robots.rtps.idl import _INSTALL_HINT

        return _err(_INSTALL_HINT)

    try:
        from strands_robots.rtps.idl import REGISTRY, get_type

        if action == "types":
            return _ok("RTPS IDL bundle types:\n" + "\n".join(sorted(REGISTRY)))

        # The operator gate is consulted here - after the backend probe, so a
        # transport that cannot publish never prompts, and before the lock, so a
        # human deciding does not hold the process-wide DDS lock and no writer
        # joins the graph on a refusal. Only ``publish`` can command:
        # ``advertise`` creates a publisher without writing a sample, and the
        # rest only read. The well-formedness condition mirrors the publish
        # branch's own, so an incomplete call is reported without asking an
        # operator about it.
        if action == "publish" and topic and type:
            refusal = gate(topic)
            if refusal is not None:
                return _err(refusal)

        with _backend.lock:
            if action == "advertise":
                if not topic or not type:
                    return _err("advertise requires topic and type")
                get_type(type)  # validate type is in the bundle before creating the writer
                _backend.writer(topic, type)
                return _ok(f"advertised {topic} ({type}) - now a publisher on the ROS 2 graph")

            if action == "publish":
                if not topic or not type:
                    return _err("publish requires topic and type")
                idl_cls = get_type(type)
                sample = _build_sample(idl_cls, fields)
                writer = _backend.writer(topic, type)
                # Brief settle so freshly-matched readers receive the first sample.
                time.sleep(0.3)
                period = 1.0 / rate if rate > 0 else 0.0
                for _ in range(count):
                    writer.write(sample)
                    if period:
                        time.sleep(period)
                return _ok(f"published {count} message(s) to {topic} ({type})")

            if action in ("subscribe", "echo"):
                if not topic or not type:
                    return _err(f"{action} requires topic and type")
                reader = _backend.reader(topic, type)
                if action == "subscribe":
                    return _ok(f"subscribed to {topic} ({type})")
                # echo: poll the reader until count samples arrive or timeout.
                samples: list[Any] = []
                # time.monotonic(): a wall-clock step during the poll would
                # return short of ``count`` samples that were still arriving.
                deadline = time.monotonic() + timeout
                while len(samples) < count and time.monotonic() < deadline:
                    for sample in reader.take(N=count - len(samples)):
                        samples.append(_sample_to_dict(sample))
                    if len(samples) < count:
                        time.sleep(0.05)
                return _ok(f"echo {topic} ({type}):\n{json.dumps(samples, indent=2, default=str)}")

            return _err(f"unknown action: {action}")
    except ImportError as exc:
        return _err(str(exc))
    except (KeyError, ValueError, AttributeError, TypeError) as exc:
        return _err(f"{action} failed: {exc}")
