"""A safety event reaches the wire even when its payload carries a reading.

:class:`~strands_robots.mesh.sensors.SensorLoopsMixin` builds nine records
addressed to a ``strands/<peer>/...`` topic. Eight of them - every ``_read_*``
reader - pass the record through ``_coerce_record`` before it is published,
because a sensor pipeline reports its readings as numpy and ``json.dumps``
refuses a ``float32``. ``publish_safety_event`` builds the ninth, and did not
coerce it: a safety payload carrying a reading was dropped before the wire while
the audit half of the same call wrote a ``sig="SERIALISE_FAILED"`` poison record
naming the failure. That is the disagreement
:func:`strands_robots.mesh.session._report_unencodable_payload` describes in its
own docstring - it raised the report from DEBUG to ERROR, and the wire half still
published nothing.

Why the existing coverage was silent: ``tests/mesh/test_sensor_readers.py``
drives ``publish_safety_event`` through a host whose ``publish`` *records* the
payload. The encoder runs one layer down, in the transport - ``json.dumps`` in
``session._put_zenoh_directly``, before ``put`` - so a recording double accepts
a payload the wire never would. The host here encodes what it is handed, which is
what lets these cells see the drop at all.
"""

from __future__ import annotations

import ast
import inspect
import json
import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.mesh import sensors as mesh_sensors
from strands_robots.mesh import session as mesh_session
from strands_robots.mesh.sensors import SensorLoopsMixin

#: A reading, its spelling, and what it must decode to on the far side. These
#: are the numeric types a sensor pipeline hands over: ``ndarray.min(axis=0)``
#: returns a ``float32``/``float64`` scalar, a joint vector is an ``ndarray``,
#: and a comparison between them is a ``np.bool_``.
_READINGS: tuple[tuple[str, Any, Any], ...] = (
    ("float32-scalar", np.float32(2.97), pytest.approx(2.97, rel=1e-6)),
    ("int64-scalar", np.int64(3), 3),
    ("bool_-scalar", np.bool_(True), True),
    ("float32-vector", np.array([1.5, -2.5], dtype=np.float32), [1.5, -2.5]),
    ("float64-vector", np.array([1.5, -2.5]), [1.5, -2.5]),
    ("list-of-float32", [np.float32(1.0), np.float32(2.0)], [1.0, 2.0]),
    ("nested-mapping", {"q": np.float32(0.5)}, {"q": pytest.approx(0.5, rel=1e-6)}),
)

_IDS = tuple(row[0] for row in _READINGS)


class _EncodingHost(SensorLoopsMixin):
    """A mixin host whose ``publish`` encodes the way the transport does.

    ``session._put_zenoh_directly`` encodes with ``json.dumps`` before handing
    the bytes to ``put``, and a payload it refuses is dropped for good - the
    failure is deterministic, so no later tick recovers it. Encoding here is
    therefore not a stricter test double than production: it is the same step,
    moved to where a unit test can observe which side of it the record landed.
    """

    def __init__(self, peer_id: str = "peer-1") -> None:
        self.robot: Any = object()
        self.peer_id = peer_id
        self._running = True
        self._stop_event = threading.Event()
        self.handed: list[tuple[str, dict[str, Any]]] = []
        self.wire: list[tuple[str, dict[str, Any]]] = []
        self.dropped: list[tuple[str, str]] = []

    def publish(self, key: str, payload: dict[str, Any]) -> None:
        self.handed.append((key, payload))
        try:
            encoded = json.dumps(payload).encode()
        except (TypeError, ValueError) as exc:
            self.dropped.append((key, f"{type(exc).__name__}: {exc}"))
            return
        self.wire.append((key, json.loads(encoded)))


class _ProviderHost(_EncodingHost):
    """An encoding host that also fronts a robot exposing sensor providers."""

    def __init__(self, **provider_attrs: Any) -> None:
        super().__init__()
        robot = type("_Robot", (), {})()
        for name, value in provider_attrs.items():
            setattr(robot, name, value)
        self.robot = robot


@pytest.fixture
def audited(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture the audit half's keyword arguments instead of writing a record."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(mesh_sensors, "log_safety_event", lambda **kw: calls.append(kw))
    return calls


def _record_builders() -> set[str]:
    """Methods of the mixin that build a record addressed to the wire.

    Derived, not listed: a record bound for a ``strands/<peer>/...`` topic
    carries the peer that produced it, so a method containing a dict literal with
    a ``"peer_id"`` key is building one. Reading the set off the source means a
    publisher added later is held to the same rule as the nine that exist.
    """
    return {name for name, fn in _mixin_methods().items() if "peer_id" in _literal_keys(fn)}


def _coercing_methods() -> set[str]:
    """Methods of the mixin that call ``_coerce_record``."""
    return {name for name, fn in _mixin_methods().items() if _calls(fn, "_coerce_record")}


def _mixin_methods() -> dict[str, ast.FunctionDef]:
    source = inspect.getsource(mesh_sensors)
    cls = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == "SensorLoopsMixin"
    )
    return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}


def _literal_keys(fn: ast.FunctionDef) -> set[str]:
    return {
        key.value
        for node in ast.walk(fn)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _calls(fn: ast.FunctionDef, name: str) -> bool:
    return any(isinstance(node, ast.Call) and name in ast.unparse(node.func) for node in ast.walk(fn))


class TestThePremiseHolds:
    """The reading really is unencodable, and the wire really encodes."""

    @pytest.mark.parametrize(("label", "value", "expected"), _READINGS, ids=_IDS)
    def test_json_refuses_the_reading_uncoerced(self, label: str, value: Any, expected: Any) -> None:
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps({"v": value})

    def test_the_transport_encodes_before_it_puts(self) -> None:
        source = inspect.getsource(mesh_session._put_zenoh_directly)
        assert "json.dumps" in source, (
            "premise: the wire encodes with json.dumps, which is why an encoding host is the "
            "faithful double and a recording one is not"
        )

    def test_a_sibling_reader_given_the_same_reading_reaches_the_wire(self) -> None:
        host = _ProviderHost(_map_info={"resolution": np.float32(0.05), "origin": np.array([1.0, 2.0])})
        info = host._read_map_info()
        assert info is not None
        host.publish(f"strands/{host.peer_id}/map/info", info)
        assert host.dropped == []
        assert host.wire[0][1]["resolution"] == pytest.approx(0.05, rel=1e-6)


class TestASafetyEventCarryingAReadingReachesTheWire:
    """The regression: the record is coerced, so the event is published."""

    @pytest.mark.parametrize(("label", "value", "expected"), _READINGS, ids=_IDS)
    def test_the_event_is_published(self, audited: list[dict[str, Any]], label: str, value: Any, expected: Any) -> None:
        host = _EncodingHost()
        host.publish_safety_event("joint_limit_tripped", severity="critical", payload={"v": value})

        assert host.dropped == [], f"a {label} reading was dropped before the wire: {host.dropped}"
        assert len(host.wire) == 1
        key, event = host.wire[0]
        assert key == "strands/peer-1/safety/event"
        assert event["payload"]["v"] == expected
        # The rest of the envelope is unchanged by the coercion.
        assert event["severity"] == "info"
        assert event["type"] == "joint_limit_tripped"
        assert event["peer_id"] == "peer-1"

    def test_a_reading_used_as_a_key_reaches_the_wire(self, audited: list[dict[str, Any]]) -> None:
        host = _EncodingHost()
        # Annotated at the call site: the parameter is ``dict[str, Any]``, and a
        # reading used as a key is exactly the case ``_jsonable`` coerces.
        keyed: dict[Any, Any] = {np.int64(7): "joint-7"}
        host.publish_safety_event("estop", payload=keyed)

        assert host.dropped == []
        assert host.wire[0][1]["payload"] == {"7": "joint-7"}


class TestBothHalvesReportTheSameEvent:
    """The wire half and the audit half carry one coerced reading, not two."""

    def test_the_audit_record_carries_the_coerced_reading(self, audited: list[dict[str, Any]]) -> None:
        host = _EncodingHost()
        host.publish_safety_event("estop", severity="critical", payload={"q": np.float32(2.97)})

        wire_payload = host.wire[0][1]["payload"]
        audit_payload = audited[0]["payload"]
        assert audit_payload["q"] == wire_payload["q"]
        # The audit half no longer needs its poison-record path for this payload either.
        json.dumps(audit_payload)


class TestTheCallersMappingIsNotEdited:
    """Coercion happens in a copy, so the caller keeps what they passed.

    The event no longer carries the caller's own mapping: it used to be inserted
    by reference, so a caller who reused or mutated the dict after the call
    changed what a fire-and-forget publish was still holding.
    """

    def test_the_supplied_mapping_is_unchanged(self, audited: list[dict[str, Any]]) -> None:
        nested = {"q": np.float32(0.5)}
        payload: dict[str, Any] = {"reading": np.float32(2.97), "nested": nested}
        host = _EncodingHost()
        host.publish_safety_event("estop", payload=payload)

        assert isinstance(payload["reading"], np.float32)
        assert payload["nested"] is nested
        assert isinstance(nested["q"], np.float32)

    def test_the_published_payload_is_not_the_callers_object(self, audited: list[dict[str, Any]]) -> None:
        payload: dict[str, Any] = {"reason": "x"}
        host = _EncodingHost()
        host.publish_safety_event("estop", payload=payload)

        assert host.handed[0][1]["payload"] is not payload
        payload["reason"] = "mutated after the call"
        assert host.handed[0][1]["payload"] == {"reason": "x"}


class TestEveryRecordBoundForTheWireIsCoerced:
    """The rule, derived from the module rather than listed here."""

    def test_the_derivation_is_not_vacuous(self) -> None:
        builders = _record_builders()
        assert len(builders) >= 9, f"expected every peer-stamped record builder, found {sorted(builders)}"
        assert "publish_safety_event" in builders
        assert {"_read_pose", "_read_health", "_read_imu", "_read_map_info"} <= builders

    def test_every_record_builder_coerces(self) -> None:
        uncoerced = _record_builders() - _coercing_methods()
        assert uncoerced == set(), (
            f"these methods build a record addressed to the wire and do not coerce it: {sorted(uncoerced)}"
        )

    def test_no_method_coerces_without_building_a_record(self) -> None:
        stray = _coercing_methods() - _record_builders()
        assert stray == set(), f"_coerce_record called where no wire record is built: {sorted(stray)}"


class TestWhatIsUnchanged:
    """Every expectation here is one the uncoerced code also met."""

    def test_a_plain_payload_is_published_verbatim(self, audited: list[dict[str, Any]]) -> None:
        host = _EncodingHost()
        host.publish_safety_event("estop", severity="critical", payload={"reason": "x", "n": 3})
        assert host.wire[0][1]["payload"] == {"reason": "x", "n": 3}

    def test_the_wire_severity_is_still_uniform(self, audited: list[dict[str, Any]]) -> None:
        host = _EncodingHost()
        host.publish_safety_event("estop", severity="critical", payload={"reason": "x"})
        assert host.wire[0][1]["severity"] == "info"
        assert host.wire[0][1]["type"] == "estop"

    def test_the_audit_record_still_carries_the_real_severity(self, audited: list[dict[str, Any]]) -> None:
        host = _EncodingHost()
        host.publish_safety_event("estop", severity="critical", payload={"reason": "x"})
        assert audited[0]["payload"]["severity"] == "critical"
        assert audited[0]["event_type"] == "estop"

    def test_no_payload_still_publishes_an_empty_one(self, audited: list[dict[str, Any]]) -> None:
        host = _EncodingHost()
        host.publish_safety_event("estop")
        assert host.wire[0][1]["payload"] == {}

    def test_a_stopped_host_still_publishes_nothing(self, audited: list[dict[str, Any]]) -> None:
        host = _EncodingHost()
        host._running = False
        host.publish_safety_event("estop", payload={"q": np.float32(1.0)})
        assert host.wire == [] and host.dropped == [] and audited == []

    def test_an_audit_failure_is_still_survived(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(**_kw: Any) -> None:
            raise RuntimeError("audit backend down")

        monkeypatch.setattr(mesh_sensors, "log_safety_event", _boom)
        host = _EncodingHost()
        host.publish_safety_event("estop", payload={"reason": "x"})
        assert len(host.wire) == 1


class TestAValueThatIsNotAReadingIsStillReported:
    """The over-reach guard: coercing must not become stringifying.

    ``_jsonable`` returns an object it cannot recognise unchanged, so the
    transport still names it. Substituting a repr would publish a record that
    misstates what the safety path saw, which is worse than a reported drop.
    """

    def test_an_unrepresentable_object_is_passed_through(self, audited: list[dict[str, Any]]) -> None:
        sentinel = object()
        host = _EncodingHost()
        host.publish_safety_event("estop", payload={"thing": sentinel})

        assert host.wire == []
        assert len(host.dropped) == 1
        assert "not JSON serializable" in host.dropped[0][1]

    def test_a_string_payload_value_is_not_taken_apart(self, audited: list[dict[str, Any]]) -> None:
        host = _EncodingHost()
        host.publish_safety_event("estop", payload={"joint": "shoulder_pan"})
        assert host.wire[0][1]["payload"]["joint"] == "shoulder_pan"
