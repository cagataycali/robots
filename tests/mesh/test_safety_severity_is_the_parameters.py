"""The severity a safety event is raised with survives a payload field of that name.

:meth:`~strands_robots.mesh.sensors.SensorLoopsMixin.publish_safety_event` sends
one event two ways. The wire copy carries a uniformly ``"info"`` severity, on
purpose: issue #272 removed per-branch severity from
``strands/+/safety/event`` so a subscriber could not read it as a
content-channel oracle for a rejection reason. The consequence the method's own
comment records is that the audit record is *the only surviving copy of the real
severity*.

That copy was built by spreading the caller's payload over the parameter::

    payload={"severity": severity, **record}

so a payload carrying a ``severity`` field replaced it - silently, with no
refusal and nothing logged. An event raised ``severity="critical"`` was audited
as whatever the payload said, and because the wire copy is uniform there is no
second record to compare against.

Eight of the nine record builders in this mixin already resolve exactly this
collision the other way. ``_stamp_local_keys`` re-asserts the keys this process
decided after a provider mapping merges in, and its own docstring names the
hazard: "Merged last, a provider mapping carrying one of those seeded names
replaces the local reading". It cites
:meth:`strands_robots.mesh.session.PeerInfo.to_dict`, which spreads the peer's
own payload *first* so the locally decided keys win. ``publish_safety_event``
built the ninth record and was the one that spread last.

Why the existing coverage was silent:
``tests/mesh/test_safety_event_payload_reaches_the_wire.py`` owns a cell named
``test_the_audit_record_still_carries_the_real_severity`` - the exact property
that failed - but it drives ``payload={"reason": "x"}``. A payload with no
``severity`` key cannot collide, so the assertion held whichever way the merge
was ordered. ``tests/mesh/test_sensor_readers.py`` asserts the same thing with
the same non-colliding payload. The property was named and never exercised.
"""

from __future__ import annotations

import ast
import inspect
import json
import threading
from typing import Any

import pytest

from strands_robots.mesh import sensors as mesh_sensors
from strands_robots.mesh.sensors import SensorLoopsMixin

#: Payload spellings of a ``severity`` field, and why each is worth its own cell.
#: A caller may name a field ``severity`` because their event has one (the audit
#: payloads ``mesh.core._emit_resume_denied`` writes carry exactly that key), and
#: the value need not even be a severity word.
_COLLIDING_PAYLOADS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("another-severity-word", {"joint": "elbow", "severity": "info"}),
    ("the-in-repo-audit-payload-shape", {"sender_id": "a", "reason": "denied", "severity": "info"}),
    ("not-a-severity-word-at-all", {"severity": 0}),
    ("severity-is-the-only-field", {"severity": None}),
    ("an-empty-string", {"severity": ""}),
)

_IDS = tuple(row[0] for row in _COLLIDING_PAYLOADS)

#: The severities the cells below raise events with. No payload above uses either
#: value, so a cell that passes cannot be passing because the two happened to
#: agree - the coincidence case has its own class.
_RAISED = "critical"
_DEFAULT_SEVERITY = "warning"


class _Host(SensorLoopsMixin):
    """A mixin host that records what each half of the call was handed."""

    def __init__(self, peer_id: str = "peer-1") -> None:
        self.robot: Any = object()
        self.peer_id = peer_id
        self._running = True
        self._stop_event = threading.Event()
        self.wire: list[tuple[str, dict[str, Any]]] = []

    def publish(self, key: str, payload: dict[str, Any]) -> None:
        self.wire.append((key, payload))


@pytest.fixture
def audited(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture the audit half's keyword arguments instead of writing a record."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(mesh_sensors, "log_safety_event", lambda **kw: calls.append(kw))
    return calls


# --- the derived rule ------------------------------------------------------


def _mixin_methods() -> dict[str, ast.FunctionDef]:
    source = inspect.getsource(mesh_sensors)
    cls = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == "SensorLoopsMixin"
    )
    return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}


def _shadowed_literal_keys(fn: ast.FunctionDef) -> set[str]:
    """Explicit keys a ``**`` spread later in the same dict literal can replace.

    Read off the literal rather than listed, so the rule is about the ordering
    the language gives the merge and not about any particular key name.
    """
    shadowed: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Dict):
            continue
        spreads = [i for i, key in enumerate(node.keys) if key is None]
        if not spreads:
            continue
        last_spread = max(spreads)
        shadowed |= {
            key.value
            for index, key in enumerate(node.keys)
            if index < last_spread and isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    return shadowed


def _merges_a_foreign_mapping(fn: ast.FunctionDef) -> bool:
    """Whether the method merges a mapping it did not build itself."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update":
            return True
        if isinstance(node, ast.Dict) and any(key is None for key in node.keys):
            return True
    return False


def _merging_methods() -> dict[str, ast.FunctionDef]:
    """Methods that merge a foreign mapping into a record they seeded.

    ``_stamp_local_keys`` is excluded because it *is* the re-assertion: its
    ``record.update(local)`` is the local keys being put back, not a foreign
    mapping being merged in.
    """
    return {
        name: fn
        for name, fn in _mixin_methods().items()
        if name != "_stamp_local_keys" and _merges_a_foreign_mapping(fn)
    }


def _calls(fn: ast.FunctionDef, name: str) -> bool:
    return any(isinstance(node, ast.Call) and name in ast.unparse(node.func) for node in ast.walk(fn))


class TestThePremisesTheFixRestsOn:
    """Measured, not assumed: why the audit copy is the one that matters."""

    def test_the_wire_copy_carries_a_uniform_severity(self, audited: list[dict[str, Any]]) -> None:
        host = _Host()
        host.publish_safety_event("estop", severity="critical", payload={"joint": "elbow"})
        assert host.wire[0][1]["severity"] == "info"

    def test_so_the_audit_record_is_the_only_copy_of_the_real_severity(self, audited: list[dict[str, Any]]) -> None:
        host = _Host()
        host.publish_safety_event("estop", severity="critical", payload={"joint": "elbow"})
        wire_event = host.wire[0][1]
        assert "critical" not in json.dumps(wire_event)
        assert audited[0]["payload"]["severity"] == "critical"

    def test_the_mixin_really_merges_foreign_mappings(self) -> None:
        merging = _merging_methods()
        assert len(merging) >= 8, f"expected every merging builder, found {sorted(merging)}"
        assert "publish_safety_event" in merging
        assert {"_read_pose", "_read_imu", "_read_lidar_summary", "_read_map_info"} <= set(merging)

    def test_the_shared_re_assertion_helper_exists_and_names_the_hazard(self) -> None:
        doc = " ".join((SensorLoopsMixin._stamp_local_keys.__doc__ or "").split())
        assert "carrying one of those seeded names replaces the local reading" in doc


class TestTheSeverityParameterSurvivesAPayloadOfTheSameName:
    """The regression. Each payload below collides; the parameter must win."""

    @pytest.mark.parametrize(("_label", "payload"), _COLLIDING_PAYLOADS, ids=_IDS)
    def test_the_audit_record_carries_the_parameter(
        self, _label: str, payload: dict[str, Any], audited: list[dict[str, Any]]
    ) -> None:
        host = _Host()
        host.publish_safety_event("estop", severity=_RAISED, payload=payload)
        assert audited[0]["payload"]["severity"] == _RAISED

    @pytest.mark.parametrize(("_label", "payload"), _COLLIDING_PAYLOADS, ids=_IDS)
    def test_the_default_severity_also_survives(
        self, _label: str, payload: dict[str, Any], audited: list[dict[str, Any]]
    ) -> None:
        host = _Host()
        host.publish_safety_event("estop", payload=payload)
        assert audited[0]["payload"]["severity"] == _DEFAULT_SEVERITY

    def test_no_payload_above_agrees_with_the_severity_it_is_graded_against(self) -> None:
        """Otherwise a cell could pass with the shadow still in place."""
        for label, payload in _COLLIDING_PAYLOADS:
            assert payload["severity"] not in (_RAISED, _DEFAULT_SEVERITY), label


class TestTheShadowWasSilentWhenTheTwoValuesAgreed:
    """Why a caller could carry the collision for a long time without noticing.

    These two cells pass with the shadow in place and with it removed, and that
    is the point rather than a gap: the merge only changed the record when the
    payload's field happened to disagree with the parameter. The audit payloads
    ``mesh.core._emit_resume_denied`` writes carry a ``severity`` field whose
    value is the severity it would also pass as the argument, so the in-repo
    shape is exactly the shape that agrees.
    """

    def test_a_payload_repeating_the_raised_severity_reads_the_same_either_way(
        self, audited: list[dict[str, Any]]
    ) -> None:
        host = _Host()
        host.publish_safety_event("estop", severity=_RAISED, payload={"severity": _RAISED})
        assert audited[0]["payload"]["severity"] == _RAISED

    def test_a_payload_repeating_the_default_severity_reads_the_same_either_way(
        self, audited: list[dict[str, Any]]
    ) -> None:
        host = _Host()
        host.publish_safety_event("estop", payload={"severity": _DEFAULT_SEVERITY})
        assert audited[0]["payload"]["severity"] == _DEFAULT_SEVERITY


class TestWhatIsUnchanged:
    """Every expectation here is one the shadowed code also met."""

    def test_a_payload_without_the_name_is_unaffected(self, audited: list[dict[str, Any]]) -> None:
        host = _Host()
        host.publish_safety_event("estop", severity="critical", payload={"joint": "elbow", "limit": 1.57})
        assert audited[0]["payload"] == {"joint": "elbow", "limit": 1.57, "severity": "critical"}

    def test_an_event_with_no_payload_still_audits_its_severity(self, audited: list[dict[str, Any]]) -> None:
        host = _Host()
        host.publish_safety_event("estop", severity="critical")
        assert audited[0]["payload"] == {"severity": "critical"}

    def test_the_audit_envelope_still_names_the_event_and_the_peer(self, audited: list[dict[str, Any]]) -> None:
        host = _Host(peer_id="alice")
        host.publish_safety_event("estop", severity="critical", payload={"joint": "elbow"})
        assert audited[0]["event_type"] == "estop"
        assert audited[0]["peer_id"] == "alice"

    def test_a_stopped_host_still_publishes_and_audits_nothing(self, audited: list[dict[str, Any]]) -> None:
        host = _Host()
        host._running = False
        host.publish_safety_event("estop", severity="critical", payload={"severity": "info"})
        assert host.wire == [] and audited == []

    def test_an_audit_failure_is_still_survived(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(**_kw: Any) -> None:
            raise OSError("audit log unavailable")

        monkeypatch.setattr(mesh_sensors, "log_safety_event", _boom)
        host = _Host()
        host.publish_safety_event("estop", severity="critical", payload={"severity": "info"})
        assert host.wire[0][1]["type"] == "estop"


class TestTheFixDoesNotReachIntoTheOtherHalf:
    """Refusing the shadow must not edit the wire copy or the caller's mapping."""

    def test_the_wire_payload_still_carries_the_callers_own_field(self, audited: list[dict[str, Any]]) -> None:
        host = _Host()
        host.publish_safety_event("estop", severity="critical", payload={"joint": "elbow", "severity": "info"})
        assert host.wire[0][1]["payload"] == {"joint": "elbow", "severity": "info"}

    def test_the_other_payload_fields_still_reach_the_audit_record(self, audited: list[dict[str, Any]]) -> None:
        host = _Host()
        host.publish_safety_event(
            "estop", severity=_RAISED, payload={"sender_id": "a", "reason": "denied", "severity": "info"}
        )
        recorded = audited[0]["payload"]
        assert recorded["sender_id"] == "a" and recorded["reason"] == "denied"
        assert set(recorded) == {"sender_id", "reason", "severity"}

    def test_the_supplied_mapping_is_unchanged(self, audited: list[dict[str, Any]]) -> None:
        payload: dict[str, Any] = {"joint": "elbow", "severity": "info"}
        host = _Host()
        host.publish_safety_event("estop", severity="critical", payload=payload)
        assert payload == {"joint": "elbow", "severity": "info"}

    def test_the_two_halves_are_not_the_same_object(self, audited: list[dict[str, Any]]) -> None:
        host = _Host()
        host.publish_safety_event("estop", severity="critical", payload={"severity": "info"})
        assert audited[0]["payload"] is not host.wire[0][1]["payload"]


class TestNoMergedMappingShadowsALocalKey:
    """The rule, derived from the module rather than listed here."""

    def test_no_merging_method_places_a_key_before_a_spread(self) -> None:
        offenders = {
            name: sorted(shadowed)
            for name, fn in _merging_methods().items()
            if (shadowed := _shadowed_literal_keys(fn))
        }
        assert offenders == {}, (
            f"these methods build a dict literal whose explicit keys a later ``**`` spread can replace: {offenders}"
        )

    def test_every_update_merge_re_asserts_the_local_keys(self) -> None:
        unstamped = sorted(
            name
            for name, fn in _merging_methods().items()
            if _calls(fn, ".update(") and not _calls(fn, "_stamp_local_keys")
        )
        assert unstamped == [], f"these methods merge with ``update`` and never re-assert their own keys: {unstamped}"

    def test_the_shadow_rule_is_not_vacuous(self) -> None:
        literal = ast.parse('{"severity": severity, **record}', mode="eval").body
        fn = ast.parse("def f():\n    return " + ast.unparse(literal)).body[0]
        assert isinstance(fn, ast.FunctionDef)
        assert _shadowed_literal_keys(fn) == {"severity"}, "the rule must flag a key placed before a spread"

    def test_the_shadow_rule_accepts_a_spread_that_comes_first(self) -> None:
        fn = ast.parse('def f():\n    return {**record, "severity": severity}').body[0]
        assert isinstance(fn, ast.FunctionDef)
        assert _shadowed_literal_keys(fn) == set(), "a spread ahead of the local key is the accepted shape"

    def test_a_key_between_two_spreads_is_still_shadowed(self) -> None:
        """Being after one spread does not help if another follows."""
        fn = ast.parse('def f():\n    return {**a, "severity": severity, **b}').body[0]
        assert isinstance(fn, ast.FunctionDef)
        assert _shadowed_literal_keys(fn) == {"severity"}
