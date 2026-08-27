"""A safety record that can never be written is announced, not logged at DEBUG.

:meth:`~strands_robots.mesh.sensors.SensorLoopsMixin.publish_safety_event` sends
one event two ways: to the wire, and to the local audit log. Its own docstring
records that the ``severity`` parameter "reaches the audit record only - the wire
copy is uniformly ``info``", so the audit record is the only copy that carries
what actually happened.

:func:`~strands_robots.mesh.session._report_unencodable_payload` states the rule
both halves are held to. A transport's fire-and-forget tolerance is scoped to a
TRANSIENT failure - "a closed session, a dropped broker, a socket-level write -
which the next tick retries" - and a permanent loss is reported at ERROR
instead, because "reporting it at DEBUG left the two halves of one call
disagreeing".

An audit write that fails is permanent in the stronger sense: a safety event is
published once, at one lockout transition, so there is no later tick at all.
These cells pin that both halves now announce a permanent loss the same way, and
that neither announces an ordinary one.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

import strands_robots.mesh.core as core_mod
import strands_robots.mesh.sensors as sensors_mod
import strands_robots.mesh.session as session_mod

#: The logger the mixin reports through.
_SENSORS_LOGGER = "strands_robots.mesh.sensors"

#: The logger the wire half's permanent-loss report comes out of.
_SESSION_LOGGER = "strands_robots.mesh.session"

#: The event this suite drives. ``remote_resume_applied`` is emitted after
#: ``_on_safety_resume`` has already cleared the lockout, so a lost record is a
#: fleet that resumed motion with nothing on the forensic trail to say so.
_EVENT_TYPE = "remote_resume_applied"

#: A severity the wire copy provably does not carry: the wire is uniformly
#: ``info``, so this value exists only in the audit record.
_REAL_SEVERITY = "critical"

_PEER = "alice"


class _AuditUnwritable(OSError):
    """What a full or read-only audit volume raises on the write."""


def _mesh(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    """A running mesh whose wire publishes are recorded rather than sent.

    Args:
        monkeypatch: Fixture used to replace the module-level ``put``.

    Returns:
        ``(mesh, wire)`` where ``wire`` accumulates every ``(key, payload)``
        the mesh published.
    """
    wire: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(core_mod, "put", lambda key, payload: wire.append((key, payload)))
    mesh = core_mod.Mesh(object(), peer_id=_PEER)
    mesh._running = True
    return mesh, wire


def _break_the_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the audit half fail the way a full volume makes it fail."""

    def refuse(**_kwargs: Any) -> None:
        raise _AuditUnwritable(28, "No space left on device")

    monkeypatch.setattr(sensors_mod, "log_safety_event", refuse)


def _records(caplog: pytest.LogCaptureFixture, *, at_least: int) -> list[logging.LogRecord]:
    """The captured records at or above ``at_least``."""
    return [r for r in caplog.records if r.levelno >= at_least]


class TestAPermanentlyLostSafetyRecordIsAnnounced:
    """The audit half's failure reaches an operator at the default level."""

    def test_the_loss_is_reported_at_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        mesh, _wire = _mesh(monkeypatch)
        _break_the_audit(monkeypatch)
        with caplog.at_level(logging.DEBUG, logger=_SENSORS_LOGGER):
            mesh.publish_safety_event(event_type=_EVENT_TYPE, severity=_REAL_SEVERITY, payload={"issuer": "bob"})
        assert [r.levelname for r in _records(caplog, at_least=logging.ERROR)] == ["ERROR"]

    def test_a_default_configured_operator_sees_it(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """WARNING is the default level, so the report must clear it."""
        mesh, _wire = _mesh(monkeypatch)
        _break_the_audit(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=_SENSORS_LOGGER):
            mesh.publish_safety_event(event_type=_EVENT_TYPE, severity=_REAL_SEVERITY, payload={"issuer": "bob"})
        assert _records(caplog, at_least=logging.WARNING), "nothing an operator would see"

    def test_the_report_names_the_event_that_was_lost(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        mesh, _wire = _mesh(monkeypatch)
        _break_the_audit(monkeypatch)
        with caplog.at_level(logging.DEBUG, logger=_SENSORS_LOGGER):
            mesh.publish_safety_event(event_type=_EVENT_TYPE, severity=_REAL_SEVERITY, payload={"issuer": "bob"})
        text = "\n".join(r.getMessage() for r in _records(caplog, at_least=logging.ERROR))
        assert _EVENT_TYPE in text

    def test_the_report_names_the_severity_the_wire_copy_does_not_carry(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The audit record was the only copy of it, so the report must say it."""
        mesh, _wire = _mesh(monkeypatch)
        _break_the_audit(monkeypatch)
        with caplog.at_level(logging.DEBUG, logger=_SENSORS_LOGGER):
            mesh.publish_safety_event(event_type=_EVENT_TYPE, severity=_REAL_SEVERITY, payload={"issuer": "bob"})
        text = "\n".join(r.getMessage() for r in _records(caplog, at_least=logging.ERROR))
        assert _REAL_SEVERITY in text

    def test_the_report_names_the_peer_and_the_reason(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        mesh, _wire = _mesh(monkeypatch)
        _break_the_audit(monkeypatch)
        with caplog.at_level(logging.DEBUG, logger=_SENSORS_LOGGER):
            mesh.publish_safety_event(event_type=_EVENT_TYPE, severity=_REAL_SEVERITY, payload={"issuer": "bob"})
        text = "\n".join(r.getMessage() for r in _records(caplog, at_least=logging.ERROR))
        assert _PEER in text
        assert "No space left on device" in text


class TestTheTwoHalvesOfOneCallAgree:
    """A permanent loss is announced the same way whichever half notices it."""

    def test_the_wire_half_announces_a_permanent_loss_at_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The premise the audit half is compared against."""

        class _NotAReading:
            """A payload value the JSON encoder refuses and coercion keeps."""

        class _Session:
            def put(self, key: str, encoded: bytes) -> None:
                raise AssertionError("an unencodable payload must not reach the wire")

        monkeypatch.setattr(session_mod, "_SESSION", _Session())
        monkeypatch.setattr(session_mod, "_unencodable_topics_warned", set())
        monkeypatch.setattr(sensors_mod, "log_safety_event", lambda **_kwargs: None)
        mesh = core_mod.Mesh(object(), peer_id=_PEER)
        mesh._running = True
        with caplog.at_level(logging.DEBUG, logger=_SESSION_LOGGER):
            mesh.publish_safety_event(
                event_type=_EVENT_TYPE, severity=_REAL_SEVERITY, payload={"tripped": _NotAReading()}
            )
        assert [r.levelname for r in _records(caplog, at_least=logging.ERROR)] == ["ERROR"]

    def test_neither_half_announces_the_loss_more_quietly_than_the_other(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One rule, two halves: the levels are read, not restated."""
        mesh, _wire = _mesh(monkeypatch)
        _break_the_audit(monkeypatch)
        with caplog.at_level(logging.DEBUG, logger=_SENSORS_LOGGER):
            mesh.publish_safety_event(event_type=_EVENT_TYPE, severity=_REAL_SEVERITY, payload={"issuer": "bob"})
        audit_levels = {r.levelno for r in caplog.records}
        assert audit_levels == {logging.ERROR}


class TestWhatIsUnchanged:
    """Announcing the loss must not change anything else about the call."""

    def test_the_call_still_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fire-and-forget: a failed audit must not break the safety path."""
        mesh, _wire = _mesh(monkeypatch)
        _break_the_audit(monkeypatch)
        mesh.publish_safety_event(event_type=_EVENT_TYPE, severity=_REAL_SEVERITY, payload={})

    def test_the_wire_copy_still_goes_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The audit failure must not suppress the half that did work."""
        mesh, wire = _mesh(monkeypatch)
        _break_the_audit(monkeypatch)
        mesh.publish_safety_event(event_type=_EVENT_TYPE, severity=_REAL_SEVERITY, payload={})
        assert [key for key, _payload in wire] == [f"strands/{_PEER}/safety/event"]

    def test_the_wire_severity_is_still_uniformly_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Issue #272's uniform wire severity is untouched by this change."""
        mesh, wire = _mesh(monkeypatch)
        _break_the_audit(monkeypatch)
        mesh.publish_safety_event(event_type=_EVENT_TYPE, severity=_REAL_SEVERITY, payload={})
        assert [payload["severity"] for _key, payload in wire] == ["info"]

    def test_a_healthy_publish_announces_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The ordinary case stays silent - ERROR must not become routine."""
        mesh, wire = _mesh(monkeypatch)
        monkeypatch.setattr(sensors_mod, "log_safety_event", lambda **_kwargs: None)
        with caplog.at_level(logging.DEBUG, logger=_SENSORS_LOGGER):
            mesh.publish_safety_event(event_type=_EVENT_TYPE, severity=_REAL_SEVERITY, payload={"issuer": "bob"})
        assert caplog.records == []
        assert len(wire) == 1

    def test_a_stopped_mesh_still_publishes_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The not-running early return precedes both halves, so it says nothing."""
        mesh, wire = _mesh(monkeypatch)
        _break_the_audit(monkeypatch)
        mesh._running = False
        with caplog.at_level(logging.DEBUG, logger=_SENSORS_LOGGER):
            mesh.publish_safety_event(event_type=_EVENT_TYPE, severity=_REAL_SEVERITY, payload={})
        assert wire == []
        assert caplog.records == []

    def test_the_callers_payload_is_left_unedited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The documented copy-not-in-place contract is unaffected."""
        mesh, _wire = _mesh(monkeypatch)
        _break_the_audit(monkeypatch)
        payload = {"issuer": "bob"}
        mesh.publish_safety_event(event_type=_EVENT_TYPE, severity=_REAL_SEVERITY, payload=payload)
        assert payload == {"issuer": "bob"}


class TestThePremisesHold:
    """What the cells above depend on, asserted rather than assumed."""

    def test_the_mixin_reads_the_module_level_audit_writer(self) -> None:
        """So replacing the module global is the seam the failure arrives through."""
        assert callable(sensors_mod.log_safety_event)

    def test_the_documented_reason_for_the_level_is_still_the_wire_halfs(self) -> None:
        """The rule is cited, not restated, so the citation must resolve."""
        doc = " ".join((session_mod._report_unencodable_payload.__doc__ or "").split())
        assert "at ERROR" in doc
        assert "TRANSIENT" in doc
