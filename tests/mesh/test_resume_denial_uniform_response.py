"""Resume-override denial contract: uniform response + no reason leak + audit resilience.

:meth:`strands_robots.mesh.core.Mesh._resume_lockout` is the operator override
path that clears an emergency-stop lockout. Its denial branches are
security-sensitive: a remote prober must not be able to use the *response* of a
rejected resume to learn anything about the lockout's internal state. The
contract these tests pin:

* **Uniform response shape.** Every denial reason -- lockout not engaged,
  override code unconfigured, wrong code -- returns the byte-identical generic
  dict ``{"status": "error", "error": "resume rejected"}``. No differential
  response leaks whether the lockout is engaged or whether an override code is
  configured.
* **No reason leak on the wire.** The broadcast safety event for a denial
  carries only an opaque ``reason_code="denied"``; the structured human reason
  ("lockout not engaged" etc.) stays in the local audit log and is never
  published to peers subscribed to ``strands/+/safety/event``.
* **Audit best-effort.** Denial still returns the generic error even when both
  audit sinks (the local ``log_safety_event`` file write and the
  ``publish_safety_event`` broadcast) raise -- an audit outage must not become a
  resume-path outage.
* **Non-engaged resume is a no-op on state.** Resuming when no lockout is
  engaged does not flip any lockout state.
"""

from __future__ import annotations

import threading

import strands_robots.mesh.core as core
from strands_robots.mesh.core import Mesh

_GENERIC_ERROR = {"status": "error", "error": "resume rejected"}
_CODE = "correct-code-1234567890abcdef00"


def _stub() -> Mesh:
    """A Mesh with just enough state for ``_resume_lockout``.

    Built via ``__new__`` (bypassing the Zenoh/transport ``__init__``) and given
    a recording ``publish_safety_event`` so tests can assert the exact wire
    payloads emitted by each denial branch.
    """
    m = Mesh.__new__(Mesh)
    m.peer_id = "p"
    m._estop_lockout = threading.Event()
    m._last_estop_ts = 0.0
    m._published_events = []  # type: ignore[attr-defined]
    m.publish_safety_event = lambda **kw: m._published_events.append(kw)  # type: ignore[method-assign, attr-defined]
    return m


class TestResumeDenialUniformResponse:
    def test_denied_when_lockout_not_engaged(self, monkeypatch):
        """Correct code but no lockout engaged -> generic error, no state change."""
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", _CODE)
        m = _stub()
        assert not m._estop_lockout.is_set()

        assert m._resume_lockout(_CODE) == _GENERIC_ERROR
        # Resuming a non-lockout must not flip lockout state either way.
        assert not m._estop_lockout.is_set()

    def test_all_denial_reasons_share_one_response_shape(self, monkeypatch):
        """Not-engaged, not-configured, and wrong-code denials are indistinguishable."""
        # Not engaged (code configured).
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", _CODE)
        not_engaged = _stub()._resume_lockout(_CODE)

        # Override code unconfigured, lockout engaged.
        monkeypatch.delenv("STRANDS_MESH_OVERRIDE_CODE", raising=False)
        m_unconfigured = _stub()
        m_unconfigured._estop_lockout.set()
        not_configured = m_unconfigured._resume_lockout("anything")

        # Wrong code, lockout engaged.
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", _CODE)
        m_bad = _stub()
        m_bad._estop_lockout.set()
        bad_code = m_bad._resume_lockout("definitely-wrong")

        assert not_engaged == not_configured == bad_code == _GENERIC_ERROR
        # Wrong code leaves the lockout engaged (denied, not cleared).
        assert m_bad._estop_lockout.is_set()

    def test_wire_event_carries_opaque_reason_code_only(self, monkeypatch):
        """The broadcast denial event never leaks the structured human reason."""
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", _CODE)
        m = _stub()  # lockout not engaged -> "lockout not engaged" reason internally

        m._resume_lockout(_CODE)

        assert len(m._published_events) == 1
        event = m._published_events[0]
        assert event["event_type"] == "resume_denied"
        payload = event["payload"]
        assert payload == {"sender_id": "p", "reason_code": "denied"}
        # The human reason text must not appear anywhere on the wire.
        assert "reason" not in payload
        assert "engaged" not in repr(event)

    def test_denial_survives_audit_sink_failures(self, monkeypatch):
        """Both audit sinks raising must not turn a denial into an exception."""
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", _CODE)
        m = _stub()

        def _raise_os(*_a, **_k):
            raise OSError("audit disk full")

        def _raise_wire(**_k):
            raise ValueError("wire publisher down")

        # Local file audit (module-level import in core) raises OSError; the
        # broadcast audit raises ValueError. Both are inside the best-effort
        # try/except in _emit_resume_denied.
        monkeypatch.setattr(core, "log_safety_event", _raise_os)
        m.publish_safety_event = _raise_wire  # type: ignore[method-assign]

        assert m._resume_lockout(_CODE) == _GENERIC_ERROR
