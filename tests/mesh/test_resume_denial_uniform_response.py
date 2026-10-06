"""Resume denial contract: uniform response + no reason leak + audit resilience.

:meth:`strands_robots.mesh.core.Mesh._resume_lockout` is the operator resume
path that clears an emergency-stop lockout. Its denial branches are
security-sensitive: a remote prober must not be able to use the *response* of a
rejected resume to learn anything about the lockout's internal state. The
contract these tests pin:

* **Uniform response shape.** Every denial reason -- lockout not engaged,
  no verification key configured, bad signature -- returns the byte-identical
  generic dict ``{"status": "error", "error": "resume rejected"}``. No
  differential response leaks whether the lockout is engaged or whether a key
  is configured.
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

from unittest.mock import MagicMock

import strands_robots.mesh.core as core
from strands_robots.mesh import resume_authority
from strands_robots.mesh.core import Mesh

from ._resume import lock, sign_for, trust_new_key

_GENERIC_ERROR = {"status": "error", "error": "resume rejected"}


def _stub() -> Mesh:
    """A Mesh with a recording ``publish_safety_event``, so tests can assert
    the exact wire payloads emitted by each denial branch."""
    m = Mesh(MagicMock(), "p")
    m._published_events = []  # type: ignore[attr-defined]
    m.publish_safety_event = lambda **kw: m._published_events.append(kw)  # type: ignore[method-assign, attr-defined]
    return m


class TestResumeDenialUniformResponse:
    def test_denied_when_lockout_not_engaged(self, monkeypatch):
        """A signed resume but no lockout engaged -> generic error, no state change."""
        key = trust_new_key(monkeypatch)
        m = _stub()
        assertion = resume_authority.sign_assertion(key, epoch=resume_authority.new_epoch(), targets=["p"])

        assert m._resume_lockout(assertion) == _GENERIC_ERROR
        assert m._resume_lockout(assertion) == _GENERIC_ERROR
        # Resuming a non-lockout must not flip lockout state either way.
        assert not m._estop_lockout.is_set()

    def test_all_denial_reasons_share_one_response_shape(self, monkeypatch):
        """Not-engaged, no-key, and bad-signature denials are indistinguishable."""
        key = trust_new_key(monkeypatch)
        not_engaged = _stub()._resume_lockout({})

        m_bad = _stub()
        lock(m_bad)
        bad_signature = m_bad._resume_lockout({**sign_for(key, m_bad), "targets": ["p", "q"]})

        monkeypatch.delenv(resume_authority.PUBLIC_KEY_ENV)
        m_unconfigured = _stub()
        lock(m_unconfigured)
        not_configured = m_unconfigured._resume_lockout(sign_for(key, m_unconfigured))

        assert not_engaged == not_configured == bad_signature == _GENERIC_ERROR
        assert m_bad._estop_lockout.is_set() and m_unconfigured._estop_lockout.is_set()

    def test_wire_event_carries_opaque_reason_code_only(self, monkeypatch):
        """The broadcast denial event never leaks the structured human reason."""
        m = _stub()  # lockout not engaged -> "lockout not engaged" reason internally

        m._resume_lockout({})

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

        assert m._resume_lockout({}) == _GENERIC_ERROR
