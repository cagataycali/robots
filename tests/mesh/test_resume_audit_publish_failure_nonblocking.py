"""Audit-publish failure must never break the remote-resume safety path.

``_on_safety_resume`` emits a forensic audit event when it refuses a remote
resume (``resume_denied``, e.g. a replayed assertion) and when a valid one
arrives with nothing to clear (``remote_resume_redundant``). Both publishes are
best-effort: a flaky or full-disk audit backend must never abort handling and
let the fleet slip into a half-state. The estop path has the equivalent
coverage in ``test_estop_audit_publish_failure_nonblocking``.
"""

import json
from unittest.mock import MagicMock

from strands_robots.mesh.core import Mesh

from ._resume import lock, resume_sample, sign_for, trust_new_key


def _mesh_whose_audit_raises(event_type: str, exc: Exception) -> tuple[Mesh, list[dict]]:
    m = Mesh(MagicMock(), "r-test")
    calls: list[dict] = []

    def audit(**kwargs):
        calls.append(kwargs)
        if kwargs.get("event_type") == event_type:
            raise exc

    m.publish_safety_event = audit  # type: ignore[assignment, method-assign]
    return m, calls


def test_a_refused_replay_with_a_failing_audit_keeps_the_lockout(monkeypatch):
    key = trust_new_key(monkeypatch)
    m, calls = _mesh_whose_audit_raises("resume_denied", OSError("audit log volume full"))
    epoch = lock(m)
    assertion = sign_for(key, m)
    m._on_safety_resume(resume_sample(assertion))
    assert not m._estop_lockout.is_set()

    lock(m, epoch)
    m._on_safety_resume(resume_sample(assertion))  # the replay; must not raise despite the audit OSError

    assert m._estop_lockout.is_set() is True
    assert [c["event_type"] for c in calls].count("resume_denied") == 1


def test_a_redundant_resume_with_a_failing_audit_does_not_raise(monkeypatch):
    key = trust_new_key(monkeypatch)
    m, calls = _mesh_whose_audit_raises("remote_resume_redundant", ValueError("bad audit payload shape"))
    epoch = lock(m)
    m._estop_lockout.clear()  # nothing to clear when the resume arrives

    m._on_safety_resume(resume_sample(sign_for(key, m, epoch=epoch)))

    assert [c["event_type"] for c in calls] == ["remote_resume_redundant"], json.dumps(calls, default=str)
    assert m._estop_lockout.is_set() is False
