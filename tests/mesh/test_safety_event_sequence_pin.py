"""Pin the safety path's observable behaviour before it is refactored.

One scripted scenario drives the REAL estop/resume handlers of real ``Mesh``
peers (minus the Zenoh transport) through every branch the two remote
handlers and the local paths have: engage, redundant engage, replay, a resume
refused for the wrong code, a resume refused for a malformed envelope, the
real HMAC resume, its replay, its redundant re-apply, and an audit sink that
raises mid-scenario. The test records the exact ordered sequence of
``(peer, event_type, severity, payload keys)`` for every mesh-published and
every local-only audit record, the lockout state after each step, and the
literal text of every refusal warning - and asserts it against the sequence
frozen below.

The frozen sequence was captured on the tree BEFORE the handlers were
restructured; the refactor is correct only while this test does not move.
"""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace

import pytest

from strands_robots.mesh import audit
from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import security as mesh_security

ISSUER = "pin-issuer"
PEER = "pin-peer"
CODE = "pin-override-code"


class _StoppableRobot:
    name = "pin-robot"

    def stop_task(self) -> dict:
        return {"ok": True, "status": "stopped"}


def _mesh(peer_id: str) -> mesh_core.Mesh:
    mesh = mesh_core.Mesh(_StoppableRobot(), peer_id=peer_id, peer_type="robot")
    mesh._running = True
    mesh._stop_event.set()
    return mesh


def _sample(payload: dict) -> SimpleNamespace:
    raw = json.dumps(payload).encode()
    return SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda r=raw: r))


@pytest.fixture
def isolated_audit(monkeypatch, tmp_path):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "pin-psk")
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", CODE)
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._AUDIT_STATE.psk_fingerprint = None
    yield
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._AUDIT_STATE.psk_fingerprint = None


def _record_everything(monkeypatch):
    """Tap the two audit sinks and the wire; return the growing trace list."""
    trace: list[tuple] = []
    original_publish = mesh_core.Mesh.publish_safety_event
    original_log = mesh_core.log_safety_event
    failing = {"publish": False}

    def publish(self, event_type, severity="warning", payload=None):
        trace.append(("wire", self.peer_id, event_type, severity, tuple(sorted((payload or {}).keys()))))
        if failing["publish"] and self.peer_id == PEER:
            raise OSError("audit sink unavailable")
        return original_publish(self, event_type, severity=severity, payload=payload)

    def log(event_type, peer_id, payload):
        trace.append(("local", peer_id, event_type, tuple(sorted(payload.keys()))))
        return original_log(event_type, peer_id, payload)

    monkeypatch.setattr(mesh_core.Mesh, "publish_safety_event", publish)
    monkeypatch.setattr(mesh_core, "log_safety_event", log)
    published: list[tuple[str, dict]] = []
    monkeypatch.setattr(mesh_core, "put", lambda key, payload: published.append((key, payload)))
    return trace, published, failing


def _refusal_templates(caplog) -> list[str]:
    """The literal format strings of every refusal warning, arguments not applied."""
    return [r.msg for r in caplog.records if r.levelno == logging.WARNING and "[safety]" in r.msg]


EXPECTED_TRACE = [
    (
        "wire",
        ISSUER,
        "emergency_stop",
        "critical",
        ("lockout_engaged", "peers_not_stopped", "responses_received", "sender_id"),
    ),
    ("wire", PEER, "remote_estop_engaged", "critical", ("issuer", "issuer_t", "trigger")),
    ("wire", PEER, "estop_replay_rejected", "warning", ("issuer", "issuer_t")),
    (
        "wire",
        ISSUER,
        "emergency_stop",
        "critical",
        ("lockout_engaged", "peers_not_stopped", "responses_received", "sender_id"),
    ),
    ("wire", PEER, "remote_estop_redundant", "info", ("issuer", "issuer_t", "lockout_engaged_since")),
    ("local", PEER, "resume_denied", ("reason", "sender_id", "severity")),
    ("wire", PEER, "resume_denied", "warning", ("reason_code", "sender_id")),
    (
        "wire",
        ISSUER,
        "emergency_stop",
        "critical",
        ("lockout_engaged", "peers_not_stopped", "responses_received", "sender_id"),
    ),
    ("wire", PEER, "remote_estop_redundant", "info", ("issuer", "issuer_t", "lockout_engaged_since")),
    ("wire", ISSUER, "resume_ok", "info", ("lockout_elapsed_s", "sender_id")),
    ("wire", PEER, "remote_resume_applied", "info", ("issuer", "issuer_t", "trigger")),
    ("wire", PEER, "resume_replay_rejected", "warning", ("issuer", "proof_nonce_prefix")),
    ("wire", ISSUER, "resume_ok", "info", ("lockout_elapsed_s", "sender_id")),
    ("wire", PEER, "remote_resume_redundant", "info", ("issuer", "issuer_t", "trigger")),
]

EXPECTED_LOCKOUT = [False, True, True, True, True, True, False, False, False]

EXPECTED_REFUSALS = [
    "[safety] %s: REJECTED remote estop -- replay of (issuer=%s, t=%s) already accepted",
    "[safety] %s: refusing remote resume -- envelope missing override_proof / proof_nonce",
    "[safety] %s: refusing remote estop -- envelope missing/invalid ``t``",
    "[safety] %s: refusing remote estop -- ``t``=%s in future (forward_skew_s=%s, now=%s)",
    "[safety] %s: refusing remote estop -- ``t``=%s too old (freshness_window_s=%s, now=%s)",
    "[safety] %s: refusing remote estop -- envelope missing/invalid ``peer_id``",
    "[safety] %s: refusing remote resume -- body source_zid present but wire source_zid absent (publisher misconfigured or attacker stripped SourceInfo)",
    "[safety] %s: resume after %.1fs lockout",
    "[safety] %s: lockout cleared via remote resume from %s",
    "[safety] %s: REJECTED remote resume -- replay of (issuer=%s, proof_nonce=%s) already accepted",
    "[safety] %s: resume after %.1fs lockout",
]


def test_safety_path_event_sequence_is_pinned(isolated_audit, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="strands_robots.mesh.core")
    trace, published, failing = _record_everything(monkeypatch)
    issuer = _mesh(ISSUER)
    peer = _mesh(PEER)
    lockout: list[bool] = []

    def estop_envelope() -> dict:
        return [p for k, p in published if k == "strands/safety/estop"][-1]

    def resume_envelope() -> dict:
        return [p for k, p in published if k == "strands/safety/resume"][-1]

    lockout.append(peer._estop_lockout.is_set())
    # 1-2
    issuer.emergency_stop()
    first = estop_envelope()
    peer._on_safety_estop(_sample(first))
    lockout.append(peer._estop_lockout.is_set())
    # 3
    peer._on_safety_estop(_sample(first))
    lockout.append(peer._estop_lockout.is_set())
    # 4 (a later ``t`` so the replay cache does not see it)
    time.sleep(0.002)
    issuer.emergency_stop()
    peer._on_safety_estop(_sample(estop_envelope()))
    lockout.append(peer._estop_lockout.is_set())
    # a command during lockout: refused with a local-only audit record (never broadcast)
    with pytest.raises(mesh_security.LockoutError):
        peer._dispatch({"action": "execute", "instruction": "keep going"})
    # 5
    assert peer._dispatch({"action": "resume", "override_code": "not-the-code"}) == {
        "status": "error",
        "error": "resume rejected",
    }
    lockout.append(peer._estop_lockout.is_set())
    # malformed resume (no proof): a warning, no audit record, lockout intact
    peer._on_safety_resume(_sample({"peer_id": ISSUER, "t": time.time()}))
    # the four envelope-domain refusals of the shared authentication phases
    peer._on_safety_estop(_sample({"peer_id": ISSUER}))
    peer._on_safety_estop(_sample({"peer_id": ISSUER, "t": time.time() + 3600}))
    peer._on_safety_estop(_sample({"peer_id": ISSUER, "t": time.time() - 3600}))
    peer._on_safety_estop(_sample({"t": time.time()}))
    peer._on_safety_resume(_sample({"peer_id": ISSUER, "t": time.time(), "source_zid": "forged"}))
    # 6
    failing["publish"] = True
    time.sleep(0.002)
    issuer.emergency_stop()
    peer._on_safety_estop(_sample(estop_envelope()))
    failing["publish"] = False
    lockout.append(peer._estop_lockout.is_set())
    # 7
    issuer._estop_lockout.set()
    issuer._last_estop_ts = time.time()
    issuer._last_estop_mono = time.monotonic()
    assert issuer._dispatch({"action": "resume", "override_code": CODE}) == {"status": "ok"}
    proof = resume_envelope()
    peer._on_safety_resume(_sample(proof))
    lockout.append(peer._estop_lockout.is_set())
    # 8
    peer._on_safety_resume(_sample(proof))
    lockout.append(peer._estop_lockout.is_set())
    # 9
    issuer._estop_lockout.set()
    assert issuer._dispatch({"action": "resume", "override_code": CODE}) == {"status": "ok"}
    peer._on_safety_resume(_sample(resume_envelope()))
    lockout.append(peer._estop_lockout.is_set())

    assert trace == EXPECTED_TRACE
    assert lockout == EXPECTED_LOCKOUT
    assert _refusal_templates(caplog) == EXPECTED_REFUSALS
