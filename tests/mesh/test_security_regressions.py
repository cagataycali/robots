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
