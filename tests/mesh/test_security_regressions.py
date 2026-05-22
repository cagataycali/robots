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
    * that the lockout is engaged (no `lockout: True` flag)
    * how long it's been engaged (no `elapsed`/`since` fields)
    The response is a generic 'command rejected'."""
    from strands_robots.mesh.core import Mesh

    m = Mesh(MagicMock(), peer_id="x")
    m._estop_lockout.set()
    m._last_estop_ts = 0  # imply long lockout
    result = m._dispatch({"action": "execute", "instruction": "go"})
    assert result == {"error": "command rejected"}
    # No leakage:
    assert "lockout" not in result
    assert "elapsed" not in result
    assert "since" not in result


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

    audit._SEQ_COUNTER = 0
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
