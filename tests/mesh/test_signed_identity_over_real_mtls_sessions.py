"""Live proof: signed identity over real mTLS Zenoh sessions, end to end.

Three Zenoh sessions in one process, each with its own leaf from one ephemeral
CA, mutually authenticated at the transport: an operator, a robot and an
attacker. The operator's signed command reaches the robot's ``_on_cmd``, is
verified and dispatched, and the robot's signed reply reaches the operator's
``_on_response`` and is accepted. The attacker, a fully admitted mTLS peer with
a valid certificate, then answers in the robot's name and is refused, and sends
a motion command the robot refuses for want of a verifiable signature. Encrypted
links did not help the finding's attacker; the signature is what stops it.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

zenoh = pytest.importorskip("zenoh")
pytest.importorskip("cryptography")

from strands_robots import _motion_grants  # noqa: E402
from strands_robots.mesh import core as mesh_core  # noqa: E402
from strands_robots.mesh import session as mesh_session  # noqa: E402
from strands_robots.mesh import wire_identity as wi  # noqa: E402
from strands_robots.mesh.core import Mesh  # noqa: E402
from tests._wire_identity import identity_for, roots_for  # noqa: E402
from tests.mesh._pki import EphemeralCA  # noqa: E402

pytestmark = pytest.mark.timeout(20)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _mtls_config(*, cert: Path, key: Path, ca: Path, listen_port: int | None = None, connect: int | None = None) -> Any:
    cfg = zenoh.Config()
    cfg.insert_json5("mode", '"client"' if connect else '"peer"')
    cfg.insert_json5("scouting/multicast/enabled", "false")
    cfg.insert_json5("scouting/gossip/enabled", "false")
    cfg.insert_json5("transport/link/protocols", json.dumps(["tls"]))
    cfg.insert_json5(
        "transport/link/tls",
        json.dumps(
            {
                "root_ca_certificate": str(ca),
                "listen_certificate": str(cert),
                "listen_private_key": str(key),
                "connect_certificate": str(cert),
                "connect_private_key": str(key),
                "enable_mtls": True,
                "verify_name_on_connect": False,
            }
        ),
    )
    if listen_port is not None:
        cfg.insert_json5("listen/endpoints", json.dumps([f"tls/127.0.0.1:{listen_port}"]))
    if connect is not None:
        cfg.insert_json5("connect/endpoints", json.dumps([f"tls/127.0.0.1:{connect}"]))
    return cfg


class _Arm:
    """A hardware-shaped robot: answers ``status`` and would ``reset`` if let."""

    tool_name_str = "so101"

    def __init__(self) -> None:
        self.resets = 0

    def get_task_status(self) -> dict[str, Any]:
        return {"status": "idle", "robot": "robot-a"}

    def reset(self) -> dict[str, Any]:
        self.resets += 1
        return {"status": "success"}

    def stop_task(self) -> dict[str, Any]:
        return {"status": "success"}


class _Peer:
    """One Zenoh session plus the un-started ``Mesh`` whose handlers it feeds."""

    def __init__(self, mesh: Mesh, session: Any, ident: wi.WireIdentity, roots: wi.TrustRoots) -> None:
        self.mesh = mesh
        self.session = session
        self.audits: list[tuple[str, dict[str, Any]]] = []
        mesh._wire_identity = ident
        mesh._trust_roots = roots
        setattr(mesh, "_audit_local", lambda event_type, payload: self.audits.append((event_type, payload)))
        # Publish on THIS session (a started Mesh would use the process session).
        setattr(mesh, "publish", lambda key, payload: session.put(key, json.dumps(payload).encode()))

    def close(self) -> None:
        self.session.close()


def _wait_for(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


@pytest.fixture
def fleet(require_signatures: EphemeralCA, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.delenv("STRANDS_ROBOT_COMMAND_ALLOW", raising=False)
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    with _motion_grants._grants_lock:
        _motion_grants._grants.clear()
    with mesh_session._PEERS_LOCK:
        mesh_session._PEERS.clear()
    ca = require_signatures
    roots = roots_for(ca)
    leaves = tmp_path / "leaves"
    idents = {cn: identity_for(ca, cn, leaves) for cn in ("op", "robot-a", "attacker")}
    port = _free_port()
    sessions: dict[str, Any] = {}
    peers: dict[str, _Peer] = {}
    try:
        for cn, mesh in (
            ("op", Mesh(_Arm(), peer_id="op", peer_type="operator")),
            ("robot-a", Mesh(_Arm(), peer_id="robot-a", peer_type="robot")),
            ("attacker", Mesh(_Arm(), peer_id="attacker", peer_type="robot")),
        ):
            cert, key = leaves / cn / f"{cn}.crt", leaves / cn / f"{cn}.key"
            cfg = _mtls_config(
                cert=cert,
                key=key,
                ca=ca.cert_path,
                listen_port=port if cn == "op" else None,
                connect=None if cn == "op" else port,
            )
            sessions[cn] = zenoh.open(cfg)
            peers[cn] = _Peer(mesh, sessions[cn], idents[cn], roots)
        # The robot answers commands on its own cmd key; the operator hears replies.
        subs = [
            sessions["robot-a"].declare_subscriber("strands/robot-a/cmd", peers["robot-a"].mesh._on_cmd),
            sessions["op"].declare_subscriber("strands/op/response/**", peers["op"].mesh._on_response),
            sessions["op"].declare_subscriber("strands/*/presence", peers["op"].mesh._on_presence),
            sessions["robot-a"].declare_subscriber("strands/*/presence", peers["robot-a"].mesh._on_presence),
        ]
        time.sleep(0.6)  # let the declarations settle across the links
        # Everyone announces itself, signed, the way the heartbeat does.
        for cn, peer in peers.items():
            peer.session.put(
                f"strands/{cn}/presence", json.dumps(peer.mesh._sign(peer.mesh._build_presence())).encode()
            )
        assert _wait_for(lambda: peers["op"].mesh.peer_cert("robot-a") is not None)
        assert _wait_for(lambda: peers["robot-a"].mesh.peer_cert("op") is not None)
        yield {"peers": peers, "subs": subs, "ca": ca}
    finally:
        for s in sessions.values():
            s.close()
    with mesh_session._PEERS_LOCK:
        mesh_session._PEERS.clear()


def _open_turn(op: Mesh, expected: str) -> tuple[str, threading.Event]:
    turn = mesh_core.uuid.uuid4().hex
    event = threading.Event()
    with op._rpc_lock:
        op._pending[turn] = event
        op._responses[turn] = []
        op._expected_responders[turn] = expected
    return turn, event


def _responses(op: Mesh, turn: str) -> list[dict[str, Any]]:
    with op._rpc_lock:
        return list(op._responses.get(turn, []))


def test_a_signed_command_and_its_signed_reply_cross_the_mtls_link(fleet: dict[str, Any]) -> None:
    op, robot = fleet["peers"]["op"], fleet["peers"]["robot-a"]
    turn, event = _open_turn(op.mesh, "robot-a")
    msg = op.mesh._sign({"sender_id": "op", "turn_id": turn, "command": {"action": "status"}, "timestamp": time.time()})
    assert "sig" in msg

    op.session.put("strands/robot-a/cmd", json.dumps(msg).encode())

    assert event.wait(5.0), robot.audits
    got = _responses(op.mesh, turn)
    assert len(got) == 1 and got[0]["responder_id"] == "robot-a"
    assert got[0]["result"] == {"status": "idle", "robot": "robot-a"}
    assert "sig" in got[0] and got[0]["sig"]["alg"] == wi.ALG_RSA
    assert [e for e, _ in op.audits] == []


def test_an_admitted_mtls_peer_cannot_answer_in_the_robots_name(fleet: dict[str, Any]) -> None:
    op, robot, attacker = (fleet["peers"][k] for k in ("op", "robot-a", "attacker"))
    turn, event = _open_turn(op.mesh, "robot-a")
    forged = attacker.mesh._sign(
        {
            "type": "response",
            "responder_id": "robot-a",
            "turn_id": turn,
            "result": {"ok": True},
            "timestamp": time.time(),
        }
    )
    bare = {
        "type": "response",
        "responder_id": "robot-a",
        "turn_id": turn,
        "result": {"ok": True},
        "timestamp": time.time(),
    }

    attacker.session.put(f"strands/op/response/robot-a/{turn}", json.dumps(forged).encode())
    attacker.session.put(f"strands/op/response/robot-a/{turn}", json.dumps(bare).encode())
    assert _wait_for(lambda: len([e for e, _ in op.audits if e == "response_hijack_rejected"]) == 2)
    assert not event.is_set() and _responses(op.mesh, turn) == []
    reasons = sorted(p["reason"] for e, p in op.audits if e == "response_hijack_rejected")
    assert reasons[0] == "message carries no signature envelope"
    assert "does not speak for" in reasons[1]
    assert sorted((p["signer"] or "") for e, p in op.audits if e == "response_hijack_rejected") == ["", "attacker"]

    # The robot's own signed answer on the same turn is still accepted.
    genuine = robot.mesh._sign(
        {
            "type": "response",
            "responder_id": "robot-a",
            "turn_id": turn,
            "result": {"ok": False},
            "timestamp": time.time(),
        }
    )
    robot.session.put(f"strands/op/response/robot-a/{turn}", json.dumps(genuine).encode())
    assert event.wait(5.0)
    assert [r["result"] for r in _responses(op.mesh, turn)] == [{"ok": False}]


def test_an_unsigned_or_misattributed_motion_command_does_not_move_the_robot(
    fleet: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    op, robot, attacker = (fleet["peers"][k] for k in ("op", "robot-a", "attacker"))
    monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", "reset@op")
    arm = robot.mesh.robot

    # Unsigned, claiming to be the operator.
    attacker.session.put(
        "strands/robot-a/cmd",
        json.dumps(
            {"sender_id": "op", "turn_id": "a" * 32, "command": {"action": "reset"}, "timestamp": time.time()}
        ).encode(),
    )
    # Signed by the attacker's own valid certificate, claiming to be the operator.
    attacker.session.put(
        "strands/robot-a/cmd",
        json.dumps(
            attacker.mesh._sign(
                {"sender_id": "op", "turn_id": "b" * 32, "command": {"action": "reset"}, "timestamp": time.time()}
            )
        ).encode(),
    )
    assert _wait_for(lambda: len([e for e, _ in robot.audits if e == "wire_motion_refused"]) == 2)
    assert arm.resets == 0
    reasons = [p["reason"] for e, p in robot.audits if e == "wire_motion_refused"]
    assert "the command carries no verifiable signature" in reasons
    assert any("does not speak for" in r for r in reasons)

    # The operator's own signed reset, pre-approved for the operator, proceeds.
    turn, event = _open_turn(op.mesh, "robot-a")
    op.session.put(
        "strands/robot-a/cmd",
        json.dumps(
            op.mesh._sign(
                {"sender_id": "op", "turn_id": turn, "command": {"action": "reset"}, "timestamp": time.time()}
            )
        ).encode(),
    )
    assert event.wait(5.0), robot.audits
    assert arm.resets == 1
