"""Acceptance: two mesh peers in two processes exchange state and RPC over mTLS.

Each peer is its own Python process holding ``Robot("so101", mode="sim",
mesh=True)``. They share a CA and nothing else: ``beta`` listens on an explicit
``tls/`` endpoint, ``alpha`` only dials it (no multicast). Over that one link
``alpha`` must see ``beta`` in its peer list, receive ``beta``'s state samples,
get an answer to a ``status`` RPC, and lock ``beta`` out with an e-stop that a
wrong resume code does not clear and the right one does. Real Zenoh, real TLS,
no doubles.
"""

from __future__ import annotations

import datetime
import ipaddress
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("zenoh")
pytest.importorskip("mujoco")
x509 = pytest.importorskip("cryptography.x509")
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID  # noqa: E402

pytestmark = pytest.mark.timeout(300)

RESUME_CODE = "acceptance-resume"

# One peer: answers one JSON request per stdin line with one JSON line.
PEER = """
import json, sys
from strands_robots import Robot

robot = Robot("so101", mode="sim", mesh=True, peer_id=sys.argv[1])
mesh, samples = robot.mesh, []
print(json.dumps({"running": mesh._running}), flush=True)
for line in sys.stdin:
    ask = json.loads(line)
    op = ask["op"]
    if op == "quit":
        break
    if op == "peers":
        out = sorted(p["peer_id"] for p in mesh.peers)
    elif op == "subscribe":
        out = mesh.subscribe(ask["topic"], lambda topic, data: samples.append(topic))
    elif op == "samples":
        out = len(samples)
    elif op == "send":
        out = mesh.send(ask["to"], ask["cmd"], timeout=10)
    elif op == "estop":
        out = [r.get("responder_id") for r in mesh.emergency_stop()]
    elif op == "locked":
        out = mesh._estop_lockout.is_set()
    print(json.dumps({"out": out}, default=str), flush=True)
robot.destroy()
"""


def _issue(name: str, ca_key: Any, ca_name: Any, key: Any) -> Any:
    now = datetime.datetime.now(datetime.UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
    )
    if key is ca_key:
        builder = builder.add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
    else:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        ).add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False
        )
    return builder.sign(ca_key, hashes.SHA256())


def _write_pki(root: Path) -> None:
    """A throwaway fleet CA and one leaf certificate per peer."""
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "acceptance-fleet-ca")])
    pem = serialization.Encoding.PEM
    (root / "ca.pem").write_bytes(_issue("acceptance-fleet-ca", ca_key, ca_name, ca_key).public_bytes(pem))
    for peer in ("alpha", "beta"):
        key = ec.generate_private_key(ec.SECP256R1())
        (root / f"{peer}.pem").write_bytes(_issue(peer, ca_key, ca_name, key).public_bytes(pem))
        key_path = root / f"{peer}.key"
        key_path.write_bytes(key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        key_path.chmod(0o600)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Peer:
    def __init__(self, name: str, root: Path, **endpoints: str) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("STRANDS_MESH", "ZENOH_"))}
        env.update(
            MUJOCO_GL=os.environ.get("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl"),
            STRANDS_MESH_TLS_CA=str(root / "ca.pem"),
            STRANDS_MESH_TLS_CERT=str(root / f"{name}.pem"),
            STRANDS_MESH_TLS_KEY=str(root / f"{name}.key"),
            STRANDS_MESH_ACCEPT_PERMISSIVE_ACL="1",
            STRANDS_MESH_OVERRIDE_CODE=RESUME_CODE,
            STRANDS_MESH_AUDIT_DIR=str(root / f"audit-{name}"),
            **endpoints,
        )
        self.log = root / f"{name}.log"
        self.proc = subprocess.Popen(
            [sys.executable, "-c", PEER, name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log.open("w"),
            text=True,
            env=env,
        )
        started = self._read()
        assert started == {"running": True}, f"{name} mesh did not start: {self.log.read_text()[-2000:]}"

    def _read(self) -> Any:
        line = self.proc.stdout.readline() if self.proc.stdout else ""
        assert line, f"peer exited: {self.log.read_text()[-2000:]}"
        return json.loads(line)

    def ask(self, op: str, **kw: Any) -> Any:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"op": op, **kw}) + "\n")
        self.proc.stdin.flush()
        return self._read()["out"]

    def close(self) -> None:
        if self.proc.poll() is None and self.proc.stdin is not None:
            self.proc.stdin.write('{"op": "quit"}\n')
            self.proc.stdin.flush()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _until(predicate: Any, seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.25)
    return False


def test_two_mesh_peers_exchange_state_and_rpc(tmp_path: Path) -> None:
    _write_pki(tmp_path)
    endpoint = f"tls/127.0.0.1:{_free_port()}"
    beta = _Peer("beta", tmp_path, ZENOH_LISTEN=endpoint)
    alpha = _Peer("alpha", tmp_path, ZENOH_CONNECT=endpoint)
    try:
        assert _until(lambda: "beta" in alpha.ask("peers") and "alpha" in beta.ask("peers"))

        alpha.ask("subscribe", topic="strands/beta/state")
        assert _until(lambda: alpha.ask("samples") > 0), "no state sample from beta reached alpha"

        status = alpha.ask("send", to="beta", cmd={"action": "status"})
        assert (status.get("responder_id"), status["result"].get("status")) == ("beta", "idle"), status

        assert "beta" in alpha.ask("estop")
        assert _until(lambda: beta.ask("locked") is True), "beta did not lock out on alpha's e-stop"

        wrong = alpha.ask("send", to="beta", cmd={"action": "resume", "override_code": "not-the-code"})
        assert (wrong["result"], beta.ask("locked")) == ({"status": "error", "error": "resume rejected"}, True)

        right = alpha.ask("send", to="beta", cmd={"action": "resume", "override_code": RESUME_CODE})
        assert right["result"] == {"status": "ok"}, right
        assert _until(lambda: beta.ask("locked") is False), "the right resume code did not clear beta"
    finally:
        alpha.close()
        beta.close()
