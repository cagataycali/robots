"""Victim robot the rogue fleet attacks.

This is a real ``Mesh`` instance composed onto a stub robot, configured
via env vars exactly as a production deployment would be. The orchestrator
sets up env + cert paths, launches us, we print a JSON line to stdout
signalling we're ready, then we sit and serve until SIGTERM.

We deliberately use the *production* :class:`strands_robots.mesh.Mesh`
class (not a stub). That way every defence the PR claims (low_pass_filter,
ACL shape gate, replay caches, audit log, ...) is exercised end-to-end
rather than mocked.

Protocol with the orchestrator:
* stdout line ``READY {<json>}`` once the mesh is up. Fields:
    - ``peer_id``
    - ``listen``  (host:port)
    - ``namespace``
    - ``audit_dir`` (path)
* SIGTERM -> graceful stop + ``GOODBYE`` line.

Env vars consumed (see ``run_fleet.py`` for the full posture matrix):
* ``VICTIM_PEER_ID``       -- defaults to ``"victim-r1"``
* ``VICTIM_LISTEN_PORT``   -- TLS listen port
* ``VICTIM_AUDIT_DIR``     -- audit log location
* All ``STRANDS_MESH_*`` vars (the same ones a real operator would set).
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
import time
from dataclasses import dataclass

# Make sure we hit the in-tree mesh, not site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from strands_robots.mesh import init_mesh  # noqa: E402


@dataclass
class StubRobot:
    """Minimal robot surface the Mesh class needs.

    Mesh asks the robot for status / state during heartbeats and via
    ``status`` RPC. We return constant data -- the rogues are testing the
    *wire* and the *payload guard*, not the action policy.
    """

    peer_id: str = "victim-r1"

    # Surface for ``Mesh.send(target, {"action": "status"})``
    def get_task_status(self) -> dict:
        return {"status": "idle", "peer_id": self.peer_id}

    # Surface used by sensors (returns empty so loops are no-ops)
    def get_state(self) -> dict | None:
        return {"position": [0.0, 0.0, 0.0], "battery": 1.0}

    # Surface checked by ``execute`` / ``start`` actions before dispatch.
    def execute(self, instruction: str, **_: object) -> dict:
        return {"ok": True, "echo": instruction}

    def start(self, **_: object) -> dict:
        return {"ok": True}

    def stop(self) -> dict:
        return {"ok": True}

    def reset(self) -> dict:
        return {"ok": True}


def main() -> int:
    peer_id = os.environ.get("VICTIM_PEER_ID", "victim-r1")
    listen_port = int(os.environ.get("VICTIM_LISTEN_PORT", "0"))

    # Wire ZENOH_LISTEN so session.get_session listens on a known port.
    # (session.py reads ZENOH_LISTEN before anything else.)
    if listen_port:
        protocol = "tls" if os.environ.get("STRANDS_MESH_AUTH_MODE", "mtls") == "mtls" else "tcp"
        os.environ["ZENOH_LISTEN"] = f"{protocol}/localhost:{listen_port}"

    robot = StubRobot(peer_id=peer_id)
    mesh = init_mesh(robot, peer_id=peer_id)

    if mesh is None:
        sys.stdout.write(
            json.dumps({"event": "FAILED", "reason": "init_mesh returned None"}) + "\n"
        )
        sys.stdout.flush()
        return 1

    ready = {
        "event": "READY",
        "peer_id": peer_id,
        "listen": f"localhost:{listen_port}",
        "namespace": os.environ.get("STRANDS_MESH_NAMESPACE", "strands"),
        "audit_dir": os.environ.get(
            "STRANDS_MESH_AUDIT_DIR", str(os.path.expanduser("~/.strands_robots"))
        ),
        "auth_mode": os.environ.get("STRANDS_MESH_AUTH_MODE", "mtls"),
    }
    sys.stdout.write(json.dumps(ready) + "\n")
    sys.stdout.flush()

    # Block until SIGTERM. We don't trap SIGINT so Ctrl-C in the
    # orchestrator still works.
    stop = {"flag": False}

    def _term(*_: object) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _term)
    try:
        while not stop["flag"]:
            time.sleep(0.1)
    finally:
        try:
            mesh.stop()
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[victim] mesh.stop raised: {e!r}\n")
        sys.stdout.write(json.dumps({"event": "GOODBYE"}) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
