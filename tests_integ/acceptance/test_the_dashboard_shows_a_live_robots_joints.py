"""Acceptance: the dashboard shows a live robot's joints byte-equal to the robot's own read.

The robot is its own Python process holding ``Robot("so101", mode="sim",
mesh=True)``; it moves the shoulder lift, steps, and prints the joint positions
it reads itself. The dashboard is the real FastAPI app with its own Zenoh
session, dialing that one peer on an explicit endpoint. ``GET /api/fleet`` must
carry every joint with the exact float the robot read - no rounding, no
reordering, no lost joint. Real MuJoCo, real Zenoh, no doubles.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("zenoh")
pytest.importorskip("mujoco")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

pytestmark = pytest.mark.timeout(300)

TOKEN = "acceptance-dashboard-bootstrap"

# The robot: move one joint, report what it reads, hold still until told to quit.
PEER = """
import json, sys
from strands_robots import Robot

robot = Robot("so101", mode="sim", mesh=True, peer_id="arm")
robot.send_action({"2": 0.5}, robot_name="so101")
robot.step(200)
obs = robot.get_observation("so101", skip_images=True)
print(json.dumps({k: v for k, v in obs.items() if not k.endswith(".vel")}), flush=True)
sys.stdin.readline()
robot.destroy()
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith(("STRANDS_MESH", "ZENOH_"))}


def test_the_dashboard_shows_a_live_robots_joints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = f"tcp/127.0.0.1:{_free_port()}"
    env = _clean_env()
    env.update(
        MUJOCO_GL=os.environ.get("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl"),
        STRANDS_MESH_LOCAL_DEV="true",
        STRANDS_MESH_AUDIT_DIR=str(tmp_path / "audit-arm"),
        ZENOH_LISTEN=endpoint,
    )
    log = tmp_path / "arm.log"
    arm = subprocess.Popen(
        [sys.executable, "-c", PEER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log.open("w"),
        text=True,
        env=env,
    )
    try:
        line = arm.stdout.readline() if arm.stdout else ""
        assert line, f"robot exited: {log.read_text()[-2000:]}"
        own = json.loads(line)
        assert sorted(own) == ["1", "2", "3", "4", "5", "6"] and abs(own["2"] - 0.5) < 0.05, own

        for key in [k for k in os.environ if k.startswith(("STRANDS_MESH", "ZENOH_"))]:
            monkeypatch.delenv(key)
        monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "true")
        monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit-dashboard"))
        monkeypatch.setenv("ZENOH_CONNECT", endpoint)
        monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", TOKEN)
        from strands_robots.dashboard.server import create_app

        shown: dict[str, float] = {}
        with TestClient(create_app()) as client:
            deadline = time.monotonic() + 30
            while not shown and time.monotonic() < deadline:
                fleet = client.get("/api/fleet", headers={"authorization": f"Bearer {TOKEN}"}).json()
                joints = (((fleet.get("peers") or {}).get("arm__so101") or {}).get("state") or {}).get("joints") or {}
                shown = {name: reading["position"] for name, reading in joints.items()}
                if not shown:
                    time.sleep(0.25)
            assert fleet["mesh_online"] is True, fleet.get("mesh_error")
        assert shown == own, f"dashboard {shown} != robot {own}"
    finally:
        if arm.poll() is None and arm.stdin is not None:
            arm.stdin.write("\n")
            arm.stdin.flush()
        try:
            arm.wait(timeout=30)
        except subprocess.TimeoutExpired:
            arm.kill()
