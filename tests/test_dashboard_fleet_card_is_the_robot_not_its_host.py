"""A spawned sim robot is one Fleet card, and a resume the peers confirm clears its badge.

Spawning a sim robot puts two peers on the mesh: the host process ``<host>`` and
the robot it holds, ``<host>__<robot>``. Every command for the robot is routed to
the host (``route_task_target``), so the host's Mesh is the one that accepts or
refuses it under a lockout. Two things followed from reading those as two robots:
the host got a card of its own (no joints, its own Run button), and the proof a
lockout cleared - a command the host accepted - was stamped on the host, so the
robot's card kept "e-stop?" forever. After a resume the dashboard now asks every
live host for ``state`` (a read a lockout refuses) and stamps the answers.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any
from unittest import mock

import pytest

from strands_robots.dashboard import safety_state
from strands_robots.dashboard.mesh_bridge import MeshBridge
from strands_robots.mesh import core as mesh_core
from tests._mesh_reply import reply_sample

HOST, ROBOT, LOCKED = "so101-sim-7487", "so101-sim-7487__so101", "arm-b"
APP = pathlib.Path(__file__).parent.parent / "strands_robots" / "dashboard" / "frontend" / "src" / "App.tsx"


def _sample(payload: dict[str, Any]) -> Any:
    sample = mock.MagicMock()
    sample.payload.to_bytes.return_value = json.dumps(payload).encode()
    return sample


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """A bridge on a loopback mesh with two real host Meshes: one clear, one still locked."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    dash = mesh_core.Mesh(None, peer_id="dash-safety", peer_type="gateway")
    hosts = {pid: mesh_core.Mesh(None, peer_id=pid) for pid in (HOST, LOCKED)}
    hosts[LOCKED]._estop_lockout.set()
    for m in (dash, *hosts.values()):
        m._running = True

    def loopback(key: str, msg: dict[str, Any]) -> None:
        parts = key.split("/")
        if parts[-1] == "cmd" and parts[1] in hosts:
            hosts[parts[1]]._on_cmd(_sample(msg))
        elif "response" in parts:
            dash._on_response(reply_sample(dash, msg))

    monkeypatch.setattr(mesh_core, "put", loopback)
    bridge = MeshBridge(peer_id="dash")
    bridge._running = True
    monkeypatch.setattr(bridge, "_safety_mesh", lambda: dash)
    now = time.time()
    bridge.peers = {pid: {"peer_id": pid, "last_seen": now, "first_seen": now - 60} for pid in (HOST, ROBOT, LOCKED)}
    bridge._lockout = safety_state.apply_event(bridge._lockout, kind="estop", data={}, now=now - 30)
    return bridge


def _badges(bridge: MeshBridge) -> dict[str, str]:
    return {pid: peer["lockout"]["state"] for pid, peer in bridge.snapshot()["peers"].items()}


def test_a_command_the_host_accepted_clears_the_robot_it_hosts(fleet) -> None:
    fleet._lockout = safety_state.apply_event(fleet._lockout, kind="resume", data={}, now=time.time() - 10)
    # The Run button on the robot's card sends to the host: that acceptance is the robot's proof.
    assert not fleet.send_cmd(HOST, {"action": "state"}, timeout=1.0).get("error")
    assert _badges(fleet) == {HOST: "clear", ROBOT: "clear", LOCKED: "unknown"}


def test_a_resume_is_confirmed_by_the_hosts_that_answer_and_only_by_them(fleet) -> None:
    sent_at = time.time()
    fleet._lockout = safety_state.apply_event(fleet._lockout, kind="resume", data={}, now=sent_at)
    assert fleet.confirm_resume(sent_at, timeout=1.0) == [HOST]
    # The peer still refusing keeps its question mark: nothing proved it clear.
    assert _badges(fleet) == {HOST: "clear", ROBOT: "clear", LOCKED: "unknown"}


def test_a_resume_this_dashboard_never_heard_proves_nothing(fleet) -> None:
    # Proof must postdate the resume the verdict is ordered on; without it no probe is sent.
    assert fleet.confirm_resume(time.time(), wait_s=0.1, timeout=1.0) == []
    assert set(_badges(fleet).values()) == {"locked"}


def test_the_fleet_renders_no_card_for_a_host_process() -> None:
    source = APP.read_text(encoding="utf-8")
    assert "const cards = useMemo(() => list.filter(p => !fleetHosts[p.peer_id])" in source
    assert "{cards.map(p => (" in source and "lockoutBanner(cards)" in source
    assert "{list.map(p => (" not in source, "a card per peer puts the host process on the Fleet as a robot"
