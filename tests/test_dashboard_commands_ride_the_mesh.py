"""A dashboard command is a ``Mesh.send``, so the mesh's audit path sees it.

``MeshBridge.send_cmd`` used to build its own envelope, publish it on the raw
session and correlate the reply in a private ``_on_response``. That copy
dropped a forged reply with a log line and no audit record, and accepted a
wait budget ``Mesh.send`` refuses. Driving a real ``Mesh`` whose transport is
a loopback pins both: the forged reply lands in ``mesh_audit.jsonl`` as
``response_hijack_rejected`` and the target's reply is the one returned.
"""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import pytest

from strands_robots.dashboard.mesh_bridge import MeshBridge
from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import session as mesh_session


def _sample(payload: dict[str, Any]) -> Any:
    sample = mock.MagicMock()
    sample.payload.to_bytes.return_value = json.dumps(payload).encode()
    return sample


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A bridge whose Mesh publishes into a loopback: an impostor answers first, then the target."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    mesh = mesh_core.Mesh(None, peer_id="dash-safety", peer_type="gateway")
    mesh._running = True
    published: list[str] = []

    def loopback(key: str, msg: dict[str, Any]) -> None:
        published.append(key)
        turn = msg["turn_id"]
        mesh._on_response(_sample({"turn_id": turn, "responder_id": "impostor", "result": {"forged": True}}))
        mesh._on_response(_sample({"turn_id": turn, "responder_id": "arm", "result": {"status": "success"}}))

    monkeypatch.setattr(mesh_core, "put", loopback)
    # The raw session is watched too, so a publish that bypasses Mesh is seen, not lost.
    monkeypatch.setattr(mesh_session, "put", lambda key, msg: published.append(key))
    bridge = MeshBridge(peer_id="dash")
    bridge._running = True
    monkeypatch.setattr(bridge, "_safety_mesh", lambda: mesh)
    return bridge, published, tmp_path / "mesh_audit.jsonl"


def test_a_forged_reply_to_a_dashboard_command_is_audited(wired):
    bridge, published, audit = wired

    result = bridge.send_cmd("arm", {"action": "status"}, timeout=1.0)

    assert published == ["strands/arm/cmd"]
    assert result["responder_id"] == "arm" and result["result"] == {"status": "success"}
    events = [json.loads(line) for line in audit.read_text().splitlines()]
    hijacks = [e for e in events if e.get("event") == "response_hijack_rejected"]
    assert len(hijacks) == 1 and hijacks[0]["payload"]["responder_id"] == "impostor", events


@pytest.mark.parametrize("timeout", [float("nan"), -1.0, float("inf")])
def test_an_unusable_wait_budget_is_refused_before_publishing(wired, timeout):
    bridge, published, _ = wired

    result = bridge.send_cmd("arm", {"action": "status"}, timeout=timeout)

    assert published == []
    assert result["ok"] is False and "timeout" in result["error"]
