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

import pytest

from strands_robots.dashboard.mesh_bridge import MeshBridge
from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import session as mesh_session
from tests._mesh_reply import reply_sample


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
        mesh._on_response(reply_sample(mesh, {"turn_id": turn, "responder_id": "impostor", "result": {"forged": True}}))
        mesh._on_response(reply_sample(mesh, {"turn_id": turn, "responder_id": "arm", "result": {"status": "success"}}))

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


def test_the_delivery_verdict_reaches_the_dashboard_caller(wired, monkeypatch):
    """With a direct transport, ``send_cmd`` passes ``delivery`` through on every shape it returns.

    An offline peer arrives as ``status == "error"`` in one round trip and is
    already reported without waiting the timeout out; the verdict beside it
    lets the console say "gone" rather than "slow".
    """
    bridge, _published, _ = wired
    mesh = bridge._safety_mesh()
    answers = iter(
        [
            {
                "status": "error",
                "error": "peer offline (iot 404)",
                "peer": "arm",
                "delivery": {"via": "direct", "confirmed": False, "latency_ms": 81.0, "reason": "offline"},
            },
            {"status": "timeout", "delivery": {"via": "direct", "confirmed": True, "latency_ms": 156.0, "reason": ""}},
            {
                "type": "response",
                "responder_id": "arm",
                "result": {"ok": 1},
                "delivery": {"via": "publish", "confirmed": False, "latency_ms": 2.0, "reason": "forbidden"},
            },
        ]
    )
    monkeypatch.setattr(mesh, "send", lambda target, cmd, timeout=30.0: next(answers))

    offline = bridge.send_cmd("arm", {"action": "status"}, timeout=1.0)
    assert offline["ok"] is False and offline["error"] == "peer offline (iot 404)"
    assert offline["delivery"]["reason"] == "offline"

    slow = bridge.send_cmd("arm", {"action": "status"}, timeout=1.0)
    assert slow["ok"] is False and slow["delivery"] == {
        "via": "direct",
        "confirmed": True,
        "latency_ms": 156.0,
        "reason": "",
    }

    answered = bridge.send_cmd("arm", {"action": "status"}, timeout=1.0)
    assert answered["result"] == {"ok": 1} and answered["delivery"]["via"] == "publish"
