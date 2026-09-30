"""Regression for GH #4172: ``robot_mesh`` returned a peer's reply cut at 600 characters.

``tell``, ``send`` and ``stop`` rendered the envelope as ``json.dumps(result)[:600]``
inside one text block. A normal ``run_policy`` result is about 1,800 characters,
so the agent read JSON cut mid key (``"stopped_early": fa``) and lost the rollout
metrics the sim tool and the dashboard both carry whole.

Pinned here: every verb that relays a peer's answer hands it back as a ``json``
content block equal to what the mesh returned, whatever its size, next to a short
text label naming the verb and the target.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.tools.robot_mesh import robot_mesh


def _call(**kwargs: Any) -> dict[str, Any]:
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = "y"
    fn = getattr(robot_mesh, "original", robot_mesh)
    return fn(tool_context=ctx, **kwargs)


@pytest.fixture
def fake_mesh():
    fake = MagicMock(name="LocalMesh")
    fake.peer_id = "local-a"
    fake.peer_type = "sim"
    fake.inbox = {}
    with (
        patch("strands_robots.mesh.get_local_robots", return_value={"local-a": fake}),
        patch("strands_robots.mesh.session.get_peers", return_value=[]),
    ):
        yield fake


def _rollout_envelope() -> dict[str, Any]:
    """An envelope the size a real rollout answers with: well over 600 characters."""
    result = {
        "status": "success",
        "steps": 150,
        "stopped_early": False,
        "duration_s": 15.02,
        "joint_names": [f"joint_{i}" for i in range(6)],
        "final_joints": [0.11 * i for i in range(6)],
        "metrics": {f"metric_{i}": float(i) / 7 for i in range(40)},
    }
    return {"type": "response", "responder_id": "peer-b", "turn_id": "t" * 32, "result": result, "timestamp": 1.5}


def _json_block(out: dict[str, Any]) -> Any:
    blocks = [block["json"] for block in out["content"] if "json" in block]
    assert len(blocks) == 1, out["content"]
    return blocks[0]


def test_tell_returns_the_whole_envelope_as_a_json_block(fake_mesh):
    envelope = _rollout_envelope()
    assert len(json.dumps(envelope)) > 600
    fake_mesh.tell.return_value = envelope
    out = _call(action="tell", target="peer-b", instruction="pick up the red cube")
    assert out["status"] == "success"
    assert _json_block(out) == envelope
    assert _json_block(out)["result"]["stopped_early"] is False
    assert any("[tell -> peer-b]" in block.get("text", "") for block in out["content"])


def test_send_returns_the_whole_envelope_as_a_json_block(fake_mesh):
    envelope = _rollout_envelope()
    fake_mesh.send.return_value = envelope
    out = _call(action="send", target="peer-b", command='{"action": "status"}')
    assert out["status"] == "success"
    assert _json_block(out) == envelope


def test_stop_returns_the_whole_envelope_as_a_json_block(fake_mesh):
    envelope = {
        "type": "response",
        "responder_id": "peer-b",
        "turn_id": "t" * 32,
        "result": {"ok": True, "stopped": True, "detail": "x" * 700},
    }
    fake_mesh.send.return_value = envelope
    out = _call(action="stop", target="peer-b")
    assert out["status"] == "success"
    assert _json_block(out) == envelope


def test_the_json_block_is_json_not_a_python_repr(fake_mesh):
    """A value ``json.dumps`` cannot encode is rendered the way the text used to render it."""
    envelope = {"type": "response", "responder_id": "peer-b", "result": {"when": object()}}
    fake_mesh.tell.return_value = envelope
    out = _call(action="tell", target="peer-b", instruction="go")
    block = _json_block(out)
    json.dumps(block)  # strictly encodable
    assert isinstance(block["result"]["when"], str)
