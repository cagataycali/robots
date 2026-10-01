"""The dashboard's sim peer tool IS the simulation tool, and a spawned peer is usable in the same turn.

The owner spawned an so101 as a mesh peer and asked the agent to add a red cube
into that sim; the agent answered that none of its tools add objects, and the
next message could not use the new peer either. Pinned here, without a mesh or
a model call:

* the proxy: a sim peer's tool advertises the mesh verbs plus every published
  simulation action the wire carries (``sim_call_allowed_actions``), never one
  it refuses, with the published params (descriptions cut to a sentence) and
  never a refused one; an action maps onto a validated ``sim_call`` command; a
  denied action names its rail before any round trip; an image the peer sent
  base64 reaches the model as bytes;
* the console: ``spawn_robot`` registers the new peer's tools into the LIVE
  agent (``Console.adopt``), the motion hook learns them, the fleet signature
  grows so the next turn does not rebuild, and the tool result says the tools
  are callable now;
* the bridge: a mesh session that refuses to open leaves its reason on the
  bridge and in ``mesh_info()``, which is what the fleet bar's banner reads.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.dashboard import agent_console, peer_tools
from strands_robots.mesh import security

# ─────────────────────────────────────────────── the proxy ─────────────────

SIM_PEER: dict[str, Any] = {
    "presence": {"robot_type": "sim", "hostname": "lab"},
    "state": {"joints": {"1": {}}},
}


def _spec() -> dict[str, Any]:
    spec = peer_tools.peer_tool_spec("so101-sim-1__so101", peer_tools.KIND_SIM, "so101_sim_1__so101")
    assert spec is not None
    return spec


def test_the_sim_proxy_offers_the_verbs_and_every_published_action_the_wire_carries() -> None:
    props = _spec()["inputSchema"]["json"]["properties"]
    actions: list[str] = props["action"]["enum"]
    assert actions[: len(peer_tools.SIM_ACTIONS)] == list(peer_tools.SIM_ACTIONS)
    assert len(actions) == len(set(actions)), "an action is offered twice"
    offered = set(actions)
    assert "add_object" in offered and "render" in offered and "list_objects" in offered
    assert offered.isdisjoint(security.SIM_CALL_DENIED_ACTIONS)
    assert "sim_call" not in offered
    assert offered - set(peer_tools.SIM_ACTIONS) == security.sim_call_allowed_actions() - set(peer_tools.SIM_ACTIONS)


def test_the_sim_proxy_carries_the_published_params_and_never_a_refused_one() -> None:
    props = _spec()["inputSchema"]["json"]["properties"]
    for name in ("name", "shape", "size", "color", "position", "camera_name", "positions"):
        assert name in props, name
    assert set(props).isdisjoint(security.SIM_CALL_DENIED_PARAMS)
    assert security.sim_call_allowed_params() <= set(props)
    for name, prop in props.items():
        description = prop.get("description")
        if isinstance(description, str) and name != "action":
            assert len(description) <= peer_tools._SIM_PARAM_DESCRIPTION_CHARS, name
    # the mesh verbs' own wording wins for a name both vocabularies use
    assert props["robot_name"]["description"].startswith("a Simulation holding several robots")


def test_every_advertised_action_maps_onto_a_command_the_wire_accepts() -> None:
    for action in _spec()["inputSchema"]["json"]["properties"]["action"]["enum"]:
        if action in peer_tools.SIM_ACTIONS:
            tool_input: dict[str, Any] = {"action": action, "instruction": "go", "target_joints": {"a": 1.0}}
        else:
            tool_input = {"action": action}
        cmd, err = peer_tools.map_invocation("p", peer_tools.KIND_SIM, tool_input)
        assert err is None and cmd is not None, (action, err)
        security.validate_command(cmd)


def test_add_object_maps_onto_a_sim_call_with_its_parameters() -> None:
    cmd, err = peer_tools.map_invocation(
        "p",
        peer_tools.KIND_SIM,
        {
            "action": "add_object",
            "name": "red_cube",
            "shape": "box",
            "size": [0.02, 0.02, 0.02],
            "color": [1, 0, 0, 1],
            "position": [0.25, 0, 0.02],
            "robot_name": None,
        },
    )
    assert err is None
    assert cmd == {
        "action": "sim_call",
        "sim_action": "add_object",
        "params": {
            "name": "red_cube",
            "shape": "box",
            "size": [0.02, 0.02, 0.02],
            "color": [1, 0, 0, 1],
            "position": [0.25, 0, 0.02],
        },
    }
    assert security.validate_command(cmd)["params"]["name"] == "red_cube"


@pytest.mark.parametrize(
    ("tool_input", "fragment"),
    [
        ({"action": "run_policy", "instruction": "wave"}, "use action='execute' on this tool"),
        ({"action": "stop_policy"}, "use action='stop' on this tool"),
        ({"action": "destroy"}, "not carried over the mesh"),
        ({"action": "open_viewer"}, "not carried over the mesh"),
        ({"action": "render", "output_path": "/tmp/x.png"}, "output_path cannot travel"),
        ({"action": "start_recording", "push_to_hub": True, "root": "/data"}, "push_to_hub, root cannot travel"),
        ({"action": "add_object", "name": "c", "target_joints": {"1": 0.1}}, "target_joints belongs to the mesh verbs"),
        ({"action": "teleport"}, "unknown action 'teleport'"),
    ],
)
def test_a_refused_sim_call_is_refused_before_the_wire_with_the_reason(
    tool_input: dict[str, Any], fragment: str
) -> None:
    cmd, err = peer_tools.map_invocation("p", peer_tools.KIND_SIM, tool_input)
    assert cmd is None and err is not None and fragment in err, err


def test_an_image_the_peer_sent_base64_reaches_the_model_as_bytes() -> None:
    png = b"\x89PNG-not-really"
    sent: list[tuple[str, dict[str, Any]]] = []

    def send_cmd(target: str, cmd: dict[str, Any], timeout: float = 30.0, *, source: str = "api") -> dict[str, Any]:
        sent.append((target, cmd))
        return {
            "status": "success",
            "content": [
                {"text": "160x120"},
                {"image": {"format": "png", "base64": base64.b64encode(png).decode("ascii"), "bytes_len": len(png)}},
                {"image": {"format": "png", "base64": "@@not-base64@@"}},
            ],
        }

    [tool] = peer_tools.build_peer_tools({"so101-sim-1__so101": SIM_PEER}, send_cmd)

    async def _run() -> list[Any]:
        events = []
        async for event in tool.stream(
            {"toolUseId": "t1", "input": {"action": "render", "camera_name": "wrist", "width": 160, "height": 120}},
            {},
        ):
            events.append(event)
        return events

    [event] = asyncio.run(_run())
    result = event.tool_result if hasattr(event, "tool_result") else event["tool_result"]
    assert result["status"] == "success"
    assert sent[0][1] == {
        "action": "sim_call",
        "sim_action": "render",
        "params": {"camera_name": "wrist", "width": 160, "height": 120},
    }
    assert result["content"][0] == {"text": "160x120"}
    assert result["content"][1] == {"image": {"format": "png", "source": {"bytes": png}}}
    assert "could not be decoded" in result["content"][2]["text"]


def _stream(tool: Any, tool_input: dict[str, Any]) -> dict[str, Any]:
    async def _run() -> list[Any]:
        return [event async for event in tool.stream({"toolUseId": "t1", "input": tool_input}, {})]

    [event] = asyncio.run(_run())
    return event.tool_result if hasattr(event, "tool_result") else event["tool_result"]


def test_the_peers_answer_inside_the_wire_envelope_is_what_the_model_reads() -> None:
    envelope = {
        "type": "response",
        "responder_id": "so101-sim-1",
        "turn_id": "t",
        "result": {"status": "success", "content": [{"text": "'red_cube' added"}]},
        "timestamp": 1.0,
    }
    [tool] = peer_tools.build_peer_tools({"so101-sim-1__so101": SIM_PEER}, lambda *a, **k: dict(envelope))
    result = _stream(tool, {"action": "list_objects"})
    assert result["status"] == "success"
    assert result["content"] == [{"text": "'red_cube' added"}]

    refused = {
        "type": "response",
        "responder_id": "so101-sim-1",
        "result": {"status": "error", "content": [{"text": "no"}]},
    }
    [tool] = peer_tools.build_peer_tools({"so101-sim-1__so101": SIM_PEER}, lambda *a, **k: dict(refused))
    assert _stream(tool, {"action": "list_objects"})["status"] == "error"

    offline = {"ok": False, "error": "mesh offline"}
    [tool] = peer_tools.build_peer_tools({"so101-sim-1__so101": SIM_PEER}, lambda *a, **k: dict(offline))
    result = _stream(tool, {"action": "list_objects"})
    assert result["status"] == "error" and "mesh offline" in result["content"][0]["text"]


# ────────────────────────────────────────────── the console ────────────────


class FakeBridge:
    def __init__(self, peers: dict[str, Any]) -> None:
        self.peers = dict(peers)
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def snapshot(self) -> dict[str, Any]:
        return {"peers": dict(self.peers)}

    def send_cmd(
        self, target: str, cmd: dict[str, Any], timeout: float = 30.0, *, source: str = "api"
    ) -> dict[str, Any]:
        self.sent.append((target, cmd))
        return {"status": "success", "content": [{"text": json.dumps(cmd)}]}


class FakeDevices:
    def __init__(self, bridge: FakeBridge) -> None:
        self.bridge = bridge

    def spawn(self, robot: str, mode: str = "sim", peer_id: str | None = None, **_: Any) -> dict[str, Any]:
        pid = peer_id or f"{robot}-sim-1"
        self.bridge.peers[pid] = {"presence": {"robot_type": "sim", "sim_robots": [robot]}}
        self.bridge.peers[f"{pid}__{robot}"] = {**SIM_PEER}
        return {"peer_id": pid, "pid": 4242, "mode": mode}

    def despawn(self, peer_id: str) -> dict[str, Any]:
        return {"ok": True, "peer_id": peer_id}

    def managed_children(self) -> list[dict[str, Any]]:
        return []


def _tool(tools: list[Any], name: str) -> Any:
    return next(t for t in tools if getattr(t, "tool_name", getattr(t, "__name__", "")) == name)


def test_spawn_robot_registers_the_new_peers_tools_into_the_live_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_console, "SPAWN_POLL_S", 0.0)
    bridge = FakeBridge({})
    console = agent_console.Console(safety=object(), model=MagicMock(), bridge=bridge, devices=FakeDevices(bridge))
    before = set(console.tool_names())
    assert not any(n.startswith("so101_sim") for n in before)

    spawn = _tool(list(console.agent.tool_registry.registry.values()), "spawn_robot")
    out = spawn("so101")  # a decorated tool stays a plain callable
    assert sorted(out["tools"]) == ["so101_sim_1", "so101_sim_1__so101"]
    assert "callable now" in out["note"]

    after = set(console.tool_names())
    assert after - before == {"so101_sim_1", "so101_sim_1__so101"}
    # the fleet did not "change" for the next turn: the change is already applied
    assert console.refresh() is False
    # the proxy in the registry is bound to the peer and reaches the bridge
    proxy = console.agent.tool_registry.registry["so101_sim_1__so101"]
    assert proxy.peer_id == "so101-sim-1__so101"
    assert console._hook is not None and console._hook._proxy_targets["so101_sim_1__so101"] == "so101-sim-1__so101"


def test_adopt_is_idempotent_and_ignores_peers_the_bridge_does_not_hold() -> None:
    bridge = FakeBridge({"arm-a__so101": SIM_PEER})
    console = agent_console.Console(safety=object(), model=MagicMock(), bridge=bridge, devices=None)
    assert console.adopt(["ghost"]) == []
    assert console.adopt(["arm-a__so101"]) == ["arm_a__so101"]  # already held: named, not registered twice
    assert console.tool_names().count("arm_a__so101") == 1


def test_adopt_without_a_bridge_registers_nothing() -> None:
    console = agent_console.Console(safety=object(), model=MagicMock())
    assert console.adopt(["x"]) == []


# ─────────────────────────────────────────────── the bridge ────────────────


def test_a_refused_mesh_session_leaves_its_reason_on_the_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    from strands_robots.dashboard import mesh_bridge
    from strands_robots.mesh import session

    def refuse() -> Any:
        raise ValueError("STRANDS_MESH_AUTH_MODE=mtls requires STRANDS_MESH_TLS_CA")

    monkeypatch.setattr(session, "get_session", refuse)
    monkeypatch.delenv("STRANDS_MESH", raising=False)
    bridge = mesh_bridge.MeshBridge(peer_id="dashboard-test")
    assert bridge.start(asyncio.new_event_loop()) is False
    assert bridge.start_error is not None and "STRANDS_MESH_TLS_CA" in bridge.start_error
    info = bridge.mesh_info()
    assert info["online"] is False and info["error"] == bridge.start_error


def test_the_kill_switch_is_a_named_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    from strands_robots.dashboard import mesh_bridge

    monkeypatch.setenv("STRANDS_MESH", "false")
    bridge = mesh_bridge.MeshBridge(peer_id="dashboard-test")
    assert bridge.start(asyncio.new_event_loop()) is False
    assert bridge.start_error is not None and "STRANDS_MESH=false" in bridge.start_error


def test_the_fleet_route_reports_the_bridges_reason() -> None:
    from fastapi import FastAPI

    from strands_robots.dashboard import routes_mesh

    class RefusingBridge:
        peer_id = "dashboard-test"
        start_error: str | None = None

        def start(self, loop: Any) -> bool:
            self.start_error = "ValueError: STRANDS_MESH_AUTH_MODE=mtls requires STRANDS_MESH_TLS_CA"
            return False

        def stop(self) -> None:
            return None

    app = FastAPI()
    app.state.bridge = RefusingBridge()
    routes_mesh.attach(app)
    [start] = [hook for hook in app.state.startup_hooks if hook.__name__ == "_start"]
    asyncio.run(start())
    assert app.state.mesh_online is False
    assert app.state.mesh_error is not None and "STRANDS_MESH_TLS_CA" in app.state.mesh_error
