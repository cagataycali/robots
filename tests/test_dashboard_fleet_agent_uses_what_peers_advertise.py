"""The fleet agent uses what each peer advertises, and the dashboard can anchor the mesh.

The sim proxy in ``dashboard/peer_tools.py`` mirrored a package copy of the
simulation's spec. With peers advertising their tool surface (presence
``tool_spec_hash``, ``describe_tool``, the ``call`` rail) the proxy becomes a
projector: the spec the peer served IS the tool's action enum and param
schema, fetched once per hash, and a function rides ``call`` judged by the
peer's own tables. A peer that advertises nothing (real hardware, an older
build) keeps the static proxy. Pinned here:

* the fetch: one ``describe_tool`` per hash, a child peer reuses its parent's,
  no hash means no fetch, a failed or malformed fetch leaves the static proxy;
* the projection: verbs first, then every advertised function, params from the
  peer with the verbs' wording winning, and never a param the peer did not list;
* the mapping: an advertised function rides ``call`` with its params; a denied
  function refuses with the peer's reason, an unknown one names the valid set,
  a field the function does not take is refused by name, all before the wire;
* the console: a changed hash changes the fleet signature; the spawned peer's
  projected tool is what ``adopt`` registers;
* the anchor: ``--mesh-listen`` is validated and exported as ``ZENOH_LISTEN``,
  the hint names a LAN address on the listened port, ``mesh_info`` carries it.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.dashboard import cli, lan_hint, peer_tools
from strands_robots.mesh import security
from strands_robots.simulation.mujoco import wire_surface


class _Sim:
    _ACTION_ALIASES = {"list_robots": "list_robots_info"}

    def list_robots_info(self) -> None: ...

    def add_object(self, name: str, shape: str = "box", mesh_path: str | None = None) -> None: ...

    def get_robot_state(self, robot_name: str | None = None) -> None: ...

    def render(
        self, camera_name: str = "default", width: int = 640, height: int = 480, output_path: str | None = None
    ) -> None: ...

    def step(self, steps: int = 1) -> None: ...


SPEC = wire_surface.build_wire_tool_spec("so101_sim", _Sim)
HASH = wire_surface.tool_spec_hash(SPEC)

SIM_PEER: dict[str, Any] = {
    "presence": {"robot_type": "sim", "sim_robots": ["so101"], "tool_spec_hash": HASH},
    "state": {"joints": {"1": 0.0}},
}


class _Bridge:
    """Records sends; answers ``describe_tool`` with SPEC inside the wire envelope."""

    def __init__(self, peers: dict[str, Any], *, describe: Any = "ok") -> None:
        self.peers = dict(peers)
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.describe = describe

    def send_cmd(self, target: str, cmd: dict[str, Any], timeout: float = 30.0, *, source: str = "api") -> Any:
        self.sent.append((target, cmd))
        if cmd.get("action") == "describe_tool":
            if self.describe == "ok":
                return {"type": "response", "result": {"tool_name": "so101_sim", "tool_spec_hash": HASH, "spec": SPEC}}
            if self.describe == "raise":
                raise TimeoutError("no answer")
            return self.describe
        return {"type": "response", "result": {"status": "success", "content": [{"text": json.dumps(cmd)}]}}


@pytest.fixture(autouse=True)
def _fresh_cache() -> Any:
    peer_tools._ADVERTISED_SPECS.clear()
    yield
    peer_tools._ADVERTISED_SPECS.clear()


# ───────────────────────────────────────────────────── the fetch ───────────


def test_the_spec_is_fetched_once_per_hash_and_a_child_reuses_its_parents() -> None:
    bridge = _Bridge({"so101-sim-1": SIM_PEER, "so101-sim-1__so101": SIM_PEER})
    tools = peer_tools.build_peer_tools(bridge.peers, bridge.send_cmd)
    describes = [t for t, c in bridge.sent if c["action"] == "describe_tool"]
    assert describes == ["so101-sim-1"]
    assert {t.tool_name for t in tools} == {"so101_sim_1", "so101_sim_1__so101"}
    assert all(t.advertised == SPEC for t in tools)
    peer_tools.build_peer_tools(bridge.peers, bridge.send_cmd)
    assert len([c for _, c in bridge.sent if c["action"] == "describe_tool"]) == 1


def test_a_peer_without_a_hash_is_never_asked_and_keeps_the_static_proxy() -> None:
    plain = {"presence": {"robot_type": "sim", "sim_robots": ["so101"]}, "state": {"joints": {"1": 0.0}}}
    bridge = _Bridge({"old-sim": plain})
    (tool,) = peer_tools.build_peer_tools(bridge.peers, bridge.send_cmd)
    assert bridge.sent == []
    assert tool.advertised is None
    enum = tool.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]
    assert "add_object" in enum and enum[: len(peer_tools.SIM_ACTIONS)] == list(peer_tools.SIM_ACTIONS)


@pytest.mark.parametrize(
    "describe", ["raise", {"error": "mesh offline"}, {"type": "response", "result": {"spec": None}}]
)
def test_a_failed_or_malformed_fetch_leaves_the_static_proxy(describe: Any) -> None:
    bridge = _Bridge({"so101-sim-1": SIM_PEER}, describe=describe)
    (tool,) = peer_tools.build_peer_tools(bridge.peers, bridge.send_cmd)
    assert tool.advertised is None
    assert peer_tools._ADVERTISED_SPECS == {}
    assert "add_object" in tool.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]


def test_a_real_robot_advertises_nothing_and_is_not_asked() -> None:
    real = {
        "presence": {"robot_type": "so101", "kind": "real", "tool_spec_hash": "not-for-hardware"},
        "state": {"joints": {"1": 0.0}},
    }
    bridge = _Bridge({"arm": real})
    (tool,) = peer_tools.build_peer_tools(bridge.peers, bridge.send_cmd)
    assert bridge.sent == []
    assert tool.peer_kind == peer_tools.KIND_REAL and tool.advertised is None


# ──────────────────────────────────────────────── the projection ───────────


def test_the_projected_tool_offers_the_verbs_then_every_advertised_function() -> None:
    spec = peer_tools.peer_tool_spec("so101-sim-1", peer_tools.KIND_SIM, "so101_sim_1", SPEC)
    assert spec is not None
    enum = spec["inputSchema"]["json"]["properties"]["action"]["enum"]
    assert enum[: len(peer_tools.SIM_ACTIONS)] == list(peer_tools.SIM_ACTIONS)
    assert enum[len(peer_tools.SIM_ACTIONS) :] == ["add_object", "get_robot_state", "list_robots", "render"]
    assert "step" in enum and enum.count("step") == 1
    assert "destroy" not in enum
    assert "advertises" in spec["description"]


def test_the_projected_params_come_from_the_peer_and_the_verbs_wording_wins() -> None:
    props = peer_tools.projected_input_schema(SPEC)["properties"]
    assert set(props) >= {"name", "shape", "camera_name", "width", "height", "robot_name", "target_joints"}
    assert "mesh_path" not in props and "output_path" not in props
    assert props["name"]["type"] == "string"
    assert len(props["name"]["description"]) <= peer_tools._SIM_PARAM_DESCRIPTION_CHARS
    assert props["robot_name"] == peer_tools._SIM_VERB_FIELDS["robot_name"]
    assert props["steps"] == peer_tools._SIM_VERB_FIELDS["steps"]


def test_every_projected_action_maps_onto_a_command_the_wire_accepts() -> None:
    for action in peer_tools.projected_actions(SPEC):
        cmd, err = peer_tools.map_invocation("p", peer_tools.KIND_SIM, {"action": action}, SPEC)
        assert err is None, (action, err)
        assert cmd is not None and cmd["action"] == "call" and cmd["function"] == action
        assert security.validate_command(cmd)["function"] == action


# ────────────────────────────────────────────────── the mapping ────────────


def test_an_advertised_function_rides_call_with_its_params() -> None:
    cmd, err = peer_tools.map_invocation(
        "p",
        peer_tools.KIND_SIM,
        {"action": "add_object", "name": "red_cube", "shape": "box", "robot_name": None},
        SPEC,
    )
    assert err is None
    assert cmd == {"action": "call", "function": "add_object", "params": {"name": "red_cube", "shape": "box"}}


def test_a_denied_function_is_refused_with_the_peers_reason_before_the_wire() -> None:
    cmd, err = peer_tools.map_invocation("p", peer_tools.KIND_SIM, {"action": "run_policy"}, SPEC)
    assert cmd is None and err is not None
    assert SPEC["denied"]["run_policy"] in err
    assert "action='execute'" in err
    cmd, err = peer_tools.map_invocation("p", peer_tools.KIND_SIM, {"action": "destroy"}, SPEC)
    assert cmd is None and err is not None and SPEC["denied"]["destroy"] in err


def test_an_unknown_function_names_what_the_peer_offers() -> None:
    cmd, err = peer_tools.map_invocation("p", peer_tools.KIND_SIM, {"action": "nod"}, SPEC)
    assert cmd is None and err is not None
    assert "unknown action 'nod'" in err and "add_object" in err and "status" in err


def test_a_field_the_function_does_not_take_is_refused_by_name() -> None:
    cmd, err = peer_tools.map_invocation(
        "p", peer_tools.KIND_SIM, {"action": "add_object", "name": "c", "mesh_path": "/tmp/x", "colour": "red"}, SPEC
    )
    assert cmd is None and err is not None
    assert "colour, mesh_path" in err and "name, shape" in err


def test_the_verbs_still_ride_their_own_actions_on_a_projected_tool() -> None:
    cmd, err = peer_tools.map_invocation(
        "p", peer_tools.KIND_SIM, {"action": "set_joints", "target_joints": {"1": 0.2}}, SPEC
    )
    assert err is None and cmd == {"action": "set_joints", "target_joints": {"1": 0.2}}
    cmd, err = peer_tools.map_invocation("p", peer_tools.KIND_SIM, {"action": "step", "steps": 3}, SPEC)
    assert err is None and cmd == {"action": "step", "steps": 3}


# ─────────────────────────────────────────────────── the console ───────────


def test_a_changed_hash_changes_the_fleet_signature() -> None:
    before = peer_tools.fleet_signature({"so101-sim-1": SIM_PEER})
    other = {**SIM_PEER, "presence": {**SIM_PEER["presence"], "tool_spec_hash": "f" * 64}}
    after = peer_tools.fleet_signature({"so101-sim-1": other})
    assert before != after
    assert {entry[:2] for entry in before} == {entry[:2] for entry in after} == {("so101-sim-1", peer_tools.KIND_SIM)}


def test_spawn_robot_registers_the_projected_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    from strands_robots.dashboard import agent_console

    monkeypatch.setattr(agent_console, "SPAWN_POLL_S", 0.0)
    bridge = _Bridge({})

    class _Devices:
        def spawn(self, robot: str, mode: str = "sim", peer_id: str | None = None, **_: Any) -> dict[str, Any]:
            pid = peer_id or f"{robot}-sim-1"
            bridge.peers[pid] = {"presence": {"robot_type": "sim", "sim_robots": [robot], "tool_spec_hash": HASH}}
            bridge.peers[f"{pid}__{robot}"] = {**SIM_PEER}
            return {"peer_id": pid, "pid": 4242, "mode": mode}

        def despawn(self, peer_id: str) -> dict[str, Any]:
            return {"ok": True, "peer_id": peer_id}

        def managed_children(self) -> list[dict[str, Any]]:
            return []

    bridge.snapshot = lambda: {"peers": dict(bridge.peers)}  # type: ignore[attr-defined]
    console = agent_console.Console(safety=object(), model=MagicMock(), bridge=bridge, devices=_Devices())
    spawn = next(
        t for t in console.agent.tool_registry.registry.values() if getattr(t, "tool_name", "") == "spawn_robot"
    )
    out = spawn("so101")
    assert sorted(out["tools"]) == ["so101_sim_1", "so101_sim_1__so101"]
    proxy = console.agent.tool_registry.registry["so101_sim_1__so101"]
    assert proxy.advertised == SPEC
    assert "add_object" in proxy.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]
    assert len([c for _, c in bridge.sent if c["action"] == "describe_tool"]) == 1
    assert console.refresh() is False


# ──────────────────────────────────────────────────── the anchor ───────────


def test_mesh_listen_is_a_parser_option_off_by_default() -> None:
    args = cli.build_parser().parse_args([])
    assert args.mesh_listen is None
    assert cli.build_parser().parse_args(["--mesh-listen", "tcp/0.0.0.0:7447"]).mesh_listen == "tcp/0.0.0.0:7447"


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        (None, True),
        ("tcp/0.0.0.0:7447", True),
        ("tls/[::]:7447", True),
        ("0.0.0.0:7447", False),
        ("http/0.0.0.0:7447", False),
        ("tcp/0.0.0.0", False),
        ("tcp/0.0.0.0:0", False),
        ("tcp/0.0.0.0:70000", False),
    ],
)
def test_mesh_listen_verdict(value: str | None, ok: bool) -> None:
    verdict = cli.mesh_listen_verdict(value)
    assert (verdict is None) is ok, verdict


def test_main_refuses_a_bad_mesh_listen_before_touching_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZENOH_LISTEN", raising=False)
    assert cli.main(["--mesh-listen", "nope"]) == 2
    assert "ZENOH_LISTEN" not in os.environ


def test_main_exports_mesh_listen_before_the_session_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    monkeypatch.delenv("ZENOH_LISTEN", raising=False)
    seen: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        seen["listen_at_serve"] = os.environ.get("ZENOH_LISTEN")

    fake_uvicorn = types.ModuleType("uvicorn")
    setattr(fake_uvicorn, "run", fake_run)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    from strands_robots.dashboard import server

    monkeypatch.setattr(server, "create_app", lambda: types.SimpleNamespace(state=types.SimpleNamespace()))
    assert cli.main(["--mesh-listen", "tcp/0.0.0.0:7447"]) == 0
    assert seen["listen_at_serve"] == "tcp/0.0.0.0:7447"


def test_the_anchor_hint_names_a_lan_address_on_the_listened_port() -> None:
    own = ["127.0.0.1", "192.168.1.20", "fe80::1%en0", "10.0.0.7"]
    assert lan_hint.mesh_anchor_hint(["tcp/0.0.0.0:7447"], own) == "ZENOH_CONNECT=tcp/192.168.1.20:7447"
    assert lan_hint.mesh_anchor_hint(["tcp/10.0.0.7:7448"], own) == "ZENOH_CONNECT=tcp/10.0.0.7:7448"
    assert lan_hint.mesh_anchor_hint(["tls/[::]:7447"], own) == "ZENOH_CONNECT=tls/192.168.1.20:7447"
    assert lan_hint.mesh_anchor_hint([], own) is None
    assert lan_hint.mesh_anchor_hint(["tcp/0.0.0.0:0"], own) is None
    assert lan_hint.mesh_anchor_hint(["tcp/0.0.0.0:7447"], ["127.0.0.1"]) is None


def test_mesh_info_carries_the_anchor_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    from strands_robots.dashboard import mesh_bridge

    monkeypatch.setattr(
        mesh_bridge, "_mesh_anchor_hint", lambda listen: "ZENOH_CONNECT=tcp/192.168.1.20:7447" if listen else None
    )
    bridge = mesh_bridge.MeshBridge.__new__(mesh_bridge.MeshBridge)
    bridge._endpoints = {"listen": ["tcp/0.0.0.0:7447"], "connect": [], "auth_mode": "none"}
    bridge._running = True
    bridge.start_error = None
    bridge.peer_id = "dash"
    bridge.peers = {}
    bridge.live_peers = lambda: []  # type: ignore[method-assign]
    assert bridge.mesh_info()["anchor_hint"] == "ZENOH_CONNECT=tcp/192.168.1.20:7447"
    bridge._endpoints = {"listen": [], "connect": [], "auth_mode": "none"}
    assert bridge.mesh_info()["anchor_hint"] is None


def test_a_spawned_child_connects_to_the_anchor_instead_of_inheriting_its_listen_role() -> None:
    from strands_robots.dashboard import device_manager

    anchored = device_manager.child_env({"ZENOH_LISTEN": "tcp/0.0.0.0:7447", "PATH": "/bin"})
    assert "ZENOH_LISTEN" not in anchored
    assert anchored["ZENOH_CONNECT"] == "tcp/127.0.0.1:7447"
    assert anchored["PATH"] == "/bin"
    told = device_manager.child_env({"ZENOH_LISTEN": "tcp/0.0.0.0:7447", "ZENOH_CONNECT": "tcp/10.0.0.1:7447"})
    assert told["ZENOH_CONNECT"] == "tcp/10.0.0.1:7447" and "ZENOH_LISTEN" not in told
    plain = device_manager.child_env({"PATH": "/bin"})
    assert plain == {"PATH": "/bin"}
