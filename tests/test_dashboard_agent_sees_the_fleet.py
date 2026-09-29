"""The dashboard agent sees and drives the FLEET, not only its in-process sim sessions.

Three layers, each pinned here without a mesh or a model call:

* the wire: ``set_joints`` is a validated sim-only mesh action (security) that
  ``Mesh._dispatch`` serves through the simulation's own published action and a
  hardware peer refuses (core);
* the proxies: a sim peer's tool advertises exactly the actions the wire carries
  and maps them onto validated commands (peer_tools);
* the console: ``fleet`` / ``spawn_robot`` / ``despawn_robot`` plus one proxy per
  peer, the HITL hook over real-arm motion, tool names the badge can report before
  the first turn, and a rebuild when the fleet changes that keeps the conversation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.dashboard import agent_console, peer_tools
from strands_robots.mesh import security

# ─────────────────────────────────────────── the wire: set_joints ──────────


def test_set_joints_is_an_allowed_action_that_needs_target_joints() -> None:
    out = security.validate_command(
        {"action": "set_joints", "target_joints": {"shoulder_lift": 0.17, "2": 1}, "hold": True, "robot_name": "so101"}
    )
    assert out == {
        "action": "set_joints",
        "target_joints": {"shoulder_lift": 0.17, "2": 1.0},
        "hold": True,
        "robot_name": "so101",
    }


@pytest.mark.parametrize(
    ("cmd", "fragment"),
    [
        ({"action": "set_joints"}, "requires `target_joints`"),
        ({"action": "set_joints", "target_joints": {}}, "at least one joint"),
        ({"action": "set_joints", "target_joints": [1.0]}, "must be a dict"),
        ({"action": "set_joints", "target_joints": {"a b": 1.0}}, "must match"),
        ({"action": "set_joints", "target_joints": {"a": "x"}}, "must be a number"),
        ({"action": "set_joints", "target_joints": {"a": 1.0}, "hold": "yes"}, "hold must be a bool"),
        ({"action": "set_joints", "target_joints": {"a": 1.0}, "robot_name": "so 101"}, "robot_name must match"),
    ],
)
def test_set_joints_refusals_name_the_field(cmd: dict[str, Any], fragment: str) -> None:
    with pytest.raises(security.ValidationError, match=fragment):
        security.validate_command(cmd)


def test_set_joints_drops_fields_it_does_not_carry() -> None:
    out = security.validate_command(
        {"action": "set_joints", "target_joints": {"a": 1.0}, "instruction": "x", "policy_port": 1}
    )
    assert set(out) == {"action", "target_joints"}


def _mesh_with(robot: Any) -> Any:
    """A Mesh object with just what ``_dispatch`` reads, no zenoh session."""
    from strands_robots.mesh.core import Mesh

    mesh = Mesh.__new__(Mesh)
    mesh.robot = robot
    mesh._estop_lockout = MagicMock(is_set=lambda: False)
    return mesh


class _Sim:
    """A Simulation as ``_dispatch`` recognises one: published actions + a world + list_robots."""

    _world = object()

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_robots(self) -> list[str]:
        return ["so101"]

    def _dispatch_action(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((action, params))
        return {"status": "success", "content": [{"text": "Set 1/1 joint positions"}]}


class _SimChild:
    """A child SimRobot peer: no actions of its own, a ``_sim_parent`` that has them."""

    def __init__(self, parent: _Sim) -> None:
        self._sim_parent = parent
        self.name = "so101"


def test_dispatch_set_joints_on_a_child_binds_its_robot_and_holds_by_default() -> None:
    sim = _Sim()
    mesh = _mesh_with(_SimChild(sim))
    out = mesh._dispatch({"action": "set_joints", "target_joints": {"2": 0.2}})
    assert out["status"] == "success"
    assert sim.calls == [("set_joint_positions", {"positions": {"2": 0.2}, "hold": True, "robot_name": "so101"})]


def test_dispatch_set_joints_on_a_simulation_forwards_robot_name_and_hold() -> None:
    sim = _Sim()
    mesh = _mesh_with(sim)
    mesh._dispatch({"action": "set_joints", "target_joints": {"a": 1.0}, "hold": False, "robot_name": "arm2"})
    assert sim.calls == [("set_joint_positions", {"positions": {"a": 1.0}, "hold": False, "robot_name": "arm2"})]


def test_dispatch_set_joints_refuses_a_hardware_peer() -> None:
    class Arm:
        def _execute_task_sync(self, *a: Any, **k: Any) -> dict[str, Any]:  # pragma: no cover - never reached
            raise AssertionError("a real arm must not take set_joints")

    out = _mesh_with(Arm())._dispatch({"action": "set_joints", "target_joints": {"a": 1.0}})
    assert "simulation-only" in out["error"] and "execute/start" in out["error"]


# ─────────────────────────────────────────── the proxies ───────────────────

SIM_PEER = {
    "presence": {"robot_type": "sim", "hostname": "h", "topics": ["health"]},
    "state": {"joints": {"1": {"position": 0.0}}},
}
REAL_PEER = {
    "presence": {"robot_type": "robot", "hw": "so101 @ /dev/tty", "topics": ["health", "state"]},
    "state": {"joints": {"1": {}}},
}


def test_sim_proxy_advertises_only_what_the_wire_carries() -> None:
    spec = peer_tools.peer_tool_spec("lane-so101__so101", peer_tools.KIND_SIM, "lane_so101__so101")
    assert spec is not None
    actions = spec["inputSchema"]["json"]["properties"]["action"]["enum"]
    assert set(actions) <= security.ALLOWED_ACTIONS
    assert "set_joints" in actions and "sim_call" not in actions
    for a in actions:
        cmd, err = peer_tools.map_invocation(
            "p", peer_tools.KIND_SIM, {"action": a, "instruction": "go", "target_joints": {"a": 1.0}}
        )
        assert err is None, (a, err)
        security.validate_command(cmd)  # the wire accepts every advertised action


@pytest.mark.parametrize(
    ("tool_input", "expected"),
    [
        (
            {"action": "set_joints", "target_joints": {"2": 0.2}, "hold": True},
            {"action": "set_joints", "target_joints": {"2": 0.2}, "hold": True},
        ),
        ({"action": "state", "robot_name": "so101", "steps": 3}, {"action": "state", "robot_name": "so101"}),
        ({"action": "step", "steps": 3}, {"action": "step", "steps": 3}),
        (
            {
                "action": "start",
                "instruction": "wave",
                "pretrained_name_or_path": "lerobot/smolvla_base",
                "policy_provider": "lerobot_local",
            },
            {
                "action": "start",
                "instruction": "wave",
                "pretrained_name_or_path": "lerobot/smolvla_base",
                "policy_provider": "lerobot_local",
            },
        ),
        (
            {"action": "execute", "instruction": "wave"},
            {"action": "execute", "instruction": "wave", "policy_provider": "mock"},
        ),
    ],
)
def test_sim_invocation_maps_onto_a_wire_command(tool_input: dict[str, Any], expected: dict[str, Any]) -> None:
    cmd, err = peer_tools.map_invocation("p", peer_tools.KIND_SIM, tool_input)
    assert err is None and cmd == expected


@pytest.mark.parametrize(
    ("tool_input", "fragment"),
    [
        ({"action": "set_joints"}, "needs target_joints"),
        ({"action": "execute"}, "needs an instruction"),
        ({"action": "add_object"}, "unknown action"),
        ({}, "needs an 'action'"),
    ],
)
def test_sim_invocation_refuses_before_the_wire(tool_input: dict[str, Any], fragment: str) -> None:
    cmd, err = peer_tools.map_invocation("p", peer_tools.KIND_SIM, tool_input)
    assert cmd is None and err is not None and fragment in err


# ─────────────────────────────────────────── the console ───────────────────


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
        return {"status": "success", "content": [{"text": "ok"}]}


class FakeDevices:
    def __init__(self, bridge: FakeBridge, *, announce: bool = True) -> None:
        self.bridge = bridge
        self.announce = announce
        self.spawned: list[tuple[str, str, str | None]] = []
        self.despawned: list[str] = []

    def spawn(self, robot: str, mode: str = "sim", peer_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.spawned.append((robot, mode, peer_id))
        pid = peer_id or f"{robot}-sim-1"
        if self.announce:
            self.bridge.peers[pid] = {"presence": {"robot_type": "sim", "sim_robots": [robot]}}
            self.bridge.peers[f"{pid}__{robot}"] = {**SIM_PEER}
        return {"peer_id": pid, "pid": 4242, "mode": mode}

    def despawn(self, peer_id: str) -> dict[str, Any]:
        self.despawned.append(peer_id)
        return {"ok": True, "peer_id": peer_id}

    def managed_children(self) -> list[dict[str, Any]]:
        return [{"peer_id": p, "robot": r} for r, _m, p in self.spawned]


def _tools_by_name(tools: list[Any]) -> dict[str, Any]:
    return {getattr(t, "tool_name", getattr(t, "__name__", "")): t for t in tools}


def test_expected_tool_names_are_honest_before_the_first_turn() -> None:
    assert agent_console.expected_tool_names(None) == [
        "robots",
        "sim_sessions",
        "sim_start",
        "sim_state",
        "sim_set_joints",
        "sim_reset",
        "sim_stop",
        "emergency_stop",
    ]
    bridge = FakeBridge(
        {
            "lane-so101": {"presence": {"robot_type": "sim"}},
            "lane-so101__so101": SIM_PEER,
            "dashboard-x-safety": {"presence": {"robot_type": "dashboard"}},
        }
    )
    names = agent_console.expected_tool_names(bridge)
    assert names[8:11] == list(agent_console.FLEET_TOOL_NAMES)
    assert set(names[11:]) == {"lane_so101", "lane_so101__so101"}  # the coordinator mints no tool


def test_fleet_lists_every_tool_worthy_peer_with_its_tool_and_state() -> None:
    bridge = FakeBridge(
        {
            "lane-so101__so101": {**SIM_PEER, "stale": True},
            "arm-1": REAL_PEER,
            "gateway-h-1": {"presence": {"robot_type": "gateway"}},
        }
    )
    fleet = _tools_by_name(agent_console.build_fleet_tools(bridge, None))["fleet"]
    out = fleet()
    rows = {r["peer_id"]: r for r in out["robots"]}
    assert set(rows) == {"lane-so101__so101", "arm-1"} and out["count"] == 2
    assert rows["lane-so101__so101"]["tool"] == "lane_so101__so101"
    assert rows["lane-so101__so101"]["kind"] == peer_tools.KIND_SIM
    assert rows["lane-so101__so101"]["stale"] is True  # listed, marked, never hidden
    assert rows["lane-so101__so101"]["joints"] == {"1": {"position": 0.0}}
    assert rows["arm-1"]["kind"] == peer_tools.KIND_REAL


def test_spawn_robot_starts_a_sim_peer_and_reports_when_it_is_on_the_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = FakeBridge({})
    devices = FakeDevices(bridge)
    spawn = _tools_by_name(agent_console.build_fleet_tools(bridge, devices))["spawn_robot"]
    out = spawn(robot="so101", peer_id="lane-so101")
    assert devices.spawned == [("so101", "sim", "lane-so101")]  # simulation only, never real
    assert out["peer_id"] == "lane-so101"
    assert out["on_mesh"] == ["lane-so101", "lane-so101__so101"]
    assert out["tools"] == ["lane_so101", "lane_so101__so101"]
    assert "card is on the dashboard" in out["note"]


def test_spawn_robot_says_so_when_no_presence_arrives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_console, "SPAWN_PRESENCE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(agent_console, "SPAWN_POLL_S", 0.01)
    bridge = FakeBridge({})
    spawn = _tools_by_name(agent_console.build_fleet_tools(bridge, FakeDevices(bridge, announce=False)))["spawn_robot"]
    out = spawn(robot="so101")
    assert out["on_mesh"] == [] and "no presence arrived" in out["note"]


def test_spawn_robot_relays_a_refusal_as_an_error() -> None:
    bridge = FakeBridge({})
    devices = FakeDevices(bridge)
    devices.spawn = lambda *a, **k: {"error": "unknown robot nope"}  # type: ignore[method-assign]
    spawn = _tools_by_name(agent_console.build_fleet_tools(bridge, devices))["spawn_robot"]
    with pytest.raises(ValueError, match="unknown robot nope"):
        spawn(robot="nope")


def test_despawn_robot_is_never_refused() -> None:
    bridge = FakeBridge({})
    devices = FakeDevices(bridge)
    despawn = _tools_by_name(agent_console.build_fleet_tools(bridge, devices))["despawn_robot"]
    assert despawn(peer_id="lane-so101") == {"ok": True, "peer_id": "lane-so101"}
    assert devices.despawned == ["lane-so101"]


def test_without_a_bridge_there_are_no_fleet_tools() -> None:
    assert agent_console.build_fleet_tools(None, None) == []


def _console(bridge: FakeBridge | None, devices: Any = None) -> agent_console.Console:
    safety = MagicMock()
    safety.store.all.return_value = []
    return agent_console.Console(safety, model=MagicMock(), bridge=bridge, devices=devices)


def test_console_holds_sim_fleet_and_proxy_tools_and_gates_real_motion() -> None:
    bridge = FakeBridge({"lane-so101__so101": SIM_PEER, "arm-1": REAL_PEER})
    console = _console(bridge, FakeDevices(bridge))
    names = set(console.tool_names())
    assert {"sim_start", "fleet", "spawn_robot", "despawn_robot", "lane_so101__so101", "arm_1"} <= names
    # The HITL rows are DERIVED from the built proxies: the real arm's execute/start, nothing for the sim.
    from strands_robots.dashboard.peer_tools import build_peer_tools, motion_actions_for

    proxies = build_peer_tools(bridge.peers, bridge.send_cmd)
    assert motion_actions_for(proxies) == {"arm_1": frozenset({"execute", "start"})}


def test_console_without_a_bridge_is_the_sim_only_agent() -> None:
    console = _console(None)
    assert set(console.tool_names()) == set(agent_console.expected_tool_names(None))
    assert console.refresh() is False


def test_console_rebuilds_when_the_fleet_changes_and_keeps_the_conversation() -> None:
    bridge = FakeBridge({})
    console = _console(bridge, FakeDevices(bridge))
    assert "lane_so101__so101" not in console.tool_names()
    console.agent.messages.append({"role": "user", "content": [{"text": "hello"}]})
    before = console.agent

    # Nothing changed: no rebuild, same agent object.
    assert console.refresh() is False and console.agent is before

    bridge.peers["lane-so101__so101"] = SIM_PEER
    assert console.refresh() is True
    assert console.agent is not before
    assert "lane_so101__so101" in console.tool_names()
    assert console.agent.messages == [{"role": "user", "content": [{"text": "hello"}]}]

    # Presence details that do not change a peer's kind do not churn the agent.
    bridge.peers["lane-so101__so101"] = {**SIM_PEER, "state": {"joints": {"1": {"position": 1.0}}}}
    assert console.refresh() is False


def test_a_resume_never_rebuilds_the_agent_that_raised_the_interrupt() -> None:
    assert agent_console._is_resume(agent_console.Console.resume("i1", True)) is True
    assert agent_console._is_resume("move it") is False


def test_agent_info_reports_the_tools_and_fleet_awareness() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from strands_robots.dashboard.server import create_app

    app = create_app()
    app.state.bridge = FakeBridge({"lane-so101__so101": SIM_PEER})
    with TestClient(app) as client:
        info = client.get("/api/agent").json()
    assert info["fleet_aware"] is True
    assert "fleet" in info["tools"] and "lane_so101__so101" in info["tools"]
