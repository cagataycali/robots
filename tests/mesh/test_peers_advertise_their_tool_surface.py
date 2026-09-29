"""A mesh peer advertises the tool it serves, and the wire invokes what was advertised.

``sim_call`` carried a package-level list of admitted simulation actions. Here
the authority moves to the peer: a Simulation's :meth:`wire_tool_spec` is its
published tool minus a deny table kept next to the spec (``wire_surface.json``,
one reason per entry); presence carries the spec's ``tool_spec_hash``;
``describe_tool`` returns the spec; ``call {function, params}`` is validated on
the wire for shape and size only (the shape ``validate_device_rpc`` enforces
for a device's own functions) and the peer refuses anything outside what it
advertises. ``sim_call`` stays as an alias onto ``call``. Pinned here:

* the surface: every published action is served or denied on purpose; a served
  function lists only params its method declares and never a denied one; the
  hash is a stable function of the canonical JSON;
* the advertisement: presence carries the hash (a child peer its parent's),
  ``describe_tool`` returns the same spec and hash, a hardware peer says it has
  none;
* the rail: ``call`` is admitted for any identifier-safe function on the wire
  and refused by the peer for a denied one (with the reason), an unknown one,
  a param the function does not take; hardware and the lockout refuse; the
  alias answers byte for byte like ``call``; the audit record names the function;
* the round trip: on a real MuJoCo world ``call add_object`` then ``call
  list_objects`` sees the cube.
"""

from __future__ import annotations

import json
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import security
from strands_robots.simulation.mujoco import wire_surface

# ────────────────────────────────────────────────── the surface ────────────


class _Sim:
    """A Simulation as the mesh recognises one: callable, a world, ``wire_tool_spec``."""

    _world = object()
    _ACTION_ALIASES = {"list_robots": "list_robots_info"}

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tool_name_str: str | None = None

    def list_robots(self) -> list[str]:
        return ["so101"]

    def list_robots_info(self) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": "so101"}]}

    def add_object(self, name: str, shape: str = "box", mesh_path: str | None = None) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": f"{name} {shape}"}]}

    def get_robot_state(self, robot_name: str | None = None) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": robot_name or "-"}]}

    def destroy(self) -> dict[str, Any]:
        raise AssertionError("a denied function must never reach the tool")

    def __call__(self, action: str = "", **kwargs: Any) -> dict[str, Any]:
        self.calls.append((action, dict(kwargs)))
        method = getattr(self, self._ACTION_ALIASES.get(action, action))
        return dict(method(**kwargs))

    def wire_tool_spec(self) -> dict[str, Any]:
        return wire_surface.build_wire_tool_spec("sim", type(self))


class _SimChild:
    """A child SimRobot peer: no tool of its own, a ``_sim_parent`` that serves one."""

    def __init__(self, parent: _Sim) -> None:
        self._sim_parent = parent
        self.name = "so101"


def test_every_published_action_is_served_or_denied_on_purpose() -> None:
    published = wire_surface.published_actions()
    served = wire_surface.served_actions()
    denied = frozenset(wire_surface.denied_actions())
    assert len(published) == 77
    assert served | denied == published
    assert not (served & denied)
    for name, reason in wire_surface.denied_actions().items():
        assert reason.strip(), name
    assert set(wire_surface.rail_for()) <= denied
    assert set(wire_surface.rail_for().values()) <= security.ALLOWED_ACTIONS
    assert frozenset(wire_surface.denied_params()) <= wire_surface.published_params()


def test_the_mesh_validator_reads_the_same_deny_tables_as_the_peer() -> None:
    """One file, two readers: the wire's early refusal and the peer's surface never disagree."""
    assert security.SIM_CALL_DENIED_ACTIONS == frozenset(wire_surface.denied_actions())
    assert security.SIM_CALL_DENIED_PARAMS == frozenset(wire_surface.denied_params())
    assert security.SIM_CALL_RAIL_FOR == wire_surface.rail_for()


def test_the_real_simulation_serves_every_admitted_action_with_its_declared_params() -> None:
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    spec = wire_surface.build_wire_tool_spec("sim", MuJoCoSimEngine)
    assert set(spec["functions"]) == set(wire_surface.served_actions())
    assert set(spec["denied"]) == set(wire_surface.denied_actions())
    for name, function in spec["functions"].items():
        assert not (set(function["params"]) & set(wire_surface.denied_params())), name
        assert set(function["params"]) <= wire_surface.published_params(), name
    add_object = spec["functions"]["add_object"]["params"]
    assert {"name", "shape", "size", "color", "position"} <= set(add_object)
    assert "mesh_path" not in add_object
    assert add_object["name"]["type"] == "string"
    for entry in add_object.values():
        assert len(entry.get("description", "")) <= wire_surface.MAX_WIRE_PARAM_DESCRIPTION_CHARS
    assert "render" in spec["functions"]
    assert "output_path" not in spec["functions"]["render"]["params"]
    assert json.dumps(spec)


def test_a_fake_serves_only_what_it_has_a_method_for_and_never_a_denied_param() -> None:
    spec = _Sim().wire_tool_spec()
    assert set(spec["functions"]) == {"add_object", "get_robot_state", "list_robots"}
    assert list(spec["functions"]["add_object"]["params"]) == ["name", "shape"]
    assert "destroy" in spec["denied"]


def test_the_hash_is_the_sha256_of_the_canonical_json_and_moves_with_the_spec() -> None:
    import hashlib

    spec = _Sim().wire_tool_spec()
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert wire_surface.tool_spec_hash(spec) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert wire_surface.tool_spec_hash(spec) == mesh_core._wire_tool_spec_hash(spec)
    reordered = json.loads(json.dumps(spec))
    reordered["functions"] = dict(reversed(list(reordered["functions"].items())))
    assert wire_surface.tool_spec_hash(reordered) == wire_surface.tool_spec_hash(spec)
    changed = json.loads(json.dumps(spec))
    changed["functions"]["add_object"]["params"].pop("shape")
    assert wire_surface.tool_spec_hash(changed) != wire_surface.tool_spec_hash(spec)


# ─────────────────────────────────────────────── the advertisement ─────────


def _mesh_with(robot: Any, *, lockout: bool = False) -> Any:
    mesh = mesh_core.Mesh.__new__(mesh_core.Mesh)
    mesh.robot = robot
    mesh.peer_id = "sim-1"
    mesh.peer_type = "so101"
    mesh._estop_lockout = MagicMock(is_set=lambda: lockout)
    return mesh


def test_presence_carries_the_hash_and_a_child_carries_its_parents() -> None:
    sim = _Sim()
    expected = wire_surface.tool_spec_hash(sim.wire_tool_spec())
    assert _mesh_with(sim)._build_presence()["tool_spec_hash"] == expected
    assert _mesh_with(_SimChild(sim))._build_presence()["tool_spec_hash"] == expected
    assert len(expected) == 64


def test_presence_of_a_hardware_peer_carries_no_hash() -> None:
    hardware = MagicMock(spec=["get_task_status", "stop_task"])
    assert "tool_spec_hash" not in _mesh_with(hardware)._build_presence()


def test_describe_tool_returns_the_spec_and_its_hash() -> None:
    sim = _Sim()
    sim.tool_name_str = "sim"
    out = _mesh_with(sim)._dispatch(security.validate_command({"action": "describe_tool"}))
    assert out["tool_name"] == "sim"
    assert out["spec"] == sim.wire_tool_spec()
    assert out["tool_spec_hash"] == wire_surface.tool_spec_hash(out["spec"])
    assert json.dumps(out)


def test_describe_tool_on_a_hardware_peer_says_it_has_none() -> None:
    hardware = MagicMock(spec=["get_task_status", "stop_task", "tool_name_str"])
    hardware.tool_name_str = "so101"
    out = _mesh_with(hardware)._dispatch({"action": "describe_tool"})
    assert out["spec"] is None
    assert out["tool_spec_hash"] is None
    assert "advertises no tool surface" in out["note"]


def test_describe_tool_is_read_only_on_the_wire() -> None:
    """A poll, like ``status``: never replay-deduplicated, never audited as an actuation."""
    sim = _Sim()
    sim.tool_name_str = "sim"
    mesh = _mesh_with(sim)
    mesh._cmd_replay_lock = threading.Lock()
    mesh._cmd_replay_cache = {}
    published: list[tuple[str, dict[str, Any]]] = []
    audited: list[tuple[str, dict[str, Any]]] = []
    mesh.publish = lambda key, payload: published.append((key, payload))  # type: ignore[method-assign]
    mesh._audit_local = lambda event, payload: audited.append((event, payload))  # type: ignore[method-assign]
    for _ in range(2):
        mesh._exec_cmd({"sender_id": "dash", "turn_id": "same-turn", "command": {"action": "describe_tool"}})
    assert len(published) == 2
    assert all(entry[1]["result"]["tool_spec_hash"] for entry in published)
    assert not [event for event, _ in audited if event.startswith("command_")]


# ─────────────────────────────────────────────────── the rail ──────────────


def _call(function: str, **params: Any) -> dict[str, Any]:
    return security.validate_command({"action": "call", "function": function, "params": params})


def test_call_is_an_allowed_action_the_wire_bounds_by_shape_not_by_list() -> None:
    assert {"call", "describe_tool"} <= security.ALLOWED_ACTIONS
    out = security.validate_command(
        {"action": "call", "function": "nod", "params": {"times": 2}, "turn_id": "t", "instruction": "dropped"}
    )
    assert out == {"action": "call", "function": "nod", "params": {"times": 2}, "turn_id": "t"}
    assert security.validate_command({"action": "call", "function": "list_objects", "params": None})["params"] == {}


@pytest.mark.parametrize(
    ("cmd", "fragment"),
    [
        ({"action": "call"}, "requires `function`"),
        ({"action": "call", "function": "a.b"}, "must match"),
        ({"action": "call", "function": "x" * 65}, "MAX_CALL_NAME_LEN"),
        ({"action": "call", "function": "f", "params": []}, "JSON object"),
        ({"action": "call", "function": "f", "params": {"bad key": 1}}, "must match"),
        ({"action": "call", "function": "f", "params": {"": 1}}, "non-empty"),
        ({"action": "call", "function": "f", "params": {"big": "x" * (64 * 1024)}}, "MAX_CALL_PARAMS_BYTES"),
    ],
)
def test_call_refusals_on_the_wire_name_the_rule(cmd: dict[str, Any], fragment: str) -> None:
    with pytest.raises(security.ValidationError, match=fragment):
        security.validate_command(cmd)


def test_the_peer_serves_an_advertised_function_through_its_own_call() -> None:
    sim = _Sim()
    out = _mesh_with(sim)._dispatch(_call("add_object", name="red_cube", shape="box"))
    assert out == {"status": "success", "content": [{"text": "red_cube box"}]}
    assert sim.calls == [("add_object", {"name": "red_cube", "shape": "box"})]


def test_the_peer_refuses_a_denied_function_with_the_tables_reason() -> None:
    sim = _Sim()
    out = _mesh_with(sim)._dispatch(_call("destroy"))
    assert out["error"].startswith("'destroy' is not served over the mesh: ")
    assert wire_surface.denied_actions()["destroy"] in out["error"]
    assert sim.calls == []


def test_the_peer_refuses_a_function_it_never_advertised() -> None:
    sim = _Sim()
    out = _mesh_with(sim)._dispatch(_call("nod"))
    assert "not a function this peer advertises" in out["error"]
    assert "describe_tool" in out["error"]
    out = _mesh_with(sim)._dispatch(_call("render"))
    assert "not a function this peer advertises" in out["error"]
    assert sim.calls == []


def test_the_peer_refuses_a_param_the_function_does_not_take_by_name() -> None:
    sim = _Sim()
    out = _mesh_with(sim)._dispatch(_call("add_object", name="c", mesh_path="/tmp/x.stl", colour="red"))
    assert "'colour', 'mesh_path'" in out["error"]
    assert "name, shape" in out["error"]
    assert sim.calls == []


def test_a_child_peer_binds_its_robot_only_when_the_function_takes_one() -> None:
    sim = _Sim()
    mesh = _mesh_with(_SimChild(sim))
    mesh._dispatch(_call("get_robot_state"))
    mesh._dispatch(_call("get_robot_state", robot_name="other"))
    mesh._dispatch(_call("list_robots"))
    assert sim.calls == [
        ("get_robot_state", {"robot_name": "so101"}),
        ("get_robot_state", {"robot_name": "other"}),
        ("list_robots", {}),
    ]


def test_hardware_refuses_a_call_and_real_motion_stays_on_its_rails() -> None:
    hardware = MagicMock(spec=["get_task_status", "stop_task"])
    out = _mesh_with(hardware)._dispatch(_call("nod"))
    assert "advertise a tool surface" in out["error"]
    assert "execute/start" in out["error"]
    hardware.stop_task.assert_not_called()


def test_the_lockout_refuses_a_call() -> None:
    sim = _Sim()
    with pytest.raises(security.LockoutError):
        _mesh_with(sim, lockout=True)._dispatch(_call("add_object", name="x"))
    assert sim.calls == []


def test_sim_call_is_an_alias_that_answers_like_call() -> None:
    via_alias = _Sim()
    via_call = _Sim()
    alias_cmd = security.validate_command(
        {"action": "sim_call", "sim_action": "add_object", "params": {"name": "red_cube", "shape": "box"}}
    )
    assert _mesh_with(via_alias)._dispatch(alias_cmd) == _mesh_with(via_call)._dispatch(
        _call("add_object", name="red_cube", shape="box")
    )
    assert via_alias.calls == via_call.calls
    hardware = MagicMock(spec=["get_task_status", "stop_task"])
    assert _mesh_with(hardware)._dispatch(alias_cmd)["error"].startswith("sim_call is a simulation-only action")


def test_an_executed_call_is_audited_with_its_function() -> None:
    sim = _Sim()
    mesh = _mesh_with(sim)
    mesh._cmd_replay_lock = threading.Lock()
    mesh._cmd_replay_cache = {}
    published: list[tuple[str, dict[str, Any]]] = []
    audited: list[tuple[str, dict[str, Any]]] = []
    mesh.publish = lambda key, payload: published.append((key, payload))  # type: ignore[method-assign]
    mesh._audit_local = lambda event, payload: audited.append((event, payload))  # type: ignore[method-assign]
    mesh._exec_cmd(
        {
            "sender_id": "dash",
            "turn_id": "turn-1",
            "command": {"action": "call", "function": "add_object", "params": {"name": "red_cube"}},
        }
    )
    assert published and published[0][1]["result"]["status"] == "success"
    events = [(event, payload.get("action"), payload.get("function")) for event, payload in audited]
    assert ("command_executed", "call", "add_object") in events


# ─────────────────────────────────────────────── the round trip ────────────


def test_a_red_cube_added_with_call_is_listed_by_the_same_world() -> None:
    pytest.importorskip("mujoco")
    from strands_robots.simulation import create_simulation

    sim: Any = create_simulation("mujoco")
    try:
        assert sim(action="create_world")["status"] == "success"
        assert sim(action="add_robot", name="so101")["status"] == "success"
        mesh = _mesh_with(sim)
        described = mesh._dispatch({"action": "describe_tool"})
        assert "add_object" in described["spec"]["functions"]
        assert described["tool_spec_hash"] == mesh._build_presence()["tool_spec_hash"]
        added = mesh._dispatch(
            _call(
                "add_object",
                name="red_cube",
                shape="box",
                size=[0.02, 0.02, 0.02],
                color=[1.0, 0.0, 0.0, 1.0],
                position=[0.25, 0.0, 0.02],
            )
        )
        assert added["status"] == "success", added
        listed = mesh._dispatch(_call("list_objects"))
        assert listed["status"] == "success"
        assert "red_cube" in listed["content"][0]["text"]
        refused = mesh._dispatch(_call("destroy"))
        assert refused["error"].startswith("'destroy' is not served over the mesh")
        assert sim._world is not None
    finally:
        sim.cleanup()
