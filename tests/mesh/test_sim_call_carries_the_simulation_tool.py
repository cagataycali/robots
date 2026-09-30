"""``sim_call`` carries the simulation tool over the mesh, one published action at a time.

A ``Robot("so101")`` in simulation exposes 77 actions to an in-process agent
(``tool_spec.json``) and answered seven verbs on the mesh, so an agent driving
it from another process could move a joint but not add a cube. Pinned here:

* the wire: ``sim_call`` is an allowed action whose ``sim_action`` must be a
  published simulation action outside :data:`SIM_CALL_DENIED_ACTIONS`, whose
  ``params`` keys must be published params outside :data:`SIM_CALL_DENIED_PARAMS`,
  identifier-safe and size-bounded; every one of the 77 published actions is
  either admitted or denied on purpose, never by omission;
* the dispatch: ``Mesh._dispatch`` serves it through the simulation's own
  ``__call__`` (a child peer binds its robot when the action takes one), a
  hardware peer refuses with a sentence, the lockout refuses it, and the result
  is made JSON-safe (image bytes travel base64, oversize blocks are named);
* the audit: a ``sim_call`` record names its ``sim_action``;
* the round trip: on a real MuJoCo world, ``add_object`` then ``list_objects``
  sees the cube, and a child peer's ``set_joint_positions`` lands on its robot.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import security

# ────────────────────────────────────────────── the wire: sim_call ─────────


def test_sim_call_is_an_allowed_action_carrying_a_published_action_and_its_params() -> None:
    assert "sim_call" in security.ALLOWED_ACTIONS
    out = security.validate_command(
        {
            "action": "sim_call",
            "sim_action": "add_object",
            "params": {"name": "red_cube", "shape": "box", "size": [0.02, 0.02, 0.02], "color": [1, 0, 0, 1]},
            "turn_id": "t-1",
            "instruction": "dropped: not a sim_call field",
        }
    )
    assert set(out) == {"action", "sim_action", "params", "turn_id"}
    assert out["params"]["name"] == "red_cube"
    assert out["params"] is not None


def test_every_published_action_is_admitted_or_denied_on_purpose() -> None:
    """The 77-name enum splits exactly into the admitted set and the deny list."""
    published = security.sim_call_published_actions()
    allowed = security.sim_call_allowed_actions()
    denied = security.SIM_CALL_DENIED_ACTIONS
    assert len(published) == 77
    assert denied <= published, sorted(denied - published)
    assert allowed | denied == published
    assert not (allowed & denied)
    for name in sorted(published):
        cmd = {"action": "sim_call", "sim_action": name}
        if name in denied:
            with pytest.raises(security.ValidationError, match="refused on the wire"):
                security.validate_command(cmd)
        else:
            assert security.validate_command(cmd)["sim_action"] == name


def test_the_deny_lists_name_only_things_the_tool_publishes() -> None:
    """A denied name that the spec no longer carries is a stale rule, not a guard."""
    assert security.SIM_CALL_DENIED_ACTIONS <= security.sim_call_published_actions()
    assert security.SIM_CALL_DENIED_PARAMS <= security.sim_call_published_params()
    assert set(security.SIM_CALL_RAIL_FOR) <= security.SIM_CALL_DENIED_ACTIONS
    assert set(security.SIM_CALL_RAIL_FOR.values()) <= security.ALLOWED_ACTIONS


@pytest.mark.parametrize(
    ("sim_action", "rail"),
    [("run_policy", "execute"), ("start_policy", "start"), ("stop_policy", "stop"), ("eval_policy", "execute")],
)
def test_a_denied_rollout_names_the_rail_it_rides(sim_action: str, rail: str) -> None:
    with pytest.raises(security.ValidationError, match=f"rides the `{rail}` action"):
        security.validate_command({"action": "sim_call", "sim_action": sim_action})


@pytest.mark.parametrize(
    ("cmd", "fragment"),
    [
        ({"action": "sim_call"}, "requires `sim_action`"),
        ({"action": "sim_call", "sim_action": ""}, "requires `sim_action`"),
        ({"action": "sim_call", "sim_action": 7}, "requires `sim_action`"),
        ({"action": "sim_call", "sim_action": "add object"}, "must match"),
        ({"action": "sim_call", "sim_action": "a" * 65}, "MAX_SIM_CALL_NAME_LEN"),
        ({"action": "sim_call", "sim_action": "no_such_action"}, "not a published simulation action"),
        ({"action": "sim_call", "sim_action": "destroy"}, "not carried over the mesh"),
        ({"action": "sim_call", "sim_action": "render", "params": {"output_path": "/tmp/x.png"}}, "peer-host path"),
        ({"action": "sim_call", "sim_action": "add_object", "params": {"mesh_path": "a.stl"}}, "peer-host path"),
        ({"action": "sim_call", "sim_action": "start_recording", "params": {"push_to_hub": True}}, "egress"),
        (
            {"action": "sim_call", "sim_action": "add_object", "params": {"bogus": 1}},
            "not a published simulation param",
        ),
        ({"action": "sim_call", "sim_action": "add_object", "params": {"na me": 1}}, "must match"),
        ({"action": "sim_call", "sim_action": "add_object", "params": {"": 1}}, "non-empty strings"),
        ({"action": "sim_call", "sim_action": "add_object", "params": [1, 2]}, "JSON object"),
        ({"action": "sim_call", "sim_action": "add_object", "params": {"name": object()}}, "not JSON-serialisable"),
        (
            {"action": "sim_call", "sim_action": "add_object", "params": {"name": "x" * (64 * 1024 + 1)}},
            "MAX_SIM_CALL_PARAMS_BYTES",
        ),
    ],
)
def test_sim_call_refusals_name_the_rule(cmd: dict[str, Any], fragment: str) -> None:
    with pytest.raises(security.ValidationError, match=fragment):
        security.validate_command(cmd)


def test_null_params_are_an_empty_object() -> None:
    out = security.validate_command({"action": "sim_call", "sim_action": "list_objects", "params": None})
    assert out["params"] == {}


# ─────────────────────────────────────────────── the dispatch ──────────────


def _mesh_with(robot: Any, *, lockout: bool = False) -> Any:
    """A Mesh object with just what ``_dispatch`` reads, no zenoh session."""
    mesh = mesh_core.Mesh.__new__(mesh_core.Mesh)
    mesh.robot = robot
    mesh.peer_id = "sim-1"
    mesh._estop_lockout = MagicMock(is_set=lambda: lockout)
    return mesh


class _Sim:
    """A Simulation as ``_dispatch`` recognises one: callable, a world, ``list_robots``."""

    _world = object()
    _ACTION_ALIASES = {"list_robots": "list_robots_info"}

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_robots(self) -> list[str]:
        return ["so101"]

    def list_robots_info(self) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": "so101"}]}

    def add_object(self, name: str, shape: str = "box") -> dict[str, Any]:
        return {"status": "success", "content": [{"text": f"{name} {shape}"}]}

    def get_robot_state(self, robot_name: str | None = None) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": robot_name or "-"}]}

    def __call__(self, action: str = "", **kwargs: Any) -> dict[str, Any]:
        self.calls.append((action, dict(kwargs)))
        method = getattr(self, self._ACTION_ALIASES.get(action, action))
        return dict(method(**kwargs))

    def wire_tool_spec(self) -> dict[str, Any]:
        from strands_robots.simulation.mujoco.wire_surface import build_wire_tool_spec

        return build_wire_tool_spec("sim", type(self))


class _SimChild:
    """A child SimRobot peer: no actions of its own, a ``_sim_parent`` that has them."""

    def __init__(self, parent: _Sim) -> None:
        self._sim_parent = parent
        self.name = "so101"


def _validated(sim_action: str, **params: Any) -> dict[str, Any]:
    return security.validate_command({"action": "sim_call", "sim_action": sim_action, "params": params})


def test_dispatch_routes_a_sim_call_through_the_simulations_own_call() -> None:
    sim = _Sim()
    out = _mesh_with(sim)._dispatch(_validated("add_object", name="red_cube", shape="box"))
    assert out == {"status": "success", "content": [{"text": "red_cube box"}]}
    assert sim.calls == [("add_object", {"name": "red_cube", "shape": "box"})]


def test_dispatch_on_a_child_binds_its_robot_only_when_the_action_takes_one() -> None:
    sim = _Sim()
    mesh = _mesh_with(_SimChild(sim))
    mesh._dispatch(_validated("get_robot_state"))
    mesh._dispatch(_validated("get_robot_state", robot_name="other"))
    mesh._dispatch(_validated("list_robots"))
    assert sim.calls == [
        ("get_robot_state", {"robot_name": "so101"}),
        ("get_robot_state", {"robot_name": "other"}),
        ("list_robots", {}),
    ]


def test_dispatch_refuses_a_hardware_peer_with_a_sentence() -> None:
    hardware = MagicMock(spec=["get_task_status", "stop_task"])
    out = _mesh_with(hardware)._dispatch(_validated("list_objects"))
    assert out["error"].startswith("sim_call is a simulation-only action")
    hardware.stop_task.assert_not_called()


def test_dispatch_refuses_a_sim_call_under_the_lockout() -> None:
    sim = _Sim()
    with pytest.raises(security.LockoutError):
        _mesh_with(sim, lockout=True)._dispatch(_validated("add_object", name="x"))
    assert sim.calls == []


def test_wire_safe_result_carries_an_image_as_base64_and_names_what_it_cuts() -> None:
    png = b"\x89PNG" + bytes(range(256))
    result = {
        "status": "success",
        "content": [
            {"text": "160x120"},
            {"image": {"format": "png", "source": {"bytes": png}}},
            {"image": {"format": "png", "source": {"bytes": b"x" * (mesh_core.SIM_CALL_MAX_IMAGE_BYTES + 1)}}},
            {"text": "y" * (mesh_core.SIM_CALL_MAX_TEXT_CHARS + 10)},
            {"other": object()},
        ],
    }
    out = mesh_core._wire_safe_result(result)
    encoded = json.dumps(out)
    assert json.loads(encoded)["status"] == "success"
    image = out["content"][1]["image"]
    assert image["format"] == "png"
    assert image["bytes_len"] == len(png)
    import base64

    assert base64.b64decode(image["base64"]) == png
    assert "not carried" in out["content"][2]["text"]
    assert str(mesh_core.SIM_CALL_MAX_IMAGE_BYTES) in out["content"][2]["text"]
    assert out["content"][3]["text"].endswith("[... 10 more characters not carried over the wire]")
    assert isinstance(out["content"][4]["other"], str)


def test_wire_safe_result_wraps_a_non_dict() -> None:
    assert mesh_core._wire_safe_result([1, 2]) == {"result": [1, 2]}


# ──────────────────────────────────────────────── the audit ────────────────


def test_an_executed_sim_call_is_audited_with_its_sim_action(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = _Sim()
    mesh = _mesh_with(sim)
    mesh._cmd_replay_lock = __import__("threading").Lock()
    mesh._cmd_replay_cache = {}
    published: list[tuple[str, dict[str, Any]]] = []
    audited: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(mesh, "publish", lambda key, payload: published.append((key, payload)))
    monkeypatch.setattr(mesh, "_audit_local", lambda event, payload: audited.append((event, payload)))
    mesh._exec_cmd(
        {
            "sender_id": "dash",
            "turn_id": "turn-1",
            "command": {"action": "sim_call", "sim_action": "add_object", "params": {"name": "red_cube"}},
        }
    )
    assert published and published[0][1]["result"]["status"] == "success"
    events = [(event, payload.get("action"), payload.get("sim_action")) for event, payload in audited]
    assert ("command_executed", "sim_call", "add_object") in events


# ─────────────────────────────────────────────── the round trip ────────────


def test_a_red_cube_added_over_the_wire_is_listed_by_the_same_world() -> None:
    pytest.importorskip("mujoco")
    from strands_robots.simulation import create_simulation

    sim: Any = create_simulation("mujoco")
    try:
        assert sim(action="create_world")["status"] == "success"
        assert sim(action="add_robot", name="so101")["status"] == "success"
        mesh = _mesh_with(sim)
        added = mesh._dispatch(
            _validated(
                "add_object",
                name="red_cube",
                shape="box",
                size=[0.02, 0.02, 0.02],
                color=[1.0, 0.0, 0.0, 1.0],
                position=[0.25, 0.0, 0.02],
            )
        )
        assert added["status"] == "success", added
        listed = mesh._dispatch(_validated("list_objects"))
        assert listed["status"] == "success"
        assert "red_cube" in listed["content"][0]["text"]
        assert json.dumps(listed)

        child = sim._world.robots["so101"]
        child._sim_parent = sim
        child_mesh = _mesh_with(child)
        moved = child_mesh._dispatch(_validated("set_joint_positions", positions={"shoulder_lift": 0.3}, hold=True))
        assert moved["status"] == "success", moved
        state = child_mesh._dispatch(_validated("get_robot_state"))
        assert "so101" in state["content"][0]["text"]
        duplicate = child_mesh._dispatch(_validated("add_object", name="red_cube", shape="box"))
        assert duplicate["status"] == "error", duplicate
    finally:
        sim.cleanup()


# ───────────────────────────────────────── two commands in a row ───────────


def _sending_mesh(published: list[tuple[str, dict[str, Any]]]) -> Any:
    """A Mesh with the send path's state and a publish that records instead of writing."""
    import threading

    mesh = mesh_core.Mesh.__new__(mesh_core.Mesh)
    mesh.peer_id = "dash"
    mesh._direct = None
    mesh._running = True
    mesh._rpc_lock = threading.Lock()
    mesh._cmd_pace_lock = threading.Lock()
    mesh._last_cmd_publish_mono = None
    mesh._stop_event = threading.Event()
    mesh._pending = {}
    mesh._responses = {}
    mesh._expected_responders = {}
    mesh._turn_sources = {}
    mesh.publish = lambda key, payload: published.append((key, payload))  # type: ignore[method-assign]
    return mesh


def test_two_commands_in_a_row_are_published_one_period_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    """The receiver drops a cmd arriving inside one period of the last; the sender never lets that happen."""
    import time

    monkeypatch.setenv("STRANDS_MESH_CMD_RATE_HZ", "20")
    published: list[tuple[str, dict[str, Any]]] = []
    mesh = _sending_mesh(published)
    stamps: list[float] = []
    original = mesh.publish

    def stamped(key: str, payload: dict[str, Any]) -> None:
        stamps.append(time.monotonic())
        original(key, payload)

    mesh.publish = stamped  # type: ignore[method-assign]
    mesh.send("sim-a", {"action": "sim_call", "sim_action": "list_objects"}, timeout=0.01)
    mesh.send("sim-b", {"action": "sim_call", "sim_action": "list_cameras"}, timeout=0.01)
    mesh.broadcast({"action": "status"}, timeout=0.01)
    assert [key for key, _ in published] == ["strands/sim-a/cmd", "strands/sim-b/cmd", "strands/broadcast"]
    gaps = [b - a for a, b in zip(stamps[:-1], stamps[1:], strict=True)]
    assert all(gap >= 1 / 20 for gap in gaps), gaps


def test_a_stopping_mesh_does_not_hold_the_pacer(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    monkeypatch.setenv("STRANDS_MESH_CMD_RATE_HZ", "0.5")  # a two second period
    mesh = _sending_mesh([])
    mesh.send("sim-a", {"action": "status"}, timeout=0.01)
    mesh._stop_event.set()
    started = time.monotonic()
    mesh.send("sim-a", {"action": "status"}, timeout=0.01)
    assert time.monotonic() - started < 1.0


def test_a_mesh_built_without_the_pacer_state_still_sends() -> None:
    published: list[tuple[str, dict[str, Any]]] = []
    mesh = _sending_mesh(published)
    del mesh._cmd_pace_lock
    mesh.send("sim-a", {"action": "status"}, timeout=0.01)
    assert published


# ──────────────────────────────────────── the card shows the cube red ───────


def test_a_published_camera_frame_keeps_its_colours() -> None:
    """A red frame leaves the peer as a red JPEG: the encoder is handed BGR, the frame is RGB."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    published: list[tuple[str, dict[str, Any]]] = []
    mesh = mesh_core.Mesh.__new__(mesh_core.Mesh)
    mesh.peer_id = "sim-1"
    mesh.publish = lambda key, payload: published.append((key, payload))  # type: ignore[method-assign]
    red = np.zeros((8, 8, 3), dtype=np.uint8)
    red[..., 0] = 255  # RGB: red
    mesh._encode_and_publish_frames({"front": red}, ["front"])
    [(key, payload)] = published
    assert key == "strands/sim-1/camera/front" and payload["encoding"] == "jpeg"
    import base64

    decoded_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(payload["data"]), np.uint8), cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = (int(decoded_rgb[4, 4, i]) for i in range(3))
    assert r > 200 and g < 60 and b < 60, (r, g, b)
