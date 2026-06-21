"""Behavior tests for the MoveIt2 ZMQ sidecar reference implementation.

Exercises :mod:`strands_robots.policies.moveit2.server.zmq_node` without a
ROS 2 environment. The sidecar keeps every ROS / ``moveit_py`` import lazy
(inside ``_build_moveit_py``, ``_plan``, and ``main``), so the module imports
cleanly on a plain venv and the heavy deps are supplied here as light fakes
through ``sys.modules`` / monkeypatch. ``zmq`` and ``msgpack`` are real (they
ship with the ``[moveit2]`` extra); the REP loop runs against a fake socket
that replays a fixed request sequence and then raises ``KeyboardInterrupt``
to break out exactly like a real Ctrl-C.

The wire protocol pinned here matches what the client
(:class:`strands_robots.policies.moveit2.MoveIt2Policy`) speaks: msgpack
request/response with ``ping`` / ``reset`` / ``plan`` endpoints and
``[time_from_start, q0..qN]`` trajectory rows.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

msgpack = pytest.importorskip(
    "msgpack",
    reason="msgpack not installed - pip install 'strands-robots[moveit2]'",
)

from strands_robots.policies.moveit2.server import zmq_node  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes for the ROS / moveit_py surface the sidecar touches.
# ---------------------------------------------------------------------------
class _FakePoint:
    def __init__(self, sec: int, nanosec: int, positions: list[float]) -> None:
        self.time_from_start = types.SimpleNamespace(sec=sec, nanosec=nanosec)
        self.positions = positions


class _FakeComponent:
    """Stand-in for a moveit_py PlanningComponent."""

    def __init__(self, *, plan_points: list[_FakePoint] | None, plan_raises: bool = False) -> None:
        self._plan_points = plan_points
        self._plan_raises = plan_raises
        self.goal: dict[str, Any] = {}
        self.start_state_set = False

    def set_start_state_to_current_state(self) -> None:
        self.start_state_set = True

    def set_goal_state(self, **kwargs: Any) -> None:
        self.goal = kwargs

    def plan(self) -> Any:
        if self._plan_raises:
            raise RuntimeError("ompl exploded")
        if self._plan_points is None:
            return None
        joint_traj = types.SimpleNamespace(points=self._plan_points)
        trajectory = types.SimpleNamespace(joint_trajectory=joint_traj)
        return types.SimpleNamespace(trajectory=trajectory)


class _FakeMoveItPy:
    def __init__(self, component: _FakeComponent | None, *, unknown_group: bool = False) -> None:
        self._component = component
        self._unknown_group = unknown_group

    def get_planning_component(self, group: str) -> _FakeComponent:
        if self._unknown_group:
            raise KeyError(group)
        assert self._component is not None  # only None in the unknown_group path
        return self._component


@pytest.fixture(autouse=True)
def fake_geometry_msgs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake ``geometry_msgs.msg`` exposing ``PoseStamped``.

    ``_plan`` imports ``PoseStamped`` at function entry for every goal branch
    (not just the pose path), so this is autouse for the whole module.
    """

    class _PoseStamped:
        def __init__(self) -> None:
            self.header = types.SimpleNamespace(frame_id="")
            self.pose = types.SimpleNamespace(
                position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=types.SimpleNamespace(w=0.0, x=0.0, y=0.0, z=0.0),
            )

    geometry_pkg = types.ModuleType("geometry_msgs")
    geometry_msg_mod = types.ModuleType("geometry_msgs.msg")
    geometry_msg_mod.PoseStamped = _PoseStamped  # type: ignore[attr-defined]
    geometry_pkg.msg = geometry_msg_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "geometry_msgs", geometry_pkg)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", geometry_msg_mod)


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------
def test_parse_args_defaults_match_client_protocol() -> None:
    args = zmq_node._parse_args([])
    assert args.host == "0.0.0.0"
    assert args.port == 5556  # MoveIt2Policy default port
    assert args.planning_group == "arm"
    assert args.log_level == "INFO"


def test_parse_args_overrides_are_applied() -> None:
    args = zmq_node._parse_args(
        ["--host", "127.0.0.1", "--port", "6000", "--planning-group", "left_arm", "--log-level", "DEBUG"]
    )
    assert args.host == "127.0.0.1"
    assert args.port == 6000
    assert args.planning_group == "left_arm"
    assert args.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# _plan: every branch of the goal/result decision tree
# ---------------------------------------------------------------------------
def test_plan_unknown_planning_group_returns_structured_failure() -> None:
    moveit_py = _FakeMoveItPy(component=None, unknown_group=True)
    resp = zmq_node._plan(
        moveit_py,
        planning_group="nope",
        joint_state=None,
        target_pose=None,
        target_joints=None,
        world_update=None,
    )
    assert resp["success"] is False
    assert resp["trajectory"] == []
    assert resp["status"].startswith("unknown_planning_group:")


def test_plan_missing_goal_is_rejected() -> None:
    moveit_py = _FakeMoveItPy(component=_FakeComponent(plan_points=[]))
    resp = zmq_node._plan(
        moveit_py,
        planning_group="arm",
        joint_state=[0.1, 0.2],  # hint accepted but unused -> exercises debug branch
        target_pose=None,
        target_joints=None,
        world_update=None,
    )
    assert resp["success"] is False
    assert resp["status"] == "missing_goal:expected_target_pose_or_target_joints"


def test_plan_with_target_joints_succeeds_and_serialises_rows() -> None:
    points = [
        _FakePoint(sec=0, nanosec=0, positions=[0.0, 0.0]),
        _FakePoint(sec=1, nanosec=500_000_000, positions=[0.5, 0.6]),
    ]
    component = _FakeComponent(plan_points=points)
    moveit_py = _FakeMoveItPy(component=component)
    resp = zmq_node._plan(
        moveit_py,
        planning_group="arm",
        joint_state=None,
        target_pose=None,
        target_joints={"j0": 0.5, "j1": 0.6},
        world_update={"depth_topic": "/camera/depth"},  # schema-free, ignored
    )
    assert resp["success"] is True
    assert resp["status"] == "ok"
    assert component.start_state_set is True
    assert component.goal == {"joint_values": {"j0": 0.5, "j1": 0.6}}
    # rows are [time_from_start_seconds, q0, q1]; nanosec folds into seconds.
    assert resp["trajectory"][0] == [0.0, 0.0, 0.0]
    assert resp["trajectory"][1][0] == pytest.approx(1.5)
    assert resp["trajectory"][1][1:] == [0.5, 0.6]


def test_plan_with_target_pose_builds_posestamped() -> None:
    component = _FakeComponent(plan_points=[_FakePoint(sec=2, nanosec=0, positions=[1.0])])
    moveit_py = _FakeMoveItPy(component=component)
    resp = zmq_node._plan(
        moveit_py,
        planning_group="arm",
        joint_state=None,
        target_pose=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        target_joints=None,
        world_update=None,
    )
    assert resp["success"] is True
    pose = component.goal["pose_stamped_msg"]
    assert component.goal["pose_link"] == "end_effector_link"
    assert pose.header.frame_id == "base_link"
    assert (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z) == (0.1, 0.2, 0.3)
    assert pose.pose.orientation.w == 1.0


def test_plan_planner_exception_is_caught() -> None:
    component = _FakeComponent(plan_points=None, plan_raises=True)
    moveit_py = _FakeMoveItPy(component=component)
    resp = zmq_node._plan(
        moveit_py,
        planning_group="arm",
        joint_state=None,
        target_pose=None,
        target_joints={"j0": 0.0},
        world_update=None,
    )
    assert resp["success"] is False
    assert resp["status"].startswith("planner_exception:")


def test_plan_empty_result_reported() -> None:
    component = _FakeComponent(plan_points=None)  # plan() -> None
    moveit_py = _FakeMoveItPy(component=component)
    resp = zmq_node._plan(
        moveit_py,
        planning_group="arm",
        joint_state=None,
        target_pose=None,
        target_joints={"j0": 0.0},
        world_update=None,
    )
    assert resp["success"] is False
    assert resp["status"] == "planner_returned_empty"


# ---------------------------------------------------------------------------
# _build_moveit_py: builder wiring (moveit_py + config builder faked)
# ---------------------------------------------------------------------------
def test_build_moveit_py_wires_optional_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    class _Builder:
        def robot_description(self, package: str) -> _Builder:
            calls["robot_description"] = package
            return self

        def moveit_cpp(self, file_path: str) -> _Builder:
            calls["moveit_cpp"] = file_path
            return self

        def to_moveit_configs(self) -> Any:
            return types.SimpleNamespace(to_dict=lambda: {"k": "v"})

    def _ConfigsBuilder(robot_name: str) -> _Builder:
        calls["robot_name"] = robot_name
        return _Builder()

    class _MoveItPy:
        def __init__(self, node_name: str, config_dict: dict) -> None:
            calls["node_name"] = node_name
            calls["config_dict"] = config_dict

        def get_planning_component_names(self) -> list[str]:
            return ["arm"]

    planning_mod = types.ModuleType("moveit.planning")
    planning_mod.MoveItPy = _MoveItPy  # type: ignore[attr-defined]
    moveit_pkg = types.ModuleType("moveit")
    moveit_pkg.planning = planning_mod  # type: ignore[attr-defined]
    configs_mod = types.ModuleType("moveit_configs_utils")
    configs_mod.MoveItConfigsBuilder = _ConfigsBuilder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "moveit", moveit_pkg)
    monkeypatch.setitem(sys.modules, "moveit.planning", planning_mod)
    monkeypatch.setitem(sys.modules, "moveit_configs_utils", configs_mod)

    args = zmq_node._parse_args(
        ["--robot-description-package", "my_robot_desc", "--moveit-config-package", "my_moveit_cfg"]
    )
    result = zmq_node._build_moveit_py(args)

    assert isinstance(result, _MoveItPy)
    assert calls["robot_name"] == "moveit2_sidecar"
    assert calls["robot_description"] == "my_robot_desc"
    assert calls["moveit_cpp"] == "my_moveit_cfg"
    assert calls["config_dict"] == {"k": "v"}


# ---------------------------------------------------------------------------
# main: the REP dispatch loop (ping / reset / plan / unknown / malformed)
# ---------------------------------------------------------------------------
class _FakeSocket:
    """ZMQ REP socket replaying ``recv_queue`` then raising KeyboardInterrupt."""

    def __init__(self, recv_queue: list[bytes]) -> None:
        self._recv_queue = list(recv_queue)
        self.sent: list[bytes] = []
        self.bound_to: str | None = None
        self.closed = False

    def bind(self, addr: str) -> None:
        self.bound_to = addr

    def recv(self) -> bytes:
        if not self._recv_queue:
            raise KeyboardInterrupt
        return self._recv_queue.pop(0)

    def send(self, data: bytes) -> None:
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True


def _install_fake_zmq_rclpy(monkeypatch: pytest.MonkeyPatch, socket: _FakeSocket) -> dict[str, bool]:
    flags = {"rclpy_init": False, "rclpy_shutdown": False, "ctx_term": False}

    class _Context:
        def socket(self, _kind: Any) -> _FakeSocket:
            return socket

        def term(self) -> None:
            flags["ctx_term"] = True

    zmq_mod = types.ModuleType("zmq")
    zmq_mod.REP = "REP"  # type: ignore[attr-defined]
    zmq_mod.Context = types.SimpleNamespace(instance=lambda: _Context())  # type: ignore[attr-defined]
    rclpy_mod = types.ModuleType("rclpy")
    rclpy_mod.init = lambda: flags.__setitem__("rclpy_init", True)  # type: ignore[attr-defined]
    rclpy_mod.shutdown = lambda: flags.__setitem__("rclpy_shutdown", True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zmq", zmq_mod)
    monkeypatch.setitem(sys.modules, "rclpy", rclpy_mod)
    return flags


def test_main_dispatches_endpoints_and_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = [
        msgpack.packb({"endpoint": "ping"}, use_bin_type=True),
        msgpack.packb({"endpoint": "reset", "data": {"options": {"seed": 7}}}, use_bin_type=True),
        msgpack.packb(
            {"endpoint": "plan", "data": {"planning_group": "arm", "target_joints": {"j0": 0.1}}},
            use_bin_type=True,
        ),
        msgpack.packb({"endpoint": "bogus"}, use_bin_type=True),
        b"\xc1",  # invalid msgpack -> malformed_request branch
    ]
    socket = _FakeSocket(requests)
    flags = _install_fake_zmq_rclpy(monkeypatch, socket)

    component = _FakeComponent(plan_points=[_FakePoint(sec=0, nanosec=0, positions=[0.1])])
    monkeypatch.setattr(zmq_node, "_build_moveit_py", lambda args: _FakeMoveItPy(component=component))

    rc = zmq_node.main(["--port", "5599"])

    assert rc == 0
    assert flags["rclpy_init"] and flags["rclpy_shutdown"] and flags["ctx_term"]
    assert socket.bound_to == "tcp://0.0.0.0:5599"
    assert socket.closed is True

    responses = [msgpack.unpackb(s, raw=False) for s in socket.sent]
    assert responses[0] == {"status": "ok"}  # ping
    assert responses[1] == {"status": "ok"}  # reset no-op
    assert responses[2]["success"] is True  # plan
    assert responses[2]["trajectory"] == [[0.0, 0.1]]
    assert responses[3] == {"error": "unknown_endpoint:bogus"}
    assert "malformed_request" in responses[4]["error"]


def test_main_returns_1_when_moveit_construction_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = _FakeSocket([])
    flags = _install_fake_zmq_rclpy(monkeypatch, socket)

    def _boom(args: Any) -> Any:
        raise RuntimeError("no ros sourced")

    monkeypatch.setattr(zmq_node, "_build_moveit_py", _boom)

    rc = zmq_node.main([])

    assert rc == 1
    assert flags["rclpy_init"] is True
    assert flags["rclpy_shutdown"] is True
    # Socket was never bound because construction failed first.
    assert socket.bound_to is None
