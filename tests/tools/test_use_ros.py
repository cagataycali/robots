"""Behavior tests for the ``use_ros`` agent tool.

The tool bridges a Strands agent to a ROS 2 graph **entirely in-process through
``rclpy``** - there is no ``ros2`` CLI shelling and no generated-code snippets.
These tests run with NO ROS 2 installed: the rclpy-facing helpers
(``_list_topics`` / ``_echo`` / ``_publish`` / ``_service_call`` / ...) and the
backend-availability probe are monkeypatched, so every action-dispatch branch,
the agent-input validation, the no-backend error path, and the structured
error-return contract are exercised hardware- and ROS-free.

It also pins package-wide contracts:

* No emoji / non-ASCII in any returned ``text``.
* ``fields`` payloads (bool / None / nested) are passed straight through to the
  rclpy helper as a real Python dict - never serialised into source - so types
  are preserved by construction.
* Backend errors surface as a structured ``{"status": "error"}`` result, never
  a raised exception.
"""

from __future__ import annotations

import sys
import types as _types
from typing import Any

import pytest

import strands_robots.tools.use_ros as ros_mod

# Reference the tool via a module-local alias rather than a second `from`
# import: the tests monkeypatch module internals through `ros_mod`, so the
# module object is the single source of truth and a dual import is avoided.
use_ros = ros_mod.use_ros


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _ascii_only(result: dict[str, Any]) -> None:
    text = _texts(result)
    assert text.isascii(), f"non-ASCII in tool output: {text!r}"


@pytest.fixture(autouse=True)
def _backend_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to a present rclpy backend; opt out where needed."""
    monkeypatch.setattr(ros_mod._backend, "available", lambda: True)


# Validation ----------------------------------------------------------------


@pytest.mark.parametrize("bad", ["/foo; rm -rf", "/a b", "/x|y", "../etc", "/a$(x)"])
def test_invalid_topic_rejected(bad: str) -> None:
    result = use_ros(action="echo", topic=bad)
    assert result["status"] == "error"
    assert "invalid topic" in _texts(result)
    _ascii_only(result)


def test_invalid_type_rejected() -> None:
    result = use_ros(action="publish", topic="/cmd_vel", type="not_a_type")
    assert result["status"] == "error"
    assert "invalid interface type" in _texts(result)


def test_invalid_service_rejected() -> None:
    result = use_ros(action="service_call", service="/spawn bad", type="turtlesim/srv/Spawn")
    assert result["status"] == "error"
    assert "invalid service" in _texts(result)


# Status --------------------------------------------------------------------


def test_status_reports_rclpy_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod._backend, "available", lambda: True)
    result = use_ros(action="status")
    assert result["status"] == "success"
    assert "backend: rclpy (in-process)" in _texts(result)
    _ascii_only(result)


def test_status_reports_none_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod._backend, "available", lambda: False)
    result = use_ros(action="status")
    assert result["status"] == "success"
    assert "backend: none" in _texts(result)
    assert "ROS 2" in _texts(result)
    _ascii_only(result)


# Listings ------------------------------------------------------------------


def test_list_topics_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_list_topics", lambda: "/turtle1/cmd_vel [geometry_msgs/msg/Twist]")
    result = use_ros(action="list_topics")
    assert result["status"] == "success"
    assert "/turtle1/cmd_vel" in _texts(result)
    _ascii_only(result)


def test_list_nodes_and_services(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_list_nodes", lambda: "/turtlesim")
    monkeypatch.setattr(ros_mod, "_list_services", lambda: "/spawn [turtlesim/srv/Spawn]")
    assert "/turtlesim" in _texts(use_ros(action="list_nodes"))
    assert "/spawn" in _texts(use_ros(action="list_services"))


# info ----------------------------------------------------------------------


def test_info_requires_target() -> None:
    result = use_ros(action="info")
    assert result["status"] == "error"
    assert "requires topic or service" in _texts(result)


def test_info_returns_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_info", lambda target: f"topic info {target}:\n  publishers: 1")
    result = use_ros(action="info", topic="/turtle1/pose")
    assert result["status"] == "success"
    assert "topic info /turtle1/pose" in _texts(result)


def test_info_miss_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_info", lambda target: None)
    result = use_ros(action="info", topic="/nope")
    assert result["status"] == "error"
    assert "no info for /nope" in _texts(result)


# echo ----------------------------------------------------------------------


def test_echo_requires_topic() -> None:
    assert use_ros(action="echo")["status"] == "error"


def test_echo_autoresolves_type_and_returns_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_resolve_topic_type", lambda topic: "turtlesim/msg/Pose")
    samples = [{"x": 5.5, "y": 1.0}, {"x": 6.0, "y": 1.0}]

    def fake_echo(topic: str, msg_type: str, timeout: float, count: int) -> list[dict[str, Any]]:
        assert msg_type == "turtlesim/msg/Pose"  # auto-resolved type reached the helper
        return samples

    monkeypatch.setattr(ros_mod, "_echo", fake_echo)
    result = use_ros(action="echo", topic="/turtle1/pose", count=2)
    assert result["status"] == "success"
    assert "turtlesim/msg/Pose" in _texts(result)
    assert "5.5" in _texts(result)
    _ascii_only(result)


def test_echo_unresolvable_type_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_resolve_topic_type", lambda topic: None)
    result = use_ros(action="echo", topic="/turtle1/pose")
    assert result["status"] == "error"
    assert "cannot resolve type" in _texts(result)


# publish / service_call ----------------------------------------------------


def test_publish_requires_topic_and_type() -> None:
    assert use_ros(action="publish", topic="/cmd_vel")["status"] == "error"


def test_service_call_requires_service_and_type() -> None:
    result = use_ros(action="service_call", service="/spawn")
    assert result["status"] == "error"
    assert "requires service and type" in _texts(result)


def test_publish_dispatches_with_real_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_publish(topic, msg_type, fields, count, rate) -> None:
        captured.update(topic=topic, msg_type=msg_type, fields=fields, count=count)

    monkeypatch.setattr(ros_mod, "_publish", fake_publish)
    result = use_ros(
        action="publish",
        topic="/turtle1/cmd_vel",
        type="geometry_msgs/msg/Twist",
        fields={"linear": {"x": 2.0}, "enabled": True, "tag": None},
        count=3,
    )
    assert result["status"] == "success"
    assert "published 3 message(s) to /turtle1/cmd_vel" in _texts(result)
    # The payload reaches the rclpy helper as a real Python dict with types intact.
    assert captured["fields"] == {"linear": {"x": 2.0}, "enabled": True, "tag": None}


def test_service_call_returns_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod, "_service_call", lambda service, srv_type, fields, timeout: {"name": "t2"})
    result = use_ros(
        action="service_call",
        service="/spawn",
        type="turtlesim/srv/Spawn",
        fields={"x": 3.0, "y": 3.0, "name": "t2"},
    )
    assert result["status"] == "success"
    assert "t2" in _texts(result)


# Error / no-backend contracts ----------------------------------------------


def test_no_backend_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ros_mod._backend, "available", lambda: False)
    result = use_ros(action="list_topics")
    assert result["status"] == "error"
    assert "ROS 2" in _texts(result) and "rclpy" in _texts(result)
    _ascii_only(result)


def test_timeout_surfaces_as_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise TimeoutError("service /spawn not available within 5.0s")

    monkeypatch.setattr(ros_mod, "_service_call", boom)
    result = use_ros(action="service_call", service="/spawn", type="turtlesim/srv/Spawn")
    assert result["status"] == "error"
    assert "not available" in _texts(result)


def test_type_resolution_failure_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    # A valid-shaped type naming a package that is not installed is an ordinary
    # agent input (e.g. an LLM-hallucinated type). rosidl_runtime_py.get_message
    # resolves it via importlib.import_module, which raises ModuleNotFoundError
    # (a subclass of ImportError) - the real production failure mode, not KeyError.
    def boom(*a: Any, **k: Any) -> Any:
        raise ModuleNotFoundError("No module named 'nonexistent_pkg'")

    monkeypatch.setattr(ros_mod, "_publish", boom)
    result = use_ros(action="publish", topic="/cmd", type="nonexistent_pkg/msg/Foo")
    assert result["status"] == "error"
    assert "publish failed" in _texts(result)


def test_unknown_action_errors() -> None:
    result = use_ros(action="warp_drive")
    assert result["status"] == "error"
    assert "unknown action" in _texts(result)


# ---------------------------------------------------------------------------
# rclpy-facing helpers (node test-double)
#
# The helpers above this point are exercised only through the monkeypatched
# `use_ros` dispatch. The blocks below drive the helpers themselves -
# `_RosBackend.available`/`_ensure_node`/`spin_for`, the graph-introspection
# formatters (`_list_*`, `_resolve_topic_type`, `_info`), and the pub/sub/
# service primitives (`_echo`/`_publish`/`_service_call`) - against a node
# test-double plus fake `rclpy` / `rosidl_runtime_py` modules. No ROS 2 is
# installed; the doubles stand in for a live graph so the real formatting,
# namespace-joining, sample-capping, and type-resolution logic runs. Behavior
# is asserted through return values, never internal state.
# ---------------------------------------------------------------------------


class _FakeNode:
    """Minimal stand-in for an rclpy Node exposing the graph API the helpers use."""

    def __init__(self) -> None:
        # Deliberately unsorted so helpers' `sorted(...)` is observable.
        self.topics = [
            ("/turtle1/pose", ["turtlesim/msg/Pose"]),
            ("/turtle1/cmd_vel", ["geometry_msgs/msg/Twist"]),
            ("/no_type", []),
        ]
        self.nodes = [
            ("turtlesim", "/"),
            ("planner", "/nav"),
            ("root_node", ""),
        ]
        self.services = [("/spawn", ["turtlesim/srv/Spawn"])]
        self.pub_counts = {"/turtle1/cmd_vel": 1}
        self.sub_counts = {"/turtle1/cmd_vel": 2}
        self.sub_cb: Any = None
        self.publishers: list[Any] = []
        self.clients: list[Any] = []
        self.destroyed: list[tuple[str, Any]] = []

    def get_topic_names_and_types(self) -> list[tuple[str, list[str]]]:
        return list(self.topics)

    def get_node_names_and_namespaces(self) -> list[tuple[str, str]]:
        return list(self.nodes)

    def get_service_names_and_types(self) -> list[tuple[str, list[str]]]:
        return list(self.services)

    def count_publishers(self, name: str) -> int:
        return self.pub_counts.get(name, 0)

    def count_subscribers(self, name: str) -> int:
        return self.sub_counts.get(name, 0)

    def create_subscription(self, cls: Any, topic: str, cb: Any, qos: int) -> Any:
        self.sub_cb = cb
        return object()

    def destroy_subscription(self, sub: Any) -> None:
        self.destroyed.append(("sub", sub))

    def create_publisher(self, cls: Any, topic: str, qos: int) -> Any:
        pub = _types.SimpleNamespace(published=[])
        pub.publish = pub.published.append
        self.publishers.append(pub)
        return pub

    def destroy_publisher(self, pub: Any) -> None:
        self.destroyed.append(("pub", pub))

    def create_client(self, cls: Any, service: str) -> Any:
        client = self.clients[0] if self.clients else None
        return client

    def destroy_client(self, client: Any) -> None:
        self.destroyed.append(("client", client))


@pytest.fixture
def fake_node(monkeypatch: pytest.MonkeyPatch) -> _FakeNode:
    """Route the helpers' node access to a test-double and make spin a no-op."""
    node = _FakeNode()
    monkeypatch.setattr(ros_mod._backend, "_ensure_node", lambda: node)
    monkeypatch.setattr(ros_mod._backend, "spin_for", lambda predicate, timeout: None)
    return node


@pytest.fixture
def fake_rosidl(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject fake rosidl_runtime_py submodules so helper lazy-imports resolve.

    Returns a record dict capturing the set_message_fields calls so tests can
    assert the payload reached the rclpy idiom unchanged.
    """
    record: dict[str, Any] = {"set_fields": []}

    class _FakeMsg:
        pass

    util = _types.ModuleType("rosidl_runtime_py.utilities")
    util.get_message = lambda type_str: _FakeMsg  # type: ignore[attr-defined]

    class _FakeSrv:
        Request = _FakeMsg

    util.get_service = lambda type_str: _FakeSrv  # type: ignore[attr-defined]

    convert = _types.ModuleType("rosidl_runtime_py.convert")
    convert.message_to_ordereddict = lambda msg: msg  # type: ignore[attr-defined]

    setmsg = _types.ModuleType("rosidl_runtime_py.set_message")

    def _set_fields(msg: Any, fields: dict[str, Any]) -> None:
        record["set_fields"].append(fields)

    setmsg.set_message_fields = _set_fields  # type: ignore[attr-defined]

    parent = _types.ModuleType("rosidl_runtime_py")
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py", parent)
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py.utilities", util)
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py.convert", convert)
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py.set_message", setmsg)
    return record


# Backend availability + node lifecycle -------------------------------------


def test_backend_available_true_when_rclpy_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rclpy", _types.ModuleType("rclpy"))
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py", _types.ModuleType("rosidl_runtime_py"))
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py.utilities", _types.ModuleType("rosidl_runtime_py.utilities"))
    backend = ros_mod._RosBackend()
    assert backend.available() is True
    # Result is cached, not re-probed.
    monkeypatch.delitem(sys.modules, "rclpy", raising=False)
    assert backend.available() is True


def test_backend_available_false_when_rclpy_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "rclpy", raising=False)
    backend = ros_mod._RosBackend()
    assert backend.available() is False


def test_ensure_node_initialises_singleton_and_spins(monkeypatch: pytest.MonkeyPatch) -> None:
    inited: list[bool] = []
    spun: list[float | None] = []

    rclpy = _types.ModuleType("rclpy")
    rclpy.ok = lambda: False  # type: ignore[attr-defined]
    rclpy.init = lambda: inited.append(True)  # type: ignore[attr-defined]

    class _FakeExecutor:
        def __init__(self) -> None:
            self.nodes: list[Any] = []

        def add_node(self, node: Any) -> None:
            self.nodes.append(node)

        def spin_once(self, timeout_sec: float | None = None) -> None:
            spun.append(timeout_sec)

    executors = _types.ModuleType("rclpy.executors")
    executors.SingleThreadedExecutor = _FakeExecutor  # type: ignore[attr-defined]
    node_mod = _types.ModuleType("rclpy.node")
    node_mod.Node = lambda name: _types.SimpleNamespace(name=name)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.executors", executors)
    monkeypatch.setitem(sys.modules, "rclpy.node", node_mod)

    backend = ros_mod._RosBackend()
    node = backend._ensure_node()
    assert node.name == "strands_robots_use_ros"
    assert inited == [True]  # rclpy.init() ran because ok() was False
    # Second call returns the cached node without re-initialising.
    assert backend._ensure_node() is node
    assert inited == [True]

    # spin_for drives the executor until the predicate flips.
    backend.spin_for(lambda: len(spun) >= 1, timeout=1.0)
    assert len(spun) >= 1


def test_spin_for_raises_without_executor() -> None:
    backend = ros_mod._RosBackend()
    with pytest.raises(RuntimeError, match="not initialised"):
        backend.spin_for(lambda: True, timeout=0.1)


# Graph introspection formatters --------------------------------------------


def test_list_helpers_format_and_sort_graph(fake_node: _FakeNode) -> None:
    topics = ros_mod._list_topics()
    # Sorted by name; "name [type]" formatting.
    assert topics.splitlines()[0].startswith("/no_type [")
    assert "/turtle1/cmd_vel [geometry_msgs/msg/Twist]" in topics

    nodes = ros_mod._list_nodes().splitlines()
    # Namespace join: "/nav" -> "/nav/planner"; "/" and "" -> "/<name>".
    assert "/nav/planner" in nodes
    assert "/turtlesim" in nodes
    assert "/root_node" in nodes

    assert ros_mod._list_services() == "/spawn [turtlesim/srv/Spawn]"


def test_resolve_topic_type_hit_and_miss(fake_node: _FakeNode) -> None:
    assert ros_mod._resolve_topic_type("/turtle1/pose") == "turtlesim/msg/Pose"
    # A topic present with an empty type list does not resolve.
    assert ros_mod._resolve_topic_type("/no_type") is None
    assert ros_mod._resolve_topic_type("/absent") is None


def test_info_reports_topic_service_and_miss(fake_node: _FakeNode) -> None:
    topic_info = ros_mod._info("/turtle1/cmd_vel")
    assert topic_info is not None
    assert "publishers: 1" in topic_info and "subscribers: 2" in topic_info

    service_info = ros_mod._info("/spawn")
    assert service_info is not None
    assert "service info /spawn" in service_info

    assert ros_mod._info("/absent") is None


# Pub / sub / service primitives --------------------------------------------


def test_echo_collects_and_caps_samples(
    monkeypatch: pytest.MonkeyPatch, fake_node: _FakeNode, fake_rosidl: dict[str, Any]
) -> None:
    pending = [{"x": 0}, {"x": 1}, {"x": 2}]

    def _deliver(predicate: Any, timeout: float) -> None:
        while pending and not predicate():
            fake_node.sub_cb(pending.pop(0))

    monkeypatch.setattr(ros_mod._backend, "spin_for", _deliver)

    samples = ros_mod._echo("/turtle1/pose", "turtlesim/msg/Pose", timeout=1.0, count=2)
    assert samples == [{"x": 0}, {"x": 1}]  # capped at count
    # The subscription is torn down after collection.
    assert fake_node.destroyed and fake_node.destroyed[0][0] == "sub"


def test_publish_sends_count_messages_with_fields(fake_node: _FakeNode, fake_rosidl: dict[str, Any]) -> None:
    ros_mod._publish(
        "/turtle1/cmd_vel",
        "geometry_msgs/msg/Twist",
        {"linear": {"x": 2.0}, "enabled": True},
        count=3,
        rate=50.0,
    )
    pub = fake_node.publishers[0]
    assert len(pub.published) == 3
    # The payload reached set_message_fields as a real dict, types intact.
    assert fake_rosidl["set_fields"] == [{"linear": {"x": 2.0}, "enabled": True}]
    assert ("pub", pub) in fake_node.destroyed


def test_service_call_helper_returns_response(fake_node: _FakeNode, fake_rosidl: dict[str, Any]) -> None:
    future = _types.SimpleNamespace(done=lambda: True, result=lambda: {"name": "t2"})
    client = _types.SimpleNamespace(
        wait_for_service=lambda timeout_sec: True,
        call_async=lambda req: future,
    )
    fake_node.clients = [client]

    resp = ros_mod._service_call("/spawn", "turtlesim/srv/Spawn", {"name": "t2"}, timeout=1.0)
    assert resp == {"name": "t2"}
    assert ("client", client) in fake_node.destroyed


def test_service_call_raises_when_service_unavailable(fake_node: _FakeNode, fake_rosidl: dict[str, Any]) -> None:
    client = _types.SimpleNamespace(
        wait_for_service=lambda timeout_sec: False,
        call_async=lambda req: None,
    )
    fake_node.clients = [client]

    with pytest.raises(TimeoutError, match="not available"):
        ros_mod._service_call("/spawn", "turtlesim/srv/Spawn", {}, timeout=0.1)
    # The client is still cleaned up on the failure path.
    assert ("client", client) in fake_node.destroyed


def test_service_call_raises_when_response_never_arrives(fake_node: _FakeNode, fake_rosidl: dict[str, Any]) -> None:
    future = _types.SimpleNamespace(done=lambda: False, result=lambda: None)
    client = _types.SimpleNamespace(
        wait_for_service=lambda timeout_sec: True,
        call_async=lambda req: future,
    )
    fake_node.clients = [client]

    with pytest.raises(TimeoutError, match="timed out"):
        ros_mod._service_call("/spawn", "turtlesim/srv/Spawn", {}, timeout=0.1)
    assert ("client", client) in fake_node.destroyed
