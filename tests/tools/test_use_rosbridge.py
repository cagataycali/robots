"""Behavior tests for the ``use_rosbridge`` agent tool.

rosbridge is a WebSocket JSON transport (roslibpy) - pure pip, no sourced ROS
environment. These tests run roslibpy-free: a fake ``roslibpy`` module is
injected via ``sys.modules`` (fake Ros/Topic/Service record all traffic), so
connection caching, every action dispatch, the validation layer, and the
structured error contract are exercised with nothing installed. The single
exception is the premise test for the double's port gate, which runs against
real roslibpy when it happens to be installed and skips otherwise.
"""

from __future__ import annotations

import enum
import sys
import types as _types
from typing import Any

import pytest

import strands_robots.tools.use_rosbridge as rb_mod

use_rosbridge = rb_mod.use_rosbridge


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


class _FakeTopic:
    def __init__(self, ros: Any, name: str, message_type: str) -> None:
        self.ros, self.name, self.message_type = ros, name, message_type
        self.advertised = False
        self.unadvertised = False
        self.unsubscribed = False
        self.published: list[dict[str, Any]] = []
        ros.topics.append(self)

    def advertise(self) -> None:
        self.advertised = True

    def unadvertise(self) -> None:
        self.unadvertised = True

    def publish(self, msg: dict[str, Any]) -> None:
        self.published.append(dict(msg))

    def subscribe(self, cb: Any) -> None:
        for m in list(type(self.ros).scripted_messages.get(self.name, [])):
            cb(m)

    def unsubscribe(self) -> None:
        self.unsubscribed = True


class _FakeService:
    def __init__(self, ros: Any, name: str, service_type: str) -> None:
        self.ros, self.name, self.service_type = ros, name, service_type

    def call(self, request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.ros.service_calls.append((self.name, self.service_type, dict(request), timeout))
        responses = type(self.ros).scripted_responses
        if self.name in responses:
            return responses[self.name]
        raise RuntimeError(f"no scripted response for {self.name}")


class _FakeRos:
    instances: list[_FakeRos] = []
    fail_next_connect = False
    scripted_responses: dict[str, dict[str, Any]] = {}
    scripted_messages: dict[str, list[dict[str, Any]]] = {}

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        # The real client gates the port in its WebSocket URL builder with this
        # expression (spelled with ``==`` there; ``is`` is the same test for
        # int and equivalent under ruff's E721). Reproduced so this double is
        # not more permissive than what it stands in for: without it, a port
        # the transport cannot address looks perfectly usable in every test
        # that runs through the fake, which is why nothing here noticed that
        # 65535 and any int subclass escaped as a bare AssertionError.
        assert port is None or (type(port) is int and port in range(0, 65535))
        self.host, self.port = host, port
        self.is_connected = False
        self.terminated = False
        self.topics: list[_FakeTopic] = []
        self.service_calls: list[tuple[str, str, dict[str, Any], float | None]] = []
        _FakeRos.instances.append(self)

    def run(self, timeout: float | None = None) -> None:
        if _FakeRos.fail_next_connect:
            raise RuntimeError("connection refused")
        self.is_connected = True

    def terminate(self) -> None:
        self.terminated = True
        self.is_connected = False


@pytest.fixture
def fake_roslibpy(monkeypatch: pytest.MonkeyPatch) -> _types.ModuleType:
    _FakeRos.instances = []
    _FakeRos.fail_next_connect = False
    _FakeRos.scripted_responses = {}
    _FakeRos.scripted_messages = {}
    mod = _types.ModuleType("roslibpy")
    mod.Ros = _FakeRos  # type: ignore[attr-defined]
    mod.Topic = _FakeTopic  # type: ignore[attr-defined]
    mod.Service = _FakeService  # type: ignore[attr-defined]
    mod.Message = dict  # type: ignore[attr-defined]
    mod.ServiceRequest = dict  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "roslibpy", mod)
    monkeypatch.setattr(rb_mod._backend, "_connections", {})
    monkeypatch.setattr(rb_mod._backend, "_available", None)
    return mod


# Validation ------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["/cmd vel", "/x|y", "../etc", "/a$(x)"])
def test_invalid_topic_rejected(bad: str) -> None:
    result = use_rosbridge(action="echo", topic=bad)
    assert result["status"] == "error"
    assert "invalid topic" in _texts(result)


def test_ros1_two_segment_type_enforced() -> None:
    # ROS1 types are pkg/Name; a ROS2-style pkg/msg/Name must be rejected so
    # agents get a correcting error instead of a silent rosbridge failure.
    result = use_rosbridge(action="publish", topic="/cmd_vel", type="geometry_msgs/msg/Twist")
    assert result["status"] == "error"
    assert "invalid interface type" in _texts(result)


def test_valid_ros1_type_accepted_shapewise(fake_roslibpy: _types.ModuleType) -> None:
    result = use_rosbridge(action="publish", topic="/cmd_vel", type="geometry_msgs/Twist")
    assert result["status"] == "success"


@pytest.mark.parametrize("bad_host", ["bad host", "h;st", ""])
def test_invalid_host_rejected(bad_host: str) -> None:
    result = use_rosbridge(action="status", host=bad_host)
    assert result["status"] == "error"
    assert "invalid host" in _texts(result)


@pytest.mark.parametrize("bad_port", [0, -1, 70000])
def test_invalid_port_rejected(bad_port: int) -> None:
    result = use_rosbridge(action="status", port=bad_port)
    assert result["status"] == "error"
    assert "invalid port" in _texts(result)


# status / availability ---------------------------------------------------------


def test_status_reports_missing_roslibpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "roslibpy", None)  # import raises deterministically
    monkeypatch.setattr(rb_mod._backend, "_available", None)
    monkeypatch.setattr(rb_mod._backend, "_connections", {})
    result = use_rosbridge(action="status")
    assert result["status"] == "success"
    assert "backend: none" in _texts(result)
    assert "strands-robots[rosbridge]" in _texts(result)


def test_status_connects_and_reports(fake_roslibpy: _types.ModuleType) -> None:
    result = use_rosbridge(action="status", host="sim.local", port=9091)
    assert result["status"] == "success"
    assert "connected to ws://sim.local:9091" in _texts(result)


def test_status_reports_unreachable_bridge(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.fail_next_connect = True  # type: ignore[attr-defined]
    result = use_rosbridge(action="status")
    assert result["status"] == "success"
    assert "not connected" in _texts(result)
    assert "rosbridge_server" in _texts(result)


def test_actions_error_without_roslibpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "roslibpy", None)
    monkeypatch.setattr(rb_mod._backend, "_available", None)
    monkeypatch.setattr(rb_mod._backend, "_connections", {})
    result = use_rosbridge(action="list_topics")
    assert result["status"] == "error"
    assert "strands-robots[rosbridge]" in _texts(result)


# Connection cache --------------------------------------------------------------


def test_connection_cached_per_host_port(fake_roslibpy: _types.ModuleType) -> None:
    use_rosbridge(action="status")
    use_rosbridge(action="status")
    use_rosbridge(action="status", port=9091)
    hosts = [(r.host, r.port) for r in fake_roslibpy.Ros.instances]  # type: ignore[attr-defined]
    assert hosts == [("localhost", 9090), ("localhost", 9091)]  # second call reused the first


def test_stale_connection_kept_and_reused_after_recovery(fake_roslibpy: _types.ModuleType) -> None:
    use_rosbridge(action="status")
    first = fake_roslibpy.Ros.instances[0]  # type: ignore[attr-defined]
    first.is_connected = False  # dropped WebSocket, bridge still down
    result = use_rosbridge(action="status", timeout=0.2)
    assert result["status"] == "success"
    assert "did not reconnect" in _texts(result)
    # The entry is never terminated NOR discarded - its factory keeps
    # retrying, and a fresh Ros after churn is unreliable in-process.
    assert not first.terminated
    # Bridge comes back: the SAME object reconnects and is reused.
    first.is_connected = True
    result = use_rosbridge(action="status")
    assert "connected to" in _texts(result)
    assert len(fake_roslibpy.Ros.instances) == 1  # type: ignore[attr-defined]


def test_stale_connection_reused_when_factory_reconnects(fake_roslibpy: _types.ModuleType) -> None:
    use_rosbridge(action="status")
    first = fake_roslibpy.Ros.instances[0]  # type: ignore[attr-defined]

    class _Flapping:
        # is_connected reads False twice (initial check + first wait poll),
        # then True - simulating the auto-reconnecting factory recovering.
        def __init__(self) -> None:
            self.reads = 0

        def __get__(self, obj: Any, objtype: Any = None) -> bool:
            self.reads += 1
            return self.reads > 2

    type(first).is_connected = _Flapping()  # type: ignore[assignment]
    try:
        result = use_rosbridge(action="status", timeout=1.0)
        assert "connected to" in _texts(result)
        assert len(fake_roslibpy.Ros.instances) == 1  # type: ignore[attr-defined]  # same object reused
    finally:
        del type(first).is_connected  # restore instance-attribute behavior


def test_failed_dial_cached_and_recovers(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.fail_next_connect = True  # type: ignore[attr-defined]
    result = use_rosbridge(action="status", timeout=0.2)
    assert "not connected" in _texts(result)
    fake_roslibpy.Ros.fail_next_connect = False  # type: ignore[attr-defined]
    # The never-connected Ros stays cached; when its factory succeeds the
    # same object is reused - no second dial is ever attempted.
    orphan = fake_roslibpy.Ros.instances[0]  # type: ignore[attr-defined]
    orphan.is_connected = True
    result = use_rosbridge(action="status")
    assert "connected to" in _texts(result)
    assert len(fake_roslibpy.Ros.instances) == 1  # type: ignore[attr-defined]


def test_unknown_action_errors(fake_roslibpy: _types.ModuleType) -> None:
    result = use_rosbridge(action="warp_drive")
    assert result["status"] == "error"
    assert "unknown action" in _texts(result)


def test_unknown_action_rejected_before_any_connection(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.fail_next_connect = True  # type: ignore[attr-defined]
    result = use_rosbridge(action="warp_drive")
    assert result["status"] == "error"
    assert "unknown action" in _texts(result)
    assert fake_roslibpy.Ros.instances == []  # type: ignore[attr-defined]  # never dialed


# rosapi-backed introspection ---------------------------------------------------


def test_list_topics_formats_sorted_pairs(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.scripted_responses["/rosapi/topics"] = {  # type: ignore[attr-defined]
        "topics": ["/zeta", "/curiosity_mars_rover/odom"],
        "types": ["std_msgs/String", "nav_msgs/Odometry"],
    }
    result = use_rosbridge(action="list_topics")
    assert result["status"] == "success"
    lines = _texts(result).splitlines()
    assert lines[0] == "/curiosity_mars_rover/odom [nav_msgs/Odometry]"  # sorted
    assert "/zeta [std_msgs/String]" in lines[1]


def test_list_services_sorted(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.scripted_responses["/rosapi/services"] = {  # type: ignore[attr-defined]
        "services": ["/b_srv", "/a_srv"],
    }
    result = use_rosbridge(action="list_services")
    assert _texts(result).splitlines() == ["/a_srv", "/b_srv"]


def test_rosapi_absence_is_actionable(fake_roslibpy: _types.ModuleType) -> None:
    # No scripted response -> the fake raises like a timed-out service; the
    # tool must convert that into a structured, named error.
    result = use_rosbridge(action="list_topics")
    assert result["status"] == "error"
    assert "/rosapi/topics" in _texts(result)


# echo / service_call ------------------------------------------------------------


def test_echo_autoresolves_type_and_caps_count(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.scripted_responses["/rosapi/topic_type"] = {"type": "nav_msgs/Odometry"}  # type: ignore[attr-defined]
    fake_roslibpy.Ros.scripted_messages["/curiosity_mars_rover/odom"] = [  # type: ignore[attr-defined]
        {"pose": {"pose": {"position": {"x": 1.0}}}},
        {"pose": {"pose": {"position": {"x": 2.0}}}},
        {"pose": {"pose": {"position": {"x": 3.0}}}},
    ]
    result = use_rosbridge(action="echo", topic="/curiosity_mars_rover/odom", count=2)
    assert result["status"] == "success"
    assert "nav_msgs/Odometry" in _texts(result)
    assert '"x": 1.0' in _texts(result) and '"x": 2.0' in _texts(result)
    assert '"x": 3.0' not in _texts(result)  # capped at count
    ros = fake_roslibpy.Ros.instances[0]  # type: ignore[attr-defined]
    assert ros.topics[-1].unsubscribed  # subscription torn down in finally


def test_echo_unresolvable_type_errors(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.scripted_responses["/rosapi/topic_type"] = {"type": ""}  # type: ignore[attr-defined]
    result = use_rosbridge(action="echo", topic="/ghost")
    assert result["status"] == "error"
    assert "cannot resolve type" in _texts(result)


def test_echo_requires_topic(fake_roslibpy: _types.ModuleType) -> None:
    assert use_rosbridge(action="echo")["status"] == "error"


def test_echo_empty_result_discloses_timeout(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.scripted_responses["/rosapi/topic_type"] = {"type": "nav_msgs/Odometry"}  # type: ignore[attr-defined]
    result = use_rosbridge(action="echo", topic="/silent", timeout=0.1)
    assert result["status"] == "success"
    assert "no messages within" in _texts(result)


def test_publish_traffic_and_unadvertise(fake_roslibpy: _types.ModuleType) -> None:
    result = use_rosbridge(
        action="publish",
        topic="/cmd_vel",
        type="geometry_msgs/Twist",
        fields={"linear": {"x": 1.5}, "angular": {"z": 0.0}},
        count=3,
    )
    assert result["status"] == "success"
    ros = fake_roslibpy.Ros.instances[0]  # type: ignore[attr-defined]
    pub = ros.topics[-1]
    assert pub.advertised and pub.unadvertised
    assert pub.published == [{"linear": {"x": 1.5}, "angular": {"z": 0.0}}] * 3


def test_service_call_returns_response(fake_roslibpy: _types.ModuleType) -> None:
    fake_roslibpy.Ros.scripted_responses["/gazebo/reset_world"] = {"ok": True}  # type: ignore[attr-defined]
    result = use_rosbridge(action="service_call", service="/gazebo/reset_world", type="std_srvs/Empty")
    assert result["status"] == "success"
    assert '"ok": true' in _texts(result)


def test_service_call_requires_service_and_type(fake_roslibpy: _types.ModuleType) -> None:
    result = use_rosbridge(action="service_call", service="/gazebo/reset_world")
    assert result["status"] == "error"
    assert "requires service and type" in _texts(result)


def test_publish_rejects_nonpositive_count(fake_roslibpy: _types.ModuleType) -> None:
    result = use_rosbridge(action="publish", topic="/cmd_vel", type="geometry_msgs/Twist", count=0)
    assert result["status"] == "error"
    assert "count" in _texts(result)


# Transport port domain -------------------------------------------------------


class _PortEnum(enum.IntEnum):
    """An ``int`` subclass port, as a settings module would export one."""

    ROSBRIDGE = 9090


# Every action, with the arguments it needs to reach the connect step.
_ACTION_ARGS: dict[str, dict[str, Any]] = {
    "status": {},
    "list_topics": {},
    "list_services": {},
    "echo": {"topic": "/odom", "type": "nav_msgs/Odometry"},
    "publish": {"topic": "/cmd_vel", "type": "geometry_msgs/Twist"},
    "service_call": {"service": "/reset", "type": "std_srvs/Empty"},
}


def test_every_action_is_covered_by_the_port_domain_cases() -> None:
    # So a seventh action cannot be added without deciding how it reports a
    # port the transport cannot address.
    assert set(_ACTION_ARGS) == set(rb_mod._ACTIONS)


@pytest.mark.skipif(not __debug__, reason="the transport gates its port with an assert, which -O strips")
def test_the_doubles_port_gate_is_the_real_transports() -> None:
    """Pin that the gate reproduced in ``_FakeRos`` is the client's own.

    Without this the behavior tests below would be asserting against invented
    limits: the point of the two refusals is that the shipped transport really
    cannot carry those ports.
    """
    roslibpy = pytest.importorskip("roslibpy", reason="the rosbridge client is an optional dependency")

    # Constructing a client does not dial - that is ros.run() - so this is a
    # pure check of the URL the transport is willing to build.
    roslibpy.Ros(host="127.0.0.1", port=65534)
    roslibpy.Ros(host="127.0.0.1", port=int(_PortEnum.ROSBRIDGE))
    with pytest.raises(AssertionError):
        roslibpy.Ros(host="127.0.0.1", port=65535)
    with pytest.raises(AssertionError):
        roslibpy.Ros(host="127.0.0.1", port=_PortEnum.ROSBRIDGE)


@pytest.mark.parametrize("action", sorted(_ACTION_ARGS))
def test_unaddressable_port_reported_through_the_envelope(action: str, fake_roslibpy: _types.ModuleType) -> None:
    # 65535 is inside this tool's accepted domain but outside the exclusive
    # range the transport's URL builder allows, and it used to leave every one
    # of these actions as a bare AssertionError.
    result = use_rosbridge(action=action, host="127.0.0.1", port=65535, timeout=0.05, **_ACTION_ARGS[action])
    assert result["status"] == "error"
    text = _texts(result)
    assert "65535" in text
    assert "cannot be dialed" in text
    assert "1-65534" in text


def test_refusal_names_the_transport_not_a_narrowed_domain(fake_roslibpy: _types.ModuleType) -> None:
    # The accepted port domain is deliberately still the OS one, shared with
    # RosbridgeRobot, the inference server CLI and the mesh session checks. So
    # the refusal must come from the transport, not from narrowing this tool's
    # own range to 1-65534 and disagreeing with all of them.
    text = _texts(use_rosbridge(action="status", host="127.0.0.1", port=65535, timeout=0.05))
    assert "invalid port" not in text
    assert "transport" in text


def test_unaddressable_port_caches_no_connection(fake_roslibpy: _types.ModuleType) -> None:
    # The cache is populated after the client is built, so a refused port must
    # leave nothing behind for a later call to find.
    assert use_rosbridge(action="status", host="127.0.0.1", port=65535, timeout=0.05)["status"] == "error"
    assert rb_mod._backend._connections == {}
    assert _FakeRos.instances == []

    assert use_rosbridge(action="status", host="127.0.0.1", port=9090, timeout=0.05)["status"] == "success"


def test_int_subclass_port_reaches_the_wire_as_a_plain_int(fake_roslibpy: _types.ModuleType) -> None:
    # The transport's gate is an identity check, so an IntEnum failed it at any
    # value - including 9090, the default rosbridge port. It is a legal port,
    # so it is normalized and dialed rather than refused.
    result = use_rosbridge(action="status", host="127.0.0.1", port=_PortEnum.ROSBRIDGE, timeout=0.05)
    assert result["status"] == "success"
    assert "connected to ws://127.0.0.1:9090" in _texts(result)

    (ros,) = _FakeRos.instances
    assert type(ros.port) is int  # equality with 9090 held before the fix too
    assert ros.port == 9090


def test_highest_addressable_port_still_connects(fake_roslibpy: _types.ModuleType) -> None:
    result = use_rosbridge(action="status", host="127.0.0.1", port=65534, timeout=0.05)
    assert result["status"] == "success"
    assert "connected to ws://127.0.0.1:65534" in _texts(result)


def test_mesh_bridge_inherits_the_transport_verdict(fake_roslibpy: _types.ModuleType) -> None:
    """RosbridgeRobot dials through this tool, so it inherits the contract.

    Asserted here rather than beside the bridge because the fix is this tool's
    and this is where the transport double lives: the mesh module's own tests
    replace ``use_rosbridge`` with a recorder, so nothing there ever reaches a
    client at all.
    """
    from strands_robots.mesh.rosbridge_robot import RosbridgeRobot

    refused = RosbridgeRobot("rover", "/cmd_vel", "/odom", host="127.0.0.1", port=65535)
    result = refused.drive(linear=0.1, count=1)
    assert result["status"] == "error"
    assert "cannot be dialed" in _texts(result)

    dialed = RosbridgeRobot("rover", "/cmd_vel", "/odom", host="127.0.0.1", port=_PortEnum.ROSBRIDGE)
    assert dialed.drive(linear=0.1, count=1)["status"] == "success"
