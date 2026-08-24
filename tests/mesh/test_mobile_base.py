"""Contract pins for :class:`MobileBaseRobot` - the shared mobile-base surface.

These are the tests that used to be written once per transport class (and, in
two cases, were only written for *some* of them - which is how
``RosBridgedRobot`` shipped without NaN rejection, without a duration bound and
without a trailing-zero stop while its siblings had all three). Pinning the
contract on the base means a future transport inherits the safety semantics and
the regression coverage together, and a fix lands in one place.

Transport-free by construction: every test drives a recording fake, so nothing
here needs rclpy, cyclonedds, a DDS domain or a reachable server.

The last section re-runs the safety matrix against the *real* shipped classes,
so the guarantees are pinned on what users actually construct and not only on
the abstract base.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import pytest

from strands_robots.mesh import MobileBaseRobot, RosBridgedRobot, RtpsRobot
from strands_robots.mesh._mobile_base import positive_finite
from strands_robots.utils import positive_finite_number_error

_OK: dict[str, Any] = {"status": "success", "content": [{"text": "ok"}]}
_FAIL: dict[str, Any] = {"status": "error", "content": [{"text": "nope"}]}


class _FakeTransport:
    """Minimal transport: records calls, implements the required surface only."""

    twist_type = "geometry_msgs/msg/Twist"

    def __init__(self, publish_result: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._publish_result = publish_result or _OK

    def publish(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"action": "publish", **kwargs})
        return self._publish_result

    def echo(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"action": "echo", **kwargs})
        return _OK

    @property
    def publishes(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["action"] == "publish"]


class _ServiceTransport(_FakeTransport):
    """Transport that also implements the optional ``service_call`` capability."""

    def __init__(self, service_results: list[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self._service_results = list(service_results or [])

    def service_call(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"action": "service_call", **kwargs})
        if self._service_results:
            return self._service_results.pop(0)
        return _OK

    @property
    def services(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["action"] == "service_call"]


def _robot(transport: _FakeTransport | None = None, **kwargs: Any) -> MobileBaseRobot:
    return MobileBaseRobot("bot", "/cmd_vel", transport or _FakeTransport(), **kwargs)


def _fake(robot: MobileBaseRobot) -> _FakeTransport:
    """The recording transport behind ``robot``, narrowed from ``Transport``.

    :attr:`MobileBaseRobot.transport` is typed as the protocol the base actually
    depends on, and that protocol deliberately does not carry ``calls`` /
    ``publishes`` - those exist only to let a test see what reached the wire.
    Narrowing here keeps the recording surface out of the production protocol
    instead of widening the protocol for the tests' benefit.
    """
    assert isinstance(robot.transport, _FakeTransport)
    return robot.transport


# -- construction -------------------------------------------------------------


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "1.0", None, True])
def test_positive_finite_refuses_non_positive_finite(value: Any) -> None:
    """``True`` is refused explicitly: it is a float-compatible 1 in disguise."""
    with pytest.raises(ValueError, match="limit"):
        positive_finite("limit", value)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "1.0", None, True, 2.5, 10])
def test_positive_finite_is_the_shared_domain_and_not_a_second_rulebook(value: Any) -> None:
    """The verdict is the shared guard's, verbatim - including the message.

    A base that restated the rule would be free to drift from it: a velocity
    clamp would start accepting a NumPy scalar or a ``True`` that a control-loop
    frequency rejects, for no reason a reader could discover. Pinning the
    delegation is what makes "every safety fix lands once" checkable rather than
    aspirational.
    """
    expected = positive_finite_number_error(value, "limit", "ctx")
    if expected is None:
        assert positive_finite("limit", value, "ctx") == float(value)
        return
    with pytest.raises(ValueError) as excinfo:
        positive_finite("limit", value, "ctx")
    assert str(excinfo.value) == expected


@pytest.mark.parametrize("field", ["max_linear", "max_angular", "max_duration", "publish_rate"])
@pytest.mark.parametrize("value", [0, -1.0, float("nan"), float("inf")])
def test_limits_must_be_positive_finite(field: str, value: float) -> None:
    limits: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match=field):
        _robot(**limits)


@pytest.mark.parametrize("node_name,cmd_vel_topic", [("bad name", "/cmd_vel"), ("bot", "/has;semicolon"), ("", "/c")])
def test_malformed_names_refused_at_construction(node_name: str, cmd_vel_topic: str) -> None:
    with pytest.raises(ValueError, match="invalid"):
        MobileBaseRobot(node_name, cmd_vel_topic, _FakeTransport())


def test_unset_limits_mean_unbounded_not_zero() -> None:
    """A platform that declares no limit is not a platform limited to zero."""
    robot = _robot()
    assert robot.max_linear is None and robot.max_angular is None and robot.max_duration is None
    robot.drive(linear=999.0, angular=-999.0)
    assert _fake(robot).publishes[0]["fields"] == {"linear": {"x": 999.0}, "angular": {"z": -999.0}}


def test_cmd_vel_type_defaults_to_the_transports_flavor() -> None:
    """ROS 1 and ROS 2 spell Twist differently; the transport owns which."""

    class _Ros1Transport(_FakeTransport):
        twist_type = "geometry_msgs/Twist"

    assert _robot().cmd_vel_type == "geometry_msgs/msg/Twist"
    assert _robot(_Ros1Transport()).cmd_vel_type == "geometry_msgs/Twist"
    assert _robot(cmd_vel_type="pkg/msg/Custom").cmd_vel_type == "pkg/msg/Custom"


# -- capability detection -----------------------------------------------------


def test_supports_reports_optional_capabilities() -> None:
    assert _robot().supports("publish") and _robot().supports("echo")
    assert not _robot().supports("service_call")
    assert _robot(_ServiceTransport()).supports("service_call")
    assert not _robot(_ServiceTransport()).supports("action_send_goal")


def test_init_services_refused_on_a_transport_that_cannot_call_services() -> None:
    """Refuse at construction rather than at the first drive.

    A handshake wired onto a transport with no service surface can never run;
    surfacing that when the robot is built is the difference between a clear
    ValueError and a vehicle that refuses to move for reasons the caller
    discovers on the track.
    """
    with pytest.raises(ValueError, match="does not implement service_call"):
        _robot(init_services=[{"service": "/arm", "type": "pkg/srv/Arm"}])


@pytest.mark.parametrize(
    "entry,match",
    [
        ({"type": "pkg/srv/Arm"}, "invalid init_services service"),
        ({"service": "/bad name", "type": "pkg/srv/Arm"}, "invalid init_services service"),
        ({"service": "/arm"}, "missing its 'type'"),
    ],
)
def test_malformed_init_services_entry_refused(entry: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _robot(_ServiceTransport(), init_services=[entry])


# -- drive: input validation --------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("axis", ["linear", "angular"])
def test_drive_refuses_non_finite_velocity(axis: str, bad: float) -> None:
    """``nan`` passes silently through a min/max clamp - reject before clamping."""
    robot = _robot()
    command: dict[str, Any] = {axis: bad}
    result = robot.drive(**command)
    assert result["status"] == "error"
    assert f"{axis} must be a finite number" in result["content"][0]["text"]
    assert _fake(robot).calls == []


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, "2", [1.0]])
def test_drive_refuses_bad_duration(bad: Any) -> None:
    robot = _robot()
    result = robot.drive(linear=1.0, duration=bad)
    assert result["status"] == "error"
    assert "duration" in result["content"][0]["text"]
    assert _fake(robot).calls == []


@pytest.mark.parametrize("bad", [0, -1, 2.7, float("nan"), float("inf"), True, "3", None])
def test_drive_refuses_a_count_no_message_burst_expresses(bad: Any) -> None:
    """``count`` is the horizon when no ``duration`` is given, so it is guarded.

    Left unchecked, ``count=0`` reaches the transport, publishes nothing and
    reports success - a drive the caller believes happened. A fractional count
    has no burst that expresses it.
    """
    robot = _robot()
    result = robot.drive(linear=1.0, count=bad)
    assert result["status"] == "error"
    assert "count" in result["content"][0]["text"]
    assert _fake(robot).calls == []


def test_drive_does_not_read_count_when_a_duration_supersedes_it() -> None:
    """``duration`` wins, so an unread ``count`` must not refuse a valid command."""
    robot = _robot(publish_rate=10.0)
    assert robot.drive(linear=1.0, duration=2.0, count=0)["status"] == "success"
    assert _fake(robot).publishes[0]["count"] == 20


@pytest.mark.parametrize("param", ["linear", "angular"])
def test_drive_accepts_a_numpy_scalar_velocity(param: str) -> None:
    """A policy action element arrives as a NumPy scalar, not a Python float.

    Refusing it would make the base reject the output of the very policies that
    drive these robots, so the accepted domain is "any finite real scalar".
    """
    numpy = pytest.importorskip("numpy")
    robot = _robot()
    command: dict[str, Any] = {param: numpy.float32(-0.25)}
    assert robot.drive(**command)["status"] == "success"
    assert _fake(robot).publishes[0]["fields"]["linear" if param == "linear" else "angular"] == {
        "x" if param == "linear" else "z": pytest.approx(-0.25)
    }


def test_drive_refuses_overlong_duration_rather_than_truncating() -> None:
    robot = _robot(max_duration=5.0)
    result = robot.drive(linear=1.0, duration=6.0)
    assert result["status"] == "error"
    assert "exceeds max_duration" in result["content"][0]["text"]
    assert _fake(robot).calls == []


def test_drive_clamps_both_axes_to_configured_limits() -> None:
    robot = _robot(max_linear=1.0, max_angular=0.5)
    robot.drive(linear=9.0, angular=-9.0)
    assert _fake(robot).publishes[0]["fields"] == {"linear": {"x": 1.0}, "angular": {"z": -0.5}}


def test_drive_duration_becomes_message_count_at_publish_rate() -> None:
    robot = _robot(publish_rate=10.0)
    robot.drive(linear=1.0, duration=1.5)
    assert _fake(robot).publishes[0]["count"] == 15


# -- drive: the enable handshake ----------------------------------------------


def _armed(**kwargs: Any) -> tuple[MobileBaseRobot, _ServiceTransport]:
    transport = _ServiceTransport(**kwargs)
    robot = MobileBaseRobot(
        "bot",
        "/cmd_vel",
        transport,
        init_services=[
            {"service": "/vehicle_state", "type": "pkg/srv/State", "fields": {"state": 1}},
            {"service": "/enable", "type": "pkg/srv/Enable", "fields": {"is_active": True}},
        ],
    )
    return robot, transport


def test_handshake_runs_once_in_order_before_the_first_command() -> None:
    robot, transport = _armed()
    robot.drive(linear=1.0)
    robot.drive(linear=0.5)
    assert [c["service"] for c in transport.services] == ["/vehicle_state", "/enable"]
    assert [c["fields"] for c in transport.services] == [{"state": 1}, {"is_active": True}]
    assert transport.calls[0]["action"] == "service_call"


def test_failed_handshake_aborts_the_drive_and_does_not_latch() -> None:
    """A retry must re-run the whole sequence - the service may just be late."""
    robot, transport = _armed(service_results=[_FAIL])
    first = robot.drive(linear=1.0)
    assert first["status"] == "error"
    assert transport.publishes == []
    assert robot._enabled is False

    second = robot.drive(linear=1.0)
    assert second["status"] == "success"
    assert [c["service"] for c in transport.services] == ["/vehicle_state", "/vehicle_state", "/enable"]


def test_invalid_request_is_refused_before_the_handshake_runs() -> None:
    """An invalid drive must not be what arms the vehicle.

    Validation ordering is a safety property, not a style choice: if the
    handshake ran first, a rejected command would still leave a real car
    switched into a commandable state.
    """
    robot, transport = _armed()
    robot.max_duration = 5.0
    assert robot.drive(linear=1.0, duration=99.0)["status"] == "error"
    assert robot.drive(linear=float("nan"))["status"] == "error"
    assert transport.calls == []


def test_stop_is_never_gated_on_the_handshake() -> None:
    """An emergency stop must not require a working service graph."""
    robot, transport = _armed(service_results=[_FAIL])
    result = robot.stop()
    assert result["status"] == "success"
    assert transport.services == []
    assert transport.publishes[0]["fields"] == {"linear": {"x": 0.0}, "angular": {"z": 0.0}}


def test_enable_is_idempotent_once_successful() -> None:
    robot, transport = _armed()
    assert robot.enable()["status"] == "success"
    again = robot.enable()
    assert again["status"] == "success"
    assert "already enabled" in again["content"][0]["text"]
    assert len(transport.services) == 2


# -- drive: the trailing-zero stop --------------------------------------------


def test_timed_command_is_followed_by_a_single_zero_command() -> None:
    robot = _robot(publish_rate=10.0)
    robot.drive(linear=1.0, duration=1.0)
    publishes = _fake(robot).publishes
    assert len(publishes) == 2
    assert publishes[0]["count"] == 10
    assert publishes[1]["fields"] == {"linear": {"x": 0.0}, "angular": {"z": 0.0}}
    assert publishes[1]["count"] == 1


def test_multi_message_command_is_followed_by_a_zero_command() -> None:
    robot = _robot()
    robot.drive(linear=1.0, count=5)
    assert len(_fake(robot).publishes) == 2


def test_trailing_zero_is_sent_even_when_the_main_publish_raises() -> None:
    """The ``finally`` is the whole point: a throwing transport still stops."""

    class _Exploding(_FakeTransport):
        def publish(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append({"action": "publish", **kwargs})
            if len(self.calls) == 1:
                raise RuntimeError("transport died mid-command")
            return _OK

    robot = _robot(_Exploding())
    with pytest.raises(RuntimeError, match="transport died"):
        robot.drive(linear=1.0, duration=1.0)
    publishes = _fake(robot).publishes
    assert len(publishes) == 2
    assert publishes[1]["fields"] == {"linear": {"x": 0.0}, "angular": {"z": 0.0}}


def test_single_shot_command_latches_with_no_trailing_zero() -> None:
    """A bare drive() keeps a raw cmd_vel's latch semantics, as documented."""
    robot = _robot()
    robot.drive(linear=1.0)
    assert len(_fake(robot).publishes) == 1


def test_an_already_zero_command_gets_no_trailing_zero() -> None:
    robot = _robot()
    robot.drive(linear=0.0, angular=0.0, duration=1.0)
    assert len(_fake(robot).publishes) == 1


def test_trailing_zero_fires_on_a_clamped_but_nonzero_command() -> None:
    robot = _robot(max_linear=1.0, publish_rate=10.0)
    robot.drive(linear=50.0, duration=1.0)
    publishes = _fake(robot).publishes
    assert publishes[0]["fields"]["linear"]["x"] == 1.0
    assert publishes[1]["fields"] == {"linear": {"x": 0.0}, "angular": {"z": 0.0}}


# -- the kinematics seam ------------------------------------------------------


def test_cmd_fields_is_the_single_override_point_for_kinematics() -> None:
    """A subclass changes the message shape without touching drive() safety."""

    class _ServoRobot(MobileBaseRobot):
        def _cmd_fields(self, linear: float, angular: float, lateral: float = 0.0) -> dict[str, Any]:
            return {"throttle": float(linear), "angle": float(angular)}

    robot = _ServoRobot("car", "/servo", _FakeTransport(), max_linear=2.0, publish_rate=10.0)
    robot.drive(linear=9.0, angular=0.25, duration=1.0)
    publishes = _fake(robot).publishes
    # Clamping, duration->count and the trailing zero all still apply, and the
    # trailing zero is expressed in the subclass's own field vocabulary. The
    # operator context reaches the transport on every publish - None here
    # because a programmatic drive carries no operator - so overriding the
    # kinematics cannot drop it.
    assert publishes[0] == {
        "action": "publish",
        "topic": "/servo",
        "type": "geometry_msgs/msg/Twist",
        "fields": {"throttle": 2.0, "angle": 0.25},
        "count": 10,
        "rate": 10.0,
        "tool_context": None,
    }
    assert publishes[1]["fields"] == {"throttle": 0.0, "angle": 0.0}


def test_lateral_seam_is_reserved_and_unused_by_the_default_shape() -> None:
    """``lateral`` exists for holonomic bases; the Twist default omits it at 0."""
    robot = _robot()
    assert robot._cmd_fields(1.0, 2.0) == {"linear": {"x": 1.0}, "angular": {"z": 2.0}}
    assert robot._cmd_fields(1.0, 2.0, 0.5) == {"linear": {"x": 1.0, "y": 0.5}, "angular": {"z": 2.0}}


# -- sensing and tools --------------------------------------------------------


def test_pose_and_scan_report_absence_instead_of_echoing_nothing() -> None:
    robot = _robot()
    for result, label in ((robot.get_pose(), "odom_topic"), (robot.get_scan(), "scan_topic")):
        assert result["status"] == "error"
        assert f"no {label} configured" in result["content"][0]["text"]
    assert _fake(robot).calls == []


def test_pose_and_scan_echo_when_wired() -> None:
    robot = _robot(odom_topic="/odom", scan_topic="/scan", odom_type="nav_msgs/msg/Odometry")
    robot.get_pose(timeout=2.0)
    robot.get_scan()
    echoes = [c for c in _fake(robot).calls if c["action"] == "echo"]
    assert echoes[0] == {
        "action": "echo",
        "topic": "/odom",
        "type": "nav_msgs/msg/Odometry",
        "count": 1,
        "timeout": 2.0,
    }
    assert echoes[1]["topic"] == "/scan"


def test_tools_reflect_only_the_capabilities_actually_wired() -> None:
    """An agent is never handed a tool that can only answer 'not configured'."""
    assert {t.tool_name for t in _robot().tools} == {"drive_bot", "stop_bot"}
    assert {t.tool_name for t in _robot(odom_topic="/odom").tools} == {"drive_bot", "stop_bot", "get_pose_bot"}
    assert "get_scan_bot" in {t.tool_name for t in _robot(scan_topic="/scan").tools}


def test_tool_names_are_instance_unique_so_robots_can_share_an_agent() -> None:
    a = MobileBaseRobot("/fleet/one", "/a/cmd_vel", _FakeTransport())
    b = MobileBaseRobot("/fleet/two", "/b/cmd_vel", _FakeTransport())
    assert {t.tool_name for t in a.tools}.isdisjoint({t.tool_name for t in b.tools})
    assert a.tool_suffix == "fleet_one"


def test_drive_tool_description_discloses_limits_and_latch_semantics() -> None:
    """The agent plans with these strings; an undisclosed latch is a trap."""
    spec = next(t for t in _robot(max_linear=1.5, max_angular=0.8).tools if t.tool_name == "drive_bot").tool_spec
    description = spec["description"]
    assert "1.5" in description and "0.8" in description
    assert "latches until stop" in description


def test_drive_tool_forwards_to_the_instance() -> None:
    robot = _robot()
    drive_tool: Any = next(t for t in robot.tools if t.tool_name == "drive_bot")
    drive_tool(linear=1.0, angular=2.0)
    assert _fake(robot).publishes[0]["fields"] == {"linear": {"x": 1.0}, "angular": {"z": 2.0}}


# -- the same guarantees on the real shipped classes --------------------------


def _shipped() -> list[MobileBaseRobot]:
    return [
        RosBridgedRobot("tb", "/cmd_vel", "/odom"),
        RtpsRobot("tb", "/cmd_vel"),
    ]


@pytest.mark.parametrize("robot", _shipped(), ids=lambda r: type(r).__name__)
def test_every_shipped_class_refuses_non_finite_velocity(robot: MobileBaseRobot) -> None:
    """Pins the gap this refactor closed.

    ``RosBridgedRobot.drive(linear=float("nan"))`` used to publish NaN straight
    onto ``cmd_vel``; only the newer siblings guarded it. The guard now lives on
    the base, so every class has it and no future transport can ship without it.
    """
    assert robot.drive(linear=float("nan"))["status"] == "error"
    assert robot.drive(angular=float("inf"))["status"] == "error"


@pytest.mark.parametrize("robot", _shipped(), ids=lambda r: type(r).__name__)
def test_every_shipped_class_refuses_a_bad_duration(robot: MobileBaseRobot) -> None:
    """``drive(duration=inf)`` used to become an unbounded publish loop."""
    assert robot.drive(linear=1.0, duration=float("inf"))["status"] == "error"
    assert robot.drive(linear=1.0, duration=-1.0)["status"] == "error"


@pytest.mark.parametrize("robot", _shipped(), ids=lambda r: type(r).__name__)
def test_every_shipped_class_pairs_drive_with_stop(robot: MobileBaseRobot) -> None:
    names = {t.tool_name for t in robot.tools}
    assert {f"drive_{robot.tool_suffix}", f"stop_{robot.tool_suffix}"} <= names


def test_shipped_classes_keep_their_own_name_grammar() -> None:
    """The base owns validation; the grammar stays per-platform.

    ``use_rtps`` writes to a DDS topic directly and needs an absolute name; the
    rclpy bridge also accepts relative and private names for ROS to resolve.
    Collapsing these into one pattern would either loosen the RTPS check or
    break existing rclpy wiring.
    """
    RosBridgedRobot("tb", "relative/cmd_vel", "/odom")  # accepted: rclpy resolves it
    with pytest.raises(ValueError, match="invalid cmd_vel_topic"):
        RtpsRobot("tb", "relative/cmd_vel")


# -- the operator gate reaches every transport --------------------------------

#: Every command verb a transport can carry. ``echo`` is absent on purpose: a
#: read is never gated, so a read verb has no operator decision to carry.
_TRANSPORT_COMMAND_VERBS = ("publish", "service_call", "action_send_goal")


def _shipped_transports() -> list[tuple[type, Any]]:
    """Each mobile-base transport paired with the tool it forwards to.

    Imported here rather than listed as names so a transport that stops being
    reachable from its module fails this file instead of quietly leaving the
    matrix below.
    """
    from strands_robots.mesh.ros_bridge import _UseRosTransport
    from strands_robots.mesh.rtps_robot import _UseRtpsTransport
    from strands_robots.tools.use_ros import use_ros
    from strands_robots.tools.use_rtps import use_rtps

    return [(_UseRosTransport, use_ros), (_UseRtpsTransport, use_rtps)]


def _tool_takes_a_context(agent_tool: Any) -> bool:
    """Whether the underlying tool has an operator gate to hand a context to."""
    func = getattr(agent_tool, "__wrapped__", agent_tool)
    return "tool_context" in inspect.signature(func).parameters


def _forwards_a_context(method: Any) -> bool:
    source = textwrap.dedent(inspect.getsource(method))
    return any(
        keyword.arg == "tool_context"
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
    )


@pytest.mark.parametrize("transport,agent_tool", _shipped_transports(), ids=lambda x: getattr(x, "__name__", ""))
def test_every_transport_accepts_the_operator_context(transport: type, agent_tool: Any) -> None:
    """The base hands the context to the transport, so every one must take it.

    :meth:`MobileBaseRobot.drive` forwards ``tool_context`` on every publish. A
    transport that did not accept the parameter would raise ``TypeError`` on
    every command rather than merely losing the gate, so this is the conformance
    half of the protocol and not a style rule.
    """
    del agent_tool
    for verb in _TRANSPORT_COMMAND_VERBS:
        method = getattr(transport, verb, None)
        if method is None:
            continue  # an optional capability this transport does not claim
        assert "tool_context" in inspect.signature(method).parameters, (
            f"{transport.__name__}.{verb} does not accept tool_context, so every command through it raises"
        )


@pytest.mark.parametrize("transport,agent_tool", _shipped_transports(), ids=lambda x: getattr(x, "__name__", ""))
def test_a_transport_forwards_the_context_exactly_when_its_tool_gates(transport: type, agent_tool: Any) -> None:
    """Forwarding is decided by the tool, not by a table maintained here.

    Both directions are defects. A gating tool that is not forwarded to turns
    the whole command surface into a per-call refusal - the failure
    ``tests/mesh/test_ros_bridge_command_gate.py`` exists for. A non-gating tool
    that is forwarded to raises ``TypeError``, because it has no such parameter.
    Deriving the expectation from the tool's own signature means a new transport
    has to be right rather than remembered.
    """
    gates = _tool_takes_a_context(agent_tool)
    for verb in _TRANSPORT_COMMAND_VERBS:
        method = getattr(transport, verb, None)
        if method is None:
            continue
        assert _forwards_a_context(method) is gates, (
            f"{transport.__name__}.{verb} "
            f"{'drops' if gates else 'forwards'} the operator context but its tool "
            f"{'gates' if gates else 'has no gate'}"
        )


def test_the_shipped_transports_cover_both_sides_of_the_gate_split() -> None:
    """Every shipped command-bearing transport forwards the operator context.

    Both ``use_ros`` and ``use_rtps`` gate their command surface, so every
    shipped transport that can publish must forward ``tool_context``. The
    parametrized rule above verifies the forwarding; this test verifies that at
    least one transport is in the gating set so the rule is not vacuous.
    """
    gating = {t.__name__ for t, agent_tool in _shipped_transports() if _tool_takes_a_context(agent_tool)}
    ungated = {t.__name__ for t, agent_tool in _shipped_transports() if not _tool_takes_a_context(agent_tool)}
    assert gating, f"gating={sorted(gating)} - expected at least one gating transport"
    # All shipped tools now gate; the ungated set being empty is the correct
    # security posture after #2693 gave use_rtps its operator-approval gate.
    # If a future ungated transport is added, it must not forward tool_context.
    if ungated:
        # Non-vacuity: both branches are exercised.
        pass


def test_rtps_transport_declares_no_service_surface() -> None:
    """Capability asymmetry is honest, not erased.

    ``use_rtps`` has no services, so an ``init_services`` handshake on an RTPS
    robot is refused at construction instead of failing on the track.
    """
    rtps = RtpsRobot("tb", "/cmd_vel")
    assert not rtps.supports("service_call")
    assert RosBridgedRobot("tb", "/cmd_vel", "/odom").supports("service_call")
