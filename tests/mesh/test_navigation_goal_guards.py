"""A navigation goal the bridge cannot honor is refused before it is sent.

:meth:`RosBridgedRobot.navigate_to` hands a goal pose to the robot's own
navigation stack, so the coordinates are the whole command. They travel inside
the action request body, which ``use_ros`` forwards verbatim - it validates the
action name and interface type, and it guards ``timeout``, but it never inspects
the pose. Without a guard on the bridge a pose that cannot be honored has two
silent outcomes: a non-finite coordinate serializes as a valid IEEE-754 float64
and the goal is accepted by the transport and handed to a planner that cannot
resolve it, or the planar-quaternion encoding raises a bare
``ValueError``/``TypeError`` out of a method whose contract is a
``{"status": ...}`` result dict - and ``navigate_to`` is exposed to an agent as a
``navigate_*`` tool, where raising escapes the dispatch contract entirely.

The pose components are signed physical quantities, so they share the accepted
domain of :meth:`RosBridgedRobot.drive`'s velocity components; the parity test
here pins that one rule rather than two.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import strands_robots.mesh.ros_bridge as ros_mod
from strands_robots.mesh import RosBridgedRobot

_NAV_ACTION = "/navigate_to_pose"

# A coordinate and a heading are signed, so only non-finite and non-numeric
# values are refusable. ``inf`` yaw is the one that raises today (``math.sin``
# rejects an infinite angle); ``nan`` yaw is the one that silently ships an
# unnormalizable quaternion.
_BAD_POSE_VALUES = [math.nan, math.inf, -math.inf, "1.0", None, [1.0], True, False]


class _Wire:
    """Stands in for ``use_ros``, recording every forwarded request."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"status": "success", "content": [{"text": "goal reached"}]}


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> tuple[RosBridgedRobot, _Wire]:
    """A nav-capable bridge plus the wire recorder it forwards goals to."""
    wire = _Wire()
    monkeypatch.setattr(ros_mod, "use_ros", wire)
    return RosBridgedRobot("tb", "/cmd_vel", "/odom", nav_action=_NAV_ACTION), wire


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


def _goal_pose(call: dict[str, Any]) -> dict[str, Any]:
    pose: dict[str, Any] = call["fields"]["pose"]["pose"]
    return pose


class TestRefusedGoalNeverReachesTheWire:
    """A refusable pose component yields an error result and sends nothing."""

    @pytest.mark.parametrize("value", _BAD_POSE_VALUES)
    @pytest.mark.parametrize("param", ["x", "y", "yaw"])
    def test_a_pose_component_the_planner_cannot_resolve_is_refused(
        self, bridge: tuple[RosBridgedRobot, _Wire], param: str, value: Any
    ) -> None:
        robot, wire = bridge
        goal: dict[str, Any] = {"x": 1.0, "y": 2.0}
        goal[param] = value
        result = robot.navigate_to(**goal)
        assert result["status"] == "error"
        assert param in _text(result)
        assert wire.calls == []

    def test_the_first_refusable_component_is_named(self, bridge: tuple[RosBridgedRobot, _Wire]) -> None:
        """Several bad components at once name one parameter, not a merged message."""
        robot, wire = bridge
        result = robot.navigate_to(x=math.nan, y=math.inf, yaw=math.nan)
        assert result["status"] == "error"
        assert "x" in _text(result)
        assert "y must" not in _text(result) and "yaw" not in _text(result)
        assert wire.calls == []

    def test_a_missing_nav_action_is_still_reported_before_the_pose(self) -> None:
        """The capability check precedes the parameter guard, as ``drive``'s does."""
        robot = RosBridgedRobot("tb", "/cmd_vel", "/odom")
        result = robot.navigate_to(x=math.nan, y=math.nan)
        assert result["status"] == "error"
        assert "no nav_action configured" in _text(result)


class TestHonoredGoalsAreStillSent:
    """The guard does not narrow the set of goals that already worked."""

    def test_a_planar_goal_is_encoded_as_a_unit_quaternion(self, bridge: tuple[RosBridgedRobot, _Wire]) -> None:
        robot, wire = bridge
        assert robot.navigate_to(x=1.5, y=-2.5, yaw=math.pi / 2)["status"] == "success"
        pose = _goal_pose(wire.calls[0])
        assert pose["position"] == {"x": 1.5, "y": -2.5}
        assert pose["orientation"]["z"] == pytest.approx(math.sin(math.pi / 4))
        assert pose["orientation"]["w"] == pytest.approx(math.cos(math.pi / 4))

    @pytest.mark.parametrize("x,y,yaw", [(0.0, 0.0, 0.0), (-12.0, 8.5, -math.pi), (1e-9, 1e9, 2.5)])
    def test_a_signed_finite_goal_is_forwarded_verbatim(
        self, bridge: tuple[RosBridgedRobot, _Wire], x: float, y: float, yaw: float
    ) -> None:
        robot, wire = bridge
        assert robot.navigate_to(x=x, y=y, yaw=yaw)["status"] == "success"
        assert _goal_pose(wire.calls[0])["position"] == {"x": x, "y": y}

    def test_an_integer_goal_is_usable(self, bridge: tuple[RosBridgedRobot, _Wire]) -> None:
        """A coordinate is continuous, so a whole-number goal is a valid one."""
        robot, wire = bridge
        assert robot.navigate_to(x=3, y=-4)["status"] == "success"
        assert _goal_pose(wire.calls[0])["position"] == {"x": 3.0, "y": -4.0}

    def test_a_numpy_scalar_goal_is_usable(self, bridge: tuple[RosBridgedRobot, _Wire]) -> None:
        """A goal read out of a pose array arrives as a NumPy scalar."""
        import numpy as np

        robot, wire = bridge
        # Splatted so the NumPy scalars reach the runtime guard as an agent
        # would supply them, rather than being narrowed to ``float`` statically.
        goal: dict[str, Any] = {"x": np.float32(0.5), "y": np.float64(-0.25)}
        assert robot.navigate_to(**goal)["status"] == "success"
        assert _goal_pose(wire.calls[0])["position"]["x"] == pytest.approx(0.5)

    def test_the_frame_and_timeout_are_still_forwarded(self, bridge: tuple[RosBridgedRobot, _Wire]) -> None:
        robot, wire = bridge
        assert robot.navigate_to(x=1.0, y=1.0, frame_id="odom", timeout=30.0)["status"] == "success"
        assert wire.calls[0]["fields"]["pose"]["header"]["frame_id"] == "odom"
        assert wire.calls[0]["timeout"] == 30.0


class TestAgentToolContract:
    """The bound ``navigate_*`` agent tool returns a result; it never raises."""

    @pytest.mark.parametrize("kwargs", [{"yaw": math.inf}, {"x": None}, {"y": [1.0]}, {"x": math.nan}])
    def test_the_bound_navigate_tool_reports_instead_of_raising(
        self, bridge: tuple[RosBridgedRobot, _Wire], kwargs: dict[str, Any]
    ) -> None:
        robot, wire = bridge
        goal: dict[str, Any] = {"x": 1.0, "y": 2.0, **kwargs}
        navigate_tool: Any = next(t for t in robot.tools if t.tool_name.startswith("navigate_"))
        result = navigate_tool(**goal)
        assert result["status"] == "error"
        assert wire.calls == []


class TestPoseAndVelocityShareOneDomain:
    """A goal coordinate and a velocity component are both signed scalars."""

    @pytest.mark.parametrize("value", [*_BAD_POSE_VALUES, 0.0, -1.0, 2.5, 1e300])
    def test_navigate_to_and_drive_return_the_same_verdict(
        self, bridge: tuple[RosBridgedRobot, _Wire], value: Any
    ) -> None:
        robot, _wire = bridge
        nav = robot.navigate_to(x=value, y=0.0)["status"]
        drive = robot.drive(linear=value)["status"]
        assert nav == drive, f"verdicts differ for {value!r}: navigate_to={nav}, drive={drive}"
