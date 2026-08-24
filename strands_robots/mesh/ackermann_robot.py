"""Ackermann ROS 2 mesh bridge - steering-geometry cars as strands robots.

An :class:`AckermannRosRobot` wraps an Ackermann-steering ROS 2 car (reference
platform: AWS DeepRacer) so an agent can drive it with the same
``Agent(tools=robot.tools)`` pattern as every other strands robot. Ackermann
platforms differ from the differential-drive bases served by
:class:`~strands_robots.mesh.ros_bridge.RosBridgedRobot` in three ways this
class absorbs:

1. Commands are normalized servo pairs (``angle``/``throttle`` in [-1, 1]),
   not ``geometry_msgs/msg/Twist``. :meth:`drive` keeps the fleet-wide
   ``(linear, angular)`` contract and converts internally with a bicycle
   model, so the agent never learns a second vocabulary.
2. The vehicle must be switched into a commandable state first (the DeepRacer
   needs a two-step manual-mode service handshake). ``init_services`` declares
   that handshake; :meth:`drive` runs it once, automatically, before the first
   command and refuses to drive if it fails.
3. There is no odometry topic on the stock platform, so there is deliberately
   no ``get_pose`` here.

All ROS 2 I/O forwards through :func:`strands_robots.tools.use_ros.use_ros`,
so the bridge owns no rclpy state and is safe to construct without ROS 2.

Commanding the car goes through ``use_ros``'s operator-approval gate, because a
servo topic that moves a vehicle and the mode services that arm it are
safety-critical surfaces. The bridge forwards an operator context to it: the
``drive_<node>`` / ``stop_<node>`` agent tools are declared ``@tool(context=True)``
and hand the context to :func:`use_ros`, so an agent driving this car prompts the
operator exactly as a direct ``use_ros`` call does. The read path
(:meth:`get_scan`) is never gated.

A **programmatic** call carries no operator context, so it is refused unless the
surfaces are pre-approved with ``STRANDS_ROS2_COMMAND_ALLOW`` (or the gate is
bypassed with ``BYPASS_TOOL_CONSENT=true``). That includes :meth:`stop` and the
trailing halt of a timed :meth:`drive`: the gate is keyed on the *surface*, not
on the payload, so a zero servo message is gated like a full-throttle one.

Typical usage::

    import os

    from strands import Agent
    from strands_robots.mesh import AckermannRosRobot

    car = AckermannRosRobot.from_deepracer(node_name="deepracer")

    # Direct, programmatic control - nobody to prompt, so the command surfaces
    # this car uses have to be pre-approved (bare names cover the namespaced
    # DeepRacer spellings):
    os.environ["STRANDS_ROS2_COMMAND_ALLOW"] = "/manual_drive,/vehicle_state,/enable_state"
    car.drive(linear=0.5, duration=2.0)   # forward half a metre per second
    print(car.get_scan())                 # reads are never gated

    # Or hand the bridge to an agent as first-class tools - the command tools
    # carry the operator context, so the gate prompts instead of refusing:
    agent = Agent(tools=car.tools)
    agent("drive forward slowly for two seconds, then read the lidar")
"""

from __future__ import annotations

import math
from typing import Any

from strands import tool
from strands.types.tools import AgentTool, ToolContext

from strands_robots.mesh.ros_bridge import _check_topic
from strands_robots.tools.use_ros import use_ros
from strands_robots.utils import (
    finite_number_error,
    partial_construction_repr,
    positive_finite_number_error,
    positive_whole_number_error,
)

_SERVO_TYPE = "deepracer_interfaces_pkg/msg/ServoCtrlMsg"

# Below this absolute linear velocity a command is treated as rest and maps to
# zero output: a stock Ackermann platform cannot yaw at rest, and the bicycle
# model's atan2 would otherwise amplify noise near v = 0 into hard steering.
_REST_VELOCITY_EPS = 1e-3


def _twist_to_servo(
    linear: float,
    angular: float,
    *,
    wheelbase_m: float,
    max_speed: float,
    max_steering_rad: float,
) -> tuple[float, float]:
    """Convert a body-frame velocity command to a normalized servo pair.

    Bicycle model: the equivalent front-wheel steering angle for yaw rate
    ``angular`` at speed ``linear`` is ``atan(wheelbase_m * angular / linear)``,
    clamped to the platform's steering limit and normalized by it. Throttle is
    ``linear`` normalized by ``max_speed``. Both outputs are in [-1, 1].

    Implemented as ``atan2`` on ``abs(linear)`` with the sign restored for
    reverse motion: the naive ``atan2(wheelbase_m * angular, linear)`` lands in
    the second quadrant when ``linear`` is negative, commanding near-full
    opposite steering while backing up instead of the mirrored angle.
    """
    v = float(linear)
    if abs(v) < _REST_VELOCITY_EPS:
        return (0.0, 0.0)
    delta = math.atan2(wheelbase_m * float(angular), abs(v))
    if v < 0:
        delta = -delta
    delta = max(-max_steering_rad, min(max_steering_rad, delta))
    throttle = max(-1.0, min(1.0, v / max_speed))
    return (delta / max_steering_rad, throttle)


class AckermannRosRobot:
    """An Ackermann-steering ROS 2 car exposed as a strands-controllable robot.

    The bridge owns no ROS 2 state; every method forwards to :func:`use_ros`.
    Constructing it never needs a ROS 2 environment - errors surface as
    structured results when a method actually runs.

    Args:
        node_name: Identifier used to name this robot's agent tools
            (``drive_<node_name>`` etc.); it does not need to match a ROS 2
            node name.
        servo_topic: Topic the vehicle's servo stack subscribes to (DeepRacer:
            ``/webserver_pkg/manual_drive``).
        scan_topic: Optional laser-scan topic. Read by :meth:`get_scan`.
        servo_type: Interface type of ``servo_topic``. Defaults to the
            DeepRacer ``ServoCtrlMsg`` (normalized ``angle``/``throttle``).
        scan_type: Interface type of ``scan_topic``. Optional - resolved from
            the live graph when omitted.
        wheelbase_m: Front-to-rear axle distance for the bicycle model.
        max_speed: Linear speed (m/s) mapped to full throttle; commands are
            clamped to this magnitude.
        max_steering_rad: Steering angle mapped to full servo deflection.
        max_duration: Longest single :meth:`drive` hold accepted; longer
            requests are rejected loudly rather than silently truncated.
        publish_rate: Command publish rate (Hz) for held :meth:`drive` calls.
        init_services: Ordered service calls (``{"service", "type", "fields"}``
            dicts) that put the vehicle into a commandable state. Run once,
            automatically, before the first :meth:`drive`. The DeepRacer
            manual-mode handshake in :meth:`from_deepracer` is the reference
            use.

    There is deliberately no ``get_pose``: the stock platform publishes no
    odometry.
    """

    def __init__(
        self,
        node_name: str,
        servo_topic: str,
        scan_topic: str | None = None,
        *,
        servo_type: str = _SERVO_TYPE,
        scan_type: str | None = None,
        wheelbase_m: float = 0.164,
        max_speed: float = 1.5,
        max_steering_rad: float = 0.5236,
        max_duration: float = 10.0,
        publish_rate: float = 20.0,
        init_services: list[dict[str, Any]] | None = None,
    ) -> None:
        self.node_name = _check_topic("node_name", node_name)
        self.servo_topic = _check_topic("servo_topic", servo_topic)
        self.scan_topic = _check_topic("scan_topic", scan_topic) if scan_topic else None
        self.servo_type = servo_type
        self.scan_type = scan_type
        for label, value in (
            ("wheelbase_m", wheelbase_m),
            ("max_speed", max_speed),
            ("max_steering_rad", max_steering_rad),
            ("max_duration", max_duration),
            ("publish_rate", publish_rate),
        ):
            if limit_err := positive_finite_number_error(value, label, type(self).__name__):
                raise ValueError(limit_err)
        self.wheelbase_m = float(wheelbase_m)
        self.max_speed = float(max_speed)
        self.max_steering_rad = float(max_steering_rad)
        self.max_duration = float(max_duration)
        self.publish_rate = float(publish_rate)
        self.init_services = list(init_services or [])
        for item in self.init_services:
            _check_topic("init_services service", item.get("service", ""))
            if not item.get("type"):
                raise ValueError(
                    f"init_services entry for {item.get('service')!r} is missing its 'type' "
                    "(expected an interface type like pkg/srv/Name)"
                )
        self._enabled = False

    @classmethod
    def from_deepracer(cls, node_name: str, **overrides: Any) -> AckermannRosRobot:
        """Construct a bridge wired for the stock AWS DeepRacer software stack.

        Servo commands go to the webserver package's manual-drive topic, the
        RPLIDAR scan topic is preconfigured, and ``init_services`` carries the
        two-step manual-mode handshake (``vehicle_state`` state=1, then
        ``enable_state`` is_active=True) the car requires before it acts on
        servo messages. Any keyword can be overridden for a modified car.
        """
        wiring: dict[str, Any] = {
            "servo_topic": "/webserver_pkg/manual_drive",
            "scan_topic": "/rplidar_ros/scan",
            "init_services": [
                {
                    "service": "/ctrl_pkg/vehicle_state",
                    "type": "deepracer_interfaces_pkg/srv/ActiveStateSrv",
                    "fields": {"state": 1},
                },
                {
                    "service": "/ctrl_pkg/enable_state",
                    "type": "deepracer_interfaces_pkg/srv/EnableStateSrv",
                    "fields": {"is_active": True},
                },
            ],
        }
        wiring.update(overrides)
        servo_topic = wiring.pop("servo_topic")
        scan_topic = wiring.pop("scan_topic")
        return cls(node_name, servo_topic, scan_topic, **wiring)

    @staticmethod
    def _error(text: str) -> dict[str, Any]:
        return {"status": "error", "content": [{"text": text}]}

    def enable(self, tool_context: ToolContext | None = None) -> dict[str, Any]:
        """Run the ``init_services`` handshake once; idempotent on success.

        Stops at the first failing call and returns its structured error
        without latching, so a later attempt retries from the start.

        ``tool_context`` is the operator context forwarded to ``use_ros``: the
        services that arm a vehicle are gated command surfaces, so a handshake
        that forwards nothing is refused rather than prompted (see :meth:`drive`).
        """
        if self._enabled:
            return {
                "status": "success",
                "content": [{"text": f"{self.node_name}: already enabled"}],
            }
        for item in self.init_services:
            result = use_ros(
                action="service_call",
                service=item["service"],
                type=item["type"],
                fields=item.get("fields", {}),
                tool_context=tool_context,
            )
            if result.get("status") != "success":
                return result
        self._enabled = True
        return {
            "status": "success",
            "content": [{"text": f"{self.node_name}: enabled ({len(self.init_services)} init call(s))"}],
        }

    def drive(
        self,
        linear: float = 0.0,
        angular: float = 0.0,
        duration: float | None = None,
        count: int = 1,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Publish a velocity command, converted through the bicycle model.

        Same contract as ``RosBridgedRobot.drive``: ``linear`` (m/s) and
        ``angular`` (rad/s), an optional ``duration`` hold (publishes
        ``round(duration * publish_rate)`` messages, takes precedence over
        ``count``). Validation runs before any side effect, in order: inputs
        must be finite, ``duration`` (when given) must be a positive finite
        number within ``max_duration``, and only then does the vehicle's
        ``init_services`` handshake run (automatically, before the first
        command; a failed handshake aborts the drive). Every timed or
        multi-message non-zero command is followed by a single zero servo
        message - even if the main publish failed - so it can never leave the
        car with a live throttle latched, and when that halt is the call that
        fails the result says so instead of reporting the drive's success. A
        bare single-shot command (no ``duration``, ``count=1``) latches like a
        raw servo command until :meth:`stop`.

        ``tool_context`` is the operator context forwarded to ``use_ros``, whose
        command gate prompts for approval on a safety-critical surface such as
        this vehicle's servo topic and mode services. The ``drive_<node_name>``
        agent tool passes the one the framework injects; a programmatic call has
        none, and the gate then refuses unless the surface is pre-approved via
        ``STRANDS_ROS2_COMMAND_ALLOW`` / ``BYPASS_TOOL_CONSENT``.
        """
        # A servo command is the one call on this bridge that physically moves
        # the car, so every knob it carries is checked before anything reaches
        # the wire, through the same shared domains the sibling bridges use.
        cmd_err = (
            finite_number_error(linear, "linear", "drive")
            or finite_number_error(angular, "angular", "drive")
            or (
                positive_finite_number_error(duration, "duration", "drive")
                if duration is not None
                # ``duration`` supersedes ``count``, so ``count`` is only the
                # effective horizon when no duration was given.
                else positive_whole_number_error(count, "count", "drive")
            )
        )
        if cmd_err:
            return self._error(cmd_err)
        # Ordered after the finiteness guard on purpose: every comparison
        # against ``nan`` is false, so a ceiling test cannot stand in for one.
        if duration is not None and duration > self.max_duration:
            return self._error(
                f"drive: duration {duration}s exceeds max_duration {self.max_duration}s "
                "- issue shorter commands instead of one long hold"
            )
        if not self._enabled and self.init_services:
            enabled = self.enable(tool_context=tool_context)
            if enabled.get("status") != "success":
                return enabled
        v = max(-self.max_speed, min(self.max_speed, float(linear)))
        angle, throttle = _twist_to_servo(
            v,
            angular,
            wheelbase_m=self.wheelbase_m,
            max_speed=self.max_speed,
            max_steering_rad=self.max_steering_rad,
        )
        n = max(1, round(duration * self.publish_rate)) if duration is not None else count
        # The trailing halt runs from ``finally`` so it goes out even if the
        # main publish raised, but its verdict is kept rather than dropped:
        # ``use_ros`` reports a transport failure as an error dict rather than
        # raising, so returning the main publish's success after a failed halt
        # would report a car still holding the commanded throttle as a drive
        # that stopped itself - and an agent reading ``success`` never issues
        # ``stop``.
        halt: dict[str, Any] | None = None
        try:
            result = self._publish_servo(angle, throttle, count=n, tool_context=tool_context)
        finally:
            if (duration is not None or n > 1) and (angle or throttle):
                halt = self._publish_servo(0.0, 0.0, count=1, tool_context=tool_context)
        if halt is not None and halt.get("status") != "success" and result.get("status") == "success":
            cause = " ".join(
                block.get("text", "") for block in halt.get("content", []) if isinstance(block, dict)
            ).strip()
            return self._error(
                f"drive: the command was published to {self.servo_topic}, but the trailing stop "
                f"failed - the car may still be holding the commanded throttle. Halt it with stop. "
                f"Halt failure: {cause or 'no detail reported'}"
            )
        return result

    def _publish_servo(
        self,
        angle: float,
        throttle: float,
        count: int,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        return use_ros(
            action="publish",
            topic=self.servo_topic,
            type=self.servo_type,
            fields={"angle": float(angle), "throttle": float(throttle)},
            count=count,
            rate=self.publish_rate,
            tool_context=tool_context,
        )

    def stop(self, tool_context: ToolContext | None = None) -> dict[str, Any]:
        """Publish a single zero servo command; it does not require :meth:`enable`.

        The halt reaches the servo topic through the same publish as
        :meth:`drive`, so it passes the same operator gate - which is why
        ``tool_context`` is forwarded here too (see :meth:`drive`).
        """
        return self._publish_servo(0.0, 0.0, count=1, tool_context=tool_context)

    def get_scan(self, timeout: float = 5.0) -> dict[str, Any]:
        """Read one sample from the laser-scan topic (error when unconfigured)."""
        if not self.scan_topic:
            return self._error("get_scan: no scan_topic configured for this robot")
        return use_ros(
            action="echo",
            topic=self.scan_topic,
            type=self.scan_type,
            count=1,
            timeout=timeout,
        )

    @property
    def tools(self) -> list[AgentTool]:
        """This robot's capabilities as named strands agent tools.

        Tools are bound to this instance and suffixed with ``node_name`` so
        multiple robots coexist in one ``Agent(tools=[...])`` call. The drive
        tool's description states the Ackermann kinematic limit (minimum
        turning radius) so the agent can plan paths the platform can follow.

        Every tool that carries a command (``drive``, ``stop``) is declared
        ``@tool(context=True)`` and forwards the injected context into
        ``use_ros``, so its operator-approval gate prompts rather than failing
        closed. ``get_scan`` takes no context because ``use_ros`` never gates
        ``echo``.
        """
        suffix = self.node_name.strip("/").replace("/", "_")
        min_radius = self.wheelbase_m / math.tan(self.max_steering_rad)

        @tool(
            name=f"drive_{suffix}",
            description=(
                f"Drive the {self.node_name} Ackermann robot (linear m/s, angular rad/s, "
                f"optional duration s). Steering geometry: cannot turn in place; "
                f"minimum turning radius ~{min_radius:.2f} m. A command with duration "
                f"stops automatically afterwards; without duration the last command "
                f"latches until stop."
            ),
            context=True,
        )
        def drive(
            linear: float = 0.0,
            angular: float = 0.0,
            duration: float | None = None,
            tool_context: ToolContext | None = None,
        ) -> dict[str, Any]:
            return self.drive(linear=linear, angular=angular, duration=duration, tool_context=tool_context)

        @tool(
            name=f"stop_{suffix}",
            description=f"Immediately stop the {self.node_name} robot.",
            context=True,
        )
        def stop(tool_context: ToolContext | None = None) -> dict[str, Any]:
            return self.stop(tool_context=tool_context)

        @tool(name=f"get_scan_{suffix}", description=f"Read one laser scan from the {self.node_name} robot.")
        def get_scan() -> dict[str, Any]:
            return self.get_scan()

        agent_tools: list[AgentTool] = [drive, stop]
        if self.scan_topic:
            agent_tools.append(get_scan)
        return agent_tools

    def __repr__(self) -> str:
        try:
            return (
                f"AckermannRosRobot(node_name={self.node_name!r}, "
                f"servo_topic={self.servo_topic!r}, scan_topic={self.scan_topic!r})"
            )
        except AttributeError:
            return partial_construction_repr(self)
