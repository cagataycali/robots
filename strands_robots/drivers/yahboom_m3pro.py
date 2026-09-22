"""Yahboom ROSMASTER M3 Pro native driver - the robot's ROS 2 interface, driven.

The M3 Pro is a mecanum chassis carrying the DOFBOT-Pro arm. Its motors live
behind an STM32 expansion board running **micro-ROS** firmware; the Jetson (or
Raspberry Pi) on the chassis runs ROS 2 Humble and a ``micro_ros_agent`` that
puts the board's topics on the graph. That graph *is* the vendor's control
interface - the desktop app, the web console and every Yahboom demo node speak
it - so this driver speaks it too, rather than the serial protocol underneath:

* ``/cmd_vel`` (``geometry_msgs/Twist``) - the base. ``linear.x`` forward,
  ``linear.y`` **strafe** (it is a mecanum base), ``angular.z`` yaw, SI units.
  The firmware zeroes the motors after roughly 200-500 ms without a fresh
  message, so a held move is a stream at :data:`PUBLISH_RATE_HZ` followed by an
  explicit zero, and a single frame is a ~0.3 s twitch by design.
* ``/arm6_joints`` (``arm_msgs/ArmJoints``) - all six servos at once:
  ``joint1..joint6`` in **integer degrees** (``int16``) plus ``time`` in
  milliseconds. Joints 1-4 travel 0-180, joint 5 0-270, joint 6 is the gripper
  and travels 30 (closed) to 180 (open). The board holds a commanded pose, so
  one message is enough - unlike ``/cmd_vel``.
* ``/arm_joint`` (``arm_msgs/ArmJoint``) - one servo: ``id`` 1-6, ``joint``
  degrees, ``time`` ms.
* ``/odom_raw`` (``nav_msgs/Odometry``), ``/imu/data_raw``
  (``sensor_msgs/Imu``) and ``/scan0`` / ``/scan1`` (``sensor_msgs/LaserScan``,
  two lidars whose sweeps overlap into near-360 degree cover) - the reads.

Three transports answer that graph, chosen by ``transport=``:

* ``"rosbridge"`` (default) - ``rosbridge_server`` on the robot, dialled over a
  WebSocket from *any* host with ``pip install 'strands-robots[rosbridge]'``:
  a laptop, CI, a fleet controller. ``port=`` is ``host[:port]`` of the bridge
  (default :data:`DEFAULT_ROSBRIDGE`). Interface types are the two-segment
  spellings rosbridge transmits.
* ``"ros2"`` - in-process ``rclpy`` for a driver running **on** the robot's
  own computer, inside its ROS environment (``ROS_DOMAIN_ID`` is 30 on the
  shipped image); ``port=`` is then ignored. The same three-segment types the
  ``use_ros`` tool speaks.
* ``"twin"`` - the ``yahboom_m3pro`` MuJoCo model, answering the same topics
  (:class:`~strands_robots.drivers.yahboom_m3pro_twin.M3ProTwinGraph`). The
  same tool, verbs and units an agent uses on the robot, over the simulation:
  what an agent learns to say to the twin it says to the hardware. ``sim=``
  hands in a built engine; otherwise one is built at the ``home`` keyframe.

The first two forward through the transports this package already owns
(:func:`strands_robots.rosbridge.rosbridge_action` /
:func:`strands_robots.ros.ros_action`), so the driver holds no socket and no
node, and every write to ``/cmd_vel`` passes the shared operator gate
(:mod:`strands_robots._command_gate`): ``/cmd_vel`` is on the blocklist, so a
base command is approved by the agent's operator, pre-approved with
``STRANDS_ROS2_COMMAND_ALLOW=/cmd_vel``, or refused - and a Python caller with
no agent sets that variable, exactly as it would for
:class:`~strands_robots.mesh.RosbridgeRobot`. The arm topics are not on the
blocklist, so arm commands pass without a prompt, like every other arm here.

**Sim parity.** :meth:`YahboomM3ProDriver.send_action` speaks the *model's*
vocabulary - ``arm1.pos .. arm5.pos`` and ``gripper.pos`` in radians, the joint
names and ranges of the ``yahboom_m3pro`` MJCF - and converts at the wire. The
description's ``home`` keyframe is written as ``q = (deg - 90) * pi / 180``,
so the inverse ``deg = 90 + degrees(q)`` maps the URDF ranges onto the servo
ranges exactly: ``+-pi/2`` onto 0-180 for joints 1-4 and ``-pi/2..pi`` onto
0-270 for joint 5 (:func:`arm_rad_to_deg`). The gripper crank ``rlink1``
travels ``-1.54`` (closed) to ``0`` (open) and maps linearly onto 30-180
(:func:`gripper_rad_to_deg`). A policy that acted in the twin acts on the
robot without a remapping layer; ``joint_signs`` flips any servo the bench
shows reversed, because polarity is the one thing a URDF cannot tell you.

Out of scope, honestly: the board publishes no arm joint-state topic this
driver can verify, so :meth:`YahboomM3ProDriver.get_observation` returns ``{}``
on the robot rather than echoing the last *command* back as a *reading* - it
reads ``/joint_states`` only where the graph carries one, which the twin's
does; the cameras
(Orbbec on the wrist, a USB camera on the chassis) are ROS image topics a
caller reads with ``use_rosbridge``/``use_ros`` ``echo`` or a lerobot camera
config, not through this driver; and no policy provider is wired, so
``start_task`` / ``run_policy`` refuse with the route (drive ``send_action`` on
your own timer).
"""

from __future__ import annotations

import json
import logging
import math
import threading
from typing import TYPE_CHECKING, Any, cast

from strands.types.tools import ToolContext

from strands_robots._command_gate import gate_command
from strands_robots.drivers.base import halt_failure_detail, undeclared_verb_error
from strands_robots.drivers.yahboom_m3pro_wire import (
    ARM_CENTER_DEG,
    ARM_CHANNELS,
    ARM_JOINTS,
    ARM_TOPIC,
    ARM_TYPE,
    BASE_CHANNELS,
    CMD_VEL_TOPIC,
    CMD_VEL_WATCHDOG_S,
    DEFAULT_MOVE_TIME_MS,
    GRIPPER_CLOSED_DEG,
    GRIPPER_CLOSED_RAD,
    GRIPPER_JOINT,
    GRIPPER_OPEN_DEG,
    GRIPPER_OPEN_RAD,
    HOME_DEG,
    IMU_TOPIC,
    IMU_TYPE,
    JOINT_STATES_TOPIC,
    JOINT_STATES_TYPE,
    JOINT_TOPIC,
    JOINT_TYPE,
    MAX_ANGULAR_RPS,
    MAX_LINEAR_MPS,
    MAX_MOVE_DURATION_S,
    MAX_MOVE_TIME_MS,
    ODOM_TOPIC,
    ODOM_TYPE,
    PUBLISH_RATE_HZ,
    REQUIRED_TOPICS,
    SCAN_TOPICS,
    SERVO_RANGES_DEG,
    TWIST_TYPE,
    arm_deg_to_rad,
    arm_rad_to_deg,
    gripper_deg_to_rad,
    gripper_rad_to_deg,
    move_time_error,
    servo_deg_error,
    twist_axis_error,
)
from strands_robots.utils import (
    boolean_flag_error,
    dial_host_error,
    finite_number_error,
    positive_finite_number_error,
    refusal_repr,
    tcp_port_error,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping

    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The robots this driver registers for (read by ``_SHIPPED_DRIVERS``).
SUPPORTED_ROBOTS: tuple[str, ...] = ("yahboom_m3pro",)

#: The two ways onto the robot's ROS 2 graph. See the module docstring.
TRANSPORTS: tuple[str, ...] = ("rosbridge", "ros2", "twin")

#: Where ``rosbridge_server`` listens on the shipped image.
DEFAULT_ROSBRIDGE = "localhost:9090"

#: What the operator gate labels a base command with, per transport - the same
#: label the transport's own agent tool uses, so one physical ``/cmd_vel`` files
#: one interrupt id and one audit source whichever surface reached it.
_GATE_TOOL_BY_TRANSPORT: dict[str, str] = {
    "rosbridge": "use_rosbridge",
    "ros2": "use_ros",
    "twin": "yahboom_m3pro_twin",
}


def _refuse(reason: str) -> dict[str, Any]:
    """One refusal envelope, so every refusal has the same shape."""
    return {"status": "error", "content": [{"text": reason}]}


def _ros2_type(two_segment: str) -> str:
    """``pkg/Name`` -> ``pkg/msg/Name``, the spelling ``rclpy`` resolves."""
    package, _, name = two_segment.partition("/")
    return f"{package}/msg/{name}"


def endpoint_error(value: object, param: str, context: str) -> str | None:
    """Report why ``value`` is not a ``host[:port]`` for rosbridge, or ``None``.

    ``port=`` is polymorphic across drivers, so the wrong *shape* is named
    here: a serial path or a URL with a scheme belongs to another driver.
    """
    if not isinstance(value, str) or not value.strip():
        return f"{context}: {param} must be the rosbridge endpoint like {DEFAULT_ROSBRIDGE!r}, got {value!r}"
    if value.startswith("/"):
        return (
            f"{context}: {param} is a rosbridge host[:port] like {DEFAULT_ROSBRIDGE!r}, got a filesystem "
            f"path {value!r} (that shape belongs to the serial arms)"
        )
    if "://" in value:
        return (
            f"{context}: {param} is a bare host[:port] like {DEFAULT_ROSBRIDGE!r}, got a URL {value!r}. "
            "The WebSocket scheme is the transport's; pass the host and port only."
        )
    host, _, port_text = value.rpartition(":") if value.count(":") == 1 else (value, "", "")
    if (host_reason := dial_host_error(host, param, context)) is not None:
        return host_reason
    if port_text:
        if not port_text.isdigit():
            return f"{context}: {param} port must be a whole number, got {port_text!r} in {value!r}"
        if (port_reason := tcp_port_error(int(port_text), param, context)) is not None:
            return port_reason
    return None


def split_endpoint(value: str) -> tuple[str, int]:
    """``host[:port]`` -> ``(host, port)``, defaulting the port to 9090."""
    if value.count(":") == 1:
        host, _, port_text = value.rpartition(":")
        return host, int(port_text)
    return value, 9090


def parse_echo(envelope: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The samples inside a transport ``echo`` envelope, or ``[]``.

    Both transports answer ``echo`` as one text block: a header line naming
    the topic and type, then the samples as a JSON array, then an optional
    note when nothing arrived. This reads the array back out, so the driver's
    reads return data rather than a string a caller has to parse.
    """
    if envelope.get("status") != "success":
        return []
    text = str((envelope.get("content") or [{}])[0].get("text", ""))
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        samples = json.loads(text[start : end + 1])
    except ValueError:
        return []
    return [sample for sample in samples if isinstance(sample, dict)] if isinstance(samples, list) else []


# --------------------------------------------------------------------------- #
# The driver.                                                                 #
# --------------------------------------------------------------------------- #


class YahboomM3ProDriver:
    """Drive a Yahboom ROSMASTER M3 Pro through its ROS 2 graph.

    Satisfies :class:`~strands_robots.drivers.base.HardwareDriver`
    structurally; the surface check
    :func:`~strands_robots.drivers.register_native_driver` runs at
    registration pins the contract.
    """

    def __init__(
        self,
        tool_name: str = "yahboom_m3pro",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        transport: str = "rosbridge",
        timeout_s: float = 5.0,
        joint_signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
        move_time_ms: int = DEFAULT_MOVE_TIME_MS,
        sim: Any | None = None,
        realtime: bool = False,
        **kwargs: Any,
    ) -> None:
        """Record configuration; :meth:`connect_eagerly` does the network work.

        Args:
            tool_name: Name the agent invokes the driver by, and the mesh peer id.
            cameras: Accepted for parity; unused - the robot's cameras are ROS
                image topics, read outside this driver.
            data_config: Accepted for parity; unused.
            port: rosbridge endpoint ``host[:port]``; ``None`` selects
                :data:`DEFAULT_ROSBRIDGE`. Ignored by the ``ros2`` transport.
            transport: One of :data:`TRANSPORTS`.
            timeout_s: Per-call transport timeout (dial, echo window), seconds.
            joint_signs: Five values, each ``+1.0`` or ``-1.0`` - the polarity
                of servos 1-5 relative to the URDF joint axes. All ``+1.0``
                until the bench says otherwise.
            move_time_ms: The ``time`` field written when a caller does not
                say - how long the servos take to reach an arm target.
            sim: ``twin`` only - a built sim engine carrying ``yahboom_m3pro``
                (what ``Robot("yahboom_m3pro", mode="sim")`` returns). ``None``
                builds one on :meth:`connect_eagerly`, at the ``home`` keyframe.
            realtime: ``twin`` only - step the world at wall-clock speed so a
                viewer sees the motion as the robot would make it. Default
                ``False``: as fast as the physics allows.
            **kwargs: Ignored; accepted so the factory can forward extras.

        Raises:
            ValueError: If any of the above is outside its domain.
        """
        del cameras, data_config  # accepted for parity; unused here
        if kwargs:
            logger.debug("YahboomM3ProDriver ignoring extra kwargs: %s", sorted(kwargs))
        context = type(self).__name__
        if transport not in TRANSPORTS:
            raise ValueError(f"{context}: transport must be one of {list(TRANSPORTS)}, got {transport!r}")
        endpoint = port or DEFAULT_ROSBRIDGE
        if reason := endpoint_error(endpoint, "port", context):
            raise ValueError(reason)
        if reason := positive_finite_number_error(timeout_s, "timeout_s", context):
            raise ValueError(reason)
        if len(joint_signs) != len(ARM_JOINTS) or any(sign not in (1.0, -1.0) for sign in joint_signs):
            raise ValueError(
                f"{context}: joint_signs is one +1.0/-1.0 per arm servo ({len(ARM_JOINTS)} values), got {joint_signs!r}"
            )
        if reason := move_time_error(move_time_ms, "move_time_ms", context):
            raise ValueError(reason)
        if reason := boolean_flag_error(realtime, "realtime", context):
            raise ValueError(reason)
        if sim is not None and transport != "twin":
            raise ValueError(f"{context}: sim= is the twin transport's engine; pass transport='twin' with it")

        self._twin: Any | None = None
        if transport == "twin":
            from strands_robots.drivers.yahboom_m3pro_twin import M3ProTwinGraph  # noqa: PLC0415 - imports this module

            self._twin = M3ProTwinGraph(sim, realtime=realtime)

        self._tool_name = tool_name
        self._transport = transport
        self._host, self._port = split_endpoint(endpoint)
        self._timeout = float(timeout_s)
        self._signs = tuple(float(sign) for sign in joint_signs)
        self._move_time_ms = int(move_time_ms)

        self._connected = False
        self._connect_error: str | None = None
        self._topics: tuple[str, ...] = ()

        self._cache_lock = threading.Lock()
        self._last_arm_deg: tuple[int, ...] | None = None
        self._last_twist: dict[str, float] | None = None
        self._last_odom: dict[str, Any] | None = None
        self._last_imu: dict[str, Any] | None = None
        #: The operator context of the agent call being served, so a base
        #: command dispatched through ``stream`` can ask the operator.
        self._operator_context: ToolContext | None = None

    # ------------------------------------------------------------------ #
    # Agent tool surface.                                                #
    # ------------------------------------------------------------------ #

    @property
    def tool_name(self) -> str:
        """The name the Strands agent invokes this driver by."""
        return self._tool_name

    @property
    def tool_type(self) -> str:
        """Always ``"robot"`` - mirrors every other driver."""
        return "robot"

    @property
    def is_connected(self) -> bool:
        """Whether the graph answered and carried the board's topics."""
        return self._connected

    @property
    def endpoint(self) -> str:
        """Where this driver reaches the graph, for a reader of ``status``."""
        if self._transport == "rosbridge":
            return f"ws://{self._host}:{self._port}"
        if self._transport == "twin":
            from strands_robots.drivers.yahboom_m3pro_twin import TWIN_ENDPOINT  # noqa: PLC0415 - imports this module

            return TWIN_ENDPOINT
        return "rclpy (in-process)"

    @property
    def sim(self) -> Any | None:
        """The engine behind the ``twin`` transport - for ``render`` and the like - else ``None``."""
        return self._twin.sim if self._twin is not None else None

    @property
    def tool_spec(self) -> ToolSpec:
        """Everything an agent may ask of the robot: reads, the arm, the base, a halt."""
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Yahboom ROSMASTER M3 Pro native driver, over the robot's ROS 2 graph. A mecanum "
                    "base (forward, strafe, yaw in SI units) carrying a 6-servo arm whose sixth servo is "
                    "the gripper. Arm targets are integer degrees per servo (1-4: 0-180, 5: 0-270, "
                    "gripper: 30 closed to 180 open) and the arm holds a pose once commanded; the base is "
                    "velocity-commanded with a firmware watchdog, so a move carries duration_s and ends "
                    "in a stop on its own."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "status: connection, the graph's topics and the last commands; "
                                    "sensors: one odometry and one IMU sample; "
                                    "arm: command all six servos (joints, degrees) over time_ms; "
                                    "gripper: open or close the gripper (open=true/false); "
                                    "home: the vendor grasp-init pose with the gripper open; "
                                    "move: drive the base (linear_x, linear_y, angular_z) for duration_s, then stop; "
                                    "stop: command a zero twist"
                                ),
                                "enum": ["status", "sensors", "arm", "gripper", "home", "move", "stop"],
                                "default": "status",
                            },
                            "joints": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 6,
                                "maxItems": 6,
                                "description": (
                                    "arm only: six servo angles in degrees, servo 1 to 6. Joints 1-4 travel "
                                    "0-180, joint 5 0-270, joint 6 (gripper) 30-180. Out of range is refused."
                                ),
                            },
                            "time_ms": {
                                "type": "integer",
                                "description": (
                                    f"arm/gripper/home: how long the servos take to arrive, default "
                                    f"{DEFAULT_MOVE_TIME_MS} ms, at most {MAX_MOVE_TIME_MS}."
                                ),
                            },
                            "open": {
                                "type": "boolean",
                                "description": "gripper only: true opens (180 deg), false closes (30 deg).",
                            },
                            "linear_x": {
                                "type": "number",
                                "description": f"move only: forward m/s, within +-{MAX_LINEAR_MPS:g}; negative reverses.",
                            },
                            "linear_y": {
                                "type": "number",
                                "description": f"move only: strafe m/s (mecanum), positive left, within +-{MAX_LINEAR_MPS:g}.",
                            },
                            "angular_z": {
                                "type": "number",
                                "description": f"move only: yaw rad/s, positive counter-clockwise, within +-{MAX_ANGULAR_RPS:g}.",
                            },
                            "duration_s": {
                                "type": "number",
                                "description": (
                                    f"move only: how long to hold the twist, at most {MAX_MOVE_DURATION_S:g} s. "
                                    "Required, because the firmware zeroes a twist it stops hearing; the driver "
                                    "streams it for this long and then sends an explicit stop."
                                ),
                            },
                        },
                        "required": ["action"],
                    }
                },
            },
        )

    async def stream(
        self,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Handle one agent invocation and yield exactly one tool result.

        The agent's ``tool_context`` is rebuilt from ``invocation_state`` the
        way :class:`~strands_robots.hardware_robot.Robot` does, and held for
        the duration of the call so a base command can reach the operator gate.
        """
        del kwargs  # forward-compat only
        tool_use_id = tool_use.get("toolUseId", "")
        request = tool_use.get("input") or {}
        action = request.get("action", "status")
        agent = invocation_state.get("agent")
        self._operator_context = (
            ToolContext(tool_use=tool_use, agent=agent, invocation_state=dict(invocation_state))
            if agent is not None
            else None
        )
        envelope: dict[str, Any]
        if action == "status":
            envelope = await self.get_status()
        elif action == "sensors":
            envelope = self._sensors_envelope()
        elif action == "arm":
            envelope = self.set_arm_degrees(request.get("joints"), time_ms=request.get("time_ms"))
        elif action == "gripper":
            envelope = self.set_gripper(request.get("open", True), time_ms=request.get("time_ms"))
        elif action == "home":
            envelope = self.home(time_ms=request.get("time_ms"))
        elif action == "move":
            envelope = self.move(
                request.get("linear_x", 0.0),
                request.get("linear_y", 0.0),
                request.get("angular_z", 0.0),
                duration_s=request.get("duration_s"),
            )
        elif action == "stop":
            envelope = self.stop_task()
        else:
            envelope = undeclared_verb_error(self, action)
        # Released here rather than in a ``finally``: every verb above answers
        # with an envelope and never raises past dispatch, so there is no path
        # that skips this line - and the dispatch chain stays a plain
        # ``if``/``elif``/``else`` at the function's top level, which is the
        # shape the fleet-wide structural test reads.
        self._operator_context = None
        yield {"toolUseId": tool_use_id, **envelope}

    def _sensors_envelope(self) -> dict[str, Any]:
        """The ``sensors`` verb: fresh odometry and IMU, or a refusal naming the silence."""
        state = self.read_state()
        if not state.get("odom") and not state.get("imu"):
            return _refuse(
                f"sensors: neither {ODOM_TOPIC} nor {IMU_TOPIC} answered within {self._timeout:g}s over "
                f"{self.endpoint}. Is the robot's bringup running, with the micro-ROS agent connected to the board?"
            )
        odom = state.get("odom") or {}
        pose = ((odom.get("pose") or {}).get("pose") or {}).get("position") or {}
        twist = ((odom.get("twist") or {}).get("twist") or {}).get("linear") or {}
        summary = (
            f"odom x {pose.get('x', '?')} y {pose.get('y', '?')} | "
            f"vel x {twist.get('x', '?')} y {twist.get('y', '?')} | imu {'ok' if state.get('imu') else 'silent'}"
        )
        return {"status": "success", "content": [{"text": summary}, {"json": state}]}

    # ------------------------------------------------------------------ #
    # Transport.                                                         #
    # ------------------------------------------------------------------ #

    def _gate(self, tool_context: ToolContext | None) -> Callable[[str, str], str | None]:
        """The operator gate for one command, labelled with this transport's tool."""
        tool = _GATE_TOOL_BY_TRANSPORT[self._transport]
        return lambda kind, target: gate_command(kind, target, tool_context=tool_context, tool=tool)

    def _call(self, action: str, *, gate: Callable[[str, str], str | None], **options: Any) -> dict[str, Any]:
        """Run one transport action, spelling the interface type for the transport.

        Args:
            action: The transport verb - ``status``, ``list_topics``, ``echo``,
                ``publish``.
            gate: The operator gate for a verb that carries a command.
            **options: ``topic``, ``type`` (two-segment), ``fields``, ``count``,
                ``rate``. ``timeout`` is the driver's.

        Returns:
            The transport's envelope, verbatim.
        """
        options.setdefault("timeout", self._timeout)
        if self._twin is not None:
            return cast("dict[str, Any]", self._twin(action, gate=gate, **options))
        if self._transport == "ros2":
            from strands_robots.ros import ros_action  # noqa: PLC0415 - rclpy is optional; imported on use

            if options.get("type"):
                options["type"] = _ros2_type(options["type"])
            return ros_action(action, gate=gate, **options)

        from strands_robots.rosbridge import rosbridge_action  # noqa: PLC0415 - roslibpy is optional; imported on use

        return rosbridge_action(action, host=self._host, port=self._port, gate=gate, **options)

    @staticmethod
    def _never(kind: str, target: str) -> str | None:
        """The gate for a read: nothing to ask an operator about."""
        del kind, target
        return None

    def _publish(
        self,
        topic: str,
        msg_type: str,
        fields: dict[str, Any],
        *,
        count: int = 1,
        rate: float = PUBLISH_RATE_HZ,
        tool_context: ToolContext | None = None,
    ) -> str | None:
        """Publish ``count`` messages; return the transport's refusal text, or ``None``."""
        result = self._call(
            "publish",
            topic=topic,
            type=msg_type,
            fields=fields,
            count=count,
            rate=rate,
            gate=self._gate(tool_context),
        )
        if result.get("status") == "success":
            return None
        return str((result.get("content") or [{}])[0].get("text", "publish failed"))

    # ------------------------------------------------------------------ #
    # Lifecycle.                                                         #
    # ------------------------------------------------------------------ #

    def connect_eagerly(self) -> str | None:
        """Reach the graph and prove the board's topics are on it.

        Returns ``None`` on success. Off hardware - no transport package, no
        bridge answering, or a graph without the board (the micro-ROS agent
        missed the STM32's announcement) - returns a reason and leaves the
        driver usable: reads return their caches and writes refuse "not
        connected". Idempotent.
        """
        if self._connected:
            return None
        probe = self._call("status", gate=self._never)
        probe_text = str((probe.get("content") or [{}])[0].get("text", ""))
        # The transports answer ``status`` as prose: rosbridge and the twin say
        # "; connected to <endpoint>" once dialled, rclpy says "backend: rclpy".
        reachable = probe.get("status") == "success" and (
            "; connected to " in probe_text or probe_text.startswith("backend: rclpy")
        )
        if not reachable:
            self._connect_error = (
                f"the {self._transport} transport did not reach the robot's graph at {self.endpoint}: {probe_text}"
            )
            return self._connect_error
        listing = self._call("list_topics", gate=self._never)
        if listing.get("status") != "success":
            self._connect_error = str((listing.get("content") or [{}])[0].get("text", "list_topics failed"))
            return self._connect_error
        topics = tuple(
            line.split(" ", 1)[0] for line in str(listing["content"][0].get("text", "")).splitlines() if line.strip()
        )
        missing = [topic for topic in REQUIRED_TOPICS if topic not in topics]
        if missing:
            self._connect_error = (
                f"reached the graph at {self.endpoint} but it carries no {', '.join(missing)} "
                f"({len(topics)} topic(s) seen). The M3 Pro's motors are behind a micro-ROS board; a graph "
                "without its topics means the micro_ros_agent missed the STM32's boot announcement - restart "
                "the agent (or the bringup launch) on the robot, not the firmware."
            )
            return self._connect_error
        self._topics = topics
        self._connected = True
        self._connect_error = None
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report reachability, the topics seen and what was last commanded."""
        with self._cache_lock:
            arm = list(self._last_arm_deg) if self._last_arm_deg else None
            twist = dict(self._last_twist) if self._last_twist else None
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "connected": self.is_connected,
                        # Part of the triple every native driver's status carries
                        # (``tool_name`` / ``connected`` / ``battery_pct``), and
                        # structurally ``None`` here rather than merely unread:
                        # this driver vouches for no reading it has not verified
                        # on the graph (the reason :meth:`get_observation` is
                        # empty too), and the board publishes no battery-percent
                        # topic this driver has confirmed. A caller that needs
                        # the chassis voltage reads the vendor's own topic with
                        # ``use_rosbridge``/``use_ros`` ``echo``.
                        "battery_pct": None,
                        "connect_error": self._connect_error,
                        "transport": self._transport,
                        "endpoint": self.endpoint,
                        "topics": list(self._topics),
                        "last_arm_deg": arm,
                        "last_twist": twist,
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """Command a zero twist, leaving the robot connected. Never raises.

        The arm holds its pose on its own; only the base can be moving.
        """
        if (detail := halt_failure_detail(self.stop_task())) is not None:
            logger.error(
                "%s.stop(): the zero twist did not reach the base, which may still be driving until "
                "the firmware watchdog zeroes it: %s",
                self._tool_name,
                detail,
            )

    def cleanup(self) -> None:
        """Zero twist, then forget the connection. Idempotent.

        The transports own their sockets and nodes, so there is nothing else
        to release here; the halt goes out because a teardown must be a stop,
        and a halt that did not send is logged rather than discarded - the
        hook returns ``None``, so the log line is the only record of it.
        """
        if self._connected and (detail := halt_failure_detail(self.stop_task())) is not None:
            logger.error(
                "%s.cleanup(): the parting zero twist did not reach the base; the firmware watchdog "
                "halts it within ~%.1fs on its own: %s",
                self._tool_name,
                CMD_VEL_WATCHDOG_S,
                detail,
            )
        self._connected = False
        if self._twin is not None:
            self._twin.close()

    # ------------------------------------------------------------------ #
    # Write path.                                                        #
    # ------------------------------------------------------------------ #

    def send_action(
        self,
        action: dict[str, Any],
        robot_name: str | None = None,
        *,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Command the arm and/or the base in the **model's** vocabulary.

        Args:
            action: Any of :data:`ARM_CHANNELS` (radians, the MJCF joint
                ranges) and :data:`BASE_CHANNELS` (m/s, m/s, rad/s). Arm keys
                are converted to servo degrees and sent as one ``ArmJoints``
                message; an arm joint the action omits keeps the last commanded
                servo angle (or its ``home`` angle before any command), because
                the wire commands all six at once. Base keys are sent as one
                Twist - a single frame, so a caller on a control loop keeps the
                base rolling by calling again inside the watchdog. An action
                with neither is refused, as is one with a key of neither kind.
            robot_name: Accepted for parity; this driver fronts one robot.
            tool_context: The operator context for the ``/cmd_vel`` gate; the
                agent path supplies it from ``stream``.

        Returns:
            A success envelope naming what went on the wire (servo degrees,
            the twist), or a refusal naming what was wrong. Nothing is sent
            on a refusal, and when both an arm and a base command are present
            the arm is sent first and a base refusal is reported with the arm
            already commanded.
        """
        if not self._connected:
            suffix = f" ({self._connect_error})" if self._connect_error else ""
            return _refuse(f"send_action: not connected - call connect_eagerly() first{suffix}")
        if not isinstance(action, dict) or not action:
            return _refuse(f"send_action: action must be a non-empty dict of {list(ARM_CHANNELS + BASE_CHANNELS)}")
        bad = sorted(set(action) - set(ARM_CHANNELS) - set(BASE_CHANNELS))
        if bad:
            return _refuse(f"send_action: unknown channel(s) {bad}; valid: {list(ARM_CHANNELS + BASE_CHANNELS)}")

        arm_keys = [key for key in ARM_CHANNELS if key in action]
        base_keys = [key for key in BASE_CHANNELS if key in action]
        sent: dict[str, Any] = {"driver": "yahboom_m3pro", "robot": robot_name or self._tool_name}

        if arm_keys:
            with self._cache_lock:
                degrees = list(self._last_arm_deg or HOME_DEG)
            for key in arm_keys:
                if (reason := finite_number_error(action[key], key, "send_action")) is not None:
                    return _refuse(reason)
                joint = key[: -len(".pos")]
                if joint == GRIPPER_JOINT:
                    degrees[5] = gripper_rad_to_deg(float(action[key]))
                else:
                    index = ARM_JOINTS.index(joint)
                    degrees[index] = arm_rad_to_deg(index + 1, float(action[key]), self._signs[index])
            for servo_id, angle in enumerate(degrees, start=1):
                if (reason := servo_deg_error(servo_id, angle, f"servo{servo_id}", "send_action")) is not None:
                    return _refuse(reason + " (converted from the model radians supplied)")
            if (refusal := self._send_arm(tuple(degrees), self._move_time_ms, tool_context)) is not None:
                return _refuse(f"send_action: {refusal}")
            sent["arm_deg"] = list(degrees)
            sent["time_ms"] = self._move_time_ms

        if base_keys:
            for key in BASE_CHANNELS:
                if key in action and (reason := twist_axis_error(action[key], key, "send_action")) is not None:
                    return _refuse(reason)
            twist = {key: float(action.get(key, 0.0)) for key in BASE_CHANNELS}
            if (refusal := self._send_twist(twist, count=1, tool_context=tool_context)) is not None:
                return _refuse(f"send_action: {refusal}")
            sent["twist"] = twist

        return {"status": "success", "content": [{"json": sent}]}

    def _send_arm(self, degrees: tuple[int, ...], time_ms: int, tool_context: ToolContext | None) -> str | None:
        """One ``ArmJoints`` message; record it on success."""
        fields = {f"joint{i}": int(angle) for i, angle in enumerate(degrees, start=1)}
        fields["time"] = int(time_ms)
        if (refusal := self._publish(ARM_TOPIC, ARM_TYPE, fields, tool_context=tool_context)) is not None:
            return refusal
        with self._cache_lock:
            self._last_arm_deg = tuple(int(angle) for angle in degrees)
        return None

    def _send_twist(self, twist: dict[str, float], *, count: int, tool_context: ToolContext | None) -> str | None:
        """``count`` Twist messages at :data:`PUBLISH_RATE_HZ`; record the twist on success."""
        fields = {
            "linear": {"x": twist["linear.x"], "y": twist["linear.y"], "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": twist["angular.z"]},
        }
        refusal = self._publish(CMD_VEL_TOPIC, TWIST_TYPE, fields, count=count, tool_context=tool_context)
        if refusal is not None:
            return refusal
        with self._cache_lock:
            self._last_twist = dict(twist)
        return None

    def set_arm_degrees(self, joints: Any, time_ms: Any = None) -> dict[str, Any]:
        """Command all six servos in the vendor's own units.

        Args:
            joints: Six angles in degrees, servo 1 to 6. Each is held to its
                servo's travel by :func:`servo_deg_error`.
            time_ms: How long the servos take to arrive; ``None`` uses the
                driver's ``move_time_ms``.

        Returns:
            A success envelope naming the degrees and time sent, or a refusal.
        """
        if not self._connected:
            return _refuse("set_arm_degrees: not connected - call connect_eagerly() first")
        if not isinstance(joints, list | tuple) or len(joints) != 6:
            return _refuse(f"set_arm_degrees: joints must be six servo angles in degrees, got {refusal_repr(joints)}")
        for servo_id, angle in enumerate(joints, start=1):
            if (reason := servo_deg_error(servo_id, angle, f"joint{servo_id}", "set_arm_degrees")) is not None:
                return _refuse(reason)
        duration = self._move_time_ms if time_ms is None else time_ms
        if (reason := move_time_error(duration, "time_ms", "set_arm_degrees")) is not None:
            return _refuse(reason)
        degrees = tuple(int(round(float(angle))) for angle in joints)
        if (refusal := self._send_arm(degrees, int(duration), self._operator_context)) is not None:
            return _refuse(f"set_arm_degrees: {refusal}")
        return {"status": "success", "content": [{"json": {"arm_deg": list(degrees), "time_ms": int(duration)}}]}

    def set_gripper(self, open: Any, time_ms: Any = None) -> dict[str, Any]:  # noqa: A002 - the verb's own word
        """Open or close the gripper, leaving the other five servos where they are.

        Args:
            open: ``True`` for :data:`GRIPPER_OPEN_DEG`, ``False`` for
                :data:`GRIPPER_CLOSED_DEG`. Read as a boolean, not for truthiness.
            time_ms: Servo travel time; ``None`` uses the driver's default.
        """
        if not isinstance(open, bool):
            return _refuse(f"set_gripper: open must be true or false, got {refusal_repr(open)}")
        with self._cache_lock:
            degrees = list(self._last_arm_deg or HOME_DEG)
        degrees[5] = GRIPPER_OPEN_DEG if open else GRIPPER_CLOSED_DEG
        return self.set_arm_degrees(degrees, time_ms=time_ms)

    def home(self, time_ms: Any = None) -> dict[str, Any]:
        """The vendor grasp-init pose (:data:`HOME_DEG`), gripper open."""
        return self.set_arm_degrees(list(HOME_DEG), time_ms=time_ms)

    def move(
        self,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        angular_z: float = 0.0,
        duration_s: float | None = None,
        *,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Hold a twist for ``duration_s``, then stop.

        The firmware zeroes the motors ~0.3 s after the last Twist, so the
        twist is streamed at :data:`PUBLISH_RATE_HZ` for the duration and an
        explicit zero follows. The answer reports both halves: a move whose
        trailing stop did not send is an error, even though the watchdog will
        stop the base on its own, because the caller was promised a stop.

        Args:
            linear_x: Forward m/s, inside ``+-MAX_LINEAR_MPS`` or refused.
            linear_y: Strafe m/s, positive left (a mecanum base), same envelope.
            angular_z: Yaw rad/s, positive counter-clockwise, inside
                ``+-MAX_ANGULAR_RPS`` or refused.
            duration_s: Required, ``(0, MAX_MOVE_DURATION_S]``.
            tool_context: Operator context for the ``/cmd_vel`` gate; the
                agent path supplies the one held by ``stream``.
        """
        context = tool_context or self._operator_context
        if duration_s is None:
            return _refuse(
                "move: duration_s is required - the base zeroes a twist it stops hearing within "
                f"{CMD_VEL_WATCHDOG_S:g}s, so a twist without a duration is a twitch, not a move"
            )
        if (reason := positive_finite_number_error(duration_s, "duration_s", "move")) is not None:
            return _refuse(reason)
        if float(duration_s) > MAX_MOVE_DURATION_S:
            return _refuse(
                f"move: duration_s is at most {MAX_MOVE_DURATION_S:g}s, got {duration_s}. A longer run "
                "belongs to repeated calls, with odometry read between legs."
            )
        twist = {"linear.x": linear_x, "linear.y": linear_y, "angular.z": angular_z}
        for key, value in twist.items():
            if (reason := twist_axis_error(value, key, "move")) is not None:
                return _refuse(reason)
        if not self._connected:
            return _refuse("move: not connected - call connect_eagerly() first")
        count = max(1, int(math.ceil(float(duration_s) * PUBLISH_RATE_HZ)))
        command = {key: float(value) for key, value in twist.items()}
        if (refusal := self._send_twist(command, count=count, tool_context=context)) is not None:
            return _refuse(f"move: {refusal}")
        stopped = self.stop_task(tool_context=context)
        outcome = {
            "commanded": command,
            "held_s": float(duration_s),
            "frames": count,
            "stopped": stopped["status"] == "success",
        }
        if stopped["status"] != "success":
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "move: the twist was streamed but the trailing stop did not send; the firmware "
                            f"watchdog halts the base within ~{CMD_VEL_WATCHDOG_S:g}s on its own"
                        )
                    },
                    {"json": outcome},
                ],
            }
        return {"status": "success", "content": [{"json": outcome}]}

    # ------------------------------------------------------------------ #
    # Task paths.                                                        #
    # ------------------------------------------------------------------ #

    def start_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        **policy_kwargs: Any,
    ) -> dict[str, Any]:
        """Refuse: no policy provider is wired to this robot yet."""
        del instruction, policy_port, policy_host, policy_provider, duration, policy_kwargs
        return _refuse(
            "start_task: no policy provider is wired to the yahboom_m3pro driver yet. A caller with a "
            "built policy drives it by calling send_action on their own timer"
        )

    def run_policy(
        self,
        policy_object: Policy,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Refuse a host-driven rollout; this driver ships the transport only."""
        del policy_object, instruction, duration, n_steps
        return _refuse(
            "run_policy: this driver sends one command per call and owns no control loop. "
            'Call send_action on your own timer, or use mode="sim" for a host-driven rollout'
        )

    def get_task_status(self) -> dict[str, Any]:
        """Report the last commands, the only task state this driver holds."""
        with self._cache_lock:
            arm = list(self._last_arm_deg) if self._last_arm_deg else None
            twist = dict(self._last_twist) if self._last_twist else None
        return {
            "status": "success",
            "content": [{"json": {"running": False, "last_arm_deg": arm, "last_twist": twist}}],
        }

    def stop_task(self, *, tool_context: ToolContext | None = None) -> dict[str, Any]:
        """Command a zero twist - the base's halt. The arm holds its own pose.

        Not exempt from the ``/cmd_vel`` gate, for the reason
        :meth:`~strands_robots.mesh.RosbridgeRobot.stop` gives: the gate is
        keyed on the surface, and a payload-shaped carve-out cannot be written
        correctly. A pre-approved ``/cmd_vel`` covers the halt with the drive.
        """
        if not self._connected:
            return _refuse("stop_task: not connected")
        zero = {key: 0.0 for key in BASE_CHANNELS}
        refusal = self._send_twist(zero, count=1, tool_context=tool_context or self._operator_context)
        if refusal is not None:
            return _refuse(f"stop_task: {refusal}")
        return {"status": "success", "content": [{"json": {"driver": "yahboom_m3pro", "twist": zero}}]}

    # ------------------------------------------------------------------ #
    # Read path.                                                         #
    # ------------------------------------------------------------------ #

    def get_observation(self) -> dict[str, float]:
        """Joint positions by name, read from ``/joint_states`` when the graph carries it.

        Returns:
            ``{"<joint>.pos": radians}`` for every model joint the topic
            reports - ``arm1..arm5``, ``gripper`` (the ``rlink1`` crank, the
            key :meth:`send_action` takes) and the three base joints - or
            ``{}`` when the graph has no ``/joint_states`` or it is silent. The
            board publishes none, so on the robot this is ``{}``: echoing the
            last *command* back as a *reading* would put a target where every
            consumer expects a measurement (:meth:`last_arm_command` answers
            that question). The twin's graph carries the topic, so there this
            is the model's state.
        """
        if not self._connected or JOINT_STATES_TOPIC not in self._topics:
            return {}
        samples = parse_echo(
            self._call("echo", topic=JOINT_STATES_TOPIC, type=JOINT_STATES_TYPE, count=1, gate=self._never)
        )
        if not samples:
            return {}
        names, positions = samples[0].get("name") or [], samples[0].get("position") or []
        reading: dict[str, float] = {}
        for name, position in zip(names, positions, strict=False):
            joint = str(name).rsplit("/", 1)[-1]
            key = GRIPPER_JOINT if joint == "rlink1" else joint
            if key in (*ARM_JOINTS, GRIPPER_JOINT, "base_x", "base_y", "base_yaw"):
                reading[f"{key}.pos"] = float(position)
        return reading

    def last_arm_command(self) -> dict[str, float] | None:
        """The last commanded arm pose in the model's radians, or ``None``.

        The inverse of the map :meth:`send_action` applies, so a caller
        recording sim-shaped actions gets them back in sim units.
        """
        with self._cache_lock:
            degrees = self._last_arm_deg
        if degrees is None:
            return None
        pose = {
            f"{name}.pos": arm_deg_to_rad(index + 1, degrees[index], self._signs[index])
            for index, name in enumerate(ARM_JOINTS)
        }
        pose[f"{GRIPPER_JOINT}.pos"] = gripper_deg_to_rad(degrees[5])
        return pose

    def read_state(self) -> dict[str, Any]:
        """One odometry and one IMU sample, fresh when the graph answers.

        Falls back to the cached samples when a topic is silent, so a caller
        always sees the last truth the robot told rather than an exception -
        the mesh publishes from here.

        Returns:
            ``{"odom": {...} | None, "imu": {...} | None}``.
        """
        if self._connected:
            for topic, msg_type, attr in ((ODOM_TOPIC, ODOM_TYPE, "_last_odom"), (IMU_TOPIC, IMU_TYPE, "_last_imu")):
                samples = parse_echo(self._call("echo", topic=topic, type=msg_type, count=1, gate=self._never))
                if samples:
                    with self._cache_lock:
                        setattr(self, attr, samples[0])
        with self._cache_lock:
            return {
                "odom": dict(self._last_odom) if self._last_odom else None,
                "imu": dict(self._last_imu) if self._last_imu else None,
            }


#: The driver, its transports, and - re-exported so a caller reads one module -
#: every fact about the wire from :mod:`strands_robots.drivers.yahboom_m3pro_wire`.
__all__ = [
    "ARM_CENTER_DEG",
    "ARM_CHANNELS",
    "ARM_JOINTS",
    "ARM_TOPIC",
    "ARM_TYPE",
    "BASE_CHANNELS",
    "CMD_VEL_TOPIC",
    "CMD_VEL_WATCHDOG_S",
    "DEFAULT_MOVE_TIME_MS",
    "DEFAULT_ROSBRIDGE",
    "GRIPPER_CLOSED_DEG",
    "GRIPPER_CLOSED_RAD",
    "GRIPPER_JOINT",
    "GRIPPER_OPEN_DEG",
    "GRIPPER_OPEN_RAD",
    "HOME_DEG",
    "IMU_TOPIC",
    "IMU_TYPE",
    "JOINT_STATES_TOPIC",
    "JOINT_STATES_TYPE",
    "JOINT_TOPIC",
    "JOINT_TYPE",
    "MAX_ANGULAR_RPS",
    "MAX_LINEAR_MPS",
    "MAX_MOVE_DURATION_S",
    "MAX_MOVE_TIME_MS",
    "ODOM_TOPIC",
    "ODOM_TYPE",
    "PUBLISH_RATE_HZ",
    "REQUIRED_TOPICS",
    "SCAN_TOPICS",
    "SERVO_RANGES_DEG",
    "SUPPORTED_ROBOTS",
    "TRANSPORTS",
    "TWIST_TYPE",
    "YahboomM3ProDriver",
    "arm_deg_to_rad",
    "arm_rad_to_deg",
    "endpoint_error",
    "gripper_deg_to_rad",
    "gripper_rad_to_deg",
    "move_time_error",
    "parse_echo",
    "servo_deg_error",
    "split_endpoint",
    "twist_axis_error",
]
