"""The M3 Pro's ROS 2 graph, answered by its MuJoCo twin.

:class:`~strands_robots.drivers.yahboom_m3pro.YahboomM3ProDriver` reaches the
robot through a *transport*: a callable that takes the same verbs the
``use_rosbridge`` / ``use_ros`` tools take (``status``, ``list_topics``,
``echo``, ``publish``) and answers with the same envelopes. Everything the
driver knows about the robot - the topics, the degree/radian maps, the
watchdog, the refusals - sits above that seam, so a third transport that
answers the graph from the ``yahboom_m3pro`` MJCF gives an agent the **same
tool, the same verbs and the same units** over the twin as over the robot.
That is what :class:`M3ProTwinGraph` is: ``Robot("yahboom_m3pro", mode="real",
transport="twin")`` builds the driver on it, and the agent that learned to
``home`` the arm and ``move`` the base in simulation says the same words to
the hardware.

What the twin answers, per topic:

* ``publish /arm6_joints`` - the six servo degrees are converted back to the
  model's radians with the driver's own inverse maps and written to the
  ``arm1..arm5`` and ``gripper`` position actuators; the world is then stepped
  for the message's ``time`` so the position servos arrive, as the board's do.
* ``publish /arm_joint`` - one servo, the same way.
* ``publish /cmd_vel`` - each frame writes the body-frame twist to the base's
  three velocity actuators, **rotated by the current yaw** because the model's
  ``base_x`` / ``base_y`` slides are world-frame (the description's DESIGN.md
  says so), and steps one publish period. After the last frame the twin does
  what the firmware does: holds the velocity for the watchdog period, then
  zeroes the motors. A single frame is therefore a ~0.3 s twitch here too.
* ``echo /odom_raw`` - a ``nav_msgs/Odometry``-shaped sample from the base
  joints: position, yaw as a quaternion, and the body-frame twist.
* ``echo /imu/data_raw`` - orientation from yaw, yaw rate, and gravity on
  ``z`` - what a level IMU on a flat floor reads.
* ``echo /joint_states`` - the model's joint positions and velocities by
  name. The twin *has* joint state where the board publishes none, so on this
  transport :meth:`~strands_robots.drivers.yahboom_m3pro.YahboomM3ProDriver.get_observation`
  is a reading rather than ``{}``.
* ``echo /scan0`` / ``/scan1`` - listed on the graph so the twin's graph is
  the robot's, but the model carries no rangefinders, so an echo answers with
  no samples - the same answer a silent lidar gives.

The operator gate is **not consulted** on this transport. The gate is a
statement about a physical command surface, and the twin has none: a
``/cmd_vel`` here moves a kinematic base in a MuJoCo world. The other two
transports consult it because they cannot know what is on the far side of the
socket; this one is the far side.

Wall clock: the twin steps physics as fast as it can, so a 2 s ``move`` returns
in milliseconds. ``realtime=True`` sleeps each publish period out instead, for
a viewer that should show the motion at the speed the robot would make it.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import TYPE_CHECKING, Any

from strands_robots.drivers.yahboom_m3pro_wire import (
    ARM_JOINTS,
    ARM_TOPIC,
    CMD_VEL_TOPIC,
    CMD_VEL_WATCHDOG_S,
    IMU_TOPIC,
    JOINT_STATES_TOPIC,
    JOINT_TOPIC,
    ODOM_TOPIC,
    SCAN_TOPICS,
    arm_deg_to_rad,
    gripper_deg_to_rad,
)
from strands_robots.utils import positive_finite_number_error

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

#: The twin's graph, as ``list_topics`` reports it - the robot's topics with
#: their ROS 2 types, plus ``/joint_states``, which the twin can answer and
#: the board cannot.
TWIN_GRAPH: tuple[tuple[str, str], ...] = (
    (ARM_TOPIC, "arm_msgs/msg/ArmJoints"),
    (JOINT_TOPIC, "arm_msgs/msg/ArmJoint"),
    (CMD_VEL_TOPIC, "geometry_msgs/msg/Twist"),
    (IMU_TOPIC, "sensor_msgs/msg/Imu"),
    (JOINT_STATES_TOPIC, "sensor_msgs/msg/JointState"),
    (ODOM_TOPIC, "nav_msgs/msg/Odometry"),
    ("/parameter_events", "rcl_interfaces/msg/ParameterEvent"),
    ("/rosout", "rcl_interfaces/msg/Log"),
    *((topic, "sensor_msgs/msg/LaserScan") for topic in SCAN_TOPICS),
)

#: The model's base actuators, in the order the world-frame twist is written.
BASE_ACTUATORS: tuple[str, ...] = ("base_x", "base_y", "base_yaw")
#: The model's gripper actuator (drives the ``rlink1`` crank).
GRIPPER_ACTUATOR = "gripper"
#: The joints ``/joint_states`` reports, in model order.
REPORTED_JOINTS: tuple[str, ...] = (*BASE_ACTUATORS, *ARM_JOINTS, "rlink1", "llink1")

#: Where the twin says it is, for ``status`` and the driver's ``endpoint``.
TWIN_ENDPOINT = "sim://yahboom_m3pro"


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": f"twin: {text}"}]}


def yaw_to_quaternion(yaw: float) -> dict[str, float]:
    """A yaw about ``z`` as the ``{x, y, z, w}`` quaternion ROS messages carry."""
    return {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2.0), "w": math.cos(yaw / 2.0)}


def body_to_world(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    """Rotate a body-frame planar velocity into the world frame the slides ride."""
    return vx * math.cos(yaw) - vy * math.sin(yaw), vx * math.sin(yaw) + vy * math.cos(yaw)


def world_to_body(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    """The inverse of :func:`body_to_world`, for the odometry twist."""
    return vx * math.cos(yaw) + vy * math.sin(yaw), -vx * math.sin(yaw) + vy * math.cos(yaw)


class M3ProTwinGraph:
    """Answer the M3 Pro's graph verbs from a ``yahboom_m3pro`` sim engine.

    Args:
        sim: A built sim engine carrying the ``yahboom_m3pro`` robot (what
            ``Robot("yahboom_m3pro", mode="sim")`` returns). ``None`` builds one
            lazily on :meth:`connect`, at the ``home`` keyframe.
        robot_name: The robot's name inside ``sim``; ``None`` takes the first.
        watchdog_s: How long the base keeps the last twist after a burst before
            the twin zeroes it - the firmware's behaviour, emulated. A positive
            finite number of seconds.
        realtime: Sleep each publish period out so the motion takes the wall
            time it would on the robot. Default ``False``: step as fast as the
            physics allows.

    Raises:
        ValueError: If ``watchdog_s`` is not a positive finite number.
    """

    def __init__(
        self,
        sim: Any | None = None,
        robot_name: str | None = None,
        *,
        watchdog_s: float = CMD_VEL_WATCHDOG_S,
        realtime: bool = False,
    ) -> None:
        if reason := positive_finite_number_error(watchdog_s, "watchdog_s", type(self).__name__):
            raise ValueError(reason)
        self._sim = sim
        self._robot_name = robot_name
        self._watchdog_s = float(watchdog_s)
        self._realtime = bool(realtime)
        self._owns_sim = sim is None
        self._build_error: str | None = None

    # ------------------------------------------------------------------ #
    # The engine.                                                        #
    # ------------------------------------------------------------------ #

    @property
    def sim(self) -> Any | None:
        """The engine behind the twin, or ``None`` before :meth:`connect` built it."""
        return self._sim

    @property
    def robot_name(self) -> str:
        """The robot's name inside the engine."""
        if self._robot_name is None:
            names = list(self._sim.list_robots()) if self._sim is not None else []
            self._robot_name = names[0] if names else "yahboom_m3pro"
        return self._robot_name

    def connect(self) -> str | None:
        """Build the engine when none was given. Returns a reason on failure, else ``None``.

        Built through :func:`strands_robots.simulation.create_simulation` and
        :meth:`~strands_robots.simulation.base.SimEngine.add_robot` - the same
        two calls ``Robot("yahboom_m3pro", mode="sim", keyframe="home")`` makes -
        rather than through the factory itself. The factory imports the driver
        registry, and the driver registry imports this robot's driver, so a twin
        that imported the factory would close a cycle around what is one table
        of facts about the robot. The simulation package imports no driver.
        """
        if self._sim is not None:
            return None
        try:
            from strands_robots.simulation import (
                create_simulation,  # noqa: PLC0415 - MuJoCo is optional; imported on use
            )

            sim = create_simulation("mujoco", tool_name="yahboom_m3pro_twin")
            for step in (sim.create_world(), sim.add_robot(name="yahboom_m3pro", keyframe="home")):
                if step.get("status") == "error":
                    sim.destroy()
                    detail = (step.get("content") or [{}])[0].get("text", str(step))
                    self._build_error = f"could not build the yahboom_m3pro twin: {detail}"
                    return self._build_error
        except Exception as exc:  # noqa: BLE001 - the reason is reported, not raised, like every connect here
            self._build_error = f"could not build the yahboom_m3pro twin ({type(exc).__name__}: {exc})"
            return self._build_error
        self._sim = sim
        self._build_error = None
        return None

    def close(self) -> None:
        """Destroy an engine this graph built; leave a caller-supplied one alone."""
        if self._sim is not None and self._owns_sim:
            try:
                self._sim.destroy()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.debug("twin: destroy() raised during close", exc_info=True)
        self._sim = None

    def _substeps(self, seconds: float) -> int:
        """Physics steps that cover ``seconds``, at least one."""
        dt = self._sim.physics_timestep() if self._sim is not None else None
        if not dt or dt <= 0:
            return 1
        return max(1, int(round(seconds / dt)))

    def _engine(self) -> Any:
        """The engine, for the verbs that run only after ``__call__`` proved it built."""
        if self._sim is None:
            raise RuntimeError("twin: the engine is not built - call connect() (or the status verb) first")
        return self._sim

    def _observation(self) -> dict[str, float]:
        """The model's scalar observation - joints and their ``.vel`` siblings."""
        obs = self._engine().get_observation(robot_name=self.robot_name, skip_images=True)
        return {key: float(value) for key, value in obs.items() if isinstance(value, int | float)}

    def _advance(self, action: dict[str, float], seconds: float) -> str | None:
        """Write ``action`` and step ``seconds``; return the engine's refusal text or ``None``."""
        result = self._engine().send_action(action, robot_name=self.robot_name, n_substeps=self._substeps(seconds))
        if result.get("status") != "success":
            return str((result.get("content") or [{}])[0].get("text", "send_action failed"))
        if self._realtime:
            time.sleep(seconds)
        return None

    # ------------------------------------------------------------------ #
    # The transport surface.                                              #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        action: str,
        *,
        gate: Callable[[str, str], str | None] | None = None,
        topic: str | None = None,
        type: str | None = None,  # noqa: A002 - the transport's own parameter name
        fields: dict[str, Any] | None = None,
        count: int = 1,
        rate: float = 10.0,
        timeout: float = 5.0,
        **_: Any,
    ) -> dict[str, Any]:
        """Run one graph verb against the twin, in the transports' envelope.

        Args:
            action: ``status``, ``list_topics``, ``echo`` or ``publish``.
            gate: Accepted for the seam and not consulted - see the module
                docstring.
            topic: The topic for ``echo`` / ``publish``.
            type: The interface type; accepted and unused, the twin knows its
                topics' types.
            fields: The message for ``publish``.
            count: Messages to publish - frames of a twist burst.
            rate: Publish rate in Hz; one frame holds ``1 / rate`` seconds.
            timeout: Accepted and unused; the twin answers synchronously.
            **_: Any other transport option, ignored.

        Returns:
            The same ``{"status", "content": [{"text"}]}`` envelope the
            rosbridge and rclpy transports answer with.
        """
        del gate, type, timeout
        if action == "status":
            if self._sim is None and (reason := self.connect()) is not None:
                return _ok(f"backend: mujoco twin; not connected - {reason}")
            return _ok(f"backend: mujoco twin; connected to {TWIN_ENDPOINT}")
        if self._sim is None:
            return _err(self._build_error or "not connected - call status first")
        if action == "list_topics":
            return _ok("\n".join(f"{name} [{kind}]" for name, kind in sorted(TWIN_GRAPH)))
        if action == "echo":
            return self._echo(topic or "", max(1, int(count)))
        if action == "publish":
            return self._publish(topic or "", dict(fields or {}), max(1, int(count)), float(rate))
        return _err(f"unknown action: {action}")

    # --- reads ---------------------------------------------------------- #

    def _echo(self, topic: str, count: int) -> dict[str, Any]:
        if topic not in dict(TWIN_GRAPH):
            return _err(f"cannot resolve type for {topic}; the twin's graph is {[name for name, _ in TWIN_GRAPH]}")
        kind = dict(TWIN_GRAPH)[topic]
        sample = self._sample(topic)
        samples = [sample] * count if sample is not None else []
        body = json.dumps(samples, indent=2)
        note = "" if samples else "\n(no messages - the twin carries no sensor on this topic)"
        return _ok(f"echo {topic} ({kind}):\n{body}{note}")

    def _sample(self, topic: str) -> dict[str, Any] | None:
        obs = self._observation()
        yaw = obs.get("base_yaw", 0.0)
        if topic == ODOM_TOPIC:
            vx, vy = world_to_body(obs.get("base_x.vel", 0.0), obs.get("base_y.vel", 0.0), yaw)
            return {
                "header": {"frame_id": "odom"},
                "child_frame_id": "base_footprint",
                "pose": {
                    "pose": {
                        "position": {"x": obs.get("base_x", 0.0), "y": obs.get("base_y", 0.0), "z": 0.0},
                        "orientation": yaw_to_quaternion(yaw),
                    }
                },
                "twist": {
                    "twist": {
                        "linear": {"x": vx, "y": vy, "z": 0.0},
                        "angular": {"x": 0.0, "y": 0.0, "z": obs.get("base_yaw.vel", 0.0)},
                    }
                },
            }
        if topic == IMU_TOPIC:
            return {
                "header": {"frame_id": "imu_link"},
                "orientation": yaw_to_quaternion(yaw),
                "angular_velocity": {"x": 0.0, "y": 0.0, "z": obs.get("base_yaw.vel", 0.0)},
                "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.80665},
            }
        if topic == JOINT_STATES_TOPIC:
            names = [joint for joint in REPORTED_JOINTS if joint in obs]
            return {
                "header": {"frame_id": ""},
                "name": names,
                "position": [obs[joint] for joint in names],
                "velocity": [obs.get(f"{joint}.vel", 0.0) for joint in names],
                "effort": [],
            }
        return None  # the lidars: listed, silent

    # --- writes --------------------------------------------------------- #

    def _publish(self, topic: str, fields: dict[str, Any], count: int, rate: float) -> dict[str, Any]:
        if topic == ARM_TOPIC:
            return self._publish_arm(fields, count)
        if topic == JOINT_TOPIC:
            return self._publish_joint(fields, count)
        if topic == CMD_VEL_TOPIC:
            return self._publish_twist(fields, count, rate)
        if topic in dict(TWIN_GRAPH):
            return _err(f"{topic} is a sensor topic; the twin publishes it, a caller cannot")
        return _err(f"the twin's graph carries no {topic}")

    @staticmethod
    def _servo_to_model(servo_id: int, degrees: float) -> tuple[str, float]:
        """One servo angle -> the actuator it drives and the model target for it."""
        if servo_id == 6:
            return GRIPPER_ACTUATOR, gripper_deg_to_rad(degrees)
        return ARM_JOINTS[servo_id - 1], arm_deg_to_rad(servo_id, degrees)

    def _publish_arm(self, fields: dict[str, Any], count: int) -> dict[str, Any]:
        try:
            targets = dict(self._servo_to_model(i, float(fields[f"joint{i}"])) for i in range(1, 7))
            seconds = max(float(fields.get("time", 0)) / 1000.0, 0.05)
        except (KeyError, TypeError, ValueError) as exc:
            return _err(f"ArmJoints needs joint1..joint6 and time, got {sorted(fields)}: {exc}")
        if (refusal := self._advance(targets, seconds)) is not None:
            return _err(refusal)
        return _ok(f"published {count} message(s) to {ARM_TOPIC}")

    def _publish_joint(self, fields: dict[str, Any], count: int) -> dict[str, Any]:
        try:
            servo_id = int(fields["id"])
            if not 1 <= servo_id <= 6:
                return _err(f"ArmJoint id must be 1-6, got {servo_id}")
            actuator, target = self._servo_to_model(servo_id, float(fields["joint"]))
            seconds = max(float(fields.get("time", 0)) / 1000.0, 0.05)
        except (KeyError, TypeError, ValueError) as exc:
            return _err(f"ArmJoint needs id, joint and time, got {sorted(fields)}: {exc}")
        if (refusal := self._advance({actuator: target}, seconds)) is not None:
            return _err(refusal)
        return _ok(f"published {count} message(s) to {JOINT_TOPIC}")

    def _publish_twist(self, fields: dict[str, Any], count: int, rate: float) -> dict[str, Any]:
        linear = fields.get("linear") or {}
        angular = fields.get("angular") or {}
        try:
            vx, vy, wz = float(linear.get("x", 0.0)), float(linear.get("y", 0.0)), float(angular.get("z", 0.0))
        except (TypeError, ValueError, AttributeError) as exc:
            return _err(f"Twist needs linear.x/y and angular.z numbers: {exc}")
        period = 1.0 / rate if rate > 0 else self._watchdog_s

        def frame(seconds: float) -> str | None:
            yaw = self._observation().get("base_yaw", 0.0)
            wx, wy = body_to_world(vx, vy, yaw)
            return self._advance({"base_x": wx, "base_y": wy, "base_yaw": wz}, seconds)

        for _ in range(count):
            if (refusal := frame(period)) is not None:
                return _err(refusal)
        # The firmware: the last twist outlives its message by the watchdog,
        # then the motors are zeroed. A zero twist re-zeroes harmlessly.
        if (vx, vy, wz) != (0.0, 0.0, 0.0) and (refusal := frame(self._watchdog_s)) is not None:
            return _err(refusal)
        if (refusal := self._advance(dict.fromkeys(BASE_ACTUATORS, 0.0), 0.0)) is not None:
            return _err(refusal)
        note = self._clamp_note({"base_x": math.hypot(vx, vy), "base_yaw": abs(wz)})
        return _ok(f"published {count} message(s) to {CMD_VEL_TOPIC}{note}")

    def _clamp_note(self, magnitudes: dict[str, float]) -> str:
        """Name the base actuators whose ``ctrlrange`` is narrower than what was asked.

        The driver's envelope is the *robot's* (the vendor teleop ceilings); the
        model's velocity actuators declare their own ``ctrlrange`` and MuJoCo
        clamps a target past it silently. A twin that drove at half the speed
        the agent asked for and said nothing would teach the agent the wrong
        robot, so the clamp is reported on the reply and logged.
        """
        model = getattr(self._sim, "mj_model", None)
        if model is None:
            return ""
        clamped: list[str] = []
        for actuator, magnitude in magnitudes.items():
            try:
                index = model.actuator(f"{self.robot_name}/{actuator}").id
            except (KeyError, ValueError, AttributeError):
                continue
            if model.actuator_ctrllimited[index]:
                ceiling = float(model.actuator_ctrlrange[index][1])
                if magnitude > ceiling + 1e-9:
                    clamped.append(f"{actuator} {magnitude:g} -> {ceiling:g}")
        if not clamped:
            return ""
        logger.warning("twin: the model's ctrlrange clamped the base command: %s", "; ".join(clamped))
        return f" (model ctrlrange clamped: {'; '.join(clamped)})"


__all__ = [
    "BASE_ACTUATORS",
    "REPORTED_JOINTS",
    "TWIN_ENDPOINT",
    "TWIN_GRAPH",
    "M3ProTwinGraph",
    "body_to_world",
    "world_to_body",
    "yaw_to_quaternion",
]
