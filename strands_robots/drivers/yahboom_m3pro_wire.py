"""The Yahboom ROSMASTER M3 Pro's wire: its ROS 2 topics, its servo units, and the maps between them and the twin.

The one module both ends of the twin transport read. The driver
(:mod:`strands_robots.drivers.yahboom_m3pro`) converts the model's radians to
servo degrees before a message goes out; the twin
(:mod:`strands_robots.drivers.yahboom_m3pro_twin`) converts them back before a
message reaches the model. Both maps live here, so the two cannot drift apart
and neither module has to import the other - a driver that imported its twin
and a twin that imported its driver would be a cycle for what is one table of
facts about the robot.

Everything here is a statement about the robot, not about a transport: the
topic names and interface types the board's micro-ROS firmware speaks, the
servo travel table, the vendor teleop ceilings, the firmware's ``/cmd_vel``
watchdog, and the degree/radian correspondence the description's ``home``
keyframe was written with. Sources are cited in the driver's module docstring.
"""

from __future__ import annotations

import math
from typing import cast

from strands_robots.utils import finite_number_error, positive_whole_number_error

# --- The graph: topics and their interface types, rosbridge (two-segment) ----- #
CMD_VEL_TOPIC = "/cmd_vel"
ARM_TOPIC = "/arm6_joints"
JOINT_TOPIC = "/arm_joint"
ODOM_TOPIC = "/odom_raw"
IMU_TOPIC = "/imu/data_raw"
#: Joint state by name. The board publishes none; the twin does, and a firmware
#: that grows one is read the same way.
JOINT_STATES_TOPIC = "/joint_states"
#: The two lidars - on the graph, read with ``use_rosbridge``/``use_ros`` echo.
SCAN_TOPICS: tuple[str, ...] = ("/scan0", "/scan1")

TWIST_TYPE = "geometry_msgs/Twist"
ARM_TYPE = "arm_msgs/ArmJoints"
JOINT_TYPE = "arm_msgs/ArmJoint"
ODOM_TYPE = "nav_msgs/Odometry"
IMU_TYPE = "sensor_msgs/Imu"
JOINT_STATES_TYPE = "sensor_msgs/JointState"

#: The topics whose presence proves the board is on the graph. When the
#: micro-ROS agent missed the STM32's boot announcement the graph carries only
#: ``/parameter_events`` and ``/rosout``, and nothing is broken but nothing
#: moves either - the agent needs a restart, not a reflash.
REQUIRED_TOPICS: tuple[str, ...] = (CMD_VEL_TOPIC, ARM_TOPIC)

# --- The arm ------------------------------------------------------------------ #
#: The model's arm joints, in servo order: ``arm1`` is servo ``joint1``.
ARM_JOINTS: tuple[str, ...] = ("arm1", "arm2", "arm3", "arm4", "arm5")
#: The model's gripper joint (the driven crank ``rlink1``, actuator ``gripper``).
GRIPPER_JOINT = "gripper"
#: Every ``send_action`` key that commands the arm, as ``<joint>.pos``.
ARM_CHANNELS: tuple[str, ...] = tuple(f"{name}.pos" for name in (*ARM_JOINTS, GRIPPER_JOINT))
#: Every ``send_action`` key that commands the base - the Twist fields, SI.
BASE_CHANNELS: tuple[str, ...] = ("linear.x", "linear.y", "angular.z")

#: Servo travel in degrees, by servo id 1-6. Joints 1-4 are 180-degree bus
#: servos, joint 5 is the 270-degree wrist servo, joint 6 is the gripper -
#: 30 is closed, 180 is open. Vendor limits, the same table every Yahboom
#: teleop node clamps to.
SERVO_RANGES_DEG: dict[int, tuple[int, int]] = {
    1: (0, 180),
    2: (0, 180),
    3: (0, 180),
    4: (0, 180),
    5: (0, 270),
    6: (30, 180),
}

#: The servo angle the model's ``q = 0`` sits at, for joints 1-5.
ARM_CENTER_DEG = 90.0
#: Gripper endpoints, in the model's crank radians and the servo's degrees.
GRIPPER_CLOSED_RAD = -1.54
GRIPPER_OPEN_RAD = 0.0
GRIPPER_CLOSED_DEG = 30
GRIPPER_OPEN_DEG = 180

#: The vendor grasp-init pose the description's ``home`` keyframe encodes
#: (servo degrees 90/120/0/0/90), with the gripper open.
HOME_DEG: tuple[int, ...] = (90, 120, 0, 0, 90, GRIPPER_OPEN_DEG)

#: Default and ceiling for the ``time`` field: how long the servos take to
#: reach the target. ``int16`` on the wire; 30 s is already a slow arm.
DEFAULT_MOVE_TIME_MS = 1500
MAX_MOVE_TIME_MS = 30_000

# --- The base ----------------------------------------------------------------- #
#: The vendor keyboard-teleop ceilings (``linear_speed_limit`` 1.0 m/s,
#: ``angular_speed_limit`` 5.0 rad/s). A magnitude past these is refused by
#: name rather than clamped, for the reason the EarthRover driver gives: a
#: velocity-commanded base at full speed is the one outcome a wrong scale must
#: not produce silently.
MAX_LINEAR_MPS = 1.0
MAX_ANGULAR_RPS = 5.0
#: The firmware's ``/cmd_vel`` watchdog: roughly 200-500 ms without a fresh
#: Twist zeroes the motors. A held move streams above this.
CMD_VEL_WATCHDOG_S = 0.3
PUBLISH_RATE_HZ = 10.0
#: The longest one ``move`` may hold a twist. A tool call is a turn, not a
#: control loop; a longer leg is repeated calls with odometry read between.
MAX_MOVE_DURATION_S = 10.0


# --------------------------------------------------------------------------- #
# Unit domains - the sim <-> servo maps, and their refusals.                  #
# --------------------------------------------------------------------------- #


def arm_rad_to_deg(joint_index: int, radians: float, sign: float = 1.0) -> int:
    """Model radians for ``arm<joint_index>`` -> the servo's integer degrees.

    ``deg = 90 + sign * degrees(q)``, the inverse of the description's
    ``q = (deg - 90) * pi / 180``. Rounded to the nearest degree because the
    wire is ``int16``.

    Args:
        joint_index: 1-5, the servo id (``arm1`` is servo 1).
        radians: The model joint position.
        sign: ``+1.0`` or ``-1.0`` - the servo's polarity relative to the URDF
            axis, from the driver's ``joint_signs``.

    Returns:
        The servo angle in whole degrees. Range is the caller's to check
        (:func:`servo_deg_error`).
    """
    del joint_index  # one formula serves every arm servo; the parameter documents the pairing
    return int(round(ARM_CENTER_DEG + sign * math.degrees(radians)))


def arm_deg_to_rad(joint_index: int, degrees: float, sign: float = 1.0) -> float:
    """Servo degrees for ``arm<joint_index>`` -> model radians (the inverse map)."""
    del joint_index
    return sign * math.radians(degrees - ARM_CENTER_DEG)


def gripper_rad_to_deg(radians: float) -> int:
    """Model crank radians (``-1.54`` closed .. ``0`` open) -> servo degrees (30 .. 180).

    Linear between the two endpoint pairs, so the model's closed end lands on
    the servo's closed end and the open ends coincide.
    """
    fraction_closed = (radians - GRIPPER_OPEN_RAD) / (GRIPPER_CLOSED_RAD - GRIPPER_OPEN_RAD)
    return int(round(GRIPPER_OPEN_DEG + fraction_closed * (GRIPPER_CLOSED_DEG - GRIPPER_OPEN_DEG)))


def gripper_deg_to_rad(degrees: float) -> float:
    """Servo degrees (30 .. 180) -> model crank radians (the inverse map)."""
    fraction_closed = (degrees - GRIPPER_OPEN_DEG) / (GRIPPER_CLOSED_DEG - GRIPPER_OPEN_DEG)
    return GRIPPER_OPEN_RAD + fraction_closed * (GRIPPER_CLOSED_RAD - GRIPPER_OPEN_RAD)


def servo_deg_error(servo_id: int, value: object, param: str, context: str) -> str | None:
    """Report why ``value`` is not a target servo ``servo_id`` can reach, or ``None``.

    Refuses rather than clamping: the range is the servo's mechanical travel,
    and a target past it on a bus servo either stalls against the stop or, on
    the gripper, crushes what it holds. A number outside the range is the
    usual symptom of radians handed to a degrees field - the caller has to be
    told, not corrected.

    Args:
        servo_id: 1-6.
        value: The candidate angle in degrees.
        param: Which field, to quote in the reason.
        context: Calling surface to quote in the reason.

    Returns:
        A reason naming the servo and its travel, or ``None``.
    """
    if (reason := finite_number_error(value, param, context)) is not None:
        return reason
    low, high = SERVO_RANGES_DEG[servo_id]
    angle = float(cast("float", value))
    if not low <= angle <= high:
        what = "the gripper" if servo_id == 6 else f"servo {servo_id}"
        return (
            f"{context}: {param}={angle:g} is outside {what}'s travel [{low}, {high}] degrees. "
            "A value in radians or on a normalised scale is the usual cause; the wire is whole degrees."
        )
    return None


def twist_axis_error(value: object, param: str, context: str) -> str | None:
    """Report why ``value`` is not a commandable Twist axis, or ``None``.

    ``linear.x`` / ``linear.y`` are bounded by :data:`MAX_LINEAR_MPS`,
    ``angular.z`` by :data:`MAX_ANGULAR_RPS`. Refused, not clamped - see the
    constants' note.
    """
    if (reason := finite_number_error(value, param, context)) is not None:
        return reason
    limit = MAX_ANGULAR_RPS if param.startswith("angular") else MAX_LINEAR_MPS
    unit = "rad/s" if param.startswith("angular") else "m/s"
    magnitude = float(cast("float", value))
    if abs(magnitude) > limit:
        return (
            f"{context}: {param}={magnitude:g} {unit} is outside the base envelope [-{limit:g}, {limit:g}] "
            f"{unit} (the vendor teleop ceiling). Refused rather than clamped, because the base holds a "
            "twist for the watchdog period at whatever speed it was given."
        )
    return None


def move_time_error(value: object, param: str, context: str) -> str | None:
    """Report why ``value`` is not a servo ``time`` in milliseconds, or ``None``."""
    if (reason := positive_whole_number_error(value, param, context)) is not None:
        return reason
    if int(cast("int", value)) > MAX_MOVE_TIME_MS:
        return f"{context}: {param}={value} ms is longer than the {MAX_MOVE_TIME_MS} ms ceiling (the wire is int16)"
    return None


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
    "TWIST_TYPE",
    "arm_deg_to_rad",
    "arm_rad_to_deg",
    "gripper_deg_to_rad",
    "gripper_rad_to_deg",
    "move_time_error",
    "servo_deg_error",
    "twist_axis_error",
]
