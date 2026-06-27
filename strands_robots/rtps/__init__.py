"""Pure-RTPS ROS 2 interop - talk to (and act as) a ROS 2 robot with no rclpy.

ROS 2 runs over DDS/RTPS. This package lets a strands agent join a ROS 2 graph
as a **first-class DDS participant** using only the pip-installable
``cyclonedds`` binding - no sourced ROS 2 distro, no ``rclpy``, no ``ros2`` CLI.
Because RTPS is stable across ROS 2 distros (Humble, Jazzy, Rolling, ...), one
implementation interoperates with every ROS 2 version.

Unlike ``use_ros`` (a *client/observer* that needs rclpy), an RTPS participant
can **act as a robot**: advertise and publish topics a real node will consume,
and subscribe to command topics - indistinguishable on the wire from hardware.

Scope (v1): topics only - ``advertise`` / ``publish`` / ``subscribe`` / ``echo``
/ discovery. Services and actions need the ROS 2 request/reply-over-DDS protocol
and are deliberately deferred to a focused follow-up.

The one hard constraint: to publish a message you must own its type definition
locally (DDS can discover remote types via XTypes, but cyclonedds-python's
dynamic-type support is not yet complete enough to rely on). So this package
ships a curated IDL bundle of the common ROS 2 messages (``strands_robots.rtps.idl``);
arbitrary custom messages are out of scope until dynamic types mature.
"""

from strands_robots.rtps.mangling import (
    dds_topic_name,
    dds_type_name,
    ros_topic_name,
)

__all__ = [
    "dds_topic_name",
    "dds_type_name",
    "ros_topic_name",
]
