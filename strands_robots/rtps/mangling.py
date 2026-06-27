"""ROS 2 <-> DDS name mangling.

ROS 2 maps its graph names onto DDS topic and type names with a fixed,
documented scheme (see the ROS 2 design doc "Topic and Service name mapping to
DDS"). Getting these exactly right is what makes a bare DDS participant
interoperable with real ROS 2 nodes.

Topic names
-----------
A ROS 2 topic ``/turtle1/cmd_vel`` becomes the DDS topic ``rt/turtle1/cmd_vel``:

* the ROS namespace separator ``/`` is preserved,
* a domain prefix is prepended: ``rt`` for topics (services use ``rq``/``rr``,
  out of scope here),
* the leading ``/`` is dropped after the prefix join (``rt`` + ``/turtle1/...``).

Type names
----------
A ROS 2 type ``geometry_msgs/msg/Twist`` becomes the DDS type
``geometry_msgs::msg::dds_::Twist_``:

* ``/`` separators become ``::``,
* the final segment gains a trailing underscore (``Twist`` -> ``Twist_``),
* a ``dds_`` segment is inserted before the final segment.

Both directions are pure string transforms with no ROS or DDS import, so they
are trivially unit-testable with no middleware present.
"""

from __future__ import annotations

import re

# ROS 2 graph names: leading slash plus alnum/_ segments (tilde/braces are
# substitution syntax that must already be resolved before mangling).
_ROS_TOPIC_RE = re.compile(r"^/[A-Za-z0-9_/]*[A-Za-z0-9_]$")
_ROS_TYPE_RE = re.compile(r"^[A-Za-z0-9_]+/(msg|srv|action)/[A-Za-z0-9_]+$")

# DDS topic prefixes per the ROS 2 mapping. Only ``rt`` (topics) is used in v1.
_TOPIC_PREFIX = "rt"


def dds_topic_name(ros_topic: str, *, prefix: str = _TOPIC_PREFIX) -> str:
    """Map a ROS 2 topic name to its DDS topic name.

    ``/turtle1/cmd_vel`` -> ``rt/turtle1/cmd_vel``

    Args:
        ros_topic: A fully-qualified ROS 2 topic (must start with ``/``).
        prefix: DDS domain prefix; ``rt`` for topics (the default).

    Raises:
        ValueError: If *ros_topic* is not a valid absolute ROS 2 topic name.
    """
    if not _ROS_TOPIC_RE.match(ros_topic):
        raise ValueError(f"invalid ROS 2 topic {ros_topic!r}: expected an absolute name like /turtle1/cmd_vel")
    return prefix + ros_topic


def ros_topic_name(dds_topic: str, *, prefix: str = _TOPIC_PREFIX) -> str:
    """Inverse of :func:`dds_topic_name`.

    ``rt/turtle1/cmd_vel`` -> ``/turtle1/cmd_vel``

    Args:
        dds_topic: A DDS topic name carrying a ROS 2 domain prefix.
        prefix: The DDS domain prefix to strip (default ``rt``).

    Raises:
        ValueError: If *dds_topic* does not begin with ``<prefix>/``.
    """
    head = prefix + "/"
    if not dds_topic.startswith(head):
        raise ValueError(f"DDS topic {dds_topic!r} does not carry the {prefix!r} ROS 2 prefix")
    return dds_topic[len(prefix) :]


def dds_type_name(ros_type: str) -> str:
    """Map a ROS 2 interface type to its DDS type name.

    ``geometry_msgs/msg/Twist`` -> ``geometry_msgs::msg::dds_::Twist_``

    Args:
        ros_type: A ``pkg/msg/Name`` (or ``srv``/``action``) interface type.

    Raises:
        ValueError: If *ros_type* is not a valid ``pkg/<kind>/Name`` triple.
    """
    if not _ROS_TYPE_RE.match(ros_type):
        raise ValueError(f"invalid ROS 2 type {ros_type!r}: expected pkg/msg/Name (or pkg/srv/Name, pkg/action/Name)")
    pkg, kind, name = ros_type.split("/")
    return f"{pkg}::{kind}::dds_::{name}_"
