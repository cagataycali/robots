"""Shared rclpy publisher for robot telemetry on a ROS 2 domain.

Both the simulation bridge (:class:`strands_robots.simulation.ros_bridge.SimRosBridge`)
and the hardware bridge (:class:`strands_robots.hardware_ros_bridge.HardwareRosBridge`)
advertise the same per-robot topics so a real arm and its digital twin look
identical on the ROS 2 graph:

* ``/<robot>/joint_states`` (``sensor_msgs/msg/JointState``) - joint names and
  positions.
* ``/<robot>/<camera>/image_raw`` (``sensor_msgs/msg/Image``, ``rgb8``) - one
  message per camera frame.

This module holds the rclpy machinery once - node ownership, topic mangling,
per-robot publisher caching, and message construction - so the sim and hardware
bridges are thin, symmetric subclasses that differ only in their default node
name. ``rclpy`` and the ROS 2 message packages are optional, system-provided
dependencies (they are not on PyPI); they are imported lazily through
:func:`strands_robots.utils.require_optional`, so importing this module - and
running with the bridge disabled - never requires ROS 2.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from strands_robots.utils import require_optional

if TYPE_CHECKING:
    import numpy as np


class RosTelemetryBridge:
    """A thin rclpy publisher for per-robot joint state and camera frames.

    Subclassed by :class:`SimRosBridge` and :class:`HardwareRosBridge`, which
    set a distinguishing default ``node_name`` but share every publish path so
    a simulated robot and a real one are byte-for-byte identical on the wire.

    Args:
        domain_id: ROS 2 domain (``ROS_DOMAIN_ID``) the bridge publishes on.
        node_name: Name of the internal rclpy node.
        qos_depth: Depth of the publishers' KEEP_LAST history.

    Raises:
        ImportError: When ``rclpy`` / the ROS 2 message packages are not
            importable, with an install hint (system ROS 2 or the docker image).
    """

    #: Default rclpy node name; subclasses override to identify their source.
    default_node_name = "strands_robots"

    def __init__(self, domain_id: int = 0, node_name: str | None = None, qos_depth: int = 10) -> None:
        # Pin the domain before rclpy reads it. Set it unconditionally so the
        # bridge publishes where the caller asked, not where the shell happened
        # to point.
        os.environ["ROS_DOMAIN_ID"] = str(int(domain_id))

        rclpy_mod: Any = require_optional(
            "rclpy", extra="ros2", purpose="the ROS 2 telemetry bridge (ros2_bridge=True)"
        )
        sensor_msgs: Any = require_optional(
            "sensor_msgs.msg", pip_install="ros-<distro>-sensor-msgs", purpose="the ROS 2 telemetry bridge"
        )
        self._rclpy = rclpy_mod
        self._JointState = sensor_msgs.JointState
        self._Image = sensor_msgs.Image

        self._owns_context = not self._rclpy.ok()
        if self._owns_context:
            self._rclpy.init()
        self._node = self._rclpy.create_node(node_name or self.default_node_name)
        self._qos_depth = qos_depth
        self._joint_pubs: dict[str, Any] = {}
        self._image_pubs: dict[str, Any] = {}

    @staticmethod
    def _safe(name: str) -> str:
        """Map a robot/camera name to a valid ROS 2 topic segment."""
        return "".join(c if (c.isalnum() or c == "_") else "_" for c in name).strip("_") or "robot"

    def _now(self) -> Any:
        return self._node.get_clock().now().to_msg()

    def _joint_publisher(self, robot: str) -> Any:
        pub = self._joint_pubs.get(robot)
        if pub is None:
            topic = f"/{self._safe(robot)}/joint_states"
            pub = self._node.create_publisher(self._JointState, topic, self._qos_depth)
            self._joint_pubs[robot] = pub
        return pub

    def _image_publisher(self, robot: str, camera: str) -> Any:
        key = f"{robot}/{camera}"
        pub = self._image_pubs.get(key)
        if pub is None:
            topic = f"/{self._safe(robot)}/{self._safe(camera)}/image_raw"
            pub = self._node.create_publisher(self._Image, topic, self._qos_depth)
            self._image_pubs[key] = pub
        return pub

    def publish_joint_states(self, robot: str, names: list[str], positions: list[float]) -> None:
        """Publish one ``JointState`` for ``robot`` on ``/<robot>/joint_states``."""
        msg = self._JointState()
        msg.header.stamp = self._now()
        msg.header.frame_id = self._safe(robot)
        msg.name = list(names)
        msg.position = [float(p) for p in positions]
        self._joint_publisher(robot).publish(msg)

    def publish_image(self, robot: str, camera: str, image: np.ndarray) -> None:
        """Publish one RGB ``Image`` on ``/<robot>/<camera>/image_raw``.

        Args:
            robot: Robot name (topic namespace).
            camera: Camera name (topic sub-namespace).
            image: ``(H, W, 3)`` uint8 RGB frame.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            return
        height, width = int(image.shape[0]), int(image.shape[1])
        msg = self._Image()
        msg.header.stamp = self._now()
        msg.header.frame_id = f"{self._safe(robot)}/{self._safe(camera)}"
        msg.height = height
        msg.width = width
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = width * 3
        msg.data = image.astype("uint8", copy=False).tobytes()
        self._image_publisher(robot, camera).publish(msg)

    def shutdown(self) -> None:
        """Destroy the node and, if this bridge initialized rclpy, shut it down."""
        node = getattr(self, "_node", None)
        if node is not None:
            try:
                node.destroy_node()
            finally:
                self._node = None
        if getattr(self, "_owns_context", False) and self._rclpy.ok():
            self._rclpy.shutdown()
            self._owns_context = False
