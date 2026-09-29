"""ROS 2 mobile-base drivers - one drive contract over three wire paths.

- :mod:`~strands_robots.drivers.ros._mobile_base` - :class:`MobileBaseRobot`, the
  shared drive/stop/pose/scan contract, and the ``Transport`` seam it talks through
- :mod:`~strands_robots.drivers.ros.ros_bridge` - :class:`RosBridgedRobot`, over ``use_ros`` (rclpy)
- :mod:`~strands_robots.drivers.ros.rosbridge_robot` - :class:`RosbridgeRobot`, over a rosbridge websocket
- :mod:`~strands_robots.drivers.ros.rtps_robot` - :class:`RtpsRobot`, over raw RTPS (no ROS install)
- :mod:`~strands_robots.drivers.ros.ackermann_robot` - :class:`AckermannRosRobot`, a car-like base
"""

from strands_robots.drivers.ros._mobile_base import ActionCapable, MobileBaseRobot, ServiceCapable, Transport
from strands_robots.drivers.ros.ackermann_robot import AckermannRosRobot
from strands_robots.drivers.ros.ros_bridge import RosBridgedRobot
from strands_robots.drivers.ros.rosbridge_robot import RosbridgeRobot
from strands_robots.drivers.ros.rtps_robot import RtpsRobot

__all__ = [
    "AckermannRosRobot",
    "ActionCapable",
    "MobileBaseRobot",
    "RosBridgedRobot",
    "RosbridgeRobot",
    "RtpsRobot",
    "ServiceCapable",
    "Transport",
]
