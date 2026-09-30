---
description: The Mesh object a robot exposes, the session and peer helpers, the bridges that put ROS 2 robots on the mesh.
---

# Mesh

The mesh puts robots on a shared Zenoh session so agents and peers discover each other, exchange commands and stop together: the `Mesh` object a robot exposes, the session and peer helpers, and the bridge classes that put ROS 2 and RTPS robots on the same mesh.

## Mesh

::: strands_robots.mesh.core
    options:
      heading_level: 3
      members:
        - init_mesh
        - get_local_robots
        - mesh_disabled_by_env

::: strands_robots.mesh.core.Mesh
    options:
      heading_level: 3
      show_root_heading: true
      filters: ["!^_"]
      members_order: source

## Session and peers

::: strands_robots.mesh.session
    options:
      heading_level: 3
      members:
        - get_session
        - release_session
        - current_session
        - session_alive
        - put
        - get_peers
        - update_peer
        - clear_peers
        - prune_peers

## Input streams

::: strands_robots.mesh.input
    options:
      heading_level: 3
      members:
        - InputPublisher
        - InputReceiver

## Bridged robots

::: strands_robots.drivers.ros.ros_bridge.RosBridgedRobot
    options:
      heading_level: 3
      show_root_heading: true

::: strands_robots.drivers.ros.rosbridge_robot.RosbridgeRobot
    options:
      heading_level: 3
      show_root_heading: true

::: strands_robots.drivers.ros.rtps_robot.RtpsRobot
    options:
      heading_level: 3
      show_root_heading: true

::: strands_robots.drivers.ros.ackermann_robot.AckermannRosRobot
    options:
      heading_level: 3
      show_root_heading: true

::: strands_robots.hardware_ros_bridge.HardwareRosBridge
    options:
      heading_level: 3
      show_root_heading: true

::: strands_robots.hardware_rtps_bridge.HardwareRtpsBridge
    options:
      heading_level: 3
      show_root_heading: true

## Safety audit

::: strands_robots.audit.log_safety_event
    options:
      heading_level: 3
      show_root_heading: true

## Device connect

::: strands_robots.device_connect
    options:
      heading_level: 3
      members:
        - init_device_connect
        - init_device_connect_sync
        - RobotDeviceDriver
        - SimulationDeviceDriver
        - ReachyMiniDriver
