# ROS 2

At the end of this page you know the three ways an agent or a `Robot` reaches a ROS 2 graph, which one fits your machine, how a real arm becomes a ROS 2 participant in both directions, and where the operator gate sits on every one of them.

```python title="sketch"
from strands import Agent
from strands_robots import use_ros, use_rosbridge, use_rtps

agent = Agent(tools=[use_rtps])                  # no ROS install needed on this machine
agent("List the topics on the graph, then echo /odom once.")
```

## Three transports

| tool | wire | needs on this machine | reaches | verbs |
|---|---|---|---|---|
| `use_ros` | in-process `rclpy` | a sourced ROS 2 distro (`source /opt/ros/jazzy/setup.bash`); `rclpy` is not on PyPI | any ROS 2 graph, with services and actions | `status`, `list_topics`, `list_nodes`, `list_services`, `list_actions`, `info`, `echo`, `publish`, `service_call`, `action_send_goal` |
| `use_rosbridge` | WebSocket to `rosbridge_server` on the robot | `pip install 'strands-robots[rosbridge]'` (roslibpy), nothing else; works from macOS and CI | ROS 1 and ROS 2 robots running rosbridge with `rosapi`; port 9090 | `status`, `list_topics`, `list_services`, `echo`, `publish`, `service_call` |
| `use_rtps` | raw DDS/RTPS as a first-class participant | `pip install 'strands-robots[ros2]'` (cyclonedds); no distro, no `rclpy` | every ROS 2 distro (Humble, Jazzy, Rolling) over one implementation; topics only | `status`, `types`, `advertise`, `publish`, `subscribe`, `echo` |

Pick `use_ros` when you are on the robot or a ROS workstation and need services or actions. Pick `use_rosbridge` when the robot already runs rosbridge (Yahboom images do) and you are on a laptop. Pick `use_rtps` when there is no ROS install anywhere near the agent, or when the agent must *be* a robot: an RTPS participant can advertise topics a real node consumes and subscribe to command topics, which `rclpy` as a client cannot do without a node.

Types are resolved dynamically on `use_ros` (`rosidl_runtime_py`, any interface installed in the distro) and on `use_rosbridge` (ROS 1 style two-segment names, `geometry_msgs/Twist`). `use_rtps` must own a type definition locally, so it ships a curated IDL bundle of common messages (`strands_robots.rtps.idl`, listed by `action="types"`); a custom message is out of scope until dynamic types in cyclonedds mature. A bare participant also carries no ROS 2 node name and no type hash, so graph metadata differs even though what a subscriber reads is identical.

rosbridge is unauthenticated by default. Use it on a network you trust.

## The gate is the same on all three

A `publish`, `service_call` or `action_send_goal` aimed at a blocklisted name goes through `strands_robots._command_gate.gate_command`, whichever transport carries it. The blocklist matches the final path segment, so `/cmd_vel` covers `/robot1/cmd_vel`:

`/cmd_vel`, `/cmd_vel_unstamped`, `/manual_drive`, `/joint_command`, `/joint_trajectory`, `/joint_trajectory_controller/joint_trajectory`, `/emergency_stop`, `/e_stop`, `/motor_enable`, `/enable_motor`, `/disable_motor`, `/vehicle_state`, `/enable_state`, `/navigate_to_pose`, `/follow_path`.

Reads are never gated; `use_rtps`'s `advertise` is not either (it creates a publisher and writes nothing). `STRANDS_ROS2_COMMAND_ALLOW` pre-approves exact names, comma-separated (`*` matches nothing here on purpose); `BYPASS_TOOL_CONSENT=true` lifts the gate with a warning. Each transport consults the gate at one fixed point: after the backend probe (a graph you cannot reach never prompts), after the verb's own arguments are checked, and before the executor lock or the WebSocket dial, so a human deciding holds nothing another caller needs. The whole decision path is on [agents](agents.md).

## A real arm on the graph

```python title="sketch"
arm = Robot("so101", mode="real", port="/dev/ttyACM0",
            ros2_bridge=True, ros2_transport="rtps", ros2_domain=0, ros2_commands=True,
            dds_security_config={"identity_ca": "file:ca.pem", "certificate": "file:arm.pem",
                                 "private_key": "file:arm.key", "governance": "file:gov.p7s",
                                 "permissions": "file:perm.p7s"})
```

`ros2_bridge=True` on the hardware `Robot` publishes `/<robot>/joint_states` (`sensor_msgs/msg/JointState`) and `/<robot>/<camera>/image_raw` (`sensor_msgs/msg/Image`, `rgb8`) from the control loop, and subscribes `/<robot>/joint_command` (`JointState`), forwarding each message into `send_action` so MoveIt, a teleop node or a trajectory replayer can drive the arm. `ros2_transport` is `"rclpy"` (`HardwareRosBridge`) or `"rtps"` (`HardwareRtpsBridge`); `ros2_commands=False` makes it publish-only; `joint_limits=` clamps inbound targets per joint.

On the `rtps` transport an enabled command surface lets any DDS participant on the domain move the arm, so it requires `dds_security_config` (identity CA, certificate, private key, governance, permissions, each a non-empty path or `file:`/`data:` URI) or the explicit opt-out `STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE=1`. A missing `rclpy` on the `rclpy` transport is a named refusal that suggests `ros2_transport='rtps'`.

## A ROS robot on the mesh

`RosBridgedRobot` (rclpy), `RosbridgeRobot` (WebSocket) and `RtpsRobot` (DDS) in `strands_robots.mesh` wrap a mobile base that speaks `/cmd_vel` and `/odom` as a mesh peer, so it answers `status`, `execute` and `stop` next to the arms ([fleet](mesh/fleet.md)). `AckermannRosRobot` does the same for a car-like base. The Yahboom M3 Pro driver is the shipped example of a whole robot driven through its ROS 2 graph ([drivers](hardware/drivers.md)).
