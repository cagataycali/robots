"""D16: three ROS 2 transports, one operator gate.

Left: the agent with use_ros, use_rosbridge and use_rtps as three cards, what each needs. Right: the
ROS 2 graph (topics, services, actions). Between them the gate (the one accent element): a publish,
service_call or action_send_goal aimed at a blocklisted name goes through gate_command whichever
transport carries it; reads are never gated. Every name is on learn/ros2.md.
"""
from scene import Scene

L, LW = 60, 380
G, GW = 480, 240
R, RW = 820, 320
ROWS = [
    ("use_ros", "in-process rclpy; a sourced distro; services and actions", "rclpy"),
    ("use_rosbridge", "WebSocket to rosbridge_server; roslibpy; works from macOS", "[rosbridge]"),
    ("use_rtps", "raw DDS participant; cyclonedds; no distro; topics only", "[ros2]"),
]


def scene() -> Scene:
    s = Scene(
        "d16_ros2_transports",
        "Three transports, one gate",
        "use_ros, use_rosbridge and use_rtps reach the same graph; a command on a blocklisted name waits for a yes on all three.",
        "Left column, three cards under the agent: use_ros, in-process rclpy with a sourced ROS 2 distro, services "
        "and actions; use_rosbridge, a WebSocket to rosbridge_server on the robot through roslibpy, works from a "
        "laptop; use_rtps, raw DDS as a first-class participant through cyclonedds, no distro, topics only. Middle, "
        "the one green element: gate_command in strands_robots._command_gate, which every publish, service_call or "
        "action_send_goal aimed at a blocklisted name passes through, whichever transport carries it; reads are "
        "never gated. Right: the ROS 2 graph, topics such as /cmd_vel and /joint_states, services, actions; and a "
        "real arm on it, Robot(ros2_bridge=True) publishing /<robot>/joint_states and /<robot>/<camera>/image_raw. "
        "Footnote: STRANDS_ROS2_COMMAND_ALLOW pre-approves names by base name; rosbridge is unauthenticated by "
        "default.",
        h=600,
    )

    s.section(L, 122, "the agent's transports")
    s.box(L, 134, LW, 56, "Agent(tools=[use_ros | use_rosbridge | use_rtps])",
          "one tool per transport; pick by what the machine has", size=13.5, subsize=11.5)
    y = 214
    for i, (name, sub, extra) in enumerate(ROWS):
        s.box(L, y, LW, 92, name, sub, size=14, subsize=11.5, id=name)
        s.chips(L + 14, y + 60, [extra])
        s.arrow([(L + LW, y + 46), (G, y + 46)], id=f"cmd_{name}")
        s.motion.append((f"cmd_{name}", "flow"))
        y += 108

    # ---------------------------------------------------------------- the gate (the one green element)
    s.box(G, 214, GW, 308, "gate_command", None, accent=True, size=14, id="gate")
    s.para(G + 14, 262, "every publish, service_call or action_send_goal aimed at a blocklisted base name waits for "
           "a yes, whichever transport carries it", GW - 28, size=11.5, cls="grot text")
    s.chips(G + 14, 346, ["/cmd_vel", "/joint_command"])
    s.chips(G + 14, 376, ["/emergency_stop"])
    s.chips(G + 14, 406, ["/motor_enable"])
    s.para(G + 14, 458, "reads are never gated, nor is advertise", GW - 28, size=11.5, cls="grot muted")
    s.chips(G + 14, 486, ["STRANDS_ROS2_COMMAND_ALLOW"])
    s.motion.insert(0, ("gate", "pulse"))

    # ---------------------------------------------------------------- the graph
    s.arrow([(G + GW, 240), (R, 240)], id="to_graph")
    s.text((G + GW + R) / 2, 230, "after a yes", cls="mono muted", size=10.5, anchor="middle")
    s.motion.append(("to_graph", "flow"))
    s.section(R, 122, "the ROS 2 graph")
    s.box(R, 134, RW, 170, "topics, services, actions",
          "Humble, Jazzy or Rolling; the payload is identical on every transport",
          size=14, subsize=12)
    s.chips(R + 14, 208, ["/cmd_vel", "/odom", "/joint_states"])
    s.chips(R + 14, 238, ["list_topics", "list_nodes", "list_services"])
    s.box(R, 330, RW, 192, "a real arm on the graph",
          "ros2_bridge=True publishes from the control loop; ros2_commands=True subscribes",
          size=14, subsize=12)
    s.chips(R + 14, 408, ["/<robot>/joint_states"])
    s.chips(R + 14, 438, ["/<robot>/<camera>/image_raw"])
    s.chips(R + 14, 468, ["dds_security_config"])

    s.footnote(574, "rosbridge is unauthenticated by default: use it on a network you trust.")
    return s
