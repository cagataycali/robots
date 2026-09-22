---
description: What use_ros validates before a name reaches rclpy, which numeric options are refused ahead of the backend probe, and the operator gate on safety-critical command surfaces.
---

# ROS 2 safety: input validation and the command gate


Agent-supplied topic, service, and type names are validated against an
allowlist before reaching the rclpy graph/type API (alphanumerics plus
`_ / ~ {}` for names; `pkg/msg/Name` or `pkg/srv/Name` for types). Because the
tool never constructs a shell command or generates source, there is no
command-injection or `eval` surface to defend - the validation simply keeps
malformed names from reaching the ROS 2 client library. Backend and timeout
failures are returned as structured `{"status": "error"}` results rather than
raised exceptions.

The numeric options an action consumes are checked in the same place, ahead of
the backend probe, so a caller mistake reports identically whether or not a ROS 2
distro is sourced and a refusal happens before a publisher joins the graph:

| Option | Consumed by | Accepted values |
|--------|-------------|-----------------|
| `count` | `echo`, `publish` | a positive integer - it is a `range()` bound, so `0` sends nothing and `2.7` or `"3"` cannot be honored |
| `rate` | `publish` | a positive finite number of Hz - the inter-message period is `1 / rate`, so `0`, a negative value, `nan` and `inf` all leave the burst unthrottled instead of paced |
| `timeout` | `echo`, `service_call`, `action_send_goal` | a positive finite number of seconds - `0` and negatives wait for nothing, `inf` never expires |

`timeout` is measured on a monotonic clock, so the budget you ask for is the budget you
get even if the host's wall clock is stepped mid-call by an NTP correction, a `date -s`
or a resume from suspend. A single `action_send_goal` deadline governs server discovery,
goal acceptance and result delivery on that one clock, which is also what keeps the
cancel sent on expiry from being cut short - it needs the executor pumped to leave the
process.

An option the requested action never reads is not second-guessed:
`use_ros(action="status", count=-1)` still reports the backend.

## Safety-critical command surfaces need operator approval

A robot is driven through three different verbs - `publish` to a topic,
`service_call` to a service and `action_send_goal` to an action server - so the
gate is keyed on the surface **name** and consulted from all three. It is also
consulted from every **transport** that reaches the graph, not just this one:
`use_rtps` publishes over raw RTPS and `use_rosbridge` over a WebSocket, and a
`Twist` on `/cmd_vel` moves the same base whichever of the three wrote it. The
blocklist and the approval decision therefore have a single owner
(`strands_robots._command_gate`) rather than a copy per tool, so a surface
refused on one transport cannot be sent on another under a different tool name. An agent
asked to "drive forward" reaches for whichever verb fits the interface it found
on the graph, so gating `publish` alone would leave `/navigate_to_pose` (a ROS 2
action) and `/emergency_stop` (usually a `std_srvs/srv/Trigger` service)
unenforceable. These surfaces are blocked by default:

| Surface | Usually reached by |
|---------|--------------------|
| `/cmd_vel`, `/cmd_vel_unstamped`, `/manual_drive` | `publish` |
| `/joint_command`, `/joint_trajectory`, `/joint_trajectory_controller/joint_trajectory` | `publish` |
| `/emergency_stop`, `/e_stop` | `service_call`, sometimes `publish` |
| `/motor_enable`, `/enable_motor`, `/disable_motor` | `service_call` |
| `/vehicle_state`, `/enable_state` | `service_call` |
| `/navigate_to_pose`, `/follow_path` | `action_send_goal` |

Matching is on the final path segment, so a namespaced form
(`/my_robot/cmd_vel`, `/fleet/robot1/emergency_stop`, and the DeepRacer's
`/webserver_pkg/manual_drive`, `/ctrl_pkg/vehicle_state`,
`/ctrl_pkg/enable_state`) is caught while a lookalike (`/cmd_vel_evil`,
`/joint_trajectory_status`) is not. The name is compared in the
form rclpy resolves it to, so the unrooted `cmd_vel` and the trailing-separator
`/cmd_vel/` are the same surface as `/cmd_vel`. Case is deliberately **not**
folded: ROS 2 graph names are case-sensitive, so `/CMD_VEL` is a genuinely
different topic that no `/cmd_vel` subscriber receives, and refusing it would
block a legitimate surface without closing a path to the robot.

Three ways through the gate, consulted in this order:

| Mode | Mechanism |
|------|-----------|
| Interactive (default) | `tool_context.interrupt()` prompts the operator; reply `y` to approve |
| Headless allowlist | `STRANDS_ROS2_COMMAND_ALLOW=/cmd_vel,/follow_path` pre-approves those surfaces and every namespaced surface sharing a base name with one of them; a surface whose base name no entry lists stays gated |
| Fully trusted | `BYPASS_TOOL_CONSENT=true` allows every blocked surface with a WARNING log |

Both lists are matched by **base name** as well as by exact name, after a
leading/trailing `/` is normalised away: a `/cmd_vel` entry matches
`/robot_b/cmd_vel` too. On the blocklist that breadth is the point - one entry
has to catch every namespaced drive topic in the graph. On the pre-approval list
it is the same breadth pointing the other way, so `STRANDS_ROS2_COMMAND_ALLOW=/cmd_vel`
lifts the gate on **every** robot's drive topic, not just the one being driven.
Name the namespace (`/turtle1/cmd_vel`) when the approval should cover one robot,
and list each surface when it should cover several. Case is never folded:
`/CMD_VEL` is a different topic that no `/cmd_vel` subscriber receives.

The gate **fails closed**: with no `tool_context` (outside an agent loop), or when
`interrupt()` is unavailable, the command is refused and the error names both
environment variables. Only the operator's approve/deny verdict is read - the
reply text is never echoed back into the agent's context.

The operator is asked **before** the transport takes its process-wide lock, so a
pending decision does not stall an unrelated read on the same graph - an odometry
`echo`, a scan, a second robot sharing the transport - for however long the human
takes to answer. All three transports consult the gate at that same point.

The reply is recorded in the local safety audit log instead, on both outcomes.
That matters because only `y` / `yes` / `approve` / `approved` count as approval,
so a reply that carries a reason (`n - not while the cell door is open`) is always
a decline - and the audit row is the one place that reason survives. An approval
is recorded too: whether a human authorised an agent to reach a physical surface
is the first thing an incident review asks. Make sure your deployment captures and
retains that log; see the [audit log](../security/audit-log.md).

Anything that wraps `use_ros` has to forward that context or it inherits the
fail-closed path for every command it sends. `RosBridgedRobot` does: its
`drive_<node>` / `stop_<node>` / `navigate_<node>` tools are declared
`@tool(context=True)` and hand the context on, so an agent driving a bridged
robot prompts the operator. A **programmatic** `robot.drive(...)` has no operator
to prompt and is refused unless the surface is pre-approved - scripts and
unattended demos set `STRANDS_ROS2_COMMAND_ALLOW` for the topics they drive.

Reading is never gated: `echo`, `info` and the `list_*` queries work on a blocked
surface, so telemetry stays available to the agent. The gate also runs *after* the
action's required arguments are validated, so an operator is never asked to
approve a call that could not have run.

