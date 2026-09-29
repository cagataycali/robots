---
description: A robot and the @tool functions in a Strands Agent, and what happens when the model asks a real robot to move.
---

# Agents

At the end of this page a Strands `Agent` holds a robot as one of its tools, you know which of the {{n:tools}} `@tool` functions to mount beside it, and what happens when the model asks a real robot to move: the gate, the variable that pre-approves it, the audit row.

```python
from strands import Agent
from strands_robots import Robot
from strands_robots import pose_tool, robot_mesh   # pose_tool needs pyserial

arm = Robot("so101")                       # sim, the default
agent = Agent(tools=[arm, pose_tool, robot_mesh])
print(sorted(agent.tool_names))   # ['pose_tool', 'robot_mesh', 'so101_sim']
```

## A robot is a tool

`Robot(...)` returns an object with the Strands `AgentTool` surface (`tool_name`, `tool_type`, `tool_spec`, `stream`), so it goes into `Agent(tools=[...])` like any tool; the model sees one tool per robot with an `action` field:

| mode | tool name | actions the model sees |
|---|---|---|
| `mode="sim"` (default) | `<name>_sim` | `get_robot_state`, `set_joint_positions`, `move_to`, `run_policy`, `render`, `step` and the world API |
| `mode="real"` | the robot's name | `get_state`, `get_robot_state`, `list_cameras`, `render`, `execute`, `start`, `status`, `stop` |

`execute` runs one rollout to completion, `start` runs it in the background; `status` and `stop` follow. Two `Robot("so101")` in one agent collide on `so101_sim`; name them with `tool_name=`.

## The tools around the robot

`strands_robots.tools` lazy-loads every tool; the ones mounted most:

| tool | what it does | gated? |
|---|---|---|
| `use_lerobot` | record, replay, train, inspect datasets ([record](data/record.md)) | no |
| `run_policy`, `train_policy` | build a policy from any provider, run or train it | no |
| `pose_tool` | named poses and joint moves on a Feetech arm | motion verbs |
| `serial_tool` | raw servo bus reads and writes | writes |
| `robot_mesh` | the fleet ([fleet](mesh/fleet.md)): read with `peers`, `status`, `inbox`; act with the six verbs in the gate table | those six |
| `use_ros`, `use_rosbridge`, `use_rtps` | a ROS 2 graph, three transports ([ROS 2](ros2.md)) | blocklisted surfaces |
| `use_unitree`, `g1_*` | Unitree G1 locomotion and arm verbs ([unitree](hardware/unitree.md)) | motion RPCs |
| `reachy_*` | Reachy Mini head, antennas, sound ([reachy](hardware/reachy-mini.md)) | no |
| `load_episode`, `sample_frames`, `write_label` | judge episodes ([label and judge](data/label-and-judge.md)) | no |

## The operator gate

Every policy rollout, ROS command, serial write and pose move the model asks for goes through `strands_robots._command_gate.gate_motion`, which decides in order:

1. The tool's allowlist variable names the command: allow silently.
2. `BYPASS_TOOL_CONSENT=true`: allow and log a WARNING.
3. No `tool_context` (outside an agent, or a host that cannot interrupt): refuse, naming the variable and value that pre-approve the call.
4. Otherwise raise a Strands interrupt named `<tool>-command-approval`. The operator answers out of band: `y`, `yes`, `approve` or `approved` proceeds, anything else declines; the model never sees it.

Reading and stopping are never gated; nor, at this commit, is the native drivers' `move_to` (`driver="strands"`); [first real arm](../start/first-real-arm.md#the-operator-gate) says so.

| caller | gated verbs | allowlist variable |
|---|---|---|
| `Robot(mode="real")` tool | `execute`, `start` | `STRANDS_ROBOT_COMMAND_ALLOW` |
| `pose_tool` | `move_motor`, `move_multiple`, `incremental_move`, `load_pose`, `reset_to_home` | `STRANDS_POSE_COMMAND_ALLOW` |
| `serial_tool` | bus writes | `STRANDS_SERIAL_COMMAND_ALLOW` |
| `use_unitree` | motion RPCs (`loco.SetVelocity`, ...) | `STRANDS_UNITREE_COMMAND_ALLOW` |
| `use_ros`, `use_rosbridge`, `use_rtps` | `publish`, `service_call`, `action_send_goal` on a blocklisted name (`/cmd_vel`, `/e_stop`, ...) | `STRANDS_ROS2_COMMAND_ALLOW` (by base name: `/cmd_vel` covers every namespace) |
| `robot_mesh` | `emergency_stop`, `broadcast`, `tell`, `send`, `stop`, `rpc` | `STRANDS_MESH_HITL_ACTIONS` selects the set |

Values are the verbs or targets the tool matches (`execute`, `/cmd_vel`, `loco.SetVelocity`), comma-separated; `*` pre-approves every command where the tool honours it; `=1` or `=true` pre-approve nothing.

## What a refusal looks like

No hardware needed: the gate precedes the port.

```python
from strands_robots.tools.pose_tool import pose_tool

result = pose_tool(action="move_motor", motor_name="shoulder_pan", position=10.0, port="/dev/ttyACM0")
print(result["content"][0]["text"])
```

```text
pose_tool: 'move_motor' moves the arm on '/dev/ttyACM0' (motor_name=shoulder_pan position=10.0); it needs operator approval before any goal position is sent. No tool_context available for operator approval. Set STRANDS_POSE_COMMAND_ALLOW=move_motor (or STRANDS_POSE_COMMAND_ALLOW=* for every pose_tool command; comma-separated) or BYPASS_TOOL_CONSENT=true to allow in headless mode.
```

Inside an agent the same call pauses the turn; the dashboard's `MotionInterruptHook` asks the operator in the browser and deposits a grant keyed on the exact tool input, which the gate spends rather than ask twice.

## Audit

Every operator verdict is one JSONL row in `~/.strands_robots/mesh_audit.jsonl` (`STRANDS_MESH_AUDIT_DIR` moves it): event `llm_tool_action`, source `<tool>_tool`, `action`, `target`, `success`, `detail: "operator approved: 'y'"` or `"operator declined: ..."`; with `STRANDS_MESH_AUDIT_PSK` set, an HMAC ([security](security.md)).

## Posture

There is no `dry_run` flag; the dry run is `mode="sim"`, the same tool surface over a MuJoCo arm. On hardware keep the gate on and pre-approve only verbs you have watched run; `BYPASS_TOOL_CONSENT=true` is for a CI box with no robot.
