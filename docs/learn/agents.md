# Agents

At the end of this page you have a Strands `Agent` that holds a robot as one of its tools, you know which of the {{n:tools}} `@tool` functions to mount next to it, and you know exactly what happens when the model asks a real robot to move: the operator gate, the environment variable that pre-approves it, and the audit row it leaves behind.

```python
from strands import Agent
from strands_robots import Robot
from strands_robots.tools import pose_tool, robot_mesh   # pose_tool needs `pip install pyserial`

arm = Robot("so101")                       # simulation, the default mode
agent = Agent(tools=[arm, pose_tool, robot_mesh])
print(sorted(agent.tool_names))            # ['pose_tool', 'robot_mesh', 'so101_sim']
```

## A robot is a tool

`Robot(...)` returns an object that satisfies the Strands `AgentTool` surface (`tool_name`, `tool_type`, `tool_spec`, `stream`). You pass it to `Agent(tools=[...])` like any other tool. The model sees one tool per robot with an `action` field:

| mode | tool name | actions the model sees |
|---|---|---|
| `mode="sim"` (default) | `<name>_sim` | the simulation verbs: `get_robot_state`, `set_joint_positions`, `move_to`, `run_policy`, `render`, `step`, and the rest of the world API |
| `mode="real"` | the canonical robot name | `get_state`, `get_robot_state`, `list_cameras`, `render`, `execute`, `start`, `status`, `stop` |

`execute` runs one policy rollout to completion and `start` runs it in the background; `status` and `stop` follow it. Two `Robot("so101")` in one agent collide on `so101_sim`; pass `tool_name=` to name them apart.

## The tools around the robot

`strands_robots.tools` lazy-loads every tool, so `from strands_robots.tools import use_lerobot` costs nothing until the call. The ones you mount most:

| tool | what it does | gated? |
|---|---|---|
| `use_lerobot` | record, replay, train, inspect LeRobot datasets ([record](data/record.md)) | no |
| `run_policy` / `train_policy` | build a policy from any provider, run or train it | no |
| `pose_tool` | named poses and direct joint moves on a Feetech arm | motion verbs |
| `serial_tool` | raw servo bus reads and writes | writes |
| `robot_mesh` | talk to the fleet: `peers`, `tell`, `send`, `broadcast`, `emergency_stop` ([fleet](mesh/fleet.md)) | six actuating actions |
| `use_ros`, `use_rosbridge`, `use_rtps` | a ROS 2 graph over three transports ([ROS 2](ros2.md)) | blocklisted surfaces |
| `use_unitree` and `g1_*` | Unitree G1 locomotion and arm verbs ([unitree](hardware/unitree.md)) | motion RPCs |
| `reachy_*` | Reachy Mini head, antennas, sound ([reachy-mini](hardware/reachy-mini.md)) | no |
| `load_episode`, `sample_frames`, `write_label` | judge recorded episodes ([label and judge](data/label-and-judge.md)) | no |

The generated list with every docstring is [reference/tools](../reference/tools.md).

## The operator gate

Every path from the model to a physical actuator goes through one function, `strands_robots._command_gate.gate_motion`. It decides in this order:

1. The tool's allowlist variable names this command: allow silently.
2. `BYPASS_TOOL_CONSENT=true`: allow, and log a WARNING.
3. No `tool_context` (the tool was called outside an agent, or the host cannot interrupt): refuse, naming the variable and value that would pre-approve the call.
4. Otherwise raise a Strands interrupt named `<tool>-command-approval`. The operator answers out of band; `y`, `yes`, `approve` or `approved` lets the command proceed, anything else declines it. The reply never reaches the model.

Reading is never gated. Stopping is never gated: `stop`, `status` and `emergency_stop` on a robot tool always run.

| caller | gated verbs | allowlist variable |
|---|---|---|
| `Robot(mode="real")` tool | `execute`, `start` | `STRANDS_ROBOT_COMMAND_ALLOW` |
| `pose_tool` | `move_motor`, `move_multiple`, `incremental_move`, `load_pose`, `reset_to_home` | `STRANDS_POSE_COMMAND_ALLOW` |
| `serial_tool` | bus writes | `STRANDS_SERIAL_COMMAND_ALLOW` |
| `use_unitree` | motion RPCs such as `loco.SetVelocity` | `STRANDS_UNITREE_COMMAND_ALLOW` |
| `use_ros`, `use_rosbridge`, `use_rtps` | `publish`, `service_call`, `action_send_goal` on a blocklisted name (`/cmd_vel`, `/joint_trajectory`, `/e_stop`, ...) | `STRANDS_ROS2_COMMAND_ALLOW` |
| `robot_mesh` | `emergency_stop`, `broadcast`, `tell`, `send`, `stop`, `rpc` | `STRANDS_MESH_HITL_ACTIONS` selects the set |

Values are the verb or target spelling the tool matches (`execute`, `/cmd_vel`, `loco.SetVelocity`), comma-separated; `*` pre-approves every command of that tool where the tool honours it. `=1` or `=true` pre-approve nothing, and the refusal says so.

## What a refusal looks like

This runs without hardware: `pose_tool` gates the motion before it opens the port.

```python
from strands_robots.tools.pose_tool import pose_tool

result = pose_tool(action="move_motor", motor_name="shoulder_pan", position=10.0, port="/dev/ttyACM0")
print(result["content"][0]["text"])
```

```text
pose_tool: 'move_motor' moves the arm on '/dev/ttyACM0' (motor_name=shoulder_pan position=10.0); it needs operator approval before any goal position is sent. No tool_context available for operator approval. Set STRANDS_POSE_COMMAND_ALLOW=move_motor (or STRANDS_POSE_COMMAND_ALLOW=* for every pose_tool command; comma-separated) or BYPASS_TOOL_CONSENT=true to allow in headless mode.
```

Inside an agent the same call pauses the turn instead. The dashboard's `MotionInterruptHook` asks the operator in the browser and deposits a grant keyed on the exact tool input; the gate spends that grant (`consume_grant`) instead of asking twice.

## Audit

Every operator verdict, approved or declined, is one JSONL row in `~/.strands_robots/mesh_audit.jsonl` (`STRANDS_MESH_AUDIT_DIR` moves it): event `llm_tool_action`, source `<tool>_tool`, and a payload with `action`, `target`, `success` and `detail: "operator approved: 'y'"` or `"operator declined: ..."`. Set `STRANDS_MESH_AUDIT_PSK` and each row carries an HMAC. See [security](security.md).

## Posture

There is no `dry_run` flag on a robot tool at this commit. The dry run is `mode="sim"`: the same tool surface, the same policy code, a MuJoCo arm at the far end. Develop the agent there, then change one argument. On hardware, keep the gate on and pre-approve only the verbs you have watched run; `BYPASS_TOOL_CONSENT=true` is for a CI box with no robot attached.
