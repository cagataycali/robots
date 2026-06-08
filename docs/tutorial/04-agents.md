---
description: Hand a Robot() to a Strands Agent and control it with plain English.
---

# 4 — AI agents

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so100")
agent = Agent(tools=[robot])   # one tool, 60+ actions

agent("Add a red cube on the table and pick it up")
# → calls robot.add_object(...) then robot.run_policy(...)
```

```bash
pip install strands-agents "strands-robots[sim-mujoco]"
```

## Add more tools

```python
from strands_robots.tools import gr00t_inference, pose_tool

agent = Agent(tools=[robot, gr00t_inference, pose_tool])
agent("Start a GR00T server on port 5555 with so100_dualcam, "
      "then pick up the cube using groot")
```

All files in `strands_robots/tools/` are `@tool`-decorated functions the agent uses directly.

## Multi-turn

```python
agent("Set up a blue ball and a red cube in opposite corners")
agent("Add a side camera looking at the workspace")
agent("Run the mock policy for 5 seconds and report which objects are still on the table")
```

## Sim → real

```python
robot = Robot("so100", mode="real",                                    # requires hardware
              cameras={"wrist": {"type": "opencv",
                                 "index_or_path": "/dev/video0", "fps": 30}})
agent = Agent(tools=[robot])
agent("Pick up the cube")
```

The agent sees the same tool spec in both modes; only the implementation changes.

## Common patterns

| Instruction | Action chain |
|-------------|--------------|
| "Reset the world" | `reset` |
| "Add a 5cm red cube" | `add_object(shape='box', size=[0.025]*3, color=[1,0,0,1])` |
| "Take a picture" | `render` |
| "Run the policy" | `run_policy(robot_name='so100', ...)` |
| "What's in the scene?" | `list_objects` + `get_state` |
| "Try 10 episodes, report success" | `eval_policy(robot_name='so100', n_episodes=10)` |
| "Record a session" | `start_recording` → `run_policy` → `stop_recording` |

Inspect the full spec: `print(robot.tool_spec)` — JSON schema with all 60+ actions.

## See also

- [Tutorial 5 — Multi-robot](05-multi-robot.md) — agent coordinates multiple robots.
- [Tutorial 8 — Real hardware](08-real-hardware.md) — real-arm bring-up checklist.
- [Hardware tools](../hardware/tools.md) — all `@tool` helpers.
- [Strands Agents documentation](https://strandsagents.com/) — provider setup, advanced patterns.
