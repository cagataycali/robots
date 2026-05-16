---
description: Wire a Robot() into a Strands Agent and drive it with English. The whole point of this library.
---

# 4 — AI agents

The `Simulation` you've been driving manually is a Strands `AgentTool`. That means you
can hand it to a Strands `Agent` and the agent will call the simulation's actions for
you, picking the right one based on natural-language input.

This is the chapter where the library earns its name.

## TL;DR

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so100")            # default: simulation
agent = Agent(tools=[robot])      # robot is registered as one tool with 35+ actions

agent("Add a red cube on the table and pick it up")
# → agent calls robot.add_object(...) and robot.run_policy(...)
```

The agent reads the docstrings and tool spec from `Simulation`, decides which actions to
fire in what order, and supplies the parameters from the user's instruction. You write
no glue code.

## Setup

```bash
pip install strands-agents
pip install "strands-robots[sim-mujoco]"
```

You also need credentials for whichever LLM Strands Agents is configured to use (Bedrock
by default — see the
[Strands Agents docs](https://strandsagents.com/) for provider setup).

## Step 1 — instantiate

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so100")
agent = Agent(tools=[robot])
```

`Robot()` returns a Simulation (or HardwareRobot in `mode="real"`); both are
`AgentTool` subclasses, both expose action specs that the agent can read.

## Step 2 — talk to it

```python
agent("List the simulation actions you have available")
# → agent calls robot.get_features() or similar discovery action

agent("Add a red cube at position (0.3, 0, 0.025) with size 5cm")
# → agent calls robot.add_object(name="cube", type="box", size=[0.025, 0.025, 0.025],
#                                pos=[0.3, 0, 0.025], rgba=[1, 0, 0, 1])

agent("Render a frame from the default camera and save it as scene.png")
# → agent calls robot.render(...) and writes the frame to disk

agent("Use the mock policy to try to pick up the cube for 10 seconds")
# → agent calls robot.run_policy(instruction="pick up the cube",
#                                policy_provider="mock", duration=10.0)
```

The agent translates fuzzy English into precise tool calls. If the user instruction is
ambiguous it asks; if it's missing parameters it provides reasonable defaults; if the
action fails it can retry or report the error.

## Step 3 — give the agent more tools

A single robot is one tool. Add more tools when you want the agent to do things outside
the simulation:

```python
from strands import Agent
from strands_robots import Robot
from strands_robots.tools import gr00t_inference, pose_tool

robot = Robot("so100")
agent = Agent(tools=[robot, gr00t_inference, pose_tool])

agent("Start a GR00T server on port 5555 with the so100_dualcam config, "
      "then have the robot pick up the cube using groot")
```

The shipped tools (`tools/*.py`) are all `@tool`-decorated functions a Strands Agent can
use directly. See [Hardware tools](../hardware/tools.md) for the full list.

## Step 4 — multi-turn conversations

Strands Agent maintains per-conversation state. Use the same agent across multiple
turns to build up a scene:

```python
agent("Set up a scene with a blue ball and a red cube in different corners of a table")
agent("Now add a wrist camera to the robot")
agent("Run the mock policy for 5 seconds and tell me what objects are still on the table")
```

The agent remembers it added the ball and cube, knows about the wrist camera, and can
report the state after the rollout.

## Step 5 — sim → real with one kwarg

Same agent code. Different mode.

```python
# Same exact instruction; switch sim → real with mode="real".
robot = Robot("so100", mode="real", cameras={
    "wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30},
})
agent = Agent(tools=[robot])
agent("Pick up the cube")
```

The agent doesn't know whether it's controlling a sim or real arm. The action spec
(`Robot`'s tool surface) is the same — the implementation differs.

For the real-hardware bring-up checklist (calibration, camera setup, safety defaults),
see [Tutorial 8 — Real hardware](08-real-hardware.md).

## What the agent actually sees

The agent reads `Simulation`'s tool spec — every action with its parameters, types, and
docstrings. The agent's LLM uses this to route the user's instruction to the right
action. You can preview it:

```python
print(robot.tool_spec)   # JSON schema with all 35+ actions
```

This is also why the system works without any prompt engineering on the user side:
the action vocabulary is rich enough that "pick up the cube" maps to `run_policy(...)`,
"add a cube" maps to `add_object(...)`, "save a frame" maps to `render(...)`, etc.

## Common patterns

| Instruction | Action chain |
|-------------|--------------|
| "Reset the world" | `reset` |
| "Add a 5cm red cube" | `add_object(type='box', size=[0.025, 0.025, 0.025], rgba=[1,0,0,1])` |
| "Take a picture" | `render` → save to disk |
| "Run the policy" | `run_policy(...)` |
| "What's in the scene?" | `list_objects` + `list_robots` (sim) + `get_state` |
| "Try 10 episodes and report success rate" | `eval_policy(num_episodes=10)` |
| "Start recording, run the policy, stop recording" | `start_recording` → `run_policy` → `stop_recording` |

## Recap

- `Agent(tools=[Robot()])` is the whole integration.
- The agent sees Simulation's 35+ actions and routes user instructions to them.
- Mix in `tools/*.py` for non-sim work (camera bring-up, GR00T container management).
- Sim → real is one `mode="real"` kwarg away.

## See also

- [Tutorial 5 — Multi-robot](05-multi-robot.md) — multiple `Robot()` instances on a
  Zenoh mesh, agent coordinates them.
- [Tutorial 8 — Real hardware](08-real-hardware.md) — the real-arm checklist.
- [Hardware tools](../hardware/tools.md) — all the `@tool` helpers an agent can use.
- [Strands Agents documentation](https://strandsagents.com/) — provider setup, prompt
  templates, advanced patterns.
