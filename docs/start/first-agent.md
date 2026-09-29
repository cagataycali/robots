---
description: A Strands Agent moves the simulated SO-101 from a sentence, then stops at the operator gate in front of a real arm.
---

# First agent

At the end of this page a Strands Agent has moved the simulated arm from a sentence you typed, and you have watched the same agent stop at the operator gate in front of a real one. The sim fences run without a model. The two fences that call `agent("...")` need a model provider configured for `strands-agents`; Bedrock is the default.

## The robot is a tool

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so101")
agent = Agent(tools=[robot], callback_handler=None)
print(agent.tool_names)
spec = robot.tool_spec
print(spec["description"][:160])
print(len(spec["inputSchema"]["json"]["properties"]["action"]["enum"]), "actions")
result = agent.tool.so101_sim(action="get_robot_state")
print(result["content"][0]["text"].splitlines()[1])
robot.cleanup()
```

You should see:

```text
['so101_sim']
Programmatic MuJoCo simulation environment (stateful session). One world per instance. The world is ALREADY CREATED and holds robot(s) 'so101' (6 joints: 1, 2,
77 actions
1 (shoulder_pan): pos=0.0000, vel=0.0000
```

Nothing was wrapped. The object `Robot()` returned is a Strands `AgentTool`: it carries a name (`so101_sim` in sim, `so101` on hardware, or whatever `tool_name=` says), a description the model reads, and one `action` enum. `agent.tool.so101_sim(...)` calls it directly, no model in the loop, and returns the same envelope the methods on [First robot](first-robot.md) returned. Two robots in one agent need two names: `Robot("so101", tool_name="left")`.

## Ask in words

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so101")
agent = Agent(tools=[robot], callback_handler=None)
result = agent("Read the arm's joint state, then move joint 1 to 0.5 rad and report where the gripper ended up.")
print(result)
robot.cleanup()
```

The model calls `get_robot_state`, then `set_joint_positions` or `actuate_robot` with some `step` calls, then `get_robot_state` again, and writes what it found. On one run on this checkout the answer reported the gripper moving from `[+0.020, -0.376, +0.259]` to `[-0.150, -0.335, +0.237]`, a 17 cm sweep along -X for a 0.5 rad pan. Your model will pick its own actions and words; the joint it reports back is read from physics, not invented.

Other tools mount the same way and are listed in the [tool reference](../reference/tools.md). `pose_tool` talks to a Feetech bus and needs `pip install pyserial` on top of the Start install:

```python
from strands import Agent
from strands_robots import Robot, run_policy, pose_tool, download_assets

robot = Robot("so101")
agent = Agent(tools=[robot, run_policy, pose_tool, download_assets], callback_handler=None)
print(agent.tool_names)
robot.cleanup()
```

```text
['so101_sim', 'run_policy', 'pose_tool', 'download_assets']
```

## The gate in front of real motion

A `mode="real"` robot built through the lerobot driver has eight actions. Six read or halt and are never gated: `get_state`, `get_robot_state`, `list_cameras`, `render`, `status`, `stop`. Two move: `execute` and `start` dispatch a policy rollout to real actuators, and both stop for a human first. This runs on a laptop because `mock=True` gives the lerobot driver a mocked servo bus:

```python
from strands import Agent
from strands_robots import Robot

arm = Robot("so101", mode="real", port="/dev/null", mock=True)
agent = Agent(tools=[arm], callback_handler=None)
result = agent("Run the mock policy on so101 for 2 seconds with the instruction 'wave'. Call the tool directly.")
print(result.stop_reason)
for interrupt in result.interrupts:
    print(interrupt.name)
    print(interrupt.reason["warning"])
    responses = [{"interruptResponse": {"interruptId": interrupt.id, "response": "n"}}]
result = agent(responses)
print(result.stop_reason)
arm.cleanup()
```

You should see the agent pause instead of finishing, and then finish once you answer:

```text
interrupt
robot-command-approval
'execute' drives the real robot 'so101' for up to 2s with 'wave' (policy mock built in this process, no server); it needs operator approval before it is dispatched. Note: MockPolicy does not read the instruction. Its actions - a test motion on every joint - are commanded to the robot whatever the task says; no status or completion that follows will mean the task was performed. Reply 'y' to approve, anything else to deny.
end_turn
```

The warning says how long the arm may move and, when the policy does not read the instruction, that the words will not shape the motion. `"y"` approves and the rollout is dispatched; anything else denies and nothing moves. `interrupt.reason["how_to_answer"]` carries the resume line, so a script that prints a paused result prints how to continue it.

With no agent, the same call is refused outright:

```python
import asyncio
from strands_robots import Robot

arm = Robot("so101", mode="real", port="/dev/null", mock=True)

async def call(action, **fields):
    tool_use = {"toolUseId": "demo", "name": arm.tool_name, "input": {"action": action, **fields}}
    async for event in arm.stream(tool_use, {}):
        return event.tool_result

refused = asyncio.run(call("execute", instruction="wave", policy_provider="mock", duration=2))
print(refused["status"])
print(refused["content"][0]["text"].split("No tool_context")[1])
arm.cleanup()
```

```text
error
 available for operator approval. Set STRANDS_ROBOT_COMMAND_ALLOW=execute (or STRANDS_ROBOT_COMMAND_ALLOW=* for every robot command; comma-separated) or BYPASS_TOOL_CONSENT=true to allow in headless mode.
```

`STRANDS_ROBOT_COMMAND_ALLOW` names pre-approved actions (`execute`, `start`, or `*`); with nobody to ask the call fails closed; every answer lands in the audit log. [The operator gate](../learn/agents.md#the-operator-gate) gives the full order, the other tools' allow variables, and the one path that is not gated yet.

## Where next

[Agents](../learn/agents.md) covers what the model sees, multi-robot agents and the dashboard's approval flow. [Policies](../learn/policies/index.md) replaces `mock` with a model that acts on the words.
