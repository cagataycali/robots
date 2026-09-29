---
description: Every agent-callable tool, one table per area, with the action values a dispatching tool accepts, read from the source.
---

# Tools

Every agent-callable tool, one table per area, read from the `@tool` decorators at build time. After this page you know which tool does what, where it lives, and the action values a dispatching tool accepts, so you can hand an `Agent` the right subset.

Tools are passed as functions:

```python title="sketch"
from strands import Agent
from strands_robots import Robot, use_lerobot, lerobot_camera, robot_mesh

robot = Robot("so101", mode="sim")
agent = Agent(tools=[robot, use_lerobot, lerobot_camera, robot_mesh])
```

A `Robot` instance is itself a tool (`robot.tool_spec`); the functions below add cameras, training, the serial bus, the mesh and the transports around it. Tools that move hardware go through [the operator gate](../learn/agents.md#the-operator-gate); the gate is part of the tool, not of this table.

Signatures and docstrings of the top-level tools: [API tools](api/tools.md).

{{tools_ref}}
