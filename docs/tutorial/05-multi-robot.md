# Chapter 5: Multi-Robot

**Time:** 20 minutes · **Hardware:** None needed · **Level:** Advanced

One robot is useful. Two robots working together can do things neither can alone — bimanual manipulation, coordinated assembly, robot-to-robot handoffs.

---

## Two Robots, One Agent

```python
from strands import Agent
from strands_robots import Robot

left_arm = Robot("so100", tool_name="left_arm")
right_arm = Robot("so100", tool_name="right_arm")

agent = Agent(tools=[left_arm, right_arm])
agent("Use left_arm and right_arm together to fold the towel")
```

Each robot gets a unique `tool_name`. The agent knows them by name and can coordinate them.

---

## Bimanual Robots

Some robots are bimanual by design — two arms in one system:

```python
robot = Robot("aloha")
agent = Agent(tools=[robot])

agent("Pick up the cup with the left arm and pour into the bowl held by the right arm")
```

The ALOHA system manages both arms internally. The agent talks to it as one tool.

---

## Mixed Robot Types

You're not limited to the same type:

```python
arm = Robot("so100", tool_name="arm")
mobile = Robot("lekiwi", tool_name="base")

agent = Agent(tools=[arm, mobile])
agent("Drive the base to the table, then use the arm to pick up the object")
```

---

## Robot Mesh Networking

For robots on different machines, use Zenoh mesh:

```python
from strands_robots import robot_mesh

# Robot 1 (on machine A)
robot_mesh.publish("arm_1", observation)

# Robot 2 (on machine B)
obs_from_arm1 = robot_mesh.subscribe("arm_1")
```

P2P communication with auto-discovery. No server needed.

---

## What You Learned

- ✅ Multiple robots with unique `tool_name`s
- ✅ Bimanual robots like ALOHA as single tools
- ✅ Mixed robot types in one agent
- ✅ Zenoh mesh for cross-machine coordination

---

**Next:** [Chapter 6: Recording Data →](06-recording.md) — Record demonstrations for training.
