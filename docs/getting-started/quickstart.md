---
description: Five minutes from pip install to robot moving. Condensed version of tutorial chapters 1-4.
---

# Quickstart

Five minutes. One terminal. The library running a robot end-to-end.

## Install

```bash
pip install "strands-robots[sim-mujoco]"
```

That pulls in `mujoco` and the rest of the simulation stack. CPU-only — no GPU
required.

## Spawn a robot

```python
from strands_robots import Robot

sim = Robot("so100")
print(sim.tool_name_str)   # 'so100_sim'
```

## Render a frame

```python
import imageio.v3 as iio

frame = sim.render(width=640, height=480)["frame"]
iio.imwrite("first_frame.png", frame)
```

If you're on a headless box, set `MUJOCO_GL=osmesa` before importing — see
[Troubleshooting](../troubleshooting.md).

## Add a cube and pick it up

```python
sim.add_object(
    name="cube", type="box", size=[0.025]*3,
    pos=[0.3, 0.0, 0.025], rgba=[1, 0, 0, 1],
)

sim.run_policy(
    instruction="pick up the red cube",
    policy_provider="mock",      # try "groot" or "lerobot_local" with a real model
    duration=10.0,
)
```

`MockPolicy` returns sinusoidal joint traces — no model, no GPU. Useful for verifying
the pipeline before plugging in a real policy.

## Add an agent

```bash
pip install strands-agents
```

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so100")
agent = Agent(tools=[robot])

agent("Add a red cube on the table and pick it up using the mock policy")
```

The agent reads the simulation's 35+ actions from its tool spec and routes the user's
instruction to the right ones.

## Where to next

- **More detail:** the [Tutorial](../tutorial/index.md) walks each step in chapter form.
- **Real hardware:** [Tutorial 8](../tutorial/08-real-hardware.md) — same code,
  `mode="real"`.
- **Real model:** [Policy providers](../policies/overview.md) — GR00T, LeRobot Local.
- **Catalog:** [Robot catalog](../robots/index.md) — 68 robots, all by name.
