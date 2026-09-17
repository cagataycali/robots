---
description: Robot control for Strands agents - name a robot and get one agent-callable tool that runs in MuJoCo simulation by default, and on real hardware when you ask for it.
---

# Strands Robots

<figure class="brand-figure" markdown="span">
  ![Strands Robots: perceive, reason, act, world - the closed control loop around a Strands Agent core](assets/hero_loop.svg){ .brand-svg }
</figure>

Name a robot, get one tool a [Strands agent](https://strandsagents.com) can drive: **{{n:robots}} robots**
in MuJoCo simulation by default, on real hardware when you ask for it.

## Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "strands-robots[sim-mujoco]"   # simulation
uv pip install "strands-robots[all]"          # sim + hardware + most policies
```

## Drive one

```python
from strands import Agent
from strands_robots import Robot

arm = Robot("so101", mode="sim")            # MuJoCo scene, CPU, no GPU
Agent(tools=[arm])("Pick up the red cube")
```

The factory returns the backend itself, so the same object is callable from Python -
`arm.get_robot_state()`, `arm.run_policy(...)` - and `mode="real"` drives a physical SO-101
through the same actions.

## Three ways in

<div class="grid cards" markdown>

-   :material-cube-outline:{ .lg .middle } **Simulate**

    ---

    Load a scene, add objects and cameras, roll out a policy, record the episode.

    [:octicons-arrow-right-24: Simulation](simulation/overview.md)

-   :material-robot-industrial:{ .lg .middle } **Drive real hardware**

    ---

    Native drivers and LeRobot buses, with an operator gate in front of every call that moves.

    [:octicons-arrow-right-24: Real hardware](hardware/robot-control.md)

-   :material-robot-happy-outline:{ .lg .middle } **Give it to an agent**

    ---

    One tool per robot; the agent chooses the action and reads back what happened.

    [:octicons-arrow-right-24: AI agents](agents.md)

</div>

## Robots at work

<div class="grid" markdown>

<figure markdown="span">
  ![SO-101 picking up a cube and lifting it clear of the table in MuJoCo](https://github.com/user-attachments/assets/b5fb7582-5bcb-4053-a9f7-9f08ec42a411){ loading=lazy }
  <figcaption><code>examples/18_so101_pick_and_lift.py</code> - a reference pick that lifts.</figcaption>
</figure>

<figure markdown="span">
  ![Unitree G1 walking forward under the whole-body-control policy provider](https://github.com/user-attachments/assets/b313e219-b985-4899-80ac-58582e0d90c5){ loading=lazy }
  <figcaption><code>run_policy(policy_provider="wbc")</code> - G1, 0 to 2.8 m in 8 s.</figcaption>
</figure>

<figure markdown="span">
  ![Pollen Microduck, a 14-DOF biped, walking across the floor of a MuJoCo scene](https://github.com/user-attachments/assets/0b75411b-6d6b-4af9-ae8d-0c215470d66c){ loading=lazy }
  <figcaption><code>microduck</code> provider - <code>alpha_walking.onnx</code> at 0.3 m/s.</figcaption>
</figure>

<figure markdown="span">
  ![Simulated SO-101 executing a SmolVLA policy rollout recorded to video](assets/run_policy_video_demo.gif){ loading=lazy }
  <figcaption>SmolVLA through <code>lerobot_local</code>, rendered headless.</figcaption>
</figure>

</div>

## What runs on what

**{{n:robots}} robots** in the registry, {{n:hardware}} of them with a hardware path, across
{{n:categories}} categories and {{n:sim_backends}} simulation backends. Which driver builds which
robot is derived from the registries rather than declared:

```python
from strands_robots.drivers import list_driver_coverage

list_driver_coverage()["so101"]     # ('lerobot', 'strands')
```

[Robot catalog](robots/index.md) · [Policy providers](policies/overview.md) ·
[Architecture](architecture.md) · [Quickstart](getting-started/quickstart.md)
