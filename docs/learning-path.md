---
description: Where to go depending on what you're trying to do.
---

# Learning path

The whole library is one `pip install`. What you read next depends on what you want.

```mermaid
graph LR
    A[See it move] --> B[Ship something] --> C[Extend it]
    classDef a fill:#2ea44f,stroke:#1b7735,color:#fff
    class A,B,C a
```

## See it move

If you're evaluating the library or just want a robot on screen, this is a half-hour.

1. [Install](getting-started/installation.md) — pick the `sim-mujoco` extra.
2. [Quickstart](getting-started/quickstart.md) — a robot moving in five minutes.
3. [Your first robot](tutorial/01-your-first-robot.md) — step physics, grab a frame.
4. [Simulation](tutorial/02-simulation.md) — cameras, objects, randomization.
5. [AI agents](tutorial/04-agents.md) — drive it with a sentence.

You end up able to spawn any of the 68 robots and have an agent control it.

## Ship something

If you're putting this into a product, a training pipeline, or research:

1. [Robot factory](getting-started/robot-factory.md) — every `Robot(...)` knob.
2. [Policies](policies/overview.md) — wire up [GR00T](policies/groot.md), [LeRobot](policies/lerobot-local.md), or [Cosmos 3](policies/cosmos3.md).
3. [Recording](tutorial/06-recording.md) — capture a LeRobot dataset.
4. [Real hardware](tutorial/08-real-hardware.md) — calibrate an arm, stream cameras.
5. [Multi-robot](tutorial/05-multi-robot.md) — two robots coordinating on the mesh.

You end up with a record → train → deploy loop on real servos and a sim twin.

## Extend it

If you're adding a policy, a backend, a robot, or a tool:

1. [Architecture](architecture.md) — the module boundaries and the one rule.
2. [Custom policies](policies/custom-policies.md) — subclass, register, use.
3. [Advanced](tutorial/09-advanced.md) — custom sim backends, data configs, tools.
4. [Robot catalog](robots/index.md) — the `robots.json` schema for adding one.
5. [Contributing](contributing.md) — hatch, lint, the PR flow.

Your change lands through the same factory and registry everyone else uses.

## Just looking something up?

[API reference](api-reference.md) · [Robot catalog](robots/index.md) · [Troubleshooting](troubleshooting.md) · [Examples](examples/overview.md)
