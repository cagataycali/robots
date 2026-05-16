---
description: Compose non-trivial scenes — multiple robots, tables, obstacles, custom MJCF.
---

# World building

The `create_world` action gives you a flat ground plane plus default lighting. From
there, `add_robot` / `add_object` / `add_camera` compose the scene incrementally. For
elaborate setups, `load_scene` swaps in a complete MJCF.

## TL;DR

```python
from strands_robots import Robot

sim = Robot("so100")                 # one arm on a flat ground plane

# Add a second arm
sim.add_robot(robot_name="so100", position=[0.0, 0.5, 0.0])

# Add a table by stacking a thin box object
sim.add_object(name="table", type="box", size=[0.5, 0.5, 0.02],
               pos=[0.0, 0.0, 0.0], rgba=[0.5, 0.3, 0.1, 1], mass=20.0)

# A wrist camera per arm
sim.add_camera(name="wrist_a", attach_to="so100", pos=[0.05, 0, 0.1])
sim.add_camera(name="wrist_b", attach_to="so100_2", pos=[0.05, 0, 0.1])
```

## Strategies

| Need | Approach |
|------|----------|
| Add a single robot to an existing world | `add_robot(...)` |
| Add objects (cubes, balls, tables) | `add_object(...)` |
| Replace the entire world with a hand-authored scene | `load_scene(xml_path=...)` |
| Generate scenes procedurally | Compose `add_robot` + `add_object` in a loop |
| Reuse a Menagerie scene | Pass its MJCF path via `load_scene` |

## Procedural composition

```python
import random

sim = Robot("so100")

# Drop 5 random cubes in a 30x30 cm region
for i in range(5):
    sim.add_object(
        name=f"cube_{i}",
        type="box",
        size=[0.025, 0.025, 0.025],
        pos=[random.uniform(0.2, 0.5),
             random.uniform(-0.15, 0.15),
             0.025],
        rgba=[random.random(), random.random(), random.random(), 1],
    )
```

This is useful for randomised manipulation tasks where the *world* (not just the
visuals) varies between rollouts.

## load_scene

If you've authored an MJCF in MuJoCo's editor or downloaded one from a benchmark suite:

```python
sim = Robot("so100")
sim.load_scene(xml_path="/path/to/my_scene.xml")
```

`load_scene` tears down the current world and rebuilds from the XML. Robots referenced
in the XML by name get registered with the sim's robot table; objects show up in
`list_objects()`.

## Per-robot positioning

When composing multiple robots, supply explicit positions so they don't overlap:

```python
sim.add_robot(robot_name="panda",  position=[0.0, 0.0, 0.0])
sim.add_robot(robot_name="ur5e",   position=[0.0, 0.6, 0.0])
sim.add_robot(robot_name="koch",   position=[0.0, -0.4, 0.0])
```

The `position` is the robot's base in world coordinates. Orientation is identity
unless you pass `quat=...`.

## Cameras: free or attached

```python
# Free camera looking at the origin
sim.add_camera(name="overhead",
               pos=[0.0, 0.0, 1.0],
               lookat=[0.0, 0.0, 0.0])

# Attached camera (moves with the robot's body/site)
sim.add_camera(name="wrist",
               attach_to="so100",   # body or site name
               pos=[0.05, 0.0, 0.1],  # offset in attach frame
               quat=[1, 0, 0, 0],
               fovy=60)
```

`attach_to` is resolved against the model's body and site names. Typical attach points
are the gripper body (for wrist cams) or the base (for chest cams on humanoids).

## Tear-down

If you're constructing many simulations in a notebook:

```python
sim.destroy()
```

`destroy()` releases MuJoCo handles, joins worker threads, and cleans up the scene.
Without it, GC eventually reaps the simulation but you can hit OOM on long sessions.

## See also

- [Simulation overview](overview.md) — every action with parameters.
- [Domain randomization](domain-randomization.md) — vary the world between rollouts.
- [Tutorial 2 — Simulation](../tutorial/02-simulation.md) — guided walkthrough.
- [LIBERO benchmark](https://github.com/strands-labs/robots/tree/main/strands_robots/benchmarks/libero) —
  a concrete example of `load_scene` driven by a benchmark.
