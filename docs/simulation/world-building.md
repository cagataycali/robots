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
sim.add_object(name="table", shape="box", size=[0.5, 0.5, 0.02],
               position=[0.0, 0.0, 0.0], color=[0.5, 0.3, 0.1, 1.0], mass=20.0)

# Free overhead camera looking at the workspace
sim.add_camera(name="overhead",
               position=[0.0, 0.0, 1.5],
               target=[0.0, 0.0, 0.0])
```

## Strategies

| Need | Approach |
|------|----------|
| Add a single robot to an existing world | `add_robot(...)` |
| Add objects (cubes, balls, tables) | `add_object(...)` |
| Replace the entire world with a hand-authored scene | `load_scene(scene_path=...)` |
| Generate scenes procedurally | Compose `add_robot` + `add_object` in a loop |
| Reuse a Menagerie scene | Pass its MJCF path via `load_scene` |
| Raw MJCF tweaks without recompile | `patch_scene_mjcf(ops)` |

## Procedural composition

```python
import random

sim = Robot("so100")

# Drop 5 random cubes in a 30x30 cm region
for i in range(5):
    sim.add_object(
        name=f"cube_{i}",
        shape="box",
        size=[0.025, 0.025, 0.025],
        position=[random.uniform(0.2, 0.5),
                  random.uniform(-0.15, 0.15),
                  0.025],
        color=[random.random(), random.random(), random.random(), 1.0],
    )
```

This is useful for randomised manipulation tasks where the *world* (not just the
visuals) varies between rollouts.

## load_scene

If you've authored an MJCF in MuJoCo's editor or downloaded one from a benchmark suite:

```python
sim = Robot("so100")
sim.load_scene(scene_path="my_scene.xml")
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

The `position` is the robot's base in world coordinates. Pass `orientation=[w,x,y,z]`
to rotate the base; identity orientation is assumed otherwise.

## Cameras

Cameras in `strands-robots` are always **free** — they look from a `position` toward
a `target`. Robot-URDF cameras (e.g. wrist cameras declared in the URDF) are
auto-discovered when `add_robot` runs; you do not need to call `add_camera` for them.

```python
# Overhead camera looking down at the workspace
sim.add_camera(name="overhead",
               position=[0.0, 0.0, 1.5],
               target=[0.0, 0.0, 0.0],
               fov=60.0,
               width=640,
               height=480)

# Side-view camera
sim.add_camera(name="side",
               position=[0.8, 0.0, 0.5],
               target=[0.0, 0.0, 0.2])
```

`position` and `target` must not be identical (that would produce a zero look-at
direction). `fov`, `width`, and `height` default to `60.0`, `640`, and `480`.

## Multi-robot policies

When running multiple robots simultaneously, use `run_multi_policy` instead of
separate `run_policy` calls. It keeps all robots synchronised to the same physics
clock and produces one merged recorded frame per step:

```python
from strands_robots.policies import create_policy

policy_a = create_policy("mock")
policy_b = create_policy("mock")

sim.run_multi_policy(
    policies={"so100": policy_a, "panda": policy_b},
    instructions={"so100": "pick cube", "panda": "hold tray"},
    duration=10.0,
)
```

Use `list_policies_running()` to inspect active policy threads started by
`start_policy`.

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
