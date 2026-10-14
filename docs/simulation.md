---
description: One world, many robots. A Simulation owns time and the scene; a Backend steps the physics.
---

# One world

{{drawing:d02_one_world}}

```python
from strands_robots import Simulation

sim = Simulation.default()        # MuJoCo, headless; every Robot given no world joins this one
sim.robots                        # {"so101": ..., "unitree_g1": ...}: ids are unique, a second so101 is "so101_2"
sim.physics                       # Physics("mujoco", timestep=0.002, substeps=10): control_dt is 20 ms
sim.control_step()                # what act() costs a simulated robot: one control step for the whole world
with sim.manual_time():           # inside, act() only writes targets; the caller calls sim.step(n)
    ...
sim.set_pose("unitree_g1", position=[2, 0, 0.8])   # teleport; velocities zeroed, time does not advance
sim.add_object("box", shape="box", position=[0.3, 0, 0.1], size=[0.03, 0.03, 0.03])
sim.camera("wrist", position=[0.15, -0.25, 0.25], target=[0.2, 0, 0.05], attach="so101")
sim.render()                      # HxWx3 uint8 from the auto-framed overview camera that sees every robot
sim.describe()                    # robots, objects, cameras, physics, time and revision, JSON-safe
```

A `Simulation` owns the `Scene` (plain data: robots, objects, cameras, gravity, timestep), the robots
in it and the time. Every scene change (`add`, `remove`, `add_object`, `camera`, `set_pose`,
`randomize`) bumps `revision`; motion does not. Adding a robot appends to the scene and rebuilds the
world, and every body that survives keeps its joint state. `sim.lock` is one re-entrant lock around
the world, so a reader and a stepping loop never see a half-built world.

`set_pose` moves a fixed base by rebuilding the world where it lands (joint state and time carried
over). A floating base is teleported in place and keeps its spawn frame, so `reset()` stands it back
on its keyframe at the spawn position.

`Simulation()` refuses what it cannot be: an unknown backend, `physics=` other than `native`,
`substeps` below 1, `num_envs` above 1, an unknown keyword.

## The backend contract

`Backend` is an abstract class: `load`, `step`, `reset`, `time`, `joint_state`, `set_targets`,
`body_pose`, `set_pose`, `set_gains`, `render`, `contacts`, `physics` and `close`. `mujoco` is the
first backend: one `MjSpec`, each robot attached under an `id/` prefix, recompiled on every scene
change with state carried by joint name. `mujoco` is imported when the first world is made, not by
`import strands_robots`. Rendering uses EGL when no display is present (`MUJOCO_GL` is respected).

Measured headless: 2000 physics steps of so101 plus unitree_g1 take 85 us each (MuJoCo 3.15, one CPU core).
