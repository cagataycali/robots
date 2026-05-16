---
description: The Simulation AgentTool — 35+ actions for worlds, cameras, objects, and randomization.
---

# 2 — Simulation

`Robot("so100")` returns a `Simulation` object. This chapter walks through the actions
that object exposes — they're what the Strands Agent will be calling in chapter 4.

## TL;DR

```python
from strands_robots import Robot

sim = Robot("so100")

# Add a wrist camera
sim.add_camera(name="wrist", attach_to="so100", pos=[0.05, 0, 0.1], fovy=60)

# Add a red cube on the table
sim.add_object(name="cube", type="box", size=[0.025, 0.025, 0.025],
               pos=[0.3, 0.0, 0.025], rgba=[1, 0, 0, 1])

# Render from the wrist camera
frame = sim.render(camera="wrist")["frame"]
```

## The action vocabulary

`Simulation` exposes 35+ AgentTool actions, grouped by topic. You can call them as
methods on the Python object (this chapter), or have a Strands Agent call them via
natural language (chapter 4). Either way, same dispatcher.

| Group | Actions |
|-------|---------|
| **World lifecycle** | `create_world`, `load_scene`, `reset`, `get_state`, `destroy` |
| **Robots** | `add_robot`, `remove_robot`, `list_robots`, `get_robot_state` |
| **Objects** | `add_object`, `remove_object`, `move_object`, `list_objects` |
| **Cameras** | `add_camera`, `remove_camera` |
| **Policies** | `run_policy`, `start_policy`, `stop_policy`, `eval_policy`, `replay_episode` |
| **Rendering** | `render`, `render_depth`, `open_viewer`, `close_viewer` |
| **Physics** | `step`, `set_gravity`, `set_timestep`, `get_contacts` |
| **Recording** | `start_recording`, `stop_recording`, `get_recording_status` |
| **Randomization** | `randomize` (colors, physics, lighting, cameras) |
| **Assets** | `list_urdfs`, `register_urdf`, `get_features` |

For the full list with parameters, see [Simulation overview](../simulation/overview.md).

## Step 1 — add a camera

Every `Simulation` starts with one default top-down camera. Add more whenever you need
a different viewpoint.

```python
sim = Robot("so100")

# Wrist camera attached to the gripper
sim.add_camera(
    name="wrist",
    attach_to="so100",       # body or site to follow
    pos=[0.05, 0.0, 0.1],    # offset from attach point
    quat=[1, 0, 0, 0],       # orientation
    fovy=60,                 # field-of-view in degrees
)

# Static side camera
sim.add_camera(name="side", pos=[0.5, -0.5, 0.3], lookat=[0.0, 0.0, 0.1])
```

`add_camera` returns a status dict with the camera id you can pass to `render`,
`remove_camera`, or `start_recording`.

## Step 2 — add objects

Objects are MuJoCo geoms (box, sphere, cylinder, mesh) with a freejoint so the physics
solver can move them.

```python
# Red cube on the table
sim.add_object(
    name="cube",
    type="box",
    size=[0.025, 0.025, 0.025],   # half-extents in metres
    pos=[0.3, 0.0, 0.025],
    rgba=[1, 0, 0, 1],
    mass=0.05,
)

# Blue sphere
sim.add_object(name="ball", type="sphere", size=[0.02], pos=[0.3, 0.1, 0.1],
               rgba=[0, 0, 1, 1])
```

`list_objects()` shows what's in the scene; `move_object("cube", pos=[...])` teleports an
object; `remove_object("cube")` deletes it.

## Step 3 — render

```python
# Default camera
frame = sim.render(width=640, height=480)["frame"]

# A specific camera
wrist_frame = sim.render(camera="wrist", width=320, height=240)["frame"]

# Depth (single-channel float)
depth = sim.render_depth(camera="wrist")["depth"]
```

The `frame` is a `numpy.ndarray` of dtype `uint8` shaped `(H, W, 3)`. Depth is `float32`
shaped `(H, W)` with values in metres.

## Step 4 — randomize

Domain randomization changes the world between rollouts so policies generalise. The
`randomize` action accepts a list of categories:

```python
sim.randomize(
    colors=True,        # random object/floor colors
    lighting=True,      # ambient + directional lighting
    physics=True,       # mass, friction, joint damping
    cameras=True,       # small camera-pose perturbations
)
```

See [Domain randomization](../simulation/domain-randomization.md) for the full
distribution it samples from.

## Step 5 — load a richer scene

`Robot("so100")` gives you a single robot on a flat ground plane. For richer scenes
(table, obstacles, multiple robots), load an MJCF or compose with `add_robot`:

```python
# Compose: start blank, add two SO-100s
sim = Robot("so100")  # also acceptable: create_simulation() directly
sim.add_robot(robot_name="so100", position=[0.0, 0.5, 0.0])  # second arm

# Or load an existing MJCF
sim.load_scene(xml="path/to/my_scene.xml")
```

See [World building](../simulation/world-building.md).

## Step 6 — physics control

```python
# Bigger timestep (faster sim, less accurate)
sim.set_timestep(0.005)

# Different gravity
sim.set_gravity([0.0, 0.0, -3.7])  # Mars

# Step many times
for _ in range(500):
    sim.step()

# Read contacts (which geoms are touching)
contacts = sim.get_contacts()["contacts"]
```

## Step 7 — run a policy (preview)

The full policy walkthrough is chapter 3, but here's a one-liner so you can see how it
plugs in:

```python
sim.run_policy(
    instruction="pick up the red cube",
    policy_provider="mock",   # MockPolicy — sinusoidal joint traces
    duration=10.0,
)
```

This is what an `Agent` will be calling in chapter 4 when you say
*"pick up the red cube"*.

## Recap

- `Simulation` is a Strands AgentTool with 35+ actions.
- Cameras, objects, and randomization are all single-action calls.
- `render` returns numpy arrays you can save with `imageio` or feed to a model.
- Multi-robot scenes are composed with `add_robot`, not different factory calls.

## See also

- [Tutorial 3 — Policies](03-policies.md) — drop a real policy into `run_policy`.
- [Simulation overview](../simulation/overview.md) — every action with full parameters.
- [World building](../simulation/world-building.md) — non-trivial scene composition.
- [Domain randomization](../simulation/domain-randomization.md) — what `randomize`
  actually samples.
