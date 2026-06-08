---
description: The Simulation AgentTool — 60+ actions for worlds, cameras, objects, and randomization.
---

# 2 — Simulation

`Robot("so100")` returns a `Simulation` object. This chapter walks through the actions
that object exposes — they're what the Strands Agent will be calling in chapter 4.

## TL;DR

```python
from strands_robots import Robot

sim = Robot("so100")

# Add a side-view camera (free camera: position + target)
sim.add_camera(name="side", position=[0.5, -0.5, 0.3], target=[0.0, 0.0, 0.1], fov=60)

# Add a red cube on the table
sim.add_object(name="cube", shape="box", size=[0.025, 0.025, 0.025],
               position=[0.3, 0.0, 0.025], color=[1, 0, 0, 1])

# Render from the side camera (returns PNG image content block)
result = sim.render(camera_name="side")
```

## The action vocabulary

`Simulation` exposes 60+ AgentTool actions, grouped by topic. You can call them as
methods on the Python object (this chapter), or have a Strands Agent call them via
natural language (chapter 4). Either way, same dispatcher.

| Group | Actions |
|-------|---------|
| **World lifecycle** | `create_world`, `load_scene`, `reset`, `get_state`, `destroy` |
| **Robots** | `add_robot`, `remove_robot`, `list_robots`, `get_robot_state` |
| **Objects** | `add_object`, `remove_object`, `move_object`, `list_objects` |
| **Cameras** | `add_camera`, `remove_camera` |
| **Policies** | `run_policy`, `start_policy`, `stop_policy`, `eval_policy`, `replay_episode`, `run_multi_policy` |
| **Rendering** | `render`, `render_depth`, `open_viewer`, `close_viewer` |
| **Physics** | `step`, `set_gravity`, `set_timestep`, `get_contacts` |
| **Recording** | `start_recording`, `stop_recording`, `get_recording_status`, `start_cameras_recording`, `stop_cameras_recording` |
| **Scene MJCF** | `replace_scene_mjcf`, `patch_scene_mjcf` |
| **Randomization** | `randomize` (randomize_colors, randomize_lighting, randomize_physics, randomize_positions) |
| **Assets** | `list_urdfs`, `register_urdf`, `get_features` |

For the full list with parameters, see [Simulation overview](../simulation/overview.md).

## Step 1 — add a camera

Every `Simulation` starts with one default top-down camera. Add more whenever you need
a different viewpoint. Cameras added via `add_camera` are free cameras defined by a
world-space `position` and a `target` point the camera looks toward. Robot-mounted
cameras (e.g. a wrist camera) come from the robot's own URDF and are auto-discovered
when `add_robot` runs; list them with `get_features(robot_name="so100")`.

```python
sim = Robot("so100")

# Side camera defined by world position and look-at target
sim.add_camera(
    name="side",
    position=[0.5, -0.5, 0.3],   # camera location in world space
    target=[0.0, 0.0, 0.1],      # point the camera looks at
    fov=60,                       # field-of-view in degrees
    width=640,
    height=480,
)

# Top-down camera
sim.add_camera(name="top", position=[0.0, 0.0, 0.8], target=[0.0, 0.0, 0.0])
```

`add_camera` returns a status dict. Pass the camera name to `render`, `remove_camera`,
or `start_recording`.

## Step 2 — add objects

Objects are MuJoCo geoms (box, sphere, cylinder, mesh) with a freejoint so the physics
solver can move them.

```python
# Red cube on the table
sim.add_object(
    name="cube",
    shape="box",
    size=[0.025, 0.025, 0.025],   # half-extents in metres
    position=[0.3, 0.0, 0.025],
    color=[1, 0, 0, 1],
    mass=0.05,
)

# Blue sphere
sim.add_object(name="ball", shape="sphere", size=[0.02], position=[0.3, 0.1, 0.1],
               color=[0, 0, 1, 1])
```

`list_objects()` shows what's in the scene; `move_object("cube", position=[...])` teleports an
object; `remove_object("cube")` deletes it.

## Step 3 — render

`render()` returns a PNG image content block — an agent receives it directly without
extra decoding. For a numpy array in your own processing loop, call `get_observation`:

```python
# render() returns a PNG image content block
result = sim.render(width=640, height=480)
# result["content"][1] is {"image": {"format": "png", "source": {"bytes": <png bytes>}}}

# For a numpy array (uint8, H×W×3), use get_observation:
obs = sim.get_observation("so100")
frame = obs["default"]            # default camera
side_frame = obs["side"]          # named camera added via add_camera

# Render from a specific camera
result_side = sim.render(camera_name="side", width=320, height=240)

# Depth (float32 H×W in metres)
depth_result = sim.render_depth(camera_name="side")
```

## Step 4 — randomize

Domain randomization changes the world between rollouts so policies generalise. The
`randomize` action accepts keyword flags:

```python
sim.randomize(
    randomize_colors=True,      # random object/floor colors
    randomize_lighting=True,    # ambient + directional lighting
    randomize_physics=True,     # mass, friction, joint damping
    randomize_positions=True,   # small object-position perturbations
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
sim.load_scene(scene_path="path/to/my_scene.xml")
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
    robot_name="so100",
    instruction="pick up the red cube",
    policy_provider="mock",   # MockPolicy — sinusoidal joint traces
    duration=10.0,
)
```

This is what an `Agent` will be calling in chapter 4 when you say
*"pick up the red cube"*.

## Recap

- `Simulation` is a Strands AgentTool with 60+ actions.
- Cameras, objects, and randomization are all single-action calls.
- `render` returns a PNG image content block; `get_observation` returns numpy arrays.
- Multi-robot scenes are composed with `add_robot`, not different factory calls.

## See also

- [Tutorial 3 — Policies](03-policies.md) — drop a real policy into `run_policy`.
- [Simulation overview](../simulation/overview.md) — every action with full parameters.
- [World building](../simulation/world-building.md) — non-trivial scene composition.
- [Domain randomization](../simulation/domain-randomization.md) — what `randomize`
  actually samples.
