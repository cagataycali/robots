---
description: Install, instantiate a robot, step physics, render a frame. Ten minutes.
---

# 1 — Your first robot

You'll spawn an SO-100 arm in MuJoCo, step the world a few times, and save a PNG frame.
By the end of this chapter you can answer: *what is the smallest possible
`strands-robots` program?*

## TL;DR

```python
from strands_robots import Robot

robot = Robot("so100")           # MuJoCo sim, mode='sim' is the safe default
robot.step()                     # advance physics one step
print(robot.tool_name_str)       # 'so100_sim'
```

That's it. No URDF path, no scene XML — `Robot("so100")` looks the name up in
`registry/robots.json`, downloads the MJCF model from MuJoCo Menagerie if needed, and
constructs a complete physics scene.

## Setup

```bash
pip install "strands-robots[sim-mujoco]"
```

The `[sim-mujoco]` extra pulls in `mujoco` and `numpy`. On Linux, you may also want
`libosmesa6-dev` and `ffmpeg` for headless rendering and video export — see
[Installation](../getting-started/installation.md).

## Step 1 — pick a robot

Every robot is addressable by name. Use `list_robots()` to see what's available:

```python
from strands_robots import list_robots

# All 68 robots
all_robots = list_robots("all")
print(f"Total: {len(all_robots)}")

# Just the arms
arms = list_robots("arm")
print([r["name"] for r in arms])
# ['arx_l5', 'fr3', 'kinova_gen3', 'koch', 'kuka_iiwa', 'panda', 'piper',
#  'so100', 'trossen_wxai', 'ur5e', ...]
```

The catalog is grouped by category: `arm`, `bimanual`, `humanoid`, `hand`, `mobile`,
`mobile_manip`, `aerial`, `expressive`. Pick one. We'll use `so100` for the rest of this
chapter — it's a 6-DOF arm, simple, fast.

## Step 2 — instantiate

```python
from strands_robots import Robot

robot = Robot("so100")
```

What just happened:

1. `"so100"` was resolved through the alias map to its canonical name (it already is the
   canonical name; aliases include things like `so-100` and `SO100`).
2. The registry entry was loaded:

   ```python
   {
     "category": "arm",
     "joints": 6,
     "asset": {
       "dir": "trs_so_arm100",
       "model_xml": "so_arm100.xml",
       "scene_xml": "scene.xml",
       "robot_descriptions_module": "trs_so_arm100_mj_description"
     },
     "aliases": ["so-100", "SO100", ...]
   }
   ```

3. The MJCF model was downloaded into `~/.strands_robots/assets/trs_so_arm100/` (cached
   on subsequent runs).
4. A `Simulation` instance was constructed and a world was created with the SO-100 plus
   a ground plane plus default lighting.

The returned object is a `strands_robots.simulation.Simulation` — a Strands `AgentTool`
with 60+ actions. We'll use a few of them now.

## Step 3 — step physics

```python
# Advance simulation one timestep (default: 0.002s)
result = robot.step()
print(result)

# Or many steps
for _ in range(100):
    robot.step()
```

`step` returns a status dict. The default timestep matches MuJoCo's recommendation; you
can override it via `Robot("so100", default_timestep=0.005)` or per-step via the
`set_timestep` action.

## Step 4 — render a frame

`render()` returns a PNG image content block that the agent receives directly. To get a
numpy array for your own processing loop, use `get_observation`:

```python
import imageio.v3 as iio

# get_observation returns a dict of numpy arrays (uint8, H×W×3) per camera
obs = robot.get_observation("so100")
frame = obs["default"]             # default top-down camera
print(frame.shape)                 # (480, 640, 3)

# Save to disk
iio.imwrite("first_frame.png", frame)

# render() returns PNG bytes as an image content block (useful for agents/display):
result = robot.render(width=640, height=480)
# result["content"][1] is {"image": {"format": "png", "source": {"bytes": <bytes>}}}
```

If you're on a headless box (CI, Docker), set `MUJOCO_GL=osmesa` before the import — see
[Troubleshooting](../troubleshooting.md).

## Step 5 — clean up (optional)

Simulations hold onto MuJoCo handles and a thread-pool executor. They self-clean on
process exit, but if you're constructing many robots in a long-running notebook, call
`destroy()`:

```python
robot.destroy()
```

## Recap

- `pip install "strands-robots[sim-mujoco]"`
- `Robot("name")` — name from the registry
- `robot.step()`, `robot.render()`, `robot.get_observation()`
- 60+ actions on every `Simulation`; we used two

## What changed from the previous decade of robotics tutorials?

- **No URDF wrangling.** The registry handles asset paths, scene composition, and
  Menagerie integration. You name a robot, you get a robot.
- **No GPU.** MuJoCo runs on CPU. A laptop is sufficient through chapter 6.
- **No agent code.** `Robot()` is a Python object you can drive directly. The
  `Agent(tools=[robot])` wrapper comes in chapter 4 — it's not required.

## See also

- [Tutorial 2 — Simulation](02-simulation.md) — load a richer scene, add cameras, add
  objects, randomize.
- [Robot factory](../getting-started/robot-factory.md) — every kwarg `Robot()` takes.
- [Robot catalog](../robots/index.md) — all 68 robots with renders.
