---
description: Spawn a robot, step physics, grab a frame. The smallest program.
---

# 1 — Your first robot

```python
from strands_robots import Robot

robot = Robot("so100")     # sim (safe default); MJCF auto-downloads, cached
robot.step()               # advance physics
print(robot.tool_name_str) # 'so100_sim'
```

```bash
pip install "strands-robots[sim-mujoco]"
```

## Pick any of 68 robots

```python
from strands_robots import list_robots
list_robots("all")    # everything
list_robots("arm")    # one category: arm | bimanual | humanoid | hand | mobile | aerial | expressive | mobile_manip
```

Name resolves through `registry/robots.json` (aliases, asset paths, Menagerie download all handled).

## Step & render

`render()` returns a PNG image content block (what the agent sees). For a numpy frame, use `get_observation`:

```python
import imageio.v3 as iio

for _ in range(100):
    robot.step()                          # default dt = 0.002s

frame = robot.get_observation("so100")["default"]   # uint8 (H, W, 3)
iio.imwrite("frame.png", frame)
```

!!! note "Headless (CI / Docker)"
    `export MUJOCO_GL=osmesa` before import. See [Troubleshooting](../troubleshooting.md).

## Cleanup

```python
robot.destroy()    # frees MuJoCo + threads; optional, auto on exit
```

## See also

- [2 — Simulation](02-simulation.md) — cameras, objects, randomize.
- [Robot factory](../getting-started/robot-factory.md) — every kwarg.
- [Robot catalog](../robots/index.md) — all 68.
