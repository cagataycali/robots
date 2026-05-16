---
description: The Simulation AgentTool — every action grouped by category, with parameters.
---

# Simulation overview

`Simulation` is the MuJoCo-backed simulation. It's a Strands `AgentTool` exposing 35+
actions a Python caller (or Strands Agent) can fire directly.

This page is the **action reference** — what each action does, what it returns, what
the parameters are. For walkthroughs, see [Tutorial 2](../tutorial/02-simulation.md).

## Construction

```python
from strands_robots import Robot
sim = Robot("so100")                         # via factory (preferred)

# Or directly:
from strands_robots.simulation import Simulation
sim = Simulation(tool_name="my_sim",
                 default_timestep=0.002,
                 default_width=640,
                 default_height=480)
```

## World lifecycle

| Action | What |
|--------|------|
| `create_world(timestep=0.002, gravity=[0,0,-9.81])` | Initialise the MuJoCo world. Implicit on `Robot()`. |
| `load_scene(xml=..., xml_path=...)` | Replace the world with an MJCF. |
| `reset()` | Reset state to t=0, keep model. |
| `get_state()` | Returns sim time, joint positions, object poses. |
| `destroy()` | Tear down model, data, and the executor. |

## Robots in the scene

| Action | What |
|--------|------|
| `add_robot(robot_name, position=[0,0,0], data_config=None, urdf_path=None)` | Add another robot to the world. |
| `remove_robot(name)` | Remove by name. |
| `list_robots()` | Names + categories of robots currently in the scene. |
| `get_robot_state(name)` | Joint positions, velocities, torques. |

## Objects

| Action | What |
|--------|------|
| `add_object(name, type="box"\|"sphere"\|"cylinder"\|"mesh", size=[...], pos=[...], rgba=[...], mass=...)` | Add a physics object. |
| `remove_object(name)` | Remove by name. |
| `move_object(name, pos=..., quat=...)` | Teleport an object. |
| `list_objects()` | Names + types of objects in the scene. |

## Cameras

| Action | What |
|--------|------|
| `add_camera(name, attach_to=..., pos=..., quat=..., fovy=60, lookat=...)` | Add a camera (free or attached to a body/site). |
| `remove_camera(name)` | Remove by name. |

## Rendering

| Action | What |
|--------|------|
| `render(camera=None, width=640, height=480)` | RGB frame as `numpy.uint8(H,W,3)`. |
| `render_depth(camera=None, ...)` | Depth as `numpy.float32(H,W)` in metres. |
| `open_viewer()` | Launch the interactive MuJoCo passive viewer. |
| `close_viewer()` | Close it. |

## Physics

| Action | What |
|--------|------|
| `step(num_steps=1)` | Advance physics. |
| `set_gravity([x, y, z])` | Update gravity. |
| `set_timestep(dt)` | Update integrator timestep. |
| `get_contacts()` | List of currently-active contacts. |

## Policies

| Action | What |
|--------|------|
| `run_policy(instruction, policy_provider="mock", policy=None, duration=10.0, ...)` | Block until duration elapses. |
| `start_policy(...)` | Async — fires up a policy thread, returns immediately. |
| `stop_policy()` | Halt a running policy. |
| `eval_policy(instruction, ..., num_episodes=10, randomize=True)` | Multi-episode benchmark with success_rate + mean_reward. |
| `replay_episode(repo_id, episode_id)` | Replay a recorded episode. |

## Recording

| Action | What |
|--------|------|
| `start_recording(repo_id, task, fps=30, ...)` | Begin LeRobot v3 recording. |
| `stop_recording()` | Finalise episode + meta files. |
| `get_recording_status()` | Current episode, frame count, output dir. |

## Randomization

| Action | What |
|--------|------|
| `randomize(colors=False, lighting=False, physics=False, cameras=False, ...)` | Domain randomization. |

See [Domain randomization](domain-randomization.md) for the full distribution it samples.

## Asset registry

| Action | What |
|--------|------|
| `list_urdfs()` | Loaded URDFs/MJCFs in the current world. |
| `register_urdf(name, path)` | Register an additional asset. |
| `get_features()` | Observation/action feature schema (used by recording). |

## Return shape

Every action returns a dict shaped:

```python
{
    "status": "success" | "error",
    "content": [{"text": "..."}, {"image": ...}, ...]
}
```

`status="success"` plus the requested data keys (e.g. `frame`, `state`). On error,
`content[0]["text"]` carries a human-readable explanation. The dispatch layer never
raises out of an action — agents always get a structured response.

## See also

- [Tutorial 2 — Simulation](../tutorial/02-simulation.md) — concrete examples.
- [World building](world-building.md) — composing scenes with multiple robots and
  objects.
- [Domain randomization](domain-randomization.md) — `randomize` distributions.
- [Architecture](../architecture.md) — how `Simulation` implements `SimEngine`.
