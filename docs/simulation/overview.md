---
description: The Simulation AgentTool — every action grouped by category, with parameters.
---

# Simulation overview

`Simulation` is the MuJoCo-backed simulation. It's a Strands `AgentTool` exposing 60+
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
| `create_world(timestep=0.002, gravity=[0,0,-9.81], ground_plane=True)` | Initialise the MuJoCo world. Implicit on `Robot()`. |
| `load_scene(scene_path=...)` | Replace the world with an MJCF. |
| `reset()` | Reset state to t=0, keep model. |
| `get_state()` | Returns sim time, joint positions, object poses. |
| `destroy()` | Tear down model, data, and the executor. |
| `export_xml()` | Serialise the current model to an MJCF string. |

## Scene MJCF

| Action | What |
|--------|------|
| `replace_scene_mjcf(xml)` | Swap the entire world XML (raw-MJCF escape hatch). |
| `patch_scene_mjcf(ops)` | Apply incremental MJCF patches without a full recompile. |
| `raycast(origin, direction, ...)` | Single ray–mesh intersection. |
| `multi_raycast(rays, ...)` | Batch ray–mesh intersections. |

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
| `add_object(name, shape="box"\|"sphere"\|"cylinder"\|"plane"\|"mesh", size=[...], position=[x,y,z], color=[r,g,b,a], orientation=[w,x,y,z], mass=0.1, is_static=False, mesh_path=None)` | Add a physics object. `plane` requires `is_static=True`. |
| `remove_object(name)` | Remove by name. |
| `move_object(name, position=..., orientation=...)` | Teleport an object. |
| `list_objects()` | Names + shapes of objects in the scene. |

## Cameras

| Action | What |
|--------|------|
| `add_camera(name, position=[x,y,z], target=[x,y,z], fov=60.0, width=640, height=480)` | Add a free camera that looks from `position` toward `target`. Robot-URDF cameras are auto-discovered on `add_robot` — no `add_camera` needed for them. |
| `remove_camera(name)` | Remove by name. |

## Rendering

| Action | What |
|--------|------|
| `render(camera_name="default", width=None, height=None)` | Returns PNG bytes inside `content` (no `frame` key). For a numpy frame in your own loop, call `sim.get_observation(robot_name)[camera_name]`. |
| `render_depth(camera_name="default", width=None, height=None)` | Depth float32 inside `content` JSON/image (no `depth` key). |
| `open_viewer()` | Launch the interactive MuJoCo passive viewer. |
| `close_viewer()` | Close it. |

## Physics

| Action | What |
|--------|------|
| `step(n_steps=1)` | Advance physics (max 100 000/call). |
| `set_gravity(gravity=[x, y, z])` | Update gravity vector. |
| `set_timestep(timestep)` | Update integrator timestep. |
| `get_contacts()` | List of currently-active contacts. |
| `get_contact_forces()` | Forces for each active contact pair. |
| `apply_force(body, force, torque, ...)` | Apply an external wrench to a body. |
| `get_jacobian(robot_name, body)` | Geometric Jacobian for a body. |
| `get_mass_matrix(robot_name)` | Joint-space mass matrix. |
| `inverse_dynamics(robot_name, ...)` | Compute required joint torques. |
| `forward_kinematics(robot_name, ...)` | Cartesian poses from joint angles. |
| `save_state()` | Snapshot full physics state. |
| `load_state(state)` | Restore a previously saved state. |
| `get_energy(robot_name)` | Kinetic and potential energy. |
| `get_sensor_data(robot_name)` | Raw MuJoCo sensor readings. |

## Policies

| Action | What |
|--------|------|
| `run_policy(robot_name, policy_provider="mock", policy_config={...}, instruction="", duration=10.0, policy_object=None, n_steps=None)` | Block until duration elapses. Pass a prebuilt policy via `policy_object=`; provider kwargs go in `policy_config`. `robot_name` is required. |
| `start_policy(...)` | Async — fires up a policy thread, returns immediately. |
| `stop_policy(robot_name)` | Halt the running policy for `robot_name`. |
| `list_policies_running()` | List all currently-running policy threads. |
| `run_multi_policy(policies={robot: Policy}, instructions=..., duration=, n_steps=)` | Synchronized multi-robot execution; produces one merged recorded frame per step. |
| `eval_policy(robot_name, policy_provider="mock", policy_config={...}, instruction="", n_episodes=1, max_steps=300, success_fn=None)` | Multi-episode evaluation reporting success rate. `robot_name` required; no `randomize=` kwarg. |
| `replay_episode(repo_id, robot_name=None, episode=0, ...)` | Replay a recorded episode. |

## Recording

| Action | What |
|--------|------|
| `start_recording(repo_id, task="", fps=30, ...)` | Begin LeRobot v3 recording. Requires `[lerobot]` extra. |
| `stop_recording(output_path=None)` | Finalise episode + meta files. |
| `get_recording_status()` | Current episode, frame count, output dir. |
| `start_cameras_recording(...)` | Plain MP4 recording via imageio-ffmpeg. Works under `[sim-mujoco]` alone — no lerobot needed. |
| `stop_cameras_recording(...)` | Stop plain MP4 recording. |
| `get_cameras_recording_status()` | Status of the plain MP4 recorder. |

## Randomization

| Action | What |
|--------|------|
| `randomize(randomize_colors=True, randomize_lighting=True, randomize_physics=False, randomize_positions=False, position_noise=0.02, color_range=(0.1,1.0), friction_range=(0.5,1.5), mass_range=(0.5,2.0), seed=None)` | Domain randomization. Destructive — recompile the scene to undo. |

See [Domain randomization](domain-randomization.md) for the full distribution it samples.

## Asset registry

| Action | What |
|--------|------|
| `list_urdfs()` | Loaded URDFs/MJCFs in the current world. |
| `register_urdf(name, path)` | Register an additional asset. |
| `get_features(robot_name=None)` | Observation/action feature schema (used by recording). |

## Return shape

Every action returns a dict shaped:

```python
{
    "status": "success" | "error",
    "content": [{"text": "..."}, {"image": ...}, ...]
}
```

`status="success"` plus any requested data embedded in `content` entries. On error,
`content[0]["text"]` carries a human-readable explanation. The dispatch layer never
raises out of an action — agents always get a structured response.

!!! note "render() returns PNG bytes, not a numpy array"
    `render()` embeds a PNG inside `content[...]["image"]["source"]["bytes"]`.
    There is no `frame` key in the response.
    For a numpy array in your own loop, use `sim.get_observation(robot_name)[camera_name]`
    which returns `numpy.uint8 (H, W, 3)`.

## See also

- [Tutorial 2 — Simulation](../tutorial/02-simulation.md) — concrete examples.
- [World building](world-building.md) — composing scenes with multiple robots and
  objects.
- [Domain randomization](domain-randomization.md) — `randomize` distributions.
- [Architecture](../architecture.md) — how `Simulation` implements `SimEngine`.
