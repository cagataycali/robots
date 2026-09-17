---
description: The Simulation AgentTool - every action grouped by category, with parameters.
---

# Simulation overview

```python
from strands_robots import Robot
sim = Robot("so100")   # preferred factory; 60+ actions as an AgentTool
```

Stepping the world and commanding actuators is [Physics and actions](physics.md);
running a policy in it is [Policy rollouts](rollouts.md). Every action answers with a tool-result dict: `status="success"` plus a `text` block, or
`status="error"` naming the parameter and the remedy. Three numeric domains run
through the whole surface, and every write is all-or-nothing - a refused value
leaves `qpos`, `qvel` and every latched wrench untouched.

| Domain | Applies to | Refused |
|--------|-----------|---------|
| Finite number | joint positions and velocities, `apply_force` vectors, `mass`, `gravity`, `timestep`, `randomize` ranges, `set_obs_noise` magnitudes, `send_action` values | `nan` / `inf`, and a `bool` / `numpy.bool_` - `float(True)` is `1.0`, so `set_gravity(True)` would configure +1 m/s^2 pointing **up** and report success. `1`, `1.0` and NumPy scalars stay accepted |
| `mjMAXVAL` (1e10) ceiling | values landing in `qpos` / `qvel`: `set_joint_positions`, `set_joint_velocities`, `move_object`, `add_object` | past it `mj_step` calls the simulation unstable and resets **every** joint and object, reporting it only on stderr. `mjMAXVAL` exactly is writable; a static object owns no `qpos` entry |
| Component count | `position`, `target`, `origin`, `force`, `torque`, `point`, `gravity`, `direction`, `orientation`, `color`, `pixels`, `send_action`'s vector form | a wrong length, and a value carrying no readable count (a 0-d NumPy array or torch tensor). Correctly sized NumPy arrays are accepted throughout |

## World

| Action | Key params | Notes |
|--------|-----------|-------|
| `create_world` | `timestep=0.002`, `gravity=[0,0,-9.81]`, `ground_plane=True` | Implicit on `Robot()` |
| `load_scene` | `scene_path` | Replace world with MJCF |
| `reset` | - | State to t=0, keep model |
| `get_state` | - | Sim time, joint positions, object poses |
| `destroy` / `cleanup` | - | Release the world, renderers, executor, ROS 2 bridge, attached teleop devices and mesh peer. A script that calls neither (nor the context manager) is covered at process exit by the MuJoCo backend's `atexit` hook, but only the explicit call hands you a result and frees GPU/GL resources where you stop needing them |
| `export_xml` | `output_path` | Serialise live scene to MJCF; reloadable via `load_scene` (assets referenced by absolute path) |

## Scene-MJCF

| Action | Notes |
|--------|-------|
| `replace_scene_mjcf(xml)` | Swap entire world XML |
| `patch_scene_mjcf(ops)` | Incremental patches, no full recompile |
| `raycast(origin, direction, ...)` | Single ray-mesh intersection |
| `multi_raycast(origin, directions, ...)` | Batch ray-mesh intersections from one origin; all-or-nothing, a direction it cannot cast refuses the batch |

## Robots

| Action | Key params |
|--------|-----------|
| `add_robot` | `robot_name`, `position=[0,0,0]`, `data_config=None`, `urdf_path=None` |
| `remove_robot` | `name` |
| `list_robots` | - each robot's asset, joint count and **live** base position, read from the physics rather than from the `add_robot` request, so a robot that walked reports where it is |
| `get_robot_state` | `name` -> joint positions, velocities, torques, plus an `end_effector` line naming the frame `move_to` drives, its world position, its measured offset `from_base` and the horizontal axis the arm extends along (`"+X"` / `"-Y"` / `null` when the arm is over its base) - so "in front of the robot" resolves to the same side for the agent and for you |

`move_to` and that `end_effector` line follow the frame `discover_ee_frame` finds:
a tool-point **site** first (`tcp`, `gripper`, `attachment_site`, ...), else a
hand/wrist **body**. A model shipping no site lands on the wrist, so a registry
entry may declare the tool point the model lacks (the shipped `so100` does - its
Menagerie model has zero sites) and the backend adds that site before the attach:

```json
"tool_frame": {"body": "Fixed_Jaw", "pos": [0.0, -0.0995, 0.001], "site": "tcp"}
```

`body` is the model's own body name, `pos` is metres in that body's frame, `site`
defaults to `tcp`. A malformed block, or one naming a body the model lacks,
refuses `add_robot` with the reason rather than falling back to the wrist. The
overlay `user_robots.json` may declare one for your own robot.

## Objects

| Action | Key params |
|--------|-----------|
| `add_object` | `name`, `shape="box"\|"sphere"\|"cylinder"\|"plane"\|"mesh"`, `size`, `position=[x,y,z]`, `color=[r,g,b,a]`, `orientation=[w,x,y,z]`, `mass=0.1`, `is_static=None`, `mesh_path=None` - omitted lets the shape decide: `plane` is made static and refuses an explicit `is_static=False`, every other shape is dynamic |
| `remove_object` | `name` |
| `move_object` | `name`, `position`, `orientation` (NOT `pos`/`quat`) |
| `list_objects` | - each object's shape, mass and **live** position, so a settled or pushed object reports where it is |

## Cameras

| Action | Key params |
|--------|-----------|
| `add_camera` | `name`, `position`, `target`, `fov=60.0`, `width=640`, `height=480` - no `attach_to`/`fovy`/`lookat` |
| `remove_camera` | `name` |
| `list_cameras` | - every name `render` / `start_recording` accepts: the built-in `"default"` free view first, then model-defined and `add_camera` cameras. Equals `describe()["cameras"]` and matches the Newton backend, so a rollout rig can be enumerated instead of guessed |

Robot-URDF cameras are auto-discovered on `add_robot`.

## Rendering

| Action | Notes |
|--------|-------|
| `render(camera_name="default", width=None, height=None)` | PNG in `content[...]["image"]["source"]["bytes"]`; no `frame` key. `output_path` is confined to a render sandbox - a bare filename (`"frame.png"`) lands there, an absolute path outside it is refused. The sandbox is `~/.strands_robots/renders`, or per instance `Robot("so101", render_dir="./shots")` / `Simulation(render_dir=...)` |
| `render_depth(...)` | Viewable grayscale depth PNG (near=bright, far=dark) + metric `depth_min`/`depth_max` (metres) in the `json` block |
| `render_all(cameras=None, ...)` | One `image` block per camera (multi-view snapshot) |
| `get_world_point(camera_name="default", pixels=[[u, v], ...])` | Ground picked pixels to metric world coordinates via the depth buffer; `point` is the median over the valid samples, `points` aligns with the input pixels |
| `get_camera_params(camera_name)` | Pinhole `K` of the frame the renderer draws. A camera declaring a physical sensor (MJCF `sensorsize` / `focal` / `principal` / `resolution`) has `K` read from MuJoCo's own frustum, so `fx != fy` and an off-centre principal point are honoured; every other camera falls back to `fovy` (square pixels, centred) |
| `open_viewer` / `close_viewer` | Interactive MuJoCo passive viewer |

For a numpy frame use `sim.get_observation(robot_name)[camera_name]` ->
`np.uint8 (H, W, 3)`.

Frame and state reads are serialised against physics: `render`, `render_depth`
and `get_frame` copy mjData under the simulation lock and hand back an
independent buffer (only the PNG encoding runs unlocked), and `get_robot_state`
reads every joint's `qpos`/`qvel`, a floating base's pose and twist and the
`move_to` frame's world position in one critical section. So a read taken while a
policy worker or the `step()` loop advances physics is a snapshot rather than a
splice of two steps. `get_observation`, `get_body_state` and the joint writers
serialise the same way.

## Recording

| Action | Notes |
|--------|-------|
| `start_recording(repo_id, task="", fps=30, ...)` | LeRobot v3 (parquet+MP4); requires `[lerobot]` extra |
| `save_episode()` | Flush the current rollout as one episode; call once per `run_policy` to record N episodes instead of one merged episode |
| `stop_recording(push_to_hub=False, bucket=None, run_id=None)` | Finalise dataset (flushes any trailing rollout) |
| `get_recording_status` | Episode, frame count, output dir |
| `start_cameras_recording(...)` | Plain MP4 via imageio-ffmpeg; `[sim-mujoco]` only, no lerobot |
| `stop_cameras_recording` / `get_cameras_recording_status` | - |

## Randomize

| Action | Key params |
|--------|-----------|
| `randomize` | `randomize_colors=True`, `randomize_lighting=True`, `randomize_physics=False`, `randomize_positions=False`, `position_noise=0.02`, `color_range=(0.1,1.0)`, `friction_range=(0.5,1.5)`, `mass_range=(0.5,2.0)`, `seed=None` |

Destructive - writes into model arrays. Recompile the scene to undo.

## Registry

| Action | Notes |
|--------|-------|
| `list_urdfs` | Built-in robot table, plus a `Registered URDFs:` section naming every `register_urdf` asset and whether it resolves |
| `register_urdf(name, path)` | Register an additional asset - it is named by `list_urdfs` from then on |
| `get_features(robot_name=None)` | Joint / actuator / camera / robot names of the scene (scoped with `robot_name`) - the source of truth for the action keys a policy must emit, and the feature schema used for recording |

Every verb above is listed in `sim.describe()["methods"]`, so an agent builds a
scene, renders it, snapshots it and reads its features from one `describe()` call
instead of guessing names.

## See also

- [Physics and actions](physics.md) - stepping, contacts, forces, state writes, `send_action`.
- [Policy rollouts](rollouts.md) - running, stopping, evaluating and watching a policy.
- [World building](world-building.md) - composing scenes.
- [Domain randomization](domain-randomization.md) - `randomize` distributions.
- [Newton](newton.md) / [Isaac](isaac.md) - the GPU backends and their parity.
- [Architecture](../architecture.md)
