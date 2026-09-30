---
description: The MuJoCo backend: install, offscreen rendering, physics queries, scene export and the MJCF editing surface.
---

# MuJoCo

By the end of this page you can run the default backend headless on a laptop, read physics quantities out of it, save and restore states, and know which calls are MuJoCo-only.

```bash
pip install 'strands-robots[sim-mujoco]'   # mujoco, robot_descriptions, imageio(-ffmpeg), mink, qpsolvers[daqp]
export MUJOCO_GL=cgl                       # macOS. Linux without a display: egl or osmesa
```

## What it is

`MuJoCoSimEngine` (`strands_robots/simulation/mujoco/`) is CPU physics at 500 Hz by default plus offscreen rendering, assembled from mixins: `physics.py` (queries and state), `scene_ops.py` and `spec_builder.py` (MjSpec editing), `rendering.py` (cameras, observations), `manipulation.py` (attach, actuate), `randomization.py`, `motion_primitives.py`, `recording.py`. Robot models come from `robot_descriptions` and the bundled menagerie assets; `create_world` reports how many are available.

```python
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco", mesh=False)
sim.create_world()
sim.add_robot("so101")
sim.add_object(name="cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.25, 0.0, 0.15])
sim.step(200)
print(sim.get_body_state("cube")["content"][0]["text"].splitlines()[1])
sim.save_state("before")
sim.apply_force("cube", force=[0.0, 0.0, 5.0])
sim.step(50)
sim.load_state("before")
print(sim.export_xml()["status"], sim.get_total_mass()["status"], sim.get_energy()["status"])
sim.cleanup()
```

You should see the cube's `pos:` line with `z` near `0.015` (it fell and settled), then three `success` values.

## Rendering

`render(camera_name="default", width=None, height=None)` returns a PNG in the content list; `get_observation(robot)` returns the raw `(H, W, 3)` array for every camera plus a scalar per joint and its `.vel` companion. `skip_images=True` skips rendering, which is the 10x throughput win a non-VLA policy gets from `requires_images = False`. `open_viewer()` opens the interactive MuJoCo viewer when a display exists.

## Physics surface

| call | returns |
|---|---|
| `get_body_state(body)` | position, `wxyz` quaternion, linear and angular velocity |
| `get_contacts()`, `get_contact_forces()` | contact pairs, normal forces |
| `raycast(origin, direction)`, `multi_raycast(...)` | first hit and distance |
| `get_jacobian(body)`, `get_mass_matrix()`, `inverse_dynamics()`, `get_energy()`, `get_total_mass()` | dynamics quantities |
| `forward_kinematics(body)` | world pose after a forward pass |
| `get_sensor_data(name)` | any `<sensor>` in the model |
| `set_joint_positions(...)`, `set_joint_velocities(...)` | write state directly |
| `set_body_properties(...)`, `set_geom_properties(...)` | mass, friction, colour at run time |
| `save_state(name)`, `load_state(name)` | named snapshots of `qpos`, `qvel`, `ctrl` |
| `set_gravity(...)`, `set_timestep(...)` | world parameters after creation |
| `get_ground_height(x, y)` | terrain height under a point |

`mj_model` and `mj_data` expose the compiled model and data when you need the raw API.

## Scene editing

The scene is an `MjSpec` that is recompiled after every structural change, so `add_robot`, `add_object` and `add_camera` work on a live world. `patch_scene_mjcf(ops)` applies `add_body`, `add_geom`, `add_site`, `set_body_pos`, `set_body_quat`, `delete_body` atomically; `replace_scene_mjcf(xml)` swaps the whole model; `load_scene(path)` starts from an MJCF file; `export_xml(path)` writes the current model out. Details and the size conventions are on [worlds and objects](worlds-and-objects.md).

## Time

`create_world(timestep=0.002)` sets physics at 500 Hz. `run_policy(control_frequency=50.0)` steps `1 / (50 * timestep)` physics substeps per action; pass `control_substeps` to pin it. `step(n)` advances `n` physics steps. `physics_timestep()` reads the live value.

When the physics diverges (a huge force, gain or timestep), MuJoCo resets the whole world to its initial pose and rewinds the clock. `step`, `send_action` and the motion primitives then answer `status="error"` with `{"diverged": true}`, naming the joint, and rollouts and evaluations stop there instead of scoring the reset world. `reset()` or `load_state(...)` recovers.

## Limits

- CPU only. One process steps one world; for batched environments use `newton` or `isaac`.
- Offscreen rendering needs a working GL backend. On a headless Linux box without EGL, set `MUJOCO_GL=osmesa` and accept slow frames.
- `mjx` is accepted as an alias but resolves to the same CPU engine; there is no JAX path.
