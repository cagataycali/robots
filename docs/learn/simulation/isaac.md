---
description: The Isaac Sim backend: what it needs, how to construct it, USD and MJCF loading, and what differs from MuJoCo.
---

# Isaac Sim

By the end of this page you know exactly what the `isaac` backend requires, how to construct it, and which MuJoCo habits do not carry over.

```bash
pip install 'strands-robots[sim-isaac]'                                            # usd-core + imageio; not Isaac Sim itself
pip install 'isaacsim[all,extscache]==6.0.*' --extra-index-url https://pypi.nvidia.com   # Isaac Sim, Python 3.12 only
export OMNI_KIT_ACCEPT_EULA=YES                                                     # first import
```

The docker route is `nvcr.io/nvidia/isaac-sim:6.0.1`. The pinned image tag and install lines live in `strands_robots/simulation/isaac/_install.py`; that file is the source of truth when they move.

## What it is

`IsaacSimulation` (`strands_robots/simulation/isaac/simulation.py`) implements `SimEngine` on NVIDIA Isaac Sim / Omniverse: photoreal rendering, synthetic data, GPU-batched sensors, USD stages. It inherits the policy orchestration (`run_policy`, `eval_policy`, benchmarks, recording) from the base class and implements the physics primitives, loaders (`isaac/loaders.py`: URDF, MJCF and USD), mesh and MJCF asset conversion, motion primitives and recording.

```python title="sketch"
from strands_robots.simulation import create_simulation
from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

ok, msg = IsaacSimulation.is_available()      # cheap probe, no stage created
print(ok, msg)

sim = create_simulation("isaac", num_envs=1, headless=True)          # shortcut kwargs
sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True, physics_dt=1 / 120))   # same thing
sim.create_world()
sim.add_robot("so101")
sim.add_camera(name="front", position=[0.6, 0.0, 0.5], target=[0.2, 0.0, 0.0])
result = sim.run_policy(robot_name="so101", policy_provider="mock", n_steps=100)
print(result["status"])
sim.destroy()
```

## Configuration

`IsaacConfig` fields, with defaults: `num_envs=1`, `device="cuda:0"`, `headless=True`, `physics_dt=1/120`, `rendering_dt=1/30`, `render_mode="headless"`, `gravity=(0, 0, -9.81)`, `ground_plane=True`, `stage_path="/World"`, `nucleus_url=None`, `camera_width=640`, `camera_height=480`, `verbose=False`, `extra={}`. Unknown keywords are rejected at construction rather than dropped, so `headles=False` is an error, not a silent default. The legacy `tool_name` and `default_timestep` shortcuts from `create_simulation` are still accepted.

## Differences from MuJoCo

| topic | Isaac |
|---|---|
| assets | URDF, MJCF (converted, `isaac/mjcf_assets.py`) and USD; meshes through `isaac/mesh_assets.py` |
| fixed base | robots import with the root welded by default (`fixed_base=True` on the internal robot record) |
| cameras | prims under the stage camera scope; `add_camera(parent_body=...)` is refused with the world-frame alternative named |
| physics rate | `physics_dt` and `rendering_dt` are separate clocks |
| WBC | cannot install the MuJoCo torque shim; a `wbc` rollout refuses unless `wbc_install_torque_control=False` on a torque-actuated scene |
| motion primitives | its own implementation in `isaac/motion_primitives.py` |
| randomization | `IsaacRandomizationMixin`, same `randomize` / `set_obs_noise` names |

`docs-old/reference/simulation/isaac-parity.md` tracked what matched and what did not at the time of writing; the table above is what the code says at this commit.

## Limits

- Python 3.12 only, an RTX-class GPU, and a multi-gigabyte install. There is no CPU fallback; `is_available()` tells you why before anything is built.
- Wrist cameras that ride with the arm are a MuJoCo and Newton feature. On Isaac, place the camera in world coordinates.
- Rendering is slower per frame than MuJoCo's offscreen path and faster per batch; the backend is for fidelity and scale, not for the inner loop of a unit test.
- `remove_robot` deletes the articulation prim from the stage, and like a dynamic `remove_object` it invalidates the tensor view: `step()` and `send_action()` refuse until the next `reset()` rebuilds it. Build the scene and reset before posing anything.
