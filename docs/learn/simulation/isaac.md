---
description: The Isaac Sim backend: what it needs, how to construct it, USD and MJCF loading, what differs from MuJoCo.
---

# Isaac Sim

By the end of this page you know what the `isaac` backend requires, how to construct it, and which MuJoCo habits do not carry over.

```bash
pip install 'strands-robots[sim-isaac]'                                            # usd-core, imageio, mujoco+mink IK (not Isaac Sim)
pip install 'isaacsim[all,extscache]==6.0.*' --extra-index-url https://pypi.nvidia.com   # Isaac Sim, Python 3.12 only
# under uv, both flags are required:
# uv pip install 'isaacsim[all,extscache]==6.0.*' --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
export OMNI_KIT_ACCEPT_EULA=YES                                                     # first import
```

Verified pip wheels: 6.0.1.0 and 6.1.0.0 (or `==6.1.*`). Both pin `numpy==2.3.1` and `torch==2.11.0`: use a fresh venv; on 6.0.x reinstall `coverage>=7.6.1` (its 7.4.4 pin breaks numba). `lerobot` (recording) needs `numpy<2.3`: install it last (2.2.6 works on 6.1). Docker: `nvcr.io/nvidia/isaac-sim:6.0.1`. Pins: `strands_robots/simulation/isaac/_install.py`.

## What it is

`IsaacSimulation` (`strands_robots/simulation/isaac/simulation.py`) implements `SimEngine` on NVIDIA Isaac Sim / Omniverse: photoreal rendering, synthetic data, GPU-rendered sensors, USD stages. It inherits the policy orchestration (`run_policy`, `eval_policy`, benchmarks, recording) from the base class and implements physics, loaders (`isaac/loaders.py`: URDF, MJCF, USD), asset conversion, motion primitives and recording.

```python title="sketch"
from strands_robots.simulation import create_simulation
from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

ok, msg = IsaacSimulation.is_available()      # cheap probe, no stage created
print(ok, msg)

sim = create_simulation("isaac", headless=True, render_mode="rtx_realtime")  # shortcut kwargs
sim = IsaacSimulation(IsaacConfig(headless=True, render_mode="rtx_realtime"))      # same thing
sim.create_world()
sim.add_robot("so101")
sim.add_camera(name="front", position=[0.6, 0.0, 0.5], target=[0.2, 0.0, 0.0])
result = sim.run_policy(robot_name="so101", policy_provider="mock", n_steps=100)
print(result["status"])
sim.destroy()
```

## Configuration

`IsaacConfig` fields, with defaults: `num_envs=1`, `device="cuda:0"`, `headless=True`, `physics_dt=1/120`, `rendering_dt=1/30`, `render_mode="headless"`, `gravity=(0, 0, -9.81)`, `ground_plane=True`, `stage_path="/World"`, `nucleus_url=None`, `camera_width=640`, `camera_height=480`, `verbose=False`, `kit_args=()`, `task_threads=None`, `boot_timeout_s=None`, `extra={}`. Unknown keywords are rejected. Legacy `tool_name` and `default_timestep` shortcuts still work.

## Differences from MuJoCo

| topic | Isaac |
|---|---|
| assets | URDF, MJCF (converted, `isaac/mjcf_assets.py`) and USD; meshes through `isaac/mesh_assets.py` |
| fixed base | robots import with the root welded (`fixed_base=True` by default) |
| cameras | world-frame prims; `add_camera(parent_body=...)` is refused |
| physics rate | `physics_dt` and `rendering_dt` are separate clocks |
| WBC | no MuJoCo torque shim; a policy declaring `requires_action_controller` (`wbc`) is refused |
| motion primitives | its own implementation in `isaac/motion_primitives.py` |
| randomization | `IsaacRandomizationMixin`, same `randomize` / `set_obs_noise` names |

## Threading

Kit updates only on the `SimulationApp` thread; `run_agent` pumps it while agent tools run elsewhere:

```python
sim = create_simulation("isaac")
sim.create_world(); sim.add_robot("so101")
agent = Agent(tools=[sim])
sim.run_agent(agent, "add a cube, reset, step 30, render")
```

Unpumped worker calls are refused; manually: `run_pump_forever(stop_event=...)` plus `run_on_main(fn)`.

## Limits

- Python 3.12 only, an RTX-class GPU, and a multi-gigabyte install. No CPU fallback; `is_available()` says why.
- Physics runs on **CPU PhysX**: `device` is reported as `device_requested` but not forwarded, because the GPU pipeline breaks incremental `add_robot`. Rendering uses the GPU.
- `render_mode="headless"` (the default) renders nothing; pass `render_mode="rtx_realtime"` (also with `headless=True`).
- Rendering is slower per frame than MuJoCo's and faster per batch: use it for fidelity, not unit-test loops.
- `remove_robot`, like a dynamic `remove_object`, invalidates the tensor view: every call reading it (`add_robot` too) refuses until `reset()`: build first, then reset.
