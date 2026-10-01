---
description: "The Newton backend: NVIDIA Warp and MuJoCo-Warp GPU physics on the same MJCF assets, with ray-traced tiled cameras."
---

# Newton

By the end of this page you know what the `newton` backend is, which solver names it accepts, and what it needs from your machine.

```bash
pip install 'strands-robots[sim-newton]'   # newton 1.5, warp-lang, mujoco 3.11, mujoco-warp, trimesh, plus [sim-mujoco]
```

## What it is

`NewtonSimEngine` (`strands_robots/simulation/newton/simulation.py`) implements `SimEngine` on newton-physics/newton, NVIDIA Warp plus MuJoCo-Warp. It ingests the same MJCF assets the MuJoCo backend uses, builds a GPU model, and steps it with one of Newton's rigid-body solvers. Rendering is headless through Newton's ray-traced `SensorTiledCamera`. Policy orchestration (`run_policy`, `eval_policy`, `replay_episode`, recording in `newton/recording.py`) is inherited from the base class.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("newton", solver="mujoco")
sim.create_world()
sim.add_robot("so100")
sim.send_action({"Rotation": 0.5}, robot_name="so100")
out = sim.render(width=320, height=240)
print(out["status"])
sim.destroy()
```

## Constructor keywords

| keyword | default | meaning |
|---|---|---|
| `solver` | `"mujoco"` | Newton solver by friendly name; see below |
| `default_timestep` | backend default | physics step in seconds |
| `substeps` | `10` | solver substeps per physics step |
| `device` | `None` | Warp device, auto-selected when `None` |
| `default_width`, `default_height` | `640`, `480` | camera resolution when a call gives none |

## Solvers

`strands_robots.simulation.newton.backend.solver_registry()` maps friendly names to Newton classes: `mujoco`, `featherstone`, `xpbd`, `semi_implicit`, `vbd`, `style3d`, `mpm`, `kamino`. `articulated_solvers()` returns the subset that can drive a robot: `mujoco`, `featherstone` and `kamino`. The rest are refused by name with the reason: `xpbd` and `semi_implicit` step without integrating rigid bodies and leave the world frozen, `vbd` needs per-body colouring the build does not apply, `style3d` is a cloth solver, `mpm` needs a config object this backend does not build.

## Same API, same rules

`add_camera(parent_body=...)` works here as on MuJoCo; a policy declaring `requires_action_controller` (the WBC torque shim) is refused rather than rolled out without it. On Newton a mesh `size` scales the mesh per axis (default `[1, 1, 1]`); MuJoCo and Isaac ignore it and still report success (#2300). `set_obs_noise` mirrors the MuJoCo signature so an identical call behaves the same. Terrain, task objects and the predicate DSL read the same observation surface. The `wbc` torque shim is MuJoCo-only, so a WBC rollout on Newton refuses unless `wbc_install_torque_control=False` against a torque-actuated scene.

## Limits

- NVIDIA GPU and CUDA-capable Warp. There is no CPU device for the articulated solvers at useful speed.
- Pinned to `mujoco>=3.11,<3.12` and `mujoco-warp` of the same series, while `[sim-mujoco]` allows any 3.5+. Install `[sim-newton]` in its own environment if you also want the newest MuJoCo release.
- Rendering is ray-traced and tiled: per-frame cost above MuJoCo's rasteriser, per-batch cost below.
