---
description: Simulation vs SimEngine, the three backends and what each needs, and a verified MuJoCo session from create_simulation to a policy rollout.
---

# Simulation

By the end of this page you have a MuJoCo world with a robot, an object and a camera running on this machine, you know the one interface every backend implements, and you know which backend needs what.

```python
from strands_robots.simulation import create_simulation, list_backends

print(list_backends())
sim = create_simulation("mujoco", mesh=False)
sim.create_world(timestep=0.002)
sim.add_robot("so101")
sim.add_object(name="cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.25, 0.0, 0.015], color=[1, 0, 0, 1])
sim.add_camera(name="front", position=[0.6, 0.0, 0.5], target=[0.2, 0.0, 0.0])
sim.step(100)
print(sim.get_state()["content"][0]["text"])
obs = sim.get_observation("so101")
print({k: v.shape for k, v in obs.items() if hasattr(v, "shape")})
sim.cleanup()
```

You should see:

```text
['isaac', 'isaac_sim', 'isaacsim', 'mj', 'mjc', 'mjx', 'mujoco', 'newton', 'nt', 'nvidia']
Simulation State
t=0.2000s (step 100)
dt=0.002s | g=[0.0, 0.0, -9.81]
Robots: 1 | Objects: 1 | Cameras: 2
Bodies: 9 | Joints: 7 | Actuators: 6
{'default': (480, 640, 3), 'front': (480, 640, 3)}
```

Every call returns an agent-tool envelope: `{"status": "success" | "error", "content": [{"text": ...}, {"json": ...}]}`. The same object is a Strands tool, so an agent gets the same surface you do.

## SimEngine and Simulation

`SimEngine` (`strands_robots/simulation/base.py`) is the abstract contract: world lifecycle (`create_world`, `reset`, `step`, `destroy`), entities (`add_robot`, `add_object`, `add_camera`), observation (`get_observation`, `render`, `get_contacts`), actuation (`send_action`), and the policy orchestration that is implemented once on the base and inherited by every backend: `run_policy`, `run_multi_policy`, `eval_policy`, `evaluate_benchmark`, `start_policy` / `stop_policy`, `replay_episode`, dataset recording.

`Simulation` is the MuJoCo engine under its historical name. `from strands_robots.simulation import Simulation` and `create_simulation("mujoco")` give you the same `MuJoCoSimEngine`. `create_simulation` is the door: it resolves an alias, imports the backend lazily, and passes the remaining keywords to the constructor.

```python
from strands_robots.simulation import SimEngine, Simulation, create_simulation

sim = create_simulation("mj", mesh=False)
print(type(sim).__name__, isinstance(sim, SimEngine), type(sim) is Simulation)
```

You should see `MuJoCoSimEngine True True`.

## Backends

| backend | aliases | install | needs | good for |
|---|---|---|---|---|
| [`mujoco`](mujoco.md) | `mj`, `mjc`, `mjx` | `strands-robots[sim-mujoco]` | a CPU; offscreen rendering via `MUJOCO_GL` | everything on this site, the default |
| [`newton`](newton.md) | `nt` | `strands-robots[sim-newton]` | an NVIDIA GPU with Warp; same MJCF assets | GPU stepping, ray-traced tiled cameras |
| [`isaac`](isaac.md) | `isaac_sim`, `isaacsim`, `nvidia` | `strands-robots[sim-isaac]` plus Isaac Sim 6.0 | Isaac Sim on Python 3.12, an RTX GPU | photoreal rendering, USD scenes, batched envs |

Built-ins win over entry-point plugins of the same name. A third-party package registers a backend under the `strands_robots.backends` entry-point group; `register_backend("my_sim", lambda: MySimEngine, aliases=["custom"])` does the same at runtime. An unknown name is a `ValueError` listing what is available and, for `newton`, `warp` and `mjwarp`, the `pip install` line.

## Constructor keywords

Keywords after the backend name go to the engine constructor. MuJoCo takes `tool_name`, `default_timestep=0.002`, `default_width=640`, `default_height=480`, `mesh`, `peer_id`, `ros2_bridge`, `ros2_domain`, `render_dir`. `mesh=False` keeps the engine off the fleet mesh, which is what a standalone script wants. Newton takes `solver="mujoco"`, `substeps=10`, `device`. Isaac takes an `IsaacConfig` or its fields as shortcuts (`num_envs`, `headless`, `physics_dt`, ...).

## Robots you can add

`add_robot(name)` resolves a registry name (`so101`, `panda`, `g1`, `go2`, ...) through `strands_robots.simulation.model_registry`; `add_robot(name="arm", data_config="franka")` gives the instance its own name. `urdf_path` loads a file, which is also how [task objects](worlds-and-objects.md) enter a scene. `keyframe="home"` spawns a canonical pose from the model's `<keyframe>`. Joint names are the model's own: the SO-101 is `1..6`, the Panda is `joint1..joint7` plus `finger_joint1`, `finger_joint2`. Read them with `sim.robot_joint_names(name)`; the [robots catalog](../../robots/index.md) lists every name.

## Where next

| to | read |
|---|---|
| build scenes: objects, cameras, articulated task objects, MJCF patches, terrain | [worlds and objects](worlds-and-objects.md) |
| stop, score and benchmark a rollout | [predicates and rollouts](predicates-and-rollouts.md) |
| randomize physics and sensors for sim to real | [randomization](randomization.md) |
| drive a robot with a policy | [policies](../policies/index.md) |
| train against the sim | [training](../training/index.md) |
