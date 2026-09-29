---
description: The mjlab backend: GPU-vectorized MuJoCo (MuJoCo-Warp) behind SimEngine, with rsl_rl training, ONNX policies and N-world evaluation.
---

# mjlab

By the end of this page you know when to pick `backend="mjlab"` over `mujoco`, what changes when `num_envs` is more than one, and how a policy trained here comes back through the `Policy` ABC.

```bash
pip install 'strands-robots[sim-mjlab]'   # mjlab 1.6, mujoco 3.11, mujoco-warp, warp-lang, onnxruntime, plus [sim-mujoco]
```

## What it is

`MjlabEngine` (`strands_robots/simulation/mjlab/simulation.py`) implements `SimEngine` on [mujocolab/mjlab](https://github.com/mujocolab/mjlab): Isaac Lab's manager-based API on MuJoCo-Warp. It loads the same MJCF assets as the MuJoCo backend, compiles them once, and steps `num_envs` copies of the scene in one Warp kernel launch on the GPU. World 0 speaks the single-world `SimEngine` contract (`get_observation`, `send_action`, `render`, `run_policy`, LeRobot recording), so `Robot("so101", backend="mjlab")` behaves like the MuJoCo backend; the batch methods add the rest.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mjlab", num_envs=256)
sim.create_world()
sim.add_robot("so101")
sim.randomize(randomize_physics=True, seed=0)   # 256 different friction / mass draws
obs = sim.get_observation_batch("so101")          # {joint: tensor (256,)} on the GPU
sim.destroy()
```

## Constructor keywords

| keyword | default | meaning |
|---|---|---|
| `num_envs` | `1` | worlds stepped together; world 0 is the classic single-world view |
| `device` | `None` | Warp / torch device, the first CUDA device when `None` |
| `default_timestep` | `0.002` | physics step in seconds |
| `env_spacing` | `2.0` | metres between world origins |
| `nconmax`, `njmax` | `None` | contact / constraint capacity per world, mjlab's estimate when `None` |
| `use_cuda_graph` | `False` | capture the step in a CUDA graph (about +16 percent at so101 N=1024, nothing below) |

## Parity with the MuJoCo backend

`tests/simulation/mjlab/test_parity_with_mujoco_backend.py` drives the same sinusoid through both backends: so101 joint trajectories agree to `8e-5` rad over 2 s, the Unitree G1 free base settles within `2e-4` m over 1 s, and `send_action(n_substeps=1)` advances physics on both, which `run_policy` relies on. `keyframe=None` spawns the zero pose on both. The same rsl_rl reach actor scores 8/16 on identical seeded targets on either engine (`examples/mjlab/vec_eval_bench.py`).

## Throughput

Steps per second of physics on a Jetson AGX Thor, so101 with 10 substeps (`scratch/throughput.py` in the lane directory, numbers in `scratch/throughput.json`):

| worlds | mujoco (CPU) | mjlab |
|---|---|---|
| 1 | 50,500 | 1,100 |
| 64 | | 60,500 |
| 1024 | | 544,000 |

One world on the GPU is slower than the CPU; the crossover is around 40 to 54 worlds. Episodes per minute for a 150-tick reach rollout, LeRobot v3 writing included: classic 227, mjlab 416 at N=16, 740 at N=256, 807 at N=1024 (`scratch/vec_eval.json` in the lane directory). Without the writer the N=1024 rollout runs at 6,300 episodes per minute; the serial LeRobot writer is the ceiling, not physics.

## Batched operations

- `get_observation_batch(robot)` and `send_action_batch(targets, robot, n_substeps)` move `(N, ...)` tensors without leaving the GPU.
- `randomize(randomize_physics=True, randomize_positions=True, seed=...)` draws per world: geom friction, body mass with inertia scaled alongside, object root pose. Repeat calls scale from the compiled defaults, so they do not compound. `randomize_colors` and `randomize_lighting` are refused: the batched worlds have no renderer, so use the MuJoCo backend for visual randomization.
- `strands_robots.training.mjlab_tasks.vec_eval.vec_rollout` runs any `Policy` provider on all worlds in lockstep (one `get_actions_batch` call per tick when the provider has it, else N concurrent `get_actions`), and `BatchedLeRobotRecorder` flushes the N episodes into one LeRobot v3 dataset through the classic recorder. Cameras are not recorded on this path.

## Training and the way back

`train_policy(provider="rsl_rl", extra={"task": "Strands-Reach-SO101"}, batch_size=512, steps=300, output_dir=...)` runs mjlab's PPO in-process and exports ONNX with a dynamic batch axis (`strands_robots/training/mjlab_tasks/export.py`). `Robot(...).execute(policy="rsl_rl_onnx", onnx_path=...)` loads it through `strands_robots/policies/rsl_rl_onnx`, which rebuilds the observation vector from the metadata mjlab attaches to the graph (joint positions minus defaults, velocities, previous action, command) and returns joint targets as Python floats. The same ONNX file runs on `mujoco`, `mjlab` and hardware. `examples/mjlab/train_then_deploy_tools.py` is the two-tool transcript: train, then roll out.

## Limits

- NVIDIA GPU only. The first build of a scene JIT-compiles Warp kernels (about 80 s for so101 on Thor; cached afterwards).
- Pinned to `mujoco>=3.11,<3.12` and `torch>=2.14` through mjlab, while `[lerobot]` pins `torch<2.12`: install `[sim-mjlab]` in its own environment, or install it second so its torch wins. `[all]` does not include it for this reason.
- No batched camera rendering. `render()` rasterises world 0 with `mujoco.Renderer` from the CPU model.
- mjlab's own ONNX export bakes a batch of 1; the provider scores such graphs one world at a time and warns once. Export through `mjlab_tasks.export` for one batched call.
