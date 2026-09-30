---
description: The mjlab backend: GPU-vectorized MuJoCo (MuJoCo-Warp) behind SimEngine, with rsl_rl training, ONNX policies and N-world evaluation.
---

# mjlab

By the end of this page you know when `backend="mjlab"` beats `mujoco`, what `num_envs` changes, and how a policy trained here comes back through the `Policy` ABC.

```bash
pip install 'strands-robots[sim-mjlab]'   # mjlab 1.6, mujoco 3.11, mujoco-warp, onnxruntime
```

## What it is

`MjlabEngine` implements `SimEngine` on [mujocolab/mjlab](https://github.com/mujocolab/mjlab): Isaac Lab's manager API on MuJoCo-Warp. The MuJoCo backend's MJCF assets, compiled once, `num_envs` copies stepped in one Warp launch. World 0 speaks the single-world contract (`get_observation`, `send_action`, `render`, `run_policy`, LeRobot recording), so `Robot("so101", backend="mjlab")` behaves like `mujoco`; batch methods add the rest.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mjlab", num_envs=256)
sim.create_world()
sim.add_robot("so101")
sim.randomize(randomize_physics=True, seed=0)   # 256 different friction / mass draws
obs = sim.get_observation_batch("so101")          # {joint: tensor (256,)} on the GPU
sim.destroy()
```

Constructor keywords: `num_envs` (1), `device` (first CUDA device), `default_timestep` (0.002), `env_spacing` (2.0 m), `nconmax` / `njmax` (mjlab's estimate), `use_cuda_graph` (False; +16 percent at so101 N=1024, nothing below).

## Parity and throughput

The parity tests drive one sinusoid through both backends: so101 joints agree to 8e-5 rad over 2 s, the Unitree G1 free base to 2e-4 m over 1 s, and the same rsl_rl reach actor scores 8/16 on identical seeded targets on both engines (`examples/mjlab/vec_eval_bench.py`).

Physics steps per second, so101, Jetson AGX Thor: mujoco CPU 50,500 at N=1; mjlab 1,100 at N=1, 60,500 at N=64, 544,000 at N=1024. One GPU world is slower than the CPU; crossover is near 50 worlds. Episodes per minute, a 150-tick reach rollout written to LeRobot v3: classic 227, mjlab 416 at N=16, 807 at N=1024 (6,300 without the writer). The serial LeRobot writer is the ceiling.

## Batched operations

- `get_observation_batch(robot)` / `send_action_batch(targets, robot, n_substeps)` move `(N, ...)` tensors on the GPU.
- `randomize(randomize_physics=True, randomize_positions=True, seed=...)` draws per world: geom friction, body mass with inertia, object pose. Repeat calls scale from compiled defaults. `randomize_colors` and `randomize_lighting` are refused (no batched renderer).
- `set_obs_noise(joint_pos_std=..., joint_vel_std=...)` adds Gaussian sensor noise to each world.
- `strands_robots.training.mjlab_tasks.vec_eval.vec_rollout` runs any `Policy` provider on all worlds in lockstep; `BatchedLeRobotRecorder` flushes the N episodes into one LeRobot v3 dataset; no cameras on this path.

## Training and the way back

`train_policy(provider="rsl_rl", extra={"task": "Strands-Reach-SO101"}, batch_size=512, steps=300, output_dir=...)` runs mjlab's PPO in-process and exports ONNX with a dynamic batch axis. `Robot(...).execute(policy="rsl_rl_onnx", onnx_path=...)` loads it on `mujoco`, `mjlab` or hardware ([rl](../policies/rl.md)). `examples/mjlab/train_then_deploy_tools.py` is the transcript.

## Limits

- NVIDIA GPU only; the first build JIT-compiles Warp kernels (about 80 s for so101 on Thor, cached after).
- mjlab pins `mujoco~=3.11` and `torch>=2.14`, `[lerobot]` pins `torch<2.12`: install `[sim-mjlab]` in its own environment or second; `[all]` leaves it out for that reason.
- No batched camera rendering; `render()` rasterises world 0 on the CPU.
- mjlab's own ONNX export bakes a batch of 1; the provider scores such graphs one world at a time, warning once.
