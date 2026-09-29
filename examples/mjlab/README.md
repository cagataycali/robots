# mjlab extreme use cases

Research examples on the mjlab (MuJoCo Warp) backend and the `rsl_rl` trainer of
strands-robots. Every number in this file was measured on one Jetson AGX Thor
(sm_110, 122 GB unified memory) while two or three sibling jobs shared the GPU,
so absolute throughput is a floor, not a ceiling. Raw json and logs behind each
table are named next to it; a number without a file behind it does not appear here.

Install (two steps, mjlab needs torch>=2.14 while the lerobot extra caps torch lower):

```bash
uv pip install "strands-robots[all]"
uv pip install "strands-robots[sim-mjlab,rl]"
export MUJOCO_GL=egl
```

| # | File | Question it answers |
|---|---|---|
| 01 | [`01_scale_sweep.py`](01_scale_sweep.py) | How far does one Thor go: env-steps/s, wall clock to a converged reach policy and GPU memory from 64 to 65,536 worlds, CUDA graph on and off |
| 02 | [`02_every_arm_one_night.py`](02_every_arm_one_night.py) | One reach recipe on every arm in the registry: build, train, export ONNX, evaluate on classic MuJoCo, one table |
| 03 | [`03_humanoid_beyond_velocity.py`](03_humanoid_beyond_velocity.py) | G1 on rough terrain and get-up, evaluated in the native mjlab play env and sim-to-sim on classic MuJoCo |
| 04 | [`04_extreme_domain_randomisation.py`](04_extreme_domain_randomisation.py) | Per-world physics randomisation pushed to the edge, judged on a perturbation grid the policy never saw |
| 07 | [`07_dataset_factory.py`](07_dataset_factory.py) | LeRobot v3 episodes per hour from 1,024 worlds with a scripted expert and with a trained actor |

Helper scripts from the backend lane live next to them (`sim2sim_reach.py`,
`sim2sim_g1_velocity.py`, `vec_eval_bench.py`, `train_then_deploy_*.py`,
`push_run_to_hf.py`).

## 01 Scale sweep

so101 reach task (`strands_robots.training.mjlab_tasks.so101_reach`), PPO through
`MjlabOnPolicyRunner`, 24 steps per env per iteration, seed 42, `at_goal` is the
fraction of worlds with the gripper within 3 cm of the target at the end of an
episode (5-iteration mean must reach 0.6). Data: [`assets/scale_sweep.json`](assets/scale_sweep.json),
plot: [`assets/scale_sweep.png`](assets/scale_sweep.png).

![scale sweep](assets/scale_sweep.png)

| num_envs | CUDA graph | iterations | collect env-steps/s | collect+learn env-steps/s | at_goal >= 0.6 | wall clock to it | peak GPU GB (torch + warp) |
|---|---|---|---|---|---|---|---|
| 64 | on | 250 | 5,910 | 3,985 | it 249 | 92 s | 0.08 |
| 64 | off | 100 | 642 | 611 | not in 100 its (final 0.00) | - | 0.10 |
| 256 | on | 158 | 19,857 | 13,879 | it 157 | 71 s | 0.10 |
| 256 | off | 100 | 2,196 | 2,105 | not in 100 its (final 0.17) | - | 0.17 |
| 1,024 | on | 201 | 60,112 | 46,169 | it 200 | 111 s | 0.16 |
| 1,024 | off | 100 | 5,068 | 4,939 | not in 100 its (final 0.03) | - | 0.47 |
| 4,096 | on | 300 | 122,834 | 97,882 | not in 300 its (final 0.43) | - | 0.41 |
| 4,096 | off | 20 | 11,410 | 11,166 | not in 20 its (final 0.00) | - | 1.65 |
| 8,192 | on | 300 | 151,790 | 120,228 | not in 300 its (final 0.38) | - | 0.74 |
| 16,384 | on | 300 | 203,573 | 157,815 | not in 300 its (final 0.35) | - | 1.40 |
| 65,536 | on | 20 | 224,336 | 172,147 | not in 20 its (final 0.00) | - | 5.35 |

What the table says:

- Collection throughput keeps climbing to 224k env-steps/s at 65,536 worlds and
  the whole scene fits in 3.5 GB; 262,144 worlds failed to allocate 10 GB while a
  CI runner held about 100 GB of the shared memory, so the true ceiling on Thor
  was not reproduced.
- CUDA graphs are the difference between a GPU simulator and a launch-bound one:
  graphs off is 9 to 12x slower at every size and the GPU sits at 3 to 5 percent
  busy. Runs without graphs at 4,096 and above were cut to 20 iterations; they
  would take hours and teach nothing new.
- More worlds buy samples per second, not fewer PPO updates: 256 and 1,024 worlds
  converge in about 200 iterations (71 s and 111 s), 4,096 and above do not
  converge in 300 iterations with the same hyperparameters (the reach
  hyperparameters were tuned at 1,024). Scaling past 1,024 needs a learning-rate
  and batch schedule to match, which is a research question, not a switch.
- 64 worlds converge too (92 s), but only with the graph on; at that size the
  CPU MuJoCo backend of strands-robots is faster (see the backend REPORT).

## 02 Every arm, one night

Results follow (running).

## 03 Humanoid beyond velocity

Results follow (running).

## 04 Extreme domain randomisation

Results follow (running).

## 07 Dataset factory

Results follow (queued).
