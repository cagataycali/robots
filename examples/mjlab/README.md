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
| 05 | [`05_curriculum_goal_box.py`](05_curriculum_goal_box.py) | Per-world curriculum: each world grows its own goal box (1x to 3x) as it succeeds; does it beat a fixed box at equal iterations |
| 06 | [`06_agent_trains_a_fleet.py`](06_agent_trains_a_fleet.py) | A Strands agent with train_policy / evaluate_policy / write_leaderboard tools trains three arms on its own and writes the leaderboard |
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

`02_every_arm_one_night.py --all-arms --num-envs 1024` walks every arm the registry
can simulate (20 today), builds an mjlab reach task for each from the registry
entry alone (end effector = the first site, else the deepest leaf body), trains
200 PPO iterations with one fixed recipe, exports ONNX, and plays 20 goals on
classic MuJoCo (`SimEngine`) through the `rsl_rl_onnx` provider. Same wall-clock
budget for every arm; nothing per-robot except three facts read from the model.

Two things the sweep needed before it was fair:

- **reach scaling.** The so101 recipe's reward stds (3 cm coarse, 15 cm fine) and
  its 2 cm success radius are tuned for a 0.4 m arm. Verbatim, arx_l5 (0.81 m)
  stalled at `at_goal` 0.014 and scored 1/20. Stds and the success radius are
  scaled by `max(1, reach / 0.4 m)`, where reach is measured by sampling 4,096
  random joint configurations on the compiled model (`reachable` in the json).
  The verbatim run is kept under `results/every_arm_verbatim/` as evidence.
- **an armature floor.** Menagerie `gen3.xml` declares no joint armature with
  kp 2000 / 500 position actuators; under random targets the wrist rings at
  347 rad/s in mjlab and 194 rad/s in classic MuJoCo (fr3, which has armature:
  2.8 rad/s), so `joint_vel_l2` dominates the reward and PPO learns to freeze
  (`at_goal` 0.003). Zero-armature actuated joints get armature 0.1 before
  compiling, in both the training spec and the evaluation plant. kinova_gen3 went
  0/20 to 10/20 with it; the no-floor run is kept under
  `results/every_arm_noarmature_kinova_gen3/`.

Rows landed so far (the sweep is still running; the table is regenerated from
`results/every_arm/*/result.json` when it finishes):


=== every_arm
| arm | DoF | end effector | reach m | at_goal (last 10 its) | train min | env-steps/s | classic MuJoCo 20 targets | median err mm | status |
|---|---|---|---|---|---|---|---|---|---|
| arx_l5 | 7 | body link7 | 0.81 | 0.59 | 8.1 | 10,271 | 15/20 | 45 | done |
| dynamixel_2r | 2 | body second_segment | 0.63 | 0.96 | 6.1 | 14,106 | 20/20 | 2 | done |
| fr3 | 7 | site attachment_site | 1.19 | 0.24 | 6.9 | 12,286 | 9/20 | 105 | done |
| fr3_v2 | 7 | body fr3v2_link8 | 1.19 | 0.26 | 6.8 | 12,583 | 8/20 | 95 | done |
| kinova_gen3 | 7 | site pinch_site | 1.18 | 0.40 | 10.4 | 8,667 | 10/20 | 88 | done |
| koch | 6 | body gripper_moving_finger | 0.33 | 0.81 | 8.8 | 9,570 | 18/20 | 6 | done |
| kuka_iiwa | 7 | site attachment_site | 1.30 | 0.82 | 6.9 | 12,702 | 20/20 | 17 | done |
| openarm | 8 | body openarm_right_finger | 0.61 | 0.00 | 12.7 | 6,888 |  |  | KeyError: 'openarm_joint1' |
| panda | 7 | body left_finger | 1.23 | 0.26 | 8.7 | 9,661 | 7/20 | 116 | done |
| piper | 7 | body link7 | 0.88 | 0.57 | 7.8 | 10,895 | 14/20 | 29 | done |
| sawyer | 7 | site attachment_site | 1.31 | 0.28 | 7.1 | 12,011 | 9/20 | 134 | done |
| so100 | 6 | body Moving_Jaw | 0.48 | 0.47 | 7.3 | 11,204 | 14/20 | 14 | done |
| so101 | 6 | site gripper | 0.57 | 0.79 | 6.6 | 12,882 | 17/20 | 7 | done |
| ur10e | 6 | site attachment_site | 1.53 | 0.09 | 6.4 | 13,415 | 0/20 | 1426 | done |
| ur5e | 6 | site attachment_site | 1.13 | 0.10 | 6.0 | 14,248 | 0/20 | 990 | done |
| vx300s | 7 | site pinch | 0.90 |  |  |  |  |  | ValueError: The observation group 'actor' returned by the en |
| wx250s | 7 | body wx250s/left_finger_link | 0.75 |  |  |  |  |  | ValueError: Not all regular expressions are matched! Please  |
| xarm7 | 7 | site link_tcp | 1.19 | 0.46 | 7.1 | 10,858 | 13/20 | 79 | done |
| yam | 7 | site tcp_site | 0.73 | 0.25 | 6.2 | 13,633 | 6/20 | 89 | done |
| z1 | 7 | body gripperMover | 0.90 | 0.51 | 5.6 | 15,168 |  |  | KeyError: 'jointGripper' |

16 arms trained; 9 reach at least 10/20 on classic MuJoCo with the one recipe.

Reading the table: a 2-DoF arm and the 7-DoF iiwa both solve the task in 200
iterations; the 1.2 m Franka-class arms (fr3, fr3_v2, panda) do not, and their
median error (95 to 116 mm) says under-training, not failure: their `at_goal`
was still climbing at iteration 200. That is the point of a fixed budget: it
tells you which arms need more than one recipe. dynamixel_2r first scored 9/20
because its MJCF lists actuators R2, R1 while joint order is R1, R2; the
example's forward kinematics now index by joint name (FINDINGS F9). openarm
fails because the registry's `model_xml` is the single arm while `scene_xml` is
the bimanual scene (F13). z1 trains (at_goal 0.51) but the classic replay dies on
`KeyError: 'jointGripper'`, the same actuated-set mismatch as openarm's second half
(F13); vx300s returns NaN observations on its first mjlab step (F19) and wx250s's
heuristic finger body does not resolve in mjlab's body table (F20).

**The rest pose was the recipe's biggest variable.** Every row above rested and reset
around qpos 0 because the example passed `keyframe=None`. A 200-reset probe in the
recipe's +-0.3 rad band (`scratch/zero_pose_probe.json` in the lane directory) shows why
ur5e and ur10e sit at 0/20: qpos 0 is the UR arm lying flat 6 cm above the floor with
the tool 0.8-1.2 m out, so half of all resets start with the tool below the floor and
the actor needs 6 x the action scale just to reach the `home` keyframe. The contact-at-
qpos-0 fraction ranks the table: kinova 100 %, panda 90 %, fr3 65 %, z1 58 %, piper
45 % against 0 % for koch, so101, iiwa and dynamixel_2r, the arms that score 17-20/20.
`--pose home` (now the default when the MJCF has a keyframe) re-runs the 12 affected arms
at the same budget (200 iterations, 1,024 environments, seed 42, the same 20 targets):

| arm | keyframe | max home-zero rad | resets with contact at qpos 0 | A at_goal | A classic 20 targets | A median mm | B at_goal | B classic 20 targets | B median mm |
|---|---|---|---|---|---|---|---|---|---|
| arx_l5 | home | 0.31 | 40 % | 0.59 | 15/20 | 45 | 0.75 | 20/20 | 15 |
| fr3 | home | 1.57 | 64 % | 0.24 | 9/20 | 105 | 0.76 | 20/20 | 15 |
| fr3_v2 | home | 1.57 | 63 % | 0.26 | 8/20 | 95 | 0.75 | 20/20 | 14 |
| kinova_gen3 | home | 3.14 | 100 % | 0.40 | 10/20 | 88 | 0.55 | 16/20 | 39 |
| panda | home | 1.57 | 90 % | 0.26 | 7/20 | 116 | 0.79 | 20/20 | 13 |
| piper | home | 1.57 | 45 % | 0.57 | 14/20 | 29 | 0.83 | 19/20 | 16 |
| sawyer | home | 3.32 | 0 % | 0.28 | 9/20 | 134 | 0.65 | 16/20 | 40 |
| ur10e | home | 1.57 | 0 % | 0.09 | 0/20 | 1426 | 0.70 | 19/20 | 42 |
| ur5e | home | 1.57 | 0 % | 0.10 | 0/20 | 990 | 0.75 | 20/20 | 22 |
| xarm7 | home | 1.16 | 30 % | 0.46 | 13/20 | 79 | 0.72 | 19/20 | 30 |
| yam | home | 1.05 | 38 % | 0.25 | 6/20 | 89 | 0.66 | 16/20 | 25 |
| z1 | home | 0.79 | 58 % |  | KeyError: 'jointGripper' |  | 0.73 | 18/20 native | 17 |

A = the zero-pose sweep above, B = the home-keyframe re-run, same budget, same 20 targets,
same seed. Every one of the 12 re-run arms improves and none gets worse: 91/220 classic
successes at qpos 0 against 205/220 from `home` over the 11 arms with both replays (z1's
classic replay dies on the F13 joint mismatch either way; its `home` ONNX scores 18/20 at
17 mm on a native mjlab replay of the same targets, so the row says so). panda 7/20 at
116 mm becomes 20/20 at 13 mm with no other change, fr3 and fr3_v2 go 9 and 8 -> 20/20,
and the two UR arms go from 0/20 to 20/20 and 19/20 (after the frame fix below), so the
Franka-class and UR "under-training" readings above were the rest pose, not the budget.
The three arms that stay below 20/20 (kinova_gen3 16, sawyer 16, yam 16) are the three
with the largest home-to-zero distance or a 7-dof redundant chain reaching a 1.2 m box;
200 iterations is the knob left to turn there. The `resets with contact at qpos 0` column
is the 200-reset probe; sawyer's 0 % with a 9/20 zero-pose score is the exception to the
contact ranking: its qpos 0 stretches the tool 1.02 m out at 0.32 m height, 3.32 rad from
`home`, the largest home-to-zero distance in the table, so the actor spent its 200
iterations travelling rather than colliding.

**The UR rows hid a second, frame bug (F16b).** With `home`, ur10e trained to `at_goal`
0.70 yet the classic replay first scored 0/20 with the tool ending 0.9-1.9 m away. A
native mjlab replay of the same ONNX on the same 20 targets (`scratch/native_replay.py`,
one world per target, no noise) scored 16/20 at 33 mm, so the actor was fine and the
replay was not. Cause: the example's forward kinematics reported the end effector in the
robot's base *body* frame, while mjlab's `ReachCommand` expresses the target in the entity
root frame, which for a fixed-base attach is the MJCF world frame. Those are the same frame
for 17 of the 20 arms and differ for exactly three: Menagerie's ur5e and ur10e rotate the
base body 180 degrees about z (`quat="0 0 0 1"`), so every replay target was mirrored in
x and y, and xarm7 lifts its base 0.12 m, so its targets were 12 cm off. The FK now
reports the world frame, the frame the `rsl_rl_onnx` provider's own site FK uses; the
same ONNX files re-evaluated: ur5e 20/20 at 22 mm, ur10e 19/20 at 42 mm, xarm7 11/20 ->
13/20. The one-line lesson for anyone wiring a reach policy across engines: the target
frame is part of the policy contract, and "base frame" is ambiguous the moment an MJCF
rotates or lifts its first body.

## 03 Humanoid beyond velocity

Results (rough done, get-up v1 and v2 done, both negative; v3 read at checkpoint 400, same crouch):

**Rough terrain, 1500 iterations, 4096 envs, 2 h 39 min on Thor** (`results/g1_rough`, shared GPU). Training ended at
mean reward 15.66 and mean episode length 926 of 1000 ticks under the terrain curriculum. Native mjlab play at the
maximum terrain difficulty, one world per command, 10 s each: `yaw_0.5` and `fwd_0.5` survive the full 10 s,
`stand` falls at 2.7 s, `fwd_1.0` at 1.4 s. Classic MuJoCo replay on a flat plane through `rsl_rl_onnx` with the
flat-plane `height_scan` builder (every ray hits z = 0): falls at 1.1-1.2 s on all four commands. The flat velocity
policy from the mjlab lane walks identically on both engines through the same harness. The control that separates the
actor from the replay: `eval-native --task rough --terrain plane` puts the same actor on a flat plane inside mjlab with
the real ray-cast sensor (`results/g1_rough/eval_native_plane.json`), where it sinks to base z 0.20 m in the first second
on all four commands and never moves; the plane height scan equals pelvis z x 0.2 (0.158 at 0.788 m, 187 equal rays, no
misses), which is exactly what the classic builder feeds, and zero action holds the robot standing at 0.734 m on both
terrains. So the classic replay is a true negative: the 1500-iteration rough actor is brittle outside the curriculum
band it ended in (F18). Two core paths had to be bypassed to get here
(F17): the exporter's one-world env overflows mjwarp's contact budget on rough terrain (`nconmax must be >= 72`), and
the provider refuses the `height_scan` term before any builder can be supplied.

**Get-up, 1000 iterations, 2048 envs, 45.7 min on the shared GPU** (`results/g1_getup`,
[`assets/g1_getup_v1_eval_native.json`](assets/g1_getup_v1_eval_native.json)). The flat velocity task with the
tracking and gait rewards removed, a height bell (target 0.72 m, std 0.3) plus a `standing` bonus (pelvis above
0.62 m and torso within 20 deg of upright), the fall termination removed, the robot dropped supine from 0.35 to
0.45 m. Mean reward went -1.3 to 17.1, and the four native play episodes tell the same story: the pelvis sits at
0.525 m from second 1 to second 9, torso not upright, 0 of 4 stood up. The reason is arithmetic, not physics: at
0.525 m the height bell still pays 66 percent, the standing bonus pays nothing until 0.62 m, so a stable crouch is
a local optimum PPO has no gradient out of (F21). `--getup-reward v2` (std 0.15, so the crouch is worth 18 percent;
standing bonus 5.0 from 0.57 m; upright weight 3.0) was the first retry.

**Get-up v2, same budget, 44 min** (`results/g1_getup_v2`,
[`assets/g1_getup_v2_eval_native.json`](assets/g1_getup_v2_eval_native.json)): mean reward -35 to 18.3, `upright`
2.97 of 3.0, `height` 0.50, `standing` 0.0000 for all 1000 iterations, and the four play episodes are byte-for-byte
the v1 trace: pelvis at 0.525 m from second 1 to 9, `upright_final` false, 0 of 4 stood up. Sharpening the height
bell could not move it, and a mid-run export of `model_600.pt` (`results/g1_getup_v2/eval_native_ckpt600.json`)
already showed the same trace, which pointed at the real mechanism: mjlab's stock `upright` term is bound to
`torso_link` (`config/g1/env_cfgs.py`), while the `standing` gate and the eval's stood-up test read the root
(pelvis) gravity. The G1's waist lets the torso stand vertical over a pelvis pitched past 20 degrees at 0.525 m, so
the crouch collects the full upright weight plus a quarter of the height bell, and the AND-gated `standing` gives no
gradient toward straightening the pelvis. `--getup-reward v3` adds a shaped `pelvis_upright` term (exp of the root's
lateral gravity, std 0.4, weight 2.0) was the last retry of this lane (phase 10, `results/g1_getup_v3`). Its
checkpoint 400 of 1000, exported and played while the run continued
([`assets/g1_getup_v3_ckpt400_eval_native.json`](assets/g1_getup_v3_ckpt400_eval_native.json)): the same 0.525 m
trace on all four drops, `upright_final` false, 0 of 4, with `pelvis_upright` at 0.54 of 2.0 (a pelvis still pitched
about 27 degrees) and `standing` still 0.0. Three reward shapes, one attractor. The trace has one more clue the
rewards do not explain: `base_z_max` is 0.85 to 1.26 m in every episode, so the supine spawn throws the pelvis up to
0.8 m into the air before the robot ever acts (the drop height is 0.35 to 0.45 m), which points at an initial-contact
impulse in the supine keyframe rather than at PPO. The next attempt should fix the spawn (settle the supine pose for
a few ticks with zero action before the episode starts, or lower the drop) before spending another 1000 iterations
on reward terms. The finished v3 run lands in `results/g1_getup_v3/eval_native.json` in the lane directory.
`eval-native` now records `z_per_second`, `base_z_max` and `upright_final` per episode so the next attempt is
readable at a glance (the v1 trace also shows the pelvis reaching 0.85 to 1.25 m inside the first second after
the drop, a launch that was not diagnosed).

## 04 Extreme domain randomisation

Two so101 reach actors, same recipe (N=1,024, 300 iterations, seed 42): `none` on
the stock task (591 s) and `extreme` with every physics term redrawn per world on
every reset (958 s; kp and kd x0.4 to x1.8, effort x0.4 to x1.0, body mass x0.5
to x2.0, friction 0.1 to 2.0, joint damping x0.3 to x4.0, encoder bias +/-50
mrad, observation noise x3). Both are then judged on classic MuJoCo (a backend
neither saw), 20 targets, seed 7, 200 ticks at 50 Hz, success = final tcp error
under 3 cm, one fresh model per episode so perturbations never stack. Data:
[`assets/dr_eval.json`](assets/dr_eval.json).

| perturbation (classic MuJoCo, unseen) | nominal success | dr success | nominal median err mm | dr median err mm |
|---|---|---|---|---|
| nominal | 12/20 | 11/20 | 13 | 19 |
| payload_100g | 12/20 | 12/20 | 13 | 19 |
| payload_250g | 12/20 | 12/20 | 13 | 19 |
| mass_x2 | 12/20 | 12/20 | 13 | 19 |
| friction_0.1 | 12/20 | 11/20 | 13 | 19 |
| kp_x0.5 | 12/20 | 12/20 | 13 | 19 |
| kp_x1.8 | 12/20 | 11/20 | 12 | 19 |
| damping_x4 | 12/20 | 11/20 | 13 | 19 |
| everything | 12/20 | 12/20 | 12 | 19 |
| encoder_bias_50mrad | 7/20 | 9/20 | 35 | 32 |
| encoder_bias_100mrad | 0/20 | 1/20 | 62 | 54 |
| obs_noise_20mrad | 12/20 | 11/20 | 14 | 19 |
| action_delay_2 | 1/20 | 4/20 | 58 | 46 |
| action_delay_5 | 0/20 | 0/20 | 160 | 146 |
| hostile | 3/20 | 5/20 | 48 | 49 |

- **Physics perturbations do not separate the actors.** Nine cells, one number:
  12/20 for `none` in every one, 11 or 12 for `extreme`. The same eight targets
  fail in every cell, with final errors of 48 to 144 mm and no approach at all
  (min error equals final error): they are outside what 300 iterations taught,
  not victims of the perturbation. The perturbations are real (a 250 g payload
  sags the held pose by 21 to 31 mrad, kp x0.5 by 5 to 20 mrad, measured on the
  stepped model), but an actor that observes joint positions and `ee_to_target`
  every 20 ms integrates a static sag away within a few ticks.
- **Extreme DR costs precision and buys nothing here.** On the successful
  episodes `extreme` lands at 9.4 mm median, `none` at 10.7 mm, but its
  distribution has a longer tail (19 mm vs 13 mm over all 20), and it needed
  62 % more wall clock for the same iterations (`set_const` per world after
  every reset).
- **What breaks a closed loop is the loop.** The second wave attacks the
  observation and the timing instead of the body: a constant 50 mrad encoder
  bias (inside the DR training range) drops `none` to 7/20 and `extreme` to
  9/20; 100 mrad kills both. Two ticks of action latency (40 ms) take `none`
  to 1/20 and `extreme` to 4/20, and the failures there pass *through* the
  target (min error 3 to 9 mm) and oscillate, the signature of a controller
  tuned for zero latency. Five ticks is 0/20 for both. Gaussian observation
  noise of 20 mrad per tick changes nothing (12/20 vs 11/20): noise averages
  out, bias and delay do not.
- **So the honest thesis is narrower than the title.** Per-world physics DR at
  this scale is cheap to run (1,024 different robots in one `step`) and the
  trained actor is a little more tolerant of encoder bias and latency, which it
  never saw explicitly. If sim-to-real is the goal, randomise the things the
  grid shows matter: observation bias, action delay and, with those, the
  gains; body mass and friction can stay nominal for a position-controlled
  reach. The delay term is the one missing from `EXTREME` today.


## 05 Curriculum goal box

`05_curriculum_goal_box.py` keeps a level per world. Every world starts with the so101 recipe's 1x goal box
(0.12..0.30 x -0.20..0.20 x 0.08..0.30 m in the base frame) and is promoted one level (1.5x, 2x, 2.5x, 3x the box
around its centre) when it reaches its target, demoted when it misses badly; the command term resamples inside the
world's current box. Training: 400 iterations, 1,024 worlds, 1,040 s on the shared GPU. Evaluation: classic MuJoCo,
20 targets per box drawn from that box, 200 ticks at 50 Hz, success within 30 mm, same seed for both actors.

| goal box | reachable targets | curriculum success | baseline_1x success | curriculum median err mm | baseline_1x median err mm |
|---|---|---|---|---|---|
| x1.0 | 0.95 | 18/20 | 18/20 | 10 | 10 |
| x1.5 | 0.75 | 10/20 | 9/20 | 31 | 45 |
| x2.0 | 0.70 | 11/20 | 8/20 | 28 | 41 |
| x2.5 | 0.50 | 9/20 | 6/20 | 70 | 106 |
| x3.0 | 0.60 | 6/20 | 3/20 | 103 | 120 |

`baseline_1x` is the `reach_none` actor from 04 (300 iterations, 1x box only). "reachable targets" is the fraction
of the 20 sampled targets that FK says the arm can reach at all (the 3x box pokes 0.6 m out of a 0.4 m arm), so
it is the ceiling for both rows.

How the population moved (worlds per level, from `curriculum_schedule.json`):

| iteration | mean level | worlds at 1x / 1.5x / 2x / 2.5x / 3x | at_goal at 1x / 3x |
|---|---|---|---|
| 100 | 0.09 | 932 / 90 / 2 / 0 / 0 | 0.10 / - |
| 200 | 0.90 | 343 / 462 / 195 / 24 / 0 | 0.56 / - |
| 300 | 1.87 | 100 / 290 / 351 / 207 / 76 | 0.67 / 0.16 |
| 399 | 2.21 | 51 / 215 / 355 / 279 / 124 | 0.63 / 0.18 |

Reading it honestly: inside the box both actors were trained for, they tie (18/20, 10 mm). Outside it the
curriculum actor wins every stage (10 vs 9, 11 vs 8, 9 vs 6, 6 vs 3; 36/80 vs 26/80 across the four wider boxes)
and its median error is 30 to 35 percent lower, but it also had 100 more iterations and never saw the 3x box for
most of training (the first world got there at iteration 250). The curriculum is a cheap way to widen the
workspace a fixed recipe covers; it is not a substitute for a recipe designed for the wider box.

## 06 Agent trains a fleet

`06_agent_trains_a_fleet.py --arms so101,koch,arx_l5` hands a Strands `Agent` three
tools: the stock `train_policy` (provider `rsl_rl`, the MuJoCo-Warp trainer),
`evaluate_policy` (plays the exported ONNX on the classic CPU MuJoCo backend for
20 seeded targets) and `write_leaderboard`. The prompt names the arms and the task
ids; the koch and arx_l5 reach tasks are registered on the fly from the registry
entry with example 02's builder. The agent decides the order, reads the numbers the
tools return and writes the table itself. `--no-llm` runs the same three tools in
the obvious order without a model, for CI.

Run 1 (claude-sonnet-4-6 on Bedrock, 9 tool uses, 38.7 min wall, three
training jobs on a GPU shared with two other sweeps). The leaderboard is the file the
agent wrote, unedited:

# Reach Policy Leaderboard — MuJoCo-Warp Fleet (2026-09-29)

> Sorted by evaluation success (descending). Training: rsl_rl, 200 steps, batch 1024, seed 0.
> Success threshold: 30 mm. Evaluation: 20 seeded episodes, CPU MuJoCo backend.

| Rank | Arm | DoF | Train wall time (min) | `Metrics/reach/at_goal` (train) | Eval success | Median final error (mm) | Status |
|------|--------|-----|----------------------:|--------------------------------:|:------------:|------------------------:|--------|
| 1 | koch | 6 | 7.97 | 0.7188 | 19 / 20 | 6.4 | ✅ OK |
| 2 | so101 | 5 | 6.51 | 0.5906 | 14 / 20 | 15.5 | ✅ OK |
| 3 | arx_l5 | 7 | 23.11 | 0.4222 | — / 20 | — | ❌ `ValueError: model has 7 outputs but 8 joint_names / 7 action_scale entries` |

Two things run 1 found, both fixed in the example afterwards:

- **Concurrent tool calls kill the second trainer.** Strands' default tool
  executor runs the model's tool calls concurrently; the agent asked for all three
  `train_policy` calls at once, two died with `Graph capture already in progress on
  this stream` (mjlab captures a CUDA graph per trainer on the default stream) and
  the agent retried them one at a time on its own. The example now passes
  `SequentialToolExecutor`.
- **An unactuated joint breaks the exported actor.** arx_l5 has 8 joints and 7
  actuators. mjlab's export metadata lists every joint of the entity while the
  actor has one output per action target, so the `rsl_rl_onnx` provider refused
  the file. The trained actor was fine: re-exported with trimmed metadata it scores
  17/20 at 18 mm median on the same 20 targets. The example now trims
  `joint_names` / `default_joint_pos` to the actuated joints after every
  `train_policy` call; the proper fix belongs in
  `strands_robots/training/mjlab_tasks/export.py` (FINDINGS F15).

Run 2, same prompt and budget with both fixes in the example (claude-sonnet-4-6,
7 tool uses, 21.8 min wall, GPU shared with the 02 home-pose re-run and the G1 get-up
training). The model again asked for all three `train_policy` calls in one turn;
`SequentialToolExecutor` ran them one after another (checkpoint stamps 04:05, 04:12,
04:19 UTC, no CUDA-graph error), the trimmed export let arx_l5 evaluate, and the
agent's table came out complete on the first pass. Unedited:

# MuJoCo-Warp Reach Policy Leaderboard
**Run date:** 2026-09-29 · **Steps:** 200 PPO iterations · **Envs:** 1 024 · **Eval:** 20 CPU episodes each

| Rank | Arm | DoF | Train time (min) | Train `at_goal` | Eval success | Median final error (mm) | Status |
|------|---------|-----|-----------------|-----------------|--------------|------------------------|--------|
| 1 | koch | 6 | 7.46 | 0.6979 | 19/20 | 6.8 | ✅ OK |
| 2 | arx_l5 | 6 | 5.87 | 0.5646 | 17/20 | 15.0 | ✅ OK |
| 3 | so101 | 5 | 6.71 | 0.6123 | 14/20 | 12.6 | ✅ OK |

> **DoF** counts actuated joints only (gripper excluded).
> **Train `at_goal`** = `Metrics/reach/at_goal` reported at the final PPO iteration.
> **Median final error** = median Euclidean end-effector distance to target at episode tick 150, across 20 seeded episodes.
> Success threshold: 30 mm for so101 & koch; 61.2 mm for arx_l5 (arm-specific, as returned by `evaluate_policy`).

Two readings of the agent's own footnotes. The DoF column is the agent's choice of
convention (run 1 wrote 7 for arx_l5, run 2 writes 6 "gripper excluded"; the entity has
8 joints and 7 actuators either way). The 61.2 mm arx_l5 threshold is the example's
reach-scaled tolerance (30 mm x 2.04, the arm's reach relative to so101) and the agent
reported it rather than hiding it. Run to run, koch and so101 land within 1/20 and 3 mm
of run 1 (19/20 and 14/20 both times); arx_l5 at 17/20 @ 15.0 mm matches the run-1 actor
re-exported by hand (17/20 @ 18 mm). Full tool-by-tool transcript:
[`assets/transcript_fleet.md`](assets/transcript_fleet.md); run-1 table kept as
[`assets/fleet_leaderboard_run1.md`](assets/fleet_leaderboard_run1.md), run 2 as
[`assets/fleet_leaderboard_run2.md`](assets/fleet_leaderboard_run2.md).

## 07 Dataset factory

Two 30-minute runs on Thor, 1,024 worlds in lockstep, 150 ticks (3 s) per episode,
fresh seeded targets and fresh per-world physics randomisation every batch, every
episode streamed into one LeRobot v3 dataset (`results/factory/*.json`; GPU shared
with the every-arm sweep and the G1 training the whole time):

| policy | minutes | episodes | frames | success | GB | episodes/h | frames/h | GB/h | rollout s | flush s | randomize s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scripted | 31.0 | 21,504 | 3,225,600 | 0.69 | 0.34 | 41,661 | 6,249,087 | 0.66 | 383.9 | 1468.0 | 5.8 |
| onnx | 31.0 | 23,552 | 3,532,800 | 0.69 | 0.37 | 45,620 | 6,842,951 | 0.72 | 274.8 | 1577.0 | 6.1 |

The ONNX run is on the Hub as `cagataydev/mjlab-factory-so101-20260929` (private,
sha `3e23649c`, 9 files, 374 MB); the episode count in the table is the one
`repo_info` reads back, not the one the process believed.

- **45,000 episodes an hour is the writer's number, not the physics'.** Per
  1,024-episode batch the ONNX actor needs 10.6 s of rollout and the recorder 66 to 69 s
  of flush; over the run 79 % (scripted) and 85 % (onnx) of the wall clock is the
  LeRobot writer. The 6,370 episodes/min rollout ceiling from the deep lane holds
  here (rollout alone would give about 300,000 actor episodes/h); the next 10x is a parallel or
  asynchronous writer, not a faster simulator.
- **The scripted expert is not faster than the actor.** Batched damped-least-squares IK
  on the CPU FK costs 15 to 17 s per batch (the `policy_s` column), the ONNX actor
  6.5 to 7 s; with the same 69 % success rate (731 vs 727 of the first 1,024) the
  learned policy is the cheaper data source once it exists, and the two datasets
  are the paired expert / self-play rows a DAgger loop would consume.
- **Per-world randomisation is free at this scale**: 0.06 to 0.29 s per batch
  (5.8 s of 31 min) for friction, mass, inertia and target pose across 1,024 worlds.
- Success is flat across batches (721 to 731 of 1,024, median final error 12 to 15 mm),
  so the randomisation is not drifting the distribution within a run; the 31 % of
  failures are the same actor-coverage limit seen in 04 (targets never approached).
