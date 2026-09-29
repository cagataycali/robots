# Laya as a System 1 typed-decision policy (research, sim only)

[Laya](https://github.com/convaiinnovations/laya) is a text-only, non-autoregressive decision model: one encoder
forward pass answers typed questions (`choice` / `score` / `noul`) about a text state with calibrated probabilities.
It has no image encoder and no continuous head, so the `laya` provider never regresses joint targets. Per tick it
serializes the observation (joint angles in degrees, gripper percent, the instruction and, through a privileged
`set_world_reader`, the cube / gripper poses), asks Laya which ONE joint to move, in which direction, by how much,
and optionally whether that step makes progress, and applies exactly one discrete primitive.

| File | What it shows | GPU |
|---|---|---|
| [`laya_gated_rollout.py`](laya_gated_rollout.py) | `create_policy("laya", confidence_gate=0.5)` on the so101, world reader wired, per-tick audit through `policy.last_tick` | Optional |
| [`experiment/`](experiment/) | The H1 / H2 / H3 study: shared scene, scripted + random primitive baselines, N=20 x 5 arms x 2 tasks runner with LeRobot v3 recording, calibration analysis, Laya-as-judge scoring | Optional |

```bash
pip install "strands-robots[sim-mujoco,laya]"
MUJOCO_GL=egl python examples/laya/laya_gated_rollout.py
```

Research findings (Jetson AGX Thor, 2026-09-28, MuJoCo so101, 20 episodes per arm x task) live in the private HF
dataset `cagataydev/laya-so101-mujoco-20260928` (`REPORT.md`, `FINDINGS.md`). The short version: zero-shot Laya is
NOT a controller (H1: every checkpoint 0/20, none beats a random primitive, each emits one primitive per checkpoint and
task text), NOT a calibrated gate (H2: judging 12,855 scripted and random proposals, AUROC 0.48-0.53, ECE 0.10-0.51),
and fast enough only on an idle GPU (H3: 17-27 ms per question, 55-105 ms for ten; CPU 2-7 s). The reusable parts are
the primitive vocabulary, the scripted/random baselines, the judge and the `confidence_gate` seam, which need a
checkpoint fine-tuned on (state, primitive) pairs before a second try.
