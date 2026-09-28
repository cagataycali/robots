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

Research findings (Jetson AGX Thor, 2026-09-28) are in the lane report; the short version: zero-shot Laya is a
usable calibrated *gate* over a proposal generator (H2) and fast enough for 10 Hz ticks with a 5-question
profile (H3), but not a controller on its own (H1).
