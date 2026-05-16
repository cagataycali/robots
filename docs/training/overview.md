---
description: Where training lives — upstream LeRobot, Isaac-GR00T, Cosmos. strands-robots ships data + inference.
---

# Training

`strands-robots` is intentionally a **data quality + inference** library, not a
training framework. Training pipelines belong upstream — they're large, diverge across
model families, and iterate fast.

## What ships

- Robot control (`Robot()` factory)
- Simulation backend (MuJoCo)
- Policy inference (Mock, GR00T, LeRobot Local)
- Dataset recording (LeRobot v3 format)
- Multi-robot mesh + safety

## What doesn't ship

- A trainer.

## What to do instead

| Want to train... | Use |
|------------------|-----|
| ACT, Pi0, SmolVLA, Diffusion Policy | `lerobot` upstream — `python -m lerobot.scripts.train ...` |
| GR00T fine-tune | `Isaac-GR00T` repo from NVIDIA |
| Cosmos | `cosmos` upstream |
| Custom architecture | Read the LeRobot v3 dataset directly with `pyarrow` / `datasets` and plug into your framework |

The recorded dataset is in LeRobot v3 format, which all of these accept (with or
without thin conversion).

## Round-trip example

```python
# 1. Record (chapter 6)
from strands_robots import Robot
sim = Robot("so100")
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=30)
for episode in range(50):
    sim.reset()
    sim.randomize(colors=True)
    sim.run_policy(instruction="pick up the cube",
                    policy_provider="mock", duration=10.0)
sim.stop_recording()

# 2. Train upstream
# bash:
#   pip install lerobot
#   python -m lerobot.scripts.train policy=act dataset.root=/tmp/my_dataset

# 3. Infer with the trained checkpoint
from strands_robots.policies import create_policy
policy = create_policy("lerobot_local",
                        pretrained_name_or_path="path/to/checkpoint")
sim.run_policy(instruction="pick up the cube", policy=policy, duration=15.0)

# 4. Deploy on real hardware (chapter 8)
from strands_robots import Robot
real_robot = Robot("so100", mode="real", cameras={...})
real_robot.run_policy(instruction="pick up the cube", policy=policy, duration=15.0)
```

Same code paths in steps 3 and 4. That's the point.

## Why split

- Trainers are heavy. Pi0 training is multi-GPU, days long, with finicky LR
  schedules per architecture. Bundling one trainer into `strands-robots` would
  freeze us at one model.
- Trainers iterate fast upstream. We don't want to fork them.
- The data + inference layer is much more stable. That's where we add value.

If upstream trainers stabilise we may add `strands_robots.trainers` thin adapters —
track [the issue tracker](https://github.com/strands-labs/robots/issues).

## See also

- [Tutorial 7 — Training](../tutorial/07-training.md) — guided walkthrough with three
  upstream paths.
- [Recording](../recording.md) — produce the dataset.
- [LerobotLocalPolicy](../policies/lerobot-local.md) — inference with a trained
  checkpoint.
