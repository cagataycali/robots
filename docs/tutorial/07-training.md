---
description: What ships in the box, what you bring yourself. Training pipelines for the datasets you recorded.
---

# 7 — Training

`strands-robots` records data in LeRobot v3 format, but it does **not** ship a trainer.
Training pipelines are intentionally external — they belong to the model author.
This chapter shows how to wire your recorded dataset into the right upstream trainer.

## TL;DR

```bash
# Record locally (chapter 6)
python scripts/record.py --repo-id user/my_dataset --episodes 50

# Push to the Hub
huggingface-cli login
python -c "from lerobot.datasets import LeRobotDataset; \
           LeRobotDataset(repo_id='user/my_dataset', \
                          root='/tmp/my_dataset').push_to_hub()"

# Train upstream (LeRobot, GR00T, etc.)
python -m lerobot.scripts.train policy=act dataset.repo_id=user/my_dataset
```

## What ships, what doesn't

| Stage | Where it lives |
|-------|----------------|
| Robot control | `strands_robots` (this library) |
| Simulation | `strands_robots.simulation` |
| Policy inference | `strands_robots.policies` (Mock / GR00T / LeRobot Local) |
| Dataset recording | `strands_robots.dataset_recorder` |
| **Training** | **upstream** — `lerobot`, `Isaac-GR00T`, `cosmos`, your code |

Why split? Training is heavy — multi-day runs on multi-GPU clusters, optimiser quirks,
LR schedules per architecture. Each model family has its own train loop and we don't
want to fork them.

## Path 1 — LeRobot training

The dataset format you recorded (LeRobot v3) is exactly what `lerobot` expects. Pick
one of its policies (`act`, `pi0`, `smolvla`, `diffusion`, etc.) and train:

```bash
pip install lerobot

# Local dataset
python -m lerobot.scripts.train \
    policy=act \
    dataset.root=/tmp/my_dataset \
    dataset.repo_id=user/my_dataset

# Or pull from Hub
python -m lerobot.scripts.train \
    policy=pi0 \
    dataset.repo_id=user/my_dataset
```

After training, load the checkpoint via `LerobotLocalPolicy`:

```python
from strands_robots import Robot
from strands_robots.policies import create_policy

policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="path/to/your/checkpoint",  # or HF model_id
)

sim = Robot("so100")
sim.run_policy(instruction="pick up the cube", policy=policy, duration=15.0)
```

Same code, your weights.

## Path 2 — GR00T fine-tuning

GR00T training lives in [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T).
Convert the recorded dataset into GR00T's expected format (the data_config you used
for recording determines the conversion):

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

# Fine-tune with your dataset
python scripts/finetune.py \
    --base-model nvidia/GR00T-N1.7-3B \
    --dataset-path /tmp/my_dataset \
    --data-config so100_dualcam
```

Once the checkpoint exists, mount it into your GR00T inference container and use
`Gr00tPolicy` (chapter 3) as before.

## Path 3 — Cosmos / your own trainer

The recorded LeRobot v3 layout is friendly to most training frameworks. Read the
parquet directly:

```python
import pyarrow.parquet as pq

table = pq.read_table("/tmp/my_dataset/data/chunk-000/episode_000000.parquet")
df = table.to_pandas()
# df has columns: observation.state, action, episode_index, frame_index, timestamp, ...
```

Or with `datasets`:

```python
from datasets import load_dataset
ds = load_dataset("/tmp/my_dataset")
```

Plug into PyTorch / JAX / your framework of choice.

## Sim-to-real considerations

If the policy you train is going on real hardware:

1. **Calibrate the real arm** (chapter 8) before any rollout — joint zeros must match
   what was recorded.
2. **Match the camera resolution** — record and infer at the same `width × height` to
   avoid silent shape mismatches.
3. **Use domain randomization** while recording so the policy doesn't overfit to one
   sim look.
4. **Keep sim and real action specs identical** — `Robot()` does this automatically;
   custom trainers need to verify.

## Why not bundle a trainer?

It would give us one trainer that works "okay" for a few architectures and quickly
becomes stale as upstream model authors iterate. Better to be the data-quality and
inference layer and let LeRobot / GR00T / Cosmos / your repo handle training.

When the upstream trainers stabilise we may add `strands_robots.trainers` thin
adapters — track [issue tracker](https://github.com/strands-labs/robots/issues) for
status.

## See also

- [Tutorial 6 — Recording](06-recording.md) — produce the dataset.
- [LerobotLocalPolicy](../policies/lerobot-local.md) — load LeRobot checkpoints back
  into a sim.
- [Gr00tPolicy](../policies/groot.md) — load GR00T fine-tunes via the inference server.
- [LeRobot training docs](https://huggingface.co/docs/lerobot) — upstream pipeline.
- [Isaac-GR00T training](https://github.com/NVIDIA/Isaac-GR00T) — upstream fine-tuning.
