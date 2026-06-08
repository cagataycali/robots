---
description: What ships in the box, what you bring yourself. Training pipelines for the datasets you recorded.
---

# 7 — Training

`strands-robots` records data; it does **not** ship a trainer. Use the upstream pipeline for whichever model you're targeting.

| Stage | Where |
|-------|-------|
| Robot control + simulation | `strands_robots` |
| Policy inference | `strands_robots.policies` (Mock / GR00T / LeRobot / Cosmos 3) |
| Dataset recording | `strands_robots.dataset_recorder` |
| **Training** | **upstream** — `lerobot`, `Isaac-GR00T`, Cosmos Framework, your code |

## LeRobot training

```bash
pip install lerobot

# Local dataset or pull from Hub — no conversion needed
python -m lerobot.scripts.train policy=act dataset.repo_id=user/my_dataset
python -m lerobot.scripts.train policy=pi0  dataset.repo_id=user/my_dataset
```

Load the checkpoint back into the sim:

```python
from strands_robots.policies import create_policy
from strands_robots import Robot

policy = create_policy("lerobot_local",
                       pretrained_name_or_path="path/to/checkpoint")  # requires GPU
sim = Robot("so100")
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_object=policy, duration=15.0)
```

## GR00T fine-tuning

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T && cd Isaac-GR00T
python scripts/finetune.py \
    --base-model nvidia/GR00T-N1.7-3B \
    --dataset-path my_dataset \
    --data-config so100_dualcam   # requires GPU
```

Mount the checkpoint into the GR00T inference container, then use `Gr00tPolicy` (chapter 3) as before.

## Cosmos / custom trainer

The LeRobot v3 parquet is readable by any framework:

```python
import pyarrow.parquet as pq
df = pq.read_table("my_dataset/data/chunk-000/episode_000000.parquet").to_pandas()
# columns: observation.state, action, episode_index, frame_index, timestamp, ...
```

After Cosmos training: `create_policy("cosmos3", embodiment="droid", port=8000)` — see [Cosmos3Policy](../policies/cosmos3.md). # requires GPU

## Sim-to-real checklist

1. Calibrate the real arm (chapter 8) — joint zeros must match recorded data.
2. Match camera resolution — record and infer at the same `width × height`.
3. Use `randomize()` while recording to avoid overfitting to one sim appearance.
4. Confirm `data_config` matches between record and infer sessions.

## See also

- [Tutorial 6 — Recording](06-recording.md) — produce the dataset.
- [LerobotLocalPolicy](../policies/lerobot-local.md) — load LeRobot checkpoints.
- [Gr00tPolicy](../policies/groot.md) — load GR00T fine-tunes via the inference server.
- [Cosmos3Policy](../policies/cosmos3.md) — NVIDIA Cosmos 3 omnimodal VLA inference.
- [LeRobot training docs](https://huggingface.co/docs/lerobot) — upstream pipeline.
- [Isaac-GR00T training](https://github.com/NVIDIA/Isaac-GR00T) — upstream fine-tuning.
