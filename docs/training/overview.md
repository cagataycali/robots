# Training Overview

Train robot policies from demonstration data.

---

## Quick Start

```python
from strands_robots import create_trainer

trainer = create_trainer("lerobot",
    policy_type="act",
    dataset_repo_id="lerobot/so100_wipe",
)
trainer.train()
```

---

## Training Pipeline

```mermaid
graph LR
    A[Demonstrations] --> B[Dataset]
    B --> C[Trainer]
    C --> D[Policy Checkpoint]
    D --> E[Deploy]

    style A fill:#0969da,stroke:#044289,color:#fff
    style B fill:#0969da,stroke:#044289,color:#fff
    style C fill:#8250df,stroke:#5a32a3,color:#fff
    style D fill:#2ea44f,stroke:#1b7735,color:#fff
    style E fill:#2ea44f,stroke:#1b7735,color:#fff
```

1. **Record** demonstrations (teleoperation or simulation)
2. **Organize** into a LeRobot dataset
3. **Train** with `create_trainer()`
4. **Evaluate** on held-out episodes
5. **Deploy** as a regular policy provider

---

## 6 Trainers

| Provider | Class | Best for |
|---|---|---|
| `lerobot` | `LerobotTrainer` | ACT, Pi0, Pi0-FAST, SmolVLA, Wall-X, X-VLA, SARM, Diffusion Policy |
| `groot` | `Gr00tTrainer` | Fine-tuning GR00T N1.6 VLA |
| `dreamgen_idm` | `DreamgenIdmTrainer` | DreamGen inverse dynamics model |
| `dreamgen_vla` | `DreamgenVlaTrainer` | DreamGen VLA fine-tuning (GR00T-Dreams) |
| `cosmos_predict` | `CosmosTrainer` | Cosmos Predict 2.5 world model |
| `cosmos_transfer` | `CosmosTransferTrainer` | Cosmos Transfer 2.5 ControlNet (sim→real) |

---

## Configuration

```python
# TrainConfig fields are auto-extracted from kwargs
trainer = create_trainer("lerobot",
    policy_type="act",
    dataset_repo_id="lerobot/so100_wipe",
    max_steps=10000,
    batch_size=32,
    learning_rate=1e-4,
    output_dir="/data/checkpoints",
)
```

---

## Examples

### LeRobot (ACT)

```python
trainer = create_trainer("lerobot",
    policy_type="act",
    dataset_repo_id="lerobot/so100_wipe",
)
trainer.train()
```

### LeRobot with PEFT/LoRA

```python
# Fine-tune with LoRA (supported in LeRobot 0.5.0+)
trainer = create_trainer("lerobot",
    policy_type="pi0",
    dataset_repo_id="lerobot/so100_wipe",
    pretrained_name_or_path="lerobot/pi0_base",
    extra_cli_args=["--policy.peft_config.use_peft=true"],
)
trainer.train()
```

### GR00T N1.6 Fine-tuning

```python
trainer = create_trainer("groot",
    base_model_path="nvidia/GR00T-N1.6-3B",
    dataset_path="/data/trajectories",
    embodiment_tag="so100",
)
trainer.train()
```

### Cosmos Predict 2.5

```python
trainer = create_trainer("cosmos_predict",
    base_model_path="nvidia/Cosmos-Predict2.5-2B",
    dataset_path="/data/robot_trajectories",
    mode="policy",
    max_steps=5000,
)
trainer.train()
```

### Cosmos Transfer 2.5 (Sim→Real)

```python
trainer = create_trainer("cosmos_transfer",
    base_model_path="nvidia/Cosmos-Transfer2-7B",
    dataset_path="/data/sim_real_pairs",
    control_type="depth",
    mode="sim2real",
)
trainer.train()
```

---

## DreamGen Data Augmentation

Not enough demos? Generate synthetic ones:

```python
from strands_robots import DreamGenPipeline

pipeline = DreamGenPipeline(
    video_model="wan2.1",
    idm_checkpoint="nvidia/gr00t-idm-so100",
    embodiment_tag="so100",
)

results = pipeline.run_full_pipeline(
    robot_dataset_path="/data/pick_and_place",
    instructions=["pour water", "fold towel"],
    num_per_prompt=50,
)
```

1 real demonstration → 50 synthetic variations → more robust training.

---

## Evaluation

```python
from strands_robots import evaluate

results = evaluate(
    policy_path="./checkpoints/policy_best.pt",
    dataset_repo_id="lerobot/so100_wipe",
)
```
