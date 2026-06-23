---
description: Train policies locally via the lerobot_train tool, or upstream with Isaac-GR00T and Cosmos. strands-robots ships data + inference + local fine-tuning.
---

# Training

`strands-robots` ships data collection and policy inference. ACT/diffusion/pi/SmolVLA
fine-tuning runs locally through the `lerobot_train` tool (a thin wrapper over
`lerobot-train`); GR00T and Cosmos training run in their upstream frameworks.

| Want to train | Use |
|---------------|-----|
| ACT, Pi0, SmolVLA, Diffusion Policy | `lerobot_train` tool (wraps `lerobot-train`) or `lerobot` directly |
| GR00T fine-tune | `Isaac-GR00T` (NVIDIA) |
| Cosmos | NVIDIA Cosmos Framework - see [Cosmos3Policy](../policies/cosmos3.md) |
| Custom architecture | Read LeRobot v3 dataset with `pyarrow` / `datasets` |

## Round-trip

```python
# 1. Record
from strands_robots import Robot

sim = Robot("so100")
sim.start_recording(repo_id="user/my_dataset", task="pick up the cube", fps=30)
for episode in range(50):
    sim.reset()
    sim.randomize(randomize_colors=True)
    sim.run_policy(robot_name="so100", instruction="pick up the cube",
                   policy_provider="mock", duration=10.0)
sim.stop_recording()

# 2. Train locally on the recorded dataset (closes record -> train -> deploy)
from strands_robots.tools import lerobot_train

# dataset_root is the directory holding meta/info.json
out = lerobot_train(
    dataset_root="~/.cache/huggingface/lerobot/user/my_dataset",
    policy_type="act",
    steps=20000,
    batch_size=8,
    val_episodes=5,        # hold out the last 5 episodes for evaluation
    output_dir="/tmp/train_out/act_cube",
)
# Background session: poll progress, then stop or let it finish.
lerobot_train(action="status", session_name=out["session_name"])

# 3. Infer with the trained checkpoint
from strands_robots.policies import create_policy

ckpt = "/tmp/train_out/act_cube/checkpoints/last/pretrained_model"
policy = create_policy("lerobot_local", pretrained_name_or_path=ckpt)
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_object=policy, duration=15.0)

# 4. Deploy (HardwareRobot has no run_policy - use start_task)
real = Robot("so100", mode="real", cameras={...})
real.start_task(instruction="pick up the cube", policy_port=5555)
```

## See also

- [Recording](../recording.md) - produce the dataset.
- [LerobotLocalPolicy](../policies/lerobot-local.md) - inference with a trained checkpoint.
- [Cosmos3Policy](../policies/cosmos3.md) - NVIDIA Cosmos 3 VLA.
