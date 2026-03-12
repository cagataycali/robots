# Chapter 7: Training

**Time:** 30 minutes · **Hardware:** GPU recommended · **Level:** Expert

You've recorded demonstrations. Now train a policy that can reproduce those behaviors autonomously.

---

## One Function Call

```python
from strands_robots import create_trainer

trainer = create_trainer("lerobot",
    policy_type="act",
    dataset_repo_id="lerobot/so100_wipe",
)
trainer.train()
```

That's the minimum. `create_trainer()` sets up everything — model architecture, data loading, optimization, checkpointing.

---

## What's Happening Inside

Training a policy means:

1. **Load** demonstration data (observations + actions)
2. **Feed** observations into a neural network
3. **Compare** the network's predicted actions to the recorded actions
4. **Update** the network to reduce the difference
5. **Repeat** thousands of times

The result: a neural network that maps observations → actions, just like the human demonstrator did.

---

## Policy Types

Different architectures for different tasks:

| Policy | Best for | Speed |
|---|---|---|
| **ACT** | Precise manipulation | Medium |
| **Diffusion Policy** | Complex multi-modal tasks | Slow |
| **Pi0** | General purpose | Fast |
| **SmolVLA** | Lightweight VLA | Fast |

```python
# ACT — Action Chunking with Transformers
trainer = create_trainer("lerobot", policy_type="act", ...)

# Diffusion Policy
trainer = create_trainer("lerobot", policy_type="diffusion", ...)
```

---

## Training Configuration

```python
trainer = create_trainer("lerobot",
    policy_type="act",
    dataset_repo_id="lerobot/so100_wipe",
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=100,
    checkpoint_dir="/data/checkpoints",
    eval_every=10,
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print(f"Success rate: {results['success_rate']:.1%}")
```

---

## Use Your Trained Policy

After training, load the checkpoint as a policy:

```python
from strands_robots import Robot, create_policy

policy = create_policy("/data/checkpoints/act_so100_wipe")
robot = Robot("so100")

obs = robot.get_observation()
action = policy.get_actions(obs, "wipe the table")
robot.apply_action(action)
```

The trained policy is now just another policy provider. Same interface.

---

## DreamGen: Train on Dreams

Don't have enough demonstrations? Record one, generate many:

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

DreamGen uses video generation + inverse dynamics to create synthetic training data from a single demonstration. 1 demo → 50 variations.

---

## What You Learned

- ✅ `create_trainer()` sets up training in one call
- ✅ Multiple policy architectures (ACT, Diffusion, Pi0)
- ✅ Training: load data → predict → compare → update → repeat
- ✅ Checkpoints become regular policies
- ✅ DreamGen for data augmentation from minimal demos

---

**Next:** [Chapter 8: Real Hardware →](08-real-hardware.md) — Deploy to physical robots.
