# LeRobot Local Policy

Run HuggingFace LeRobot models directly on your machine. Supports ACT, Pi0, Pi0-FAST, SmolVLA, Wall-X, X-VLA, SARM, Diffusion Policy, and any LeRobot-compatible model.

---

## Quick Start

```python
from strands_robots import create_policy

policy = create_policy("lerobot/act_aloha_sim_transfer_cube_human")
```

That's it. The policy auto-resolves the model type from `config.json` on HuggingFace Hub using `PreTrainedConfig.from_pretrained()`.

---

## How It Works

1. You pass a HuggingFace model ID or local path
2. `PreTrainedConfig.from_pretrained()` loads the config and resolves the policy type via draccus
3. `PreTrainedPolicy.from_pretrained()` loads weights into the correct class
4. Falls back to manual `config.json` parsing + module import for third-party policies

No hardcoded policy lists — any policy registered in LeRobot (including third-party plugins installed via `pip install lerobot_policy_mypolicy`) automatically works.

---

## Supported Architectures

All LeRobot 0.5.0+ policies work automatically:

| Policy Type | Description |
|---|---|
| `act` | Action Chunking Transformer |
| `pi0` | Pi-Zero foundation model |
| `pi0_fast` | Pi-Zero FAST (faster inference variant) |
| `smolvla` | SmolVLA (lightweight VLA) |
| `wall_x` | Wall-X policy |
| `xvla` | X-VLA (cross-embodiment VLA) |
| `sarm` | SARM policy |
| `diffusion` | Diffusion Policy |

Third-party policies installed as LeRobot plugins are also supported.

---

## Configuration

```python
policy = create_policy("lerobot_local",
    pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human",
    device="cuda",              # "cuda", "mps", or "cpu"
    actions_per_step=10,        # Action chunking
    policy_type=None,           # Auto-detect from config.json
)
```

### Language-Conditioned Models (VLAs)

VLA models (SmolVLA, X-VLA) need language tokens. The policy auto-detects this from the model config and tokenizes instructions automatically:

```python
policy = create_policy("lerobot/smolvla_base")
actions = policy.get_actions_sync(obs, instruction="pick up the red cube")
```

Requires the `vla` extra: `pip install "strands-robots[vla]"` (needs `transformers>=5.0.0`).

---

## Processor Bridge

If the robot has processor configs (observation/action normalization), the policy auto-loads them:

```python
# Processor bridge handles obs normalization + action denormalization
# This happens transparently — you pass raw obs, get raw actions
actions = policy.get_actions_sync(obs, instruction="pick up cube")
```

---

## Install

```bash
pip install "strands-robots[lerobot]"    # LeRobot + servos
pip install "strands-robots[vla]"        # + VLA models (transformers>=5.0.0)
```
