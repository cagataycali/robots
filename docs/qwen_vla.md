# Qwen-VLA Integration

[Qwen-VLA](https://arxiv.org/abs/2605.30280) is a unified vision-language-action
model: a Qwen3.5-4B VLM backbone paired with a 1.15B-param DiT flow-matching
action expert. Its **only** platform-specific interface is an embodiment-aware
text prompt, so the same checkpoint drives manipulation, navigation, and
trajectory prediction across robots.

This page covers the **inference provider** (Phase A). Training (the 4-stage
T2A -> CPT -> SFT -> RL recipe) ships incrementally under
`strands_robots/training/qwen_vla/`.

## Install

```bash
# Service mode only (connect to a running server over ZMQ - no model deps)
pip install 'strands-robots[qwen-vla-service]'

# Local mode (in-process model load - requires GPU + upstream package)
pip install 'strands-robots[qwen-vla]'
```

## Quickstart

### Service mode (recommended)

Connect a `QwenVlaPolicy` to a running Qwen-VLA ZMQ server. No model
dependencies are needed on the client.

```python
from strands_robots.policies.qwen_vla import QwenVlaPolicy

policy = QwenVlaPolicy(data_config="so100", host="127.0.0.1", port=5556)

obs = {
    "webcam": frame,            # (H, W, 3) uint8
    "single_arm": joint_state,  # (6,) float
    "gripper": gripper_state,   # (1,) float
}
actions = policy.get_actions_sync(obs, "pick up the red cube")
# actions: list of H per-timestep dicts, e.g. {"single_arm": [...], "gripper": [...]}
```

### Local mode

```python
policy = QwenVlaPolicy(
    data_config="aloha_bimanual",
    model_path="Qwen/Qwen-VLA-Base",
    device="cuda",
    denoising_steps=4,
)
```

### Via the factory / `Robot()`

`qwen_vla` is registered in `registry/policies.json` with shorthands
`qwen`, `qwen-vla`, `qwenvla`:

```python
from strands_robots.policies import create_policy

policy = create_policy("qwen-vla", data_config="so100", port=5556)
```

## The embodiment prompt (section 2.3)

The prompt is the sole platform-specific input. It is built from the
`QwenVlaDataConfig` morphology fields:

```
The robot is {robot_tag} with {single arm / dual arms}[, waist][, and mobile
base]. The control frequency is {FPS} Hz. Please predict the next {chunk_size}
control actions to execute the following task: {instruction}.
```

Deploying to a new robot = a new prompt (new data config), **not** a new model
head. Build one directly:

```python
from strands_robots.policies.qwen_vla import build_embodiment_prompt

build_embodiment_prompt(
    robot_tag="unitree_g1", arm_config="dual", fps=30, chunk_size=16,
    instruction="walk to the table", has_waist=True, has_mobile_base=True,
)
```

## Data configs

Seed embodiments live in `policies/qwen_vla/data_configs.json` (resolved via
the `_extends` inheritance mechanism, mirroring GR00T):

| Config | Arms | Waist | Mobile base | FPS |
|---|---|---|---|---|
| `so100`, `so100_dualcam` | single | no | no | 30 |
| `aloha_bimanual` (`aloha`) | dual | no | no | 50 |
| `widowx` | single | no | no | 5 |
| `unitree_g1` (`g1`) | dual | yes | no | 30 |
| `unitree_g1_mobile` | dual | yes | yes | 30 |
| `franka_panda` | single | no | no | 20 |
| `libero_panda` (`libero`) | single | no | no | 20 |

Register a custom embodiment at runtime with `create_custom_data_config(...)`.

## Normalization & the unified action layout

Action channels are quantile-normalized to `[-1, 1]` (eq. 5) and packed into a
fixed-width `Y in R^{H x K}` tensor with a per-channel binary mask excluding
zero-padding (section 2.4). Helpers in `policies/qwen_vla/normalize.py`:
`compute_quantile_stats`, `normalize` / `unnormalize`, `build_channel_mask`,
`pad_to_width` / `unpad_from_width`.

## Managing a server (agent tool)

```python
from strands_robots.tools import qwen_vla_inference

qwen_vla_inference(action="status", port=5556)
qwen_vla_inference(action="ping", host="127.0.0.1", port=5556)
qwen_vla_inference(
    action="start",
    data_config="so100",
    model_path="/path/to/checkpoint",
    server_command="python -m qwen_vla.serve",  # your server entrypoint
)
```

The tool validates every input against allowlists (no shell metacharacters in
`host`, loopback-only bind, path-traversal / protected-directory rejection on
`model_path`) and binds to `127.0.0.1` by default.

## Configuration (env vars)

| Variable | Purpose |
|---|---|
| `QWEN_VLA_API_TOKEN` | ZMQ auth token for SERVICE mode (sent with each request). |

## Status

- **Phase A (inference provider)**: shipped — prompt, data configs, normalize,
  client, policy (LOCAL + SERVICE), registry entry, agent tool, unit +
  integration tests.
- **Phase B-G (training)**: incremental — see `PLAN.md`.

> Note: the upstream Qwen-VLA model package / public checkpoint is not yet
> released. LOCAL mode raises a clear, actionable error until it is; SERVICE
> mode works today against any server speaking the documented ZMQ envelope
> (see `tests_integ/qwen_vla/` for a reference stub server).
