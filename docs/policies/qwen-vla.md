# Qwen-VLA Integration

[Qwen-VLA](https://arxiv.org/abs/2605.30280) is a unified vision-language-action
model: a Qwen3.5-4B VLM backbone paired with a 1.15B-param DiT flow-matching
action expert. Its **only** platform-specific interface is an embodiment-aware
text prompt, so the same checkpoint drives manipulation, navigation, and
trajectory prediction across robots.

This page covers the **inference policy provider**: connect a `QwenVlaPolicy`
to a running Qwen-VLA server over ZMQ (SERVICE mode), or load a checkpoint
in-process (LOCAL mode). The design mirrors the existing GR00T provider:
explicit observation/action mappings, two inference modes, and `reset(seed=)`
forwarding for reproducible evaluation.

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

In the MuJoCo sim, drive the `so100` arm directly:

```python
from strands_robots import Robot

sim = Robot("so100")
sim.run_policy(
    "so100",
    policy_provider="qwen_vla",
    policy_config={"data_config": "so100", "host": "127.0.0.1", "port": 5556},
    instruction="pick up the red cube",
)
```

The provider auto-bridges the sim's per-joint observation schema (one scalar
per actuator + the rendered camera frame) into the model's grouped
`video` / `state` modality keys, and flattens the model's grouped action
vectors back to one scalar per robot joint - so it drives per-joint actuators
out of the box.

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

### Splitting a unified `Y[H, K]` chunk (per-family widths)

When the inference server returns a **single** `Y[H, K]` tensor (the paper's
unified layout, §2.4) rather than per-family arrays, the policy must split it
back into action families. Splitting requires the **exact per-family channel
widths** — an even split would silently mis-route channels for any embodiment
whose families differ in width (e.g. so100: `single_arm` 6 + `gripper` 1; an
even 3/4 split sends gripper commands to arm joints). To avoid that
data-corruption class, the policy **never guesses**:

- Seed embodiments declare widths in `data_configs.json` via `action_dims`
  (e.g. so100 `{"single_arm": 6, "gripper": 1}`), so the auto-inferred mapping
  splits correctly out of the box.
- For a custom embodiment, pass widths through `action_mapping` using the
  `"robot_key:width"` syntax, or `create_custom_data_config(..., action_dims=...)`:

  ```python
  policy = QwenVlaPolicy(
      data_config="so100",
      action_mapping={"action.single_arm": "arm:6", "action.gripper": "grip:1"},
  )
  ```

- Without widths, a multi-family unified chunk raises `ValueError` rather than
  mis-routing. Servers that already return per-family action dicts never hit
  this path (no split needed).

## Configuration (env vars)

| Variable | Purpose |
|---|---|
| `QWEN_VLA_API_TOKEN` | ZMQ auth token for SERVICE mode (sent with each request). |

## Status

- **Inference provider**: shipped — prompt, data configs, normalize, client,
  policy (LOCAL + SERVICE), registry entry, unit + integration tests.

> Note: the upstream Qwen-VLA model package / public checkpoint is not yet
> released. LOCAL mode raises a clear, actionable error until it is; SERVICE
> mode works today against any server speaking the documented ZMQ envelope
> (see `tests_integ/qwen_vla/test_qwen_vla_inference.py` for a reference stub
> server).
