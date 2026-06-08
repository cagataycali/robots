---
description: HuggingFace LeRobot direct inference — ACT, Pi0, SmolVLA, Diffusion Policy, MolmoAct2. RTC + processor bridge.
---

# LeRobot Local

`LerobotLocalPolicy` runs a HuggingFace LeRobot policy in-process — no separate server.
Supports the full LeRobot model zoo (ACT, Pi0, SmolVLA, Diffusion Policy, etc.) plus
MolmoAct2.

## TL;DR

```bash
export STRANDS_TRUST_REMOTE_CODE=1
```

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/pi0_so100",   # any HF model_id or local path
    device="cuda",                                  # "cuda" | "cpu" | "mps"
)
```

## Setup

```bash
pip install "strands-robots[lerobot]"
```

That pulls in `lerobot` + `torch`. Test the install:

```python
from strands_robots.policies import create_policy
policy = create_policy("lerobot_local",
                        pretrained_name_or_path="lerobot/pi0_so100")
```

## Supported policies

Anything LeRobot's `make_policy(...)` understands, plus MolmoAct2. As of LeRobot 0.5:

- **ACT** — Action Chunking Transformer
- **Pi0** — VLA from Physical Intelligence
- **Pi0.5** — newer Pi0 variant
- **SmolVLA** — small VLA from HuggingFace
- **Diffusion Policy** — flow-matching alternative
- **VQ-BeT** — discrete action tokenisation
- **MolmoAct2** — transformers-native VLA for SO100/SO101; configured via
  `norm_tag`, `image_keys`, and `inference_action_mode`

The exact list depends on the LeRobot version installed. The policy auto-detects the
class from the checkpoint's config.

## Constructor parameters

```python
LerobotLocalPolicy(
    pretrained_name_or_path: str = "",         # HF model_id or local checkpoint dir
    policy_type: str | None = None,            # override auto-detected policy class
    device: str | None = None,                 # torch device ("cuda", "cpu", "mps")
    actions_per_step: int = 1,                 # actions to consume per control tick
    use_processor: bool = True,                # enable observation processor bridge
    processor_overrides: dict | None = None,   # override processor defaults
    tokenizer_max_length: int = 48,            # instruction tokenization length
    tokenizer_padding_side: str = "right",     # "left" | "right"
    rtc_enabled: bool | None = None,           # enable Real-Time Chunk smoothing
    rtc_execution_horizon: int | None = None,  # RTC execution horizon
    rtc_max_guidance_weight: float | None = None,  # RTC max guidance weight
    inference_kwargs: dict | None = None,      # extra kwargs for model.forward()
    embodiment: str | None = None,             # embodiment tag override
    norm_tag: str | None = None,               # normalisation tag (MolmoAct2)
    image_keys: list[str] | None = None,       # camera key override (MolmoAct2)
    inference_action_mode: str = "continuous", # "continuous" | "discrete"
)
```

`pretrained_name_or_path` is the only required argument. Everything else has sensible
defaults.

## Trust remote code

Most LeRobot models on the Hub use `trust_remote_code=True` for custom architectures.
The factory enforces an explicit opt-in:

```bash
export STRANDS_TRUST_REMOTE_CODE=1
```

Without that, `create_policy("lerobot_local", ...)` raises
`UntrustedRemoteCodeError`. This matters most on real hardware — an attacker
publishing a malicious checkpoint could otherwise execute code with your servo
permissions.

Set the env var only after vetting the model source. See
`strands_robots/policies/factory.py` for the gate logic.

## Processor bridge

LeRobot 0.4 and 0.5 use slightly different observation/action processor pipelines.
`LerobotLocalPolicy` includes a bridge (`processor.py`) that handles both. You can
override processor behaviour via `processor_overrides`:

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/pi0_so100",
    processor_overrides={"normalize": False},
)
```

Most callers don't need this — the defaults match the model's training config.

## MolmoAct2

MolmoAct2 is a transformers-native VLA designed for SO100/SO101 setups. Configure it
using `norm_tag`, `image_keys`, and `inference_action_mode`:

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="your-org/molmoact2-so101",
    device="cuda",
    norm_tag="so101",
    image_keys=["wrist_camera", "front_camera"],
    inference_action_mode="continuous",
)
```

See `examples/molmoact2_so101_pickplace.py` for a full rollout example.

## RTC (Real-Time Chunk)

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/pi0_so100",
    rtc_enabled=True,
    rtc_execution_horizon=16,
    rtc_max_guidance_weight=1.0,
)
```

RTC overlaps action-chunk generation so the robot keeps moving while the next chunk
is being computed — especially useful for diffusion-based policies with ~200ms
inference latency.

## Resolution: 0.4 vs 0.5

LeRobot's policy class registration changed between 0.4 and 0.5. The resolution logic
(`policies/lerobot_local/resolution.py`) handles both versions automatically — the
caller doesn't pin to one.

If you have a `lerobot==0.4.x` install, the resolution still works for the policies
that existed at 0.4. Newer policies (post-0.5) require a 0.5+ install.

## Loading from a local checkpoint

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="/path/to/your/local/checkpoint",  # not an HF id
)
```

The policy detects local paths vs HF ids automatically. Local checkpoints don't
require `STRANDS_TRUST_REMOTE_CODE=1` since you control the bytes on disk.

## See also

- [Tutorial 3 — Policies](../tutorial/03-policies.md) — full walkthrough.
- [Tutorial 7 — Training](../tutorial/07-training.md) — train a checkpoint, load it
  back here.
- [GR00T](groot.md) — server-based alternative.
- [LeRobot project](https://github.com/huggingface/lerobot) — upstream library.
