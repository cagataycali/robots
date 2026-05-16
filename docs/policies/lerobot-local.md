---
description: HuggingFace LeRobot direct inference — ACT, Pi0, SmolVLA, Diffusion Policy. RTC + processor bridge.
---

# LeRobot Local

`LerobotLocalPolicy` runs a HuggingFace LeRobot policy in-process — no separate server.
Supports the full LeRobot model zoo (ACT, Pi0, SmolVLA, Diffusion Policy, etc.).

## TL;DR

```bash
export STRANDS_TRUST_REMOTE_CODE=1
```

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/pi0_so100",   # any HF model_id or local path
    device="cuda",                                 # "cuda" | "cpu" | "mps"
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

Anything LeRobot's `make_policy(...)` understands. As of LeRobot 0.5:

- **ACT** — Action Chunking Transformer
- **Pi0** — VLA from Physical Intelligence
- **Pi0.5** — newer Pi0 variant
- **SmolVLA** — small VLA from HuggingFace
- **Diffusion Policy** — flow-matching alternative
- **VQ-BeT** — discrete action tokenisation

The exact list depends on the LeRobot version installed. The policy auto-detects the
class from the checkpoint's config.

## Constructor parameters

```python
LerobotLocalPolicy(
    pretrained_name_or_path: str,    # HF model_id OR local checkpoint dir
    device: str = "cuda",            # torch device
    use_amp: bool = False,           # auto mixed precision
    rtc: bool = False,               # Real-Time Chunk smoothing
    processor_overrides: dict | None = None,
    **kwargs,                        # passed to LeRobot's make_policy
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

## RTC (Real-Time Chunk)

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/pi0_so100",
    rtc=True,
    chunk_size=16,
)
```

Same idea as GR00T's RTC: overlapping action chunks smoothed between inference calls,
so the robot doesn't stall between chunks. Especially useful for slower diffusion-based
policies.

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
