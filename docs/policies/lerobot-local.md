---
description: HuggingFace LeRobot direct inference - ACT, Pi0, SmolVLA, Diffusion Policy, MolmoAct2. RTC + processor bridge.
---

# LeRobot Local

```bash
uv pip install "strands-robots[lerobot]"
export STRANDS_TRUST_REMOTE_CODE=1        # required; raises UntrustedRemoteCodeError otherwise
```

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/pi0_so100",   # HF model_id or local path
    device="cuda",
)
```

## Parameters

```python
LerobotLocalPolicy(
    pretrained_name_or_path="",          # HF model_id or local checkpoint dir (required)
    policy_type=None,                    # override auto-detected class
    device=None,                         # "cuda" | "cpu" | "mps"
    actions_per_step=1,
    use_processor=True,                  # observation/action processor bridge
    processor_overrides=None,
    tokenizer_max_length=48,
    tokenizer_padding_side="right",
    rtc_enabled=None,                    # Real-Time Chunk smoothing (NOT rtc=)
    rtc_execution_horizon=None,
    rtc_max_guidance_weight=None,
    inference_kwargs=None,
    embodiment=None,
    norm_tag=None,                       # MolmoAct2 normalisation tag
    image_keys=None,                     # MolmoAct2 camera key override
    inference_action_mode="continuous",  # "continuous" | "discrete"
)
```

## Supported models

| Model | Notes |
|-------|-------|
| ACT | Action Chunking Transformer |
| Pi0 / Pi0.5 | Physical Intelligence VLA |
| SmolVLA | HuggingFace small VLA |
| Diffusion Policy | flow-matching |
| VQ-BeT | discrete action tokenisation |
| MolmoAct2 | transformers-native SO100/SO101; **requires lerobot from source** (see below) |

## MolmoAct2

> **Important:** MolmoAct2 requires lerobot installed **from source** (git main).
> The `MolmoAct2Policy` class was added after lerobot 0.5.1 (the latest PyPI
> release as of June 2025). A plain `pip install strands-robots[lerobot]` resolves
> lerobot 0.5.1, which does NOT include MolmoAct2.

Install lerobot from source before using MolmoAct2:

```bash
# Standard (x86_64, macOS):
uv pip install "lerobot[feetech] @ git+https://github.com/huggingface/lerobot.git"

# Jetson / aarch64 (pyav wheel may fail to build - skip it, lerobot uses torchcodec):
uv pip install "lerobot[feetech] @ git+https://github.com/huggingface/lerobot.git" --no-build-isolation
# If pyav still blocks the install, exclude it and add torchcodec manually:
uv pip install torchcodec>=0.7
uv pip install "lerobot[feetech] @ git+https://github.com/huggingface/lerobot.git" --no-deps
uv pip install -r <(pip show lerobot 2>/dev/null | grep Requires | sed 's/Requires: //;s/, /\n/g' | grep -v "^av$")
```

Once lerobot from source is installed, MolmoAct2 works:

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="your-org/molmoact2-so101",
    device="cuda",
    norm_tag="so101",
    image_keys=["wrist_camera", "front_camera"],
    inference_action_mode="continuous",
)
# see examples/molmoact2_so101_pickplace.py
```

This requirement will go away once HuggingFace publishes lerobot >= 0.5.2 to PyPI
(which will include MolmoAct2 natively). At that point the standard
`pip install strands-robots[lerobot]` path will work.

## Processor bridge and normalization

`use_processor=True` (default) wraps the policy in a processor bridge that
normalizes observations going into the model and unnormalizes actions coming
back out, so the robot sees commands in physical joint units.

The bridge loads the model's own pipeline configs in priority order:

1. `policy_preprocessor.json` / `policy_postprocessor.json` - LeRobot's standard
   saved pipelines (most lerobot-native checkpoints).
2. **`norm_stats.json` fallback** - checkpoints that ship only a stats file (no
   standard pipeline configs), such as the MolmoAct2 SO-100/101 family. The
   bridge detects the `molmoact2_norm_stats.v1` schema and builds the
   normalizers itself.

Without the fallback in (2), a stats-only checkpoint would silently pass data
through un-normalized: state reaches the policy in raw degrees and predicted
actions reach the motors still in the model's normalized space, producing
off-policy / micro-motion trajectories.

The fallback supports the `q01_q99`, `q10_q90`, `min_max` and `mean_std`
normalization modes declared by `norm_mode`. For `q01_q99`:

```
state_norm  = clip(2 * (state - q01) / (q99 - q01) - 1, -1, 1)
action_unnorm = (clip(action, -1, 1) + 1) * (q99 - q01) / 2 + q01
```

When a stats file declares multiple embodiment tags, pass `norm_tag=` to select
one; a single-tag file is auto-detected.

## RTC

```python
policy = create_policy("lerobot_local", pretrained_name_or_path="lerobot/pi0_so100",
                        rtc_enabled=True, rtc_execution_horizon=16, rtc_max_guidance_weight=1.0)
```

## See also

- [Policy providers](../policies/overview.md)
- [Training](../training/overview.md)
- [GR00T](groot.md)
- [cuRobo](curobo.md)
- [LeRobot project](https://github.com/huggingface/lerobot)
