---
description: HuggingFace LeRobot direct inference - ACT, Pi0, SmolVLA, Diffusion Policy, MolmoAct2. RTC + processor bridge.
---

# LeRobot Local

```bash
uv pip install "strands-robots[smolvla]"  # [lerobot] plus transformers, which SmolVLA needs at load
export STRANDS_TRUST_REMOTE_CODE=1        # required; raises UntrustedRemoteCodeError otherwise
```

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/smolvla_base",   # HF model_id or local path
    device="cuda",
)
```

`lerobot_local` loads a LeRobot checkpoint in-process and drives it through the
same `get_actions` contract as every other provider. The policies themselves,
their configs and their processor pipelines are LeRobot's - see the
[LeRobot policy docs](https://huggingface.co/docs/lerobot). This page covers
what strands adds around them.

## Parameters

```python
LerobotLocalPolicy(
    pretrained_name_or_path="",          # HF model_id or local checkpoint dir (required)
    policy_type=None,                    # override auto-detected class
    device=None,                         # "cuda" | "cpu" | "mps"
    actions_per_step=1,                  # positive int; auto-set from config.n_action_steps if left at 1
                                         #   (a value BELOW that chunk is warned about - see RTC)
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
    camera_key_map=None,                 # {robot_cam_name: policy_image_key}
    obs_rename_override=None,            # {runtime_obs_key: "observation.images.*"} merged over embodiment.obs_rename
    strict_keys=False,                   # raise instead of a degraded camera OR joint-state binding
    cache_model=True,                    # reuse a process-cached model across instances
    revision=None,                       # pin a HF Hub revision (branch/tag/commit SHA)
)
```

| Parameter | What strands does with it | Refused |
|---|---|---|
| `policy_type` | auto-detected from the checkpoint config; `list_policy_types()` enumerates what the installed lerobot resolves | a type lerobot cannot resolve |
| `revision` | threaded to `from_pretrained(revision=...)`; part of the cache key | - |
| `device` | remaps a checkpoint's baked `device_processor.device` (e.g. `"cuda"`) to the requested device instead of failing on a CPU-only host | an unavailable device |
| `tokenizer_max_length` | instruction token budget | anything but a positive `int` (a count below one truncates the instruction away) |
| `image_keys` | MolmoAct2 camera declaration | a bare string (`"wrist"` would be read as five one-letter names) |
| `strict_keys` | turns the degraded camera / joint-state bindings into raises (see [observation binding](lerobot-local-observations.md)) | non-boolean |
| `cache_model` | see Model caching | non-boolean |

## Model caching

Loading a large VLA (MolmoAct2 SO-100/101 ships 1,295 weight files) takes a
minute or more. Models are cached process-wide, keyed by
`(pretrained_name_or_path, policy_type, device, revision)` and by the RTC
request (`rtc_enabled`, `rtc_execution_horizon`, `rtc_max_guidance_weight`); a
second `create_policy` with the same key reuses the weights. RTC is part of the
key because it is configured on the model rather than beside it, so an RTC-on
and an RTC-off policy from one checkpoint hold one resident copy each - which is
what lets an on/off comparison run in a single process without either arm
rewriting the other's RTC. Call `clear_model_cache()` between the two arms when
the memory matters more than the reload. Every instance records
`load_cache_hit` (`bool`) and `load_time_s` (`float`, near `0.0` on a hit),
and `run_policy` reports them as
`policy_load_cache_hit` / `policy_load_time_s` in its result block.

```python
from strands_robots.policies.lerobot_local import clear_model_cache, list_cached_models
clear_model_cache()  # evict cached models and free their GPU/CPU memory
for entry in list_cached_models():
    print(entry["namespace"], entry["pretrained_name_or_path"], entry["device"])
```

## Supported models

`policy_type` accepts any type the installed lerobot resolves:

```python
from strands_robots import list_policy_types
list_policy_types()
```

| `policy_type` | Model |
|---------------|-------|
| `act` | Action Chunking Transformer |
| `diffusion` | Diffusion Policy (visuomotor) |
| `vqbet` | VQ-BeT - discrete action tokenisation |
| `tdmpc` | TD-MPC model-based control |
| `smolvla` | SmolVLA - HuggingFace small VLA |
| `pi0` / `pi05` / `pi0_fast` | Physical Intelligence VLA family |
| `groot` | NVIDIA GR00T |
| `molmoact2` | transformers-native SO100/SO101 VLA; `pip install 'strands-robots[molmoact2]'` (see below) |
| `eo1` | EO-1 VLA |
| `xvla` | X-VLA |
| `wall_x` | Wall-X VLA |
| `vla_jepa` | VLA-JEPA |
| `multi_task_dit` | Multi-task Diffusion Transformer |
| `gaussian_actor` | Gaussian actor |

## MolmoAct2

MolmoAct2 ships in lerobot >= 0.6 and resolves straight from PyPI; the
`[molmoact2]` extra layers its transformers range on top of `[lerobot]`:

```bash
uv pip install "strands-robots[molmoact2]"
```

```python
sim.run_policy(
    robot_name="so101_follower",
    policy_provider="lerobot_local",
    policy_config={
        "pretrained_name_or_path": "your-org/molmoact2-so101",
        "norm_tag": "so101",
        "inference_action_mode": "continuous",
        "actions_per_step": 30,   # explicit; matches the trained chunk size
    },
    instruction="pick up the cube",
    action_horizon=30,            # do not truncate the 30-step chunk
)
```

Action contract, units and motion diagnostics: [MolmoAct2](molmoact2.md).

## Observation binding

The processor bridge and its normalization stats, the unit frame those stats are
in, the joint keys that compose `observation.state`, and the camera routed onto
each declared image feature: [observation
binding](lerobot-local-observations.md).

## RTC

Real-Time Chunking (LeRobot's `rtc_*` config) blends each new action chunk into
the still-unexecuted tail of the previous one and lets inference overlap
execution. Enable it per policy:

```python
policy = create_policy("lerobot_local", pretrained_name_or_path="lerobot/smolvla_base",
                        rtc_enabled=True, rtc_execution_horizon=16, rtc_max_guidance_weight=1.0)
```

Every public flow-matching checkpoint ships `config.rtc_config = None` - RTC
is an inference-time choice, not a training artifact - so `rtc_enabled=True`
builds that config from your `rtc_execution_horizon` /
`rtc_max_guidance_weight` (lerobot's defaults for the rest) and hands it to
lerobot's `init_rtc_processor()`. Only the flow-matching config classes
declare the field: ask ACT or Diffusion for RTC and the provider warns and
runs `select_action()`. `rtc_enabled=None` (the default) follows whatever the
checkpoint itself was saved with.

The sim consumes `policy.execution_horizon` actions from each chunk before
re-querying - `rtc_execution_horizon` (default 10) for an RTC policy, the full
chunk otherwise.

RTC - not a smaller `actions_per_step` - is how you shorten that interval.
Pinning `actions_per_step` below the checkpoint's `config.n_action_steps`
truncates every chunk to its prefix and re-queries from a state the model was
never trained to replay from, so each seam is a discontinuity in the commanded
trajectory; the provider now names that when it happens. `rtc_enabled=True`
re-queries just as often and blends the unexecuted tail into the next chunk
instead, leaving `actions_per_step` at the trained chunk. For relative-action checkpoints (pi0 / pi0.5 / pi0-FAST
trained with `RelativeActionsProcessorStep`) the carried prefix is re-anchored
to the state at the new query, so the seam does not double-apply the offset.

### Synchronous vs async chunk execution in sim

`run_policy` / `PolicyRunner.run` accept `async_rtc`:

| `async_rtc` | Behaviour | Use when |
| --- | --- | --- |
| `None` (default) | Auto-resolve from `policy.is_chunk_emitting()`: chunk-emitting policies get the async overlap, single-step policies stay synchronous. An explicit `True`/`False` always wins. | The common case - let the policy decide. |
| `False` | Query the policy, drain `execution_horizon` actions, then re-query. Seam blending works because the RTC policy is re-queried mid-chunk, but inference and execution do **not** overlap. | Single-step policies, deterministic regression runs, or any policy whose `get_actions` reads live sim state. |
| `True` | While the current chunk drains, fire the next `get_actions` on a background worker once the chunk is ~50% consumed, then atomically swap it in. | Chunk-emitting VLA / flow-matching policies (pi0, pi0.5, pi0-FAST, SmolVLA, MolmoAct2) where sim per-step timing should track real hardware. |

```python
sim.run_policy(robot_name="so101", policy_provider="lerobot_local",
               policy_config={"pretrained_name_or_path": "lerobot/smolvla_base", "rtc_enabled": True},
               action_horizon=8, async_rtc=False)
```

### Deterministic inference delay

RTC slices the next chunk by how many control steps elapsed during inference.
`PolicyRunner` passes the exact count via `policy.set_rtc_observed_delay(steps)`
before each query (`0` in the synchronous loop; the remaining drain of the
current chunk in the async pipeline), so a runner-driven eval is
bit-reproducible regardless of machine load. Driven without a runner (async
real hardware), leave it `None` and the policy uses its wall-clock p95
estimate. The override accepts `None` or a non-negative `int` only, and
`set_control_frequency(hz)` a finite positive number - `nan`, `inf`, fractions
and `True` are refused where they arrive.

## See also

- [Observation binding](lerobot-local-observations.md) - normalization stats, unit frames, state and camera routing
- [MolmoAct2 (SO-100/101)](molmoact2.md) - action contract, units, and motion diagnostics
- [Policy providers](../policies/overview.md)
- [Training](../training/overview.md)
- [LeRobot policy docs](https://huggingface.co/docs/lerobot) - configs, processors, RTC
