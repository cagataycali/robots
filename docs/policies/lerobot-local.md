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
| `strict_keys` | turns the degraded camera / joint-state bindings below into raises | non-boolean |
| `cache_model` | see Model caching | non-boolean |

## Model caching

Loading a large VLA (MolmoAct2 SO-100/101 ships 1,295 weight files) takes a
minute or more. Models are cached process-wide, keyed by
`(pretrained_name_or_path, policy_type, device, revision)`; a second
`create_policy` with the same key reuses the weights. Every instance records
`load_cache_hit` (`bool`) and `load_seconds`, and `run_policy` reports them as
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

## Processor bridge and normalization

`use_processor=True` (default) wraps the policy in the checkpoint's
pre/post-processor pipelines, so observations are normalized going in and
actions come back in physical joint units. The bridge takes them from
`policy_preprocessor.json` / `policy_postprocessor.json` when the checkpoint
ships them; a pre-processor-era checkpoint that instead carries
`normalize_inputs.*` / `unnormalize_outputs.*` buffers in `model.safetensors`
(still the case for zoo checkpoints such as
`lerobot/act_aloha_sim_transfer_cube_human`) has both pipelines rebuilt from
those buffers with lerobot's own `extract_normalization_stats` +
`make_pre_post_processors`. MolmoAct2 checkpoints take neither path: lerobot's
own factory reads their `norm_stats.json`, and a `norm_tag` the file does not
declare is refused by lerobot naming the tags it does (see
[MolmoAct2](#molmoact2)). `processor_overrides` is keyed by step name (a step's
`registry_name`, or its class name when it is not registry-backed) and each
override is routed to the pipeline that owns that step:

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/smolvla_base",
    policy_type="smolvla",
    processor_overrides={
        "normalizer_processor": {"stats": dataset_stats},    # observation.state
        "unnormalizer_processor": {"stats": dataset_stats},  # action
    },
)
```

Naming only one leaves the other inert, and the diagnostic keeps reporting
whichever half is still unnormalized. Fine-tuning the checkpoint writes stats
under the canonical keys and needs no override at all.

Supplied stats must be as wide as the features the checkpoint declares - a
6-DOF SO-101, a 7-DOF arm and a 14-DOF bimanual all have `observation.state`,
so the wrong dataset's stats are an easy reach. A mismatch is refused at load,
naming the feature and both widths, instead of surfacing as LeRobot's tensor
broadcast error on the first inference after the robot has been commanded.
Visual `(C,)` stats are exempt (LeRobot reshapes them to `(C, 1, 1)`):

```
ValueError: lerobot_local: lerobot/smolvla_base was given normalization stats
that do not match the widths the checkpoint declares: ["observation.state
(STATE/MEAN_STD): feature declares width 6, stats 'observation.state.mean'
supply 7", ...]
```

## State routing

`observation.state` is composed from `robot_state_keys`, set with
`set_robot_state_keys([...])` or by an `embodiment`. When only some keys are
present the vector is partly bound and the policy reports the absent keys, the
keys the observation does carry and the remedy; when none are present it falls
back to the observation's own state vector. Both degradations are logged, and
`strict_keys=True` turns them into raises.

## Camera routing

Observations use bare camera names (`top`, `wrist`); the policy declares image
inputs as `observation.images.*`. Each camera is routed by, in order:

1. `camera_key_map={"front": "observation.images.top", ...}` when given;
2. the embodiment's `obs_rename` map (below);
3. a name heuristic (exact stem, then substring).

### Embodiment `obs_rename` and the pre-flight check

`embodiment="so101"` routes cameras from the embodiment's `obs_rename`
(`{runtime_camera_name: "observation.images.*"}`), merged under any
`obs_rename_override`. Before the model is built the policy checks that every
declared image feature has a source in the runtime observation and refuses
otherwise, naming both sides:

```text
Embodiment 'so101' cannot route cameras to the model's image feature(s)
['observation.images.image', 'observation.images.wrist_image']: none of the
expected source key(s) ['front', 'wrist'] are in the runtime observation, which
provides [...]. Either: (a) rename your sim cameras to one of ['front', 'wrist']
..., or (b) pass policy_config={'obs_rename_override': {...}} ...
```

A single-camera checkpoint needs no embodiment: declare the joint names with
`set_robot_state_keys([...])` and the policy synthesizes a state-only embodiment
that routes the one declared image feature to the one camera.

## RTC

Real-Time Chunking (LeRobot's `rtc_*` config) blends each new action chunk into
the still-unexecuted tail of the previous one and lets inference overlap
execution. Enable it per policy:

```python
policy = create_policy("lerobot_local", pretrained_name_or_path="lerobot/pi0_so100",
                        rtc_enabled=True, rtc_execution_horizon=16, rtc_max_guidance_weight=1.0)
```

The sim consumes `policy.execution_horizon` actions from each chunk before
re-querying - `rtc_execution_horizon` (default 10) for an RTC policy, the full
chunk otherwise. For relative-action checkpoints (pi0 / pi0.5 / pi0-FAST
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
               policy_config={"pretrained_name_or_path": "lerobot/pi0_so100", "rtc_enabled": True},
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

- [MolmoAct2 (SO-100/101)](molmoact2.md) - action contract, units, and motion diagnostics
- [Policy providers](../policies/overview.md)
- [Training](../training/overview.md)
- [LeRobot policy docs](https://huggingface.co/docs/lerobot) - configs, processors, RTC
