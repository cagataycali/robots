---
description: kimodo samples whole-body Unitree G1 motion from an English prompt with NVIDIA's text-to-motion diffusion model.
---

# kimodo

!!! warning "Deprecated, removed in 0.7"
    No in-tree replacement: generate the motion offline and replay it as joint targets.

This page turns a sentence such as "a person walking forward with confident strides" into a 29-joint kinematic reference for the Unitree G1 and plays it in simulation.

```bash
pip install 'strands-robots[kimodo]'    # torch, diffusers, transformers, huggingface_hub, accelerate, scipy
export STRANDS_TRUST_REMOTE_CODE=1      # the diffusers pipeline loads with trust_remote_code=True
```

## What it is

`KimodoPolicy` wraps NVIDIA's Kimodo (`nvidia/Kimodo-G1-RP-v1`), a text-conditioned motion diffusion model. One diffusion pass samples a per-frame full-body `qpos` sequence for the G1; the policy resamples it from the model's native frame rate to the control rate and emits one action dict per tick, keyed by the canonical WBC joint ordering for all 29 leg, waist and arm joints. Any English motion description is a goal; there is no fixed clip vocabulary. `requires_images` is `False`.

Standalone in MuJoCo the targets are applied directly, which is the faithful kinematic reference. Closing the loop through physics needs a controller that tracks that reference over the same 29 joints. That is a cascade, and `CompositePolicy` (which merges disjoint joint groups) cannot express it. [protomotions](protomotions.md) is the tracker; `wbc` cannot track a pose because its only input is a base velocity.

```python title="sketch"
from strands_robots.policies import create_policy

policy = create_policy("kimodo", diffusion_steps=100, guidance_scale=7.5, num_frames=120)
policy = create_policy("text2motion")   # same provider, all defaults
```

## Constructor keywords

{{providers:kwargs:kimodo}}

Everything after `config` is keyword-only and mirrors a `KimodoConfig` field; an explicit keyword wins over the config object. No `**kwargs`: a misspelled keyword is a `TypeError`, but a misspelled key inside `config={...}` is dropped silently by `KimodoConfig.from_dict`. Defaults: `model_id="nvidia/Kimodo-G1-RP-v1"`, `diffusion_steps=100`, `guidance_scale=7.5`, `num_frames=120`, `dtype="fp16"` (`bf16` and `fp32` accepted). `motion_agent` injects a sampler and is how NVIDIA's bare-weights checkpoint, which targets its own runtime rather than a `diffusers` pipeline, is driven.

## Per-call keywords

| keyword | meaning |
|---|---|
| `text_prompt` | overrides `instruction` as the prompt |
| `diffusion_steps`, `guidance_scale` | override the config for this sample |
| `seed` | seeds the sampler for this episode; `reset(seed)` does the same between episodes |

## Run it

Needs the extra, the environment gate, a CUDA GPU and a first-use download.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco", mesh=False)
sim.create_world()
sim.add_robot("g1")
result = sim.run_policy(
    robot_name="g1",
    policy_provider="kimodo",
    policy_config={"diffusion_steps": 100, "guidance_scale": 7.5, "num_frames": 120},
    instruction="a person walking forward with confident strides",
    n_steps=200,
    control_frequency=50.0,
)
print(result["status"])
```

On hardware, `strands_robots.policies.kimodo.hardware.kimodo_action_to_lerobot_g1` renames the 29 canonical joints onto the LeRobot G1 driver's motor names.

## Driving the NVIDIA checkpoint

The bare weights load through NVIDIA's own `kimodo` runtime, shipped with the model rather than on PyPI. Wrap it as the `motion_agent` and hand the policy to `run_policy` as a built object; `seed` must reach `torch.manual_seed`, because the runtime draws from the global generator and an adapter that ignores it defeats the per-episode seed:

```python title="sketch"
import numpy as np
import torch

from strands_robots.policies.kimodo import KimodoPolicy


class NativeKimodoAgent:
    """Samples through NVIDIA's kimodo runtime instead of diffusers."""

    def __init__(self, device: str = "cuda") -> None:
        from kimodo.exports.mujoco import MujocoQposConverter
        from kimodo.model.load_model import load_model

        self._model = load_model("kimodo-g1-rp", device=device)
        self._converter = MujocoQposConverter(self._model.skeleton)
        self._device = device

    def sample(self, prompt, num_frames, diffusion_steps, guidance_scale, seed):
        if seed is not None:
            torch.manual_seed(seed)
        output = self._model([prompt.strip().rstrip(".") + "."], [num_frames], num_denoising_steps=diffusion_steps, num_samples=1, return_numpy=True)
        qpos = np.asarray(self._converter.dict_to_qpos(output, self._device))
        return qpos[0].astype(np.float32) if qpos.ndim == 3 else qpos.astype(np.float32)


sim.run_policy(robot_name="g1", policy_object=KimodoPolicy(motion_agent=NativeKimodoAgent()), instruction="a person walking forward", n_steps=200, control_frequency=50)
```

`MujocoQposConverter` turns the runtime's rotation matrices into the `(num_frames, 7 + 29)` array; `guidance_scale` has no counterpart there and is ignored.

## Limits

- Unitree G1 only; the output is the G1's 29-joint layout.
- Kinematic, not dynamic. Played directly, the reference ignores contact and balance. Track it with [protomotions](protomotions.md) for physics.
- A sample is one diffusion pass over `num_frames` at the native rate, so a long prompt is a long first call; `transition_frames` blends consecutive samples.
- `trust_remote_code=True` is how the `diffusers` pipeline loads, hence the environment gate.
