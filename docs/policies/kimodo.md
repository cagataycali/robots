# Kimodo — text-to-motion diffusion for the Unitree G1

`KimodoPolicy` wraps NVIDIA's Kimodo (`nvidia/Kimodo-G1-RP-v1`) text-conditioned
motion diffusion model. Given a natural-language prompt it samples per-frame
full-body `qpos` sequences for the Unitree G1 in a single diffusion pass, then
streams them one frame per tick as G1 joint targets.

Kimodo sits in the same seat as [`MotionBricksPolicy`](./motionbricks.md) — it
is a *kinematic motion generator* that emits motion targets, not torques. A
tracking controller (WBC / PD) turns those into physics — see
[`WBC`](./wbc.md) or compose via `policy_provider="composite"` (see
[Custom Policies](./custom-policies.md)).

## When to use

| | Kimodo | MotionBricks |
|---|---|---|
| Control input | free-form text prompt | style token + heading |
| Prompt vocabulary | anything English | fixed clip modes |
| Sampler | diffusion (multi-step) | autoregressive one-shot |
| Wall clock (Jetson AGX-class, 100 steps, 120 frames) | ~8 s | ~1 s |
| Best for | novel motions, prompt engineering | known styles, low latency |

## Install

```bash
pip install "strands-robots[kimodo]"
```

The extra installs the `diffusers` loader, which drives any checkpoint published
in *diffusers pipeline layout*. `trust_remote_code=True` is forwarded for a
pipeline that ships custom code, and the factory gates it behind an explicit
opt-in:

```bash
export STRANDS_TRUST_REMOTE_CODE=1
```

Weights are fetched from HuggingFace on first use under the NVIDIA Open Model
License; nothing is bundled with `strands_robots`.

!!! important "`nvidia/Kimodo-G1-RP-v1` is not a diffusers pipeline"

    NVIDIA publishes the Kimodo weights bare — `config.yaml`,
    `model.safetensors` and `stats/`, with `library_name: kimodo` on the Hub.
    There is no `model_index.json`, so `DiffusionPipeline.from_pretrained`
    cannot load it and the default `model_id` is refused at construction. To
    run the NVIDIA checkpoint, supply its sampler through `motion_agent=` — see
    [Driving the NVIDIA checkpoint](#driving-the-nvidia-checkpoint).

## Quick start

```python
import os; os.environ["MUJOCO_GL"] = "egl"  # headless GL on Jetson/Docker
from strands_robots import Robot

sim = Robot("g1", mesh=False)
sim.add_camera(name="front", position=[3.0, 0.0, 1.2], target=[0.0, 0.0, 0.8])

sim.run_policy(
    robot_name="g1",
    policy_provider="kimodo",
    policy_config={
        "diffusion_steps": 100,
        "guidance_scale": 7.5,
        "num_frames": 120,
        "device": "cuda",
        "dtype": "fp16",
    },
    instruction="a person walking forward with confident strides",
    n_steps=200,
    control_frequency=50,
    video={"path": "walk.mp4", "camera": "front", "fps": 25},
)
```

## Composing with a physics tracker

Kimodo is kinematic. To close the loop through physics, compose it with WBC:

```python
sim.run_policy(
    robot_name="g1",
    policy_provider="composite",
    policy_config={
        "layers": [
            {"provider": "kimodo", "config": {"diffusion_steps": 100}},
            {"provider": "wbc"},
        ],
    },
    instruction="walking forward",
    n_steps=500,
)
```

## Config reference

`KimodoConfig` (`strands_robots.policies.kimodo.KimodoConfig`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `model_id` | str | `nvidia/Kimodo-G1-RP-v1` | HF model id |
| `diffusion_steps` | int | 100 | 25–200 useful range |
| `guidance_scale` | float | 7.5 | CFG weight |
| `num_frames` | int | 120 | ≤196 (RP-v1 max) |
| `native_fps` | int | 30 | Sampler native rate |
| `tracker_fps` | int | 50 | SLERP upsample target |
| `device` | str \| None | auto | `"cuda"` / `"cpu"` |
| `dtype` | str | `"fp16"` | `"fp16"` / `"bf16"` / `"fp32"` |
| `seed` | int \| None | None | Reproducible sampling |

Every field above is also an explicit keyword argument of `KimodoPolicy`, so it
can be set three interchangeable ways:

```python
from strands_robots import create_policy
from strands_robots.policies.kimodo import KimodoConfig, KimodoPolicy

create_policy("kimodo", diffusion_steps=25)          # flat, through the factory
KimodoPolicy(config=KimodoConfig(diffusion_steps=25))  # a config object
KimodoPolicy(config={"diffusion_steps": 25})           # a plain dict
```

Precedence is per-field override > `config` field > the default in the table. A
merged value is re-validated by `KimodoConfig`, so `diffusion_steps=0` is
refused whichever way it arrives. There is no `**kwargs`: a misspelled knob
raises `TypeError` at construction instead of being silently ignored.

## When the checkpoint is not a Kimodo checkpoint

`model_id` is accepted verbatim so an alternate revision can be pinned. Two
distinct refusals guard that freedom.

**At load time**, a target carrying no `model_index.json` is not a diffusers
pipeline at all, so no amount of sampling will help. Rather than surface a bare
404 for a file that will never exist, the loader names the layout mismatch and
the remedy:

```text
RuntimeError: Kimodo model_id 'nvidia/Kimodo-G1-RP-v1' is not a diffusers
pipeline: it carries no model_index.json, so DiffusionPipeline.from_pretrained
cannot load it. NVIDIA's Kimodo checkpoints publish bare weights (config.yaml
plus model.safetensors) for their own runtime - the Hub declares library_name
'kimodo', not 'diffusers'. Pass motion_agent= with a sampler that loads this
checkpoint through its own runtime and returns a (num_frames, 7+29) qpos array,
or point model_id at a checkpoint published in diffusers pipeline layout.
```

A transport failure is *not* reported this way — a 401 or a 503 re-raises
untouched, so a network problem is never misread as a layout problem.

**At sample time**, a pipeline that loaded but names its output something other
than `motion` is refused with a `RuntimeError` naming the `model_id` and the
fields the output *did* carry:

```text
RuntimeError: Kimodo pipeline output for model_id 'acme/not-kimodo' carries no
'motion' field: got _SampleOutput with fields sample. Kimodo emits per-frame
qpos under 'motion' - point model_id at a Kimodo checkpoint, or pass
motion_agent= to adapt a sampler that names its output differently.
```

The remedies are the two the message names: point `model_id` at a Kimodo
checkpoint, or pass a `motion_agent=` adapter that reads the sampler's own
output field and returns the `(num_frames, 7+29)` `qpos` array this policy
expects.

## Driving the NVIDIA checkpoint

`nvidia/Kimodo-G1-RP-v1` loads through NVIDIA's own `kimodo` runtime, which is
distributed with the model rather than on PyPI. Wrap it in a `KimodoMotionAgent`
and hand the policy to `run_policy` as a built object:

```python
import numpy as np
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
        output = self._model(
            [prompt.strip().rstrip(".") + "."],
            [num_frames],
            num_denoising_steps=diffusion_steps,
            num_samples=1,
            return_numpy=True,
        )
        qpos = np.asarray(self._converter.dict_to_qpos(output, self._device))
        return qpos[0].astype(np.float32) if qpos.ndim == 3 else qpos.astype(np.float32)


sim.run_policy(
    robot_name="g1",
    policy_object=KimodoPolicy(motion_agent=NativeKimodoAgent()),
    instruction="a person walking forward with confident strides",
    n_steps=200,
    control_frequency=50,
)
```

The runtime emits a dict of rotation matrices and root positions, so the
`MujocoQposConverter` step is what produces the `(num_frames, 7+29)` qpos array
the agent protocol expects. `guidance_scale` has no counterpart in that runtime
(its classifier-free-guidance knob is a per-stage `cfg_weight` list) and is
ignored by this adapter.

## Unit testing without weights

Inject a `KimodoMotionAgent` stub — no torch/diffusers/CUDA needed. See
`tests/policies/kimodo/test_kimodo_policy.py` for the pattern.

## References

* Kimodo: <https://huggingface.co/nvidia/Kimodo-G1-RP-v1>
* Sibling policy: [`motionbricks`](./motionbricks.md)
