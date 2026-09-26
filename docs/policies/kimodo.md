# Kimodo — text-to-motion diffusion for the Unitree G1

`KimodoPolicy` wraps NVIDIA's Kimodo (`nvidia/Kimodo-G1-RP-v1`) text-conditioned
motion diffusion model. Given a natural-language prompt it samples per-frame
full-body `qpos` sequences for the Unitree G1 in a single diffusion pass, then
streams them one frame per tick as G1 joint targets. It is a *kinematic motion
generator*: a whole-body reference over all 29 leg + waist + arm joints, not
torques. It takes free-form English and pays for it with a multi-step diffusion
sampler (~8 s for 100 steps, 120 frames on a Jetson AGX-class device).

## Install

```bash
pip install "strands-robots[kimodo,sim-mujoco]"
```

`[kimodo]` carries the sampler's own dependencies only; the Quick start below
runs the motion in MuJoCo, so it needs `[sim-mujoco]` too - without it
`Robot("g1")` raises `ImportError: 'mujoco' is required for MuJoCo simulation`
before any motion is sampled.

The extra installs the `diffusers` loader, which drives any checkpoint published
in *diffusers pipeline layout*. Two independent opt-ins guard that loader, both
off by default. The first decides whether the provider may be built at all:

```bash
export STRANDS_TRUST_REMOTE_CODE=1
```

The second decides whether a checkpoint's own Python code may run when it is
loaded. It is per call, defaults to `False`, and is never set by the environment
variable above:

```python
sim.run_policy(
    policy_provider="kimodo",
    policy_config={
        "model_id": "your-org/kimodo-pipeline",
        "trust_remote_code": True,      # only for a repo you have vetted
    },
    instruction="walk forward",
)
```

The flag is a real boolean, not a spelling of one. A JSON `policy_config` that
writes `"trust_remote_code": "false"` is refused at construction, naming the key
and the value, rather than forwarded to a `from_pretrained` that reads it by
truthiness.

Weights are fetched from HuggingFace on first use under the NVIDIA Open Model
License; nothing is bundled with `strands_robots`.

!!! important "`nvidia/Kimodo-G1-RP-v1` is not a diffusers pipeline"

    NVIDIA publishes the Kimodo weights bare — `config.yaml`,
    `model.safetensors` and `stats/`, `library_name: kimodo` on the Hub, no
    `model_index.json` — so `DiffusionPipeline.from_pretrained` cannot load it
    and the default `model_id` is refused on the first sample: the policy
    constructs, and the refusal arrives when a motion is asked for, so
    `run_policy` reports the rollout as an error having applied no action and
    written no video. To run the NVIDIA checkpoint, supply its sampler through
    `motion_agent=` — see
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

## Tracking the reference under physics

Run standalone (the example above) the 29 joint targets are applied directly -
with no tracker in between, but still to the G1's position actuators under
gravity, and the clip's own 7 root values are not applied. So the motion plays in
place and a standing G1 handed a walking clip topples. Replaying the clip through
`set_joint_positions` holds the root, so that is the route to *look* at a sample
(the [chaining snippet](kimodo-sampling.md#chaining-prompts-into-a-long-horizon-sequence));
making the robot follow it under physics needs a controller that **tracks the
reference**, in series over the same joints:

```text
prompt -> Kimodo -> 29 joint targets -> reference tracker -> torques -> robot
```

That tracker is [`ProtoMotionsPolicy`](./protomotions.md), which takes a `qpos`
clip and emits balanced PD targets. It is a cascade, not a composition:
[`CompositePolicy`](./custom-policies.md) merges two policies over **disjoint**
joint groups, so handing it a whole-body generator and a whole-body controller
is refused with the shadowed joints named. [`WBCPolicy`](./wbc.md) is not a
reference tracker either — its only command is a target base velocity — so
composing it with Kimodo cannot track a Kimodo motion.

## Config reference

`KimodoConfig` (`strands_robots.policies.kimodo.KimodoConfig`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `model_id` | str | `nvidia/Kimodo-G1-RP-v1` | HF model id |
| `diffusion_steps` | int | 100 | 25–200 useful range, ≤500 (the count multiplies the cost of every sample) |
| `guidance_scale` | float | 7.5 | CFG weight, positive and finite |
| `num_frames` | int | 120 | ≤196 (RP-v1 max) |
| `transition_frames` | int | 5 | Native frames a chained segment is eased over |
| `native_fps` | int | 30 | Sampler native rate |
| `tracker_fps` | int | 50 | SLERP upsample target |
| `device` | str \| None | auto | `"cuda"` / `"cpu"` |
| `dtype` | str | `"fp16"` | `"fp16"` / `"bf16"` / `"fp32"` |
| `seed` | int \| None | None | Reproducible sampling. A whole number, either sign, or `None` for fresh entropy |

Every field is also an explicit keyword argument of `KimodoPolicy`, so it can be
set three interchangeable ways, with precedence per-field override > `config`
field > default; a merged value is re-validated by `KimodoConfig`:

```python
from strands_robots import create_policy
from strands_robots.policies.kimodo import KimodoConfig, KimodoPolicy

create_policy("kimodo", diffusion_steps=25)          # flat, through the factory
KimodoPolicy(config=KimodoConfig(diffusion_steps=25))  # a config object
KimodoPolicy(config={"diffusion_steps": 25})           # a plain dict
```

A misspelled knob is refused by the two keyword forms (`TypeError`) and dropped
by the dict form: `KimodoConfig.from_dict` drops unknown keys for forward
compatibility, so `config={"diffusion_stpes": 25}` builds with the default 100
and no warning. Pass the knob as a keyword if a typo should be refused.

A config can also come from a JSON object on disk; `from_json` expands `~`,
names the file in every refusal, and does not check the extension:

```python
KimodoPolicy(config=KimodoConfig.from_json("~/kimodo.json"))
```

## When the checkpoint is not a Kimodo checkpoint

`model_id` is accepted verbatim so an alternate revision can be pinned. Two
refusals guard that freedom, both `RuntimeError`s naming the `model_id` and the
two remedies (point `model_id` at a Kimodo checkpoint, or pass a `motion_agent=`
adapter that returns the `(num_frames, 7+29)` `qpos` array): **at load time**, a
target with no `model_index.json` is not a diffusers pipeline, so the loader
names the layout mismatch instead of a bare 404 (a 401 or 503 re-raises
untouched, so a network problem is never misread as a layout problem); **at
sample time**, a pipeline whose output carries no `motion` field is refused
naming the fields it did carry.

## Driving the NVIDIA checkpoint

`nvidia/Kimodo-G1-RP-v1` loads through NVIDIA's own `kimodo` runtime, which is
distributed with the model rather than on PyPI. Wrap it in a `KimodoMotionAgent`
and hand the policy to `run_policy` as a built object:

```python
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

`MujocoQposConverter` turns the runtime's rotation matrices into the
`(num_frames, 7+29)` array; `guidance_scale` has no counterpart there and is
ignored. `seed` goes through `torch.manual_seed` because the runtime draws from
the global generator — an adapter that accepts `seed` and ignores it still
satisfies the protocol, raises nothing, and defeats the
[per-episode seed](kimodo-sampling.md#when-the-sampler-runs-again).

## Driving the real robot

Kimodo names joints the way the URDF does (`left_hip_pitch_joint`); lerobot's
`UnitreeG1` driver names its action keys after its own enum (`kLeftHipPitch.q`).
The hardware path is a key rename between the two:

```python
from strands_robots.policies.kimodo.hardware import build_lerobot_g1_action_dict

for policy_action in await policy.get_actions(observation, instruction):
    robot.send_action(build_lerobot_g1_action_dict(policy_action))
```

`get_joint_map()` returns the table itself. Both are lerobot-only helpers
(`pip install "strands-robots[lerobot]"`); the physical robot also needs Unitree's
`unitree_sdk2`. Joints are paired by name, never by enum position, because the
driver silently ignores keys it does not know; a driver-side rename or DOF change
is refused with the unmatched joints named on both sides. The rename is one-way —
`get_observation()` already reports `<motor>.q` keys.

## Unit testing without weights

Inject a `KimodoMotionAgent` stub — no torch/diffusers/CUDA needed. See
`tests/policies/kimodo/test_kimodo_policy.py` for the pattern.

## References

* Kimodo: <https://huggingface.co/nvidia/Kimodo-G1-RP-v1>

## See also

- [Kimodo sampling and chaining](kimodo-sampling.md) - when the sampler runs
  again, the seed contract, and chaining prompts into a long-horizon sequence.
- [ProtoMotions](protomotions.md) - the tracker that follows a Kimodo clip under
  physics.
- [Policy providers](overview.md) - what `policy_provider` may name.
