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
pip install "strands-robots[kimodo]"
```

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

Weights are fetched from HuggingFace on first use under the NVIDIA Open Model
License; nothing is bundled with `strands_robots`.

!!! important "`nvidia/Kimodo-G1-RP-v1` is not a diffusers pipeline"

    NVIDIA publishes the Kimodo weights bare — `config.yaml`,
    `model.safetensors` and `stats/`, `library_name: kimodo` on the Hub, no
    `model_index.json` — so `DiffusionPipeline.from_pretrained` cannot load it
    and the default `model_id` is refused at construction. To run the NVIDIA
    checkpoint, supply its sampler through `motion_agent=` — see
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

Run standalone (the example above) the 29 targets are applied directly, which is
the faithful visualisation of the generated motion. Making the robot follow it
under physics needs a controller that **tracks the reference**, in series over
the same joints:

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

## When the sampler runs again

One `sample()` call produces a motion buffer that `get_actions` drains one frame
per control tick, holding the last frame once exhausted. The buffer is keyed on
the four inputs that determine it — the prompt plus `diffusion_steps`,
`guidance_scale` and `seed` — so the sampler runs again as soon as any of them
differs, and otherwise the buffered frames are reused:

```python
await policy.get_actions({}, "walking forward")                     # samples
await policy.get_actions({}, "walking forward")                     # drains
await policy.get_actions({}, "waving")                              # samples
await policy.get_actions({}, "waving", diffusion_steps=25)          # samples
policy.reset()                                                      # rewinds
policy.reset(seed=7); await policy.get_actions({}, "waving")        # samples
```

Per-call overrides keep the config fields' domains and are checked before the
key is built, so a refused override costs neither a diffusion run nor a frame.
The seed must be a whole number on every surface that sets one: a fractional
seed would key as `int(seed)`, making `2.5` and `2.9` replay one motion. This is
what makes a multi-episode `eval_policy` meaningful — `PolicyRunner.evaluate`
hands each episode its own seed through `reset(seed=...)`, so every episode
samples its own motion and the run replays exactly at the same master `seed=`.

## Chaining prompts into a long-horizon sequence

Because a changed prompt samples the next segment and the stream simply
continues, a long-horizon episode is a rollout that changes the instruction as
it goes; a `policy_object` driven directly is the smallest version:

```python
import asyncio

from strands_robots import Robot
from strands_robots.policies.kimodo import KIMODO_G1_JOINTS, KimodoPolicy

CHAIN = [
    ("a person walking forward with confident strides", 90),
    ("a person turning to the left", 60),
    ("a person waving with the right hand", 60),
    ("a person crouching down to pick an object off the floor", 90),
    ("a person walking forward with confident strides", 90),
]

sim = Robot("g1", mesh=False)
policy = KimodoPolicy()
policy.set_robot_state_keys(list(KIMODO_G1_JOINTS))

for instruction, ticks in CHAIN:
    for _ in range(ticks):
        action = asyncio.run(policy.get_actions({}, instruction))[0]
        sim.set_joint_positions(action, robot_name="g1")
```

Each segment is sampled once, on the tick its instruction first appears. Kimodo
samples every motion from its own canonical start pose, so a new segment is
eased off the pose last commanded across `transition_frames` native frames
(default 5, the sampler's own `num_transition_frames`; minimum 1). Without it,
across the 600 ordered pairs of a 25-motion corpus the median seam moved a joint
1.6 rad in one tick. Easing shifts where a segment starts, not how it moves (the
root orientation takes the rotational form of the same offset, so a turn keeps
its rate); it removes the discontinuity without re-planning the motion. An
episode boundary is not a seam: `reset()` forgets the last commanded pose.

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
satisfies the protocol, raises nothing, and defeats the per-episode seed above.

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
