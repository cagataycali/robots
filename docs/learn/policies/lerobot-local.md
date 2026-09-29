---
description: lerobot_local runs any LeRobot checkpoint in process. Install extra, constructor keywords, embodiments, camera naming rules, and limits.
---

# lerobot_local

By the end of this page you can run a HuggingFace LeRobot checkpoint (ACT, diffusion, pi0, SmolVLA, GR00T N1.7, MolmoAct2) on a simulated or real arm in this process, and you know the two naming rules that decide whether the model sees your cameras and joints.

```bash
pip install 'strands-robots[lerobot]'          # lerobot[feetech,dataset] + psutil
pip install 'strands-robots[smolvla]'          # adds lerobot[smolvla]
pip install 'strands-robots[molmoact2]'        # adds lerobot[molmoact2]
export STRANDS_TRUST_REMOTE_CODE=1             # required: models load with trust_remote_code=True
```

## What it is

`LerobotLocalPolicy` hands the checkpoint to LeRobot's own factory: the policy type is read from the model's `config.json`, so any class LeRobot registers works without a change here. The model's processor pipeline (`preprocessor.json` / `postprocessor.json`) normalises observations and unnormalises actions. Flow-matching models get Real-Time Chunking when the config declares it: the runtime tells the policy its control rate and the steps consumed during inference, and the policy blends the next chunk onto the seam.

Build it by name or smart string; a non-`nvidia` HuggingFace id resolves here.

```python title="sketch"
from strands_robots.policies import create_policy

policy = create_policy("lerobot_local", pretrained_name_or_path="robotfuel/act_so101_t16b", embodiment="so101")
policy = create_policy("robotfuel/act_so101_t16b", embodiment="so101")   # same thing
```

## Constructor keywords

{{providers:kwargs:lerobot_local}}

The remedy for inert normalization has two halves: `processor_overrides={"normalizer_processor": {"stats": ...}}` replaces the stats, and `state_units` / `action_units` (`degrees` or `radians`) say which unit they were recorded in. `so100` and `so101` declare `state_units='degrees'`, which is correct only against degree-recorded stats; a checkpoint whose stats are in radians needs the unit half beside the stats half.

The registry marks `pretrained_name_or_path` as required for `run_policy`. `actions_per_step` left at `1` is auto-raised to the model's trained `n_action_steps`; pass a value above 1 to pin it. `cache_model=True` keeps loaded weights across policies in this process; `clear_model_cache()` and `list_cached_models()` in `strands_robots.policies.lerobot_local` manage that cache.

## Embodiments

An embodiment is a declared key map between what the robot emits and what the model was trained on: `state_keys`, `action_keys`, `obs_rename`, and a `dim_policy` (`strict`, `pad`, or `truncate`) for when the model's state width differs from the robot's. The maps live in `strands_robots/policies/lerobot_local/embodiments.json`. Sim entries use the bare MuJoCo joint names from the robot's XML; `*_real` entries use LeRobot driver motor names with the `.pos` suffix. Known embodiments and aliases:

{{providers:embodiments}}

## Rule 1: state keys

Without `set_robot_state_keys`, the policy infers the state vector from the observation's insertion order over its numeric scalars. The sim backends write `obs[joint]` then `obs[f"{joint}.vel"]`, so `strands_robots.policies._state_keys.drop_velocity_siblings` removes each `.vel` whose position companion is present and keeps one that has none (LeKiwi declares `x.vel`, `y.vel`, `theta.vel` as state). Every provider that infers an ordering shares this rule; an explicit `robot_state_keys` list is never filtered.

## Rule 2: camera names

A checkpoint declares image features such as `observation.images.image`. The embodiment's `obs_rename` maps the camera key you attach onto that feature. If you name a sim camera after the model card (`realsense_top`) instead of the embodiment's source key (`front`), the rename never fires and inference fails late. `preflight` runs before any weights download and refuses with the expected source keys.

Generated from `embodiments.json`:

{{providers:cameras}}

Two ways to satisfy the check:

```python title="sketch"
# 1. Name the cameras as the embodiment expects.
sim.add_camera(name="front", position=[0.22, 0.025, 0.6], target=[0.22, 0.025, 0])
sim.add_camera(name="wrist", parent_body="so101/gripper", position=[0.058, 0.0, -0.029], target=[-0.024, 0.0, -0.297])

# 2. Keep your names and route them (camera_key_map, then obs_rename_override, merge over obs_rename).
sim.run_policy(
    robot_name="so101",
    policy_provider="lerobot_local",
    policy_config={
        "pretrained_name_or_path": "allenai/MolmoAct2-SO100_101",
        "embodiment": "so101",
        "obs_rename_override": {"realsense_top": "observation.images.image", "realsense_side": "observation.images.wrist_image"},
    },
    instruction="pick up the cube",
)
```

`parent_body` mounts a camera on a link so a wrist view rides with the arm; `position` and `target` are then in that frame, both required. It works on `mujoco` and `newton`; `isaac` refuses it and names the world-frame alternative.

## Run it

Needs the extra above and an 865 MB download on first use. `smolvla_base` declares `camera1..3` and ships no SO-101 stats, so `embodiment="so101"` (which converts degrees) is refused after the download; an inline embodiment with native units runs:

```python
import os
os.environ["STRANDS_TRUST_REMOTE_CODE"] = "1"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco")
sim.create_world()
sim.add_robot("so101")
sim.add_camera(name="front", position=[0.22, 0.025, 0.6], target=[0.22, 0.025, 0])
sim.add_camera(name="wrist", parent_body="so101/gripper", position=[0.058, 0.0, -0.029], target=[-0.024, 0.0, -0.297])
joints = sim.robot_joint_names("so101")
embodiment = {"name": "so101_native", "state_keys": joints, "action_keys": joints, "dim_policy": "pad",
              "obs_rename": {"front": "observation.images.camera1", "wrist": "observation.images.camera2",
                             "default": "observation.images.camera3"}}
result = sim.run_policy(robot_name="so101", policy_provider="lerobot_local",
                        policy_config={"pretrained_name_or_path": "lerobot/smolvla_base", "embodiment": embodiment},
                        instruction="pick up the cube", n_steps=30, control_frequency=30.0)
print(result["status"])
sim.cleanup()
```

A checkpoint fine-tuned on an SO-101 carries its stats, and `embodiment="so101"` then converts units both ways in sim and binds the arm's `.pos` keys on hardware ([First policy](../../start/first-policy.md)). A real arm's tool takes the same dict as `policy_config`; host/port stay `policy_host`/`policy_port`:

```json
{"action": "execute", "policy_provider": "lerobot_local",
 "policy_config": {"pretrained_name_or_path": "robotfuel/act_so101_t16b", "embodiment": "so101",
                   "obs_rename_override": {"front": null, "wrist": "observation.images.wrist"}}}
```

## Limits

- Torch and the model share this process. One interpreter cannot hold `lerobot` (`transformers>=5`) and NVIDIA's Isaac-GR00T (`transformers==4.57.3`); for GR00T in process use `policy_type="groot"` here, or the [groot](groot.md) service.
- `trust_remote_code=True` is unconditional for this provider, hence the environment gate. Only load checkpoints from organisations you trust.
- `dim_policy="pad"` and `"truncate"` adapt the state vector to the model width by design, and most shipped embodiments declare `pad`; `strict` refuses a width mismatch and names the two opt-ins.
- An embodiment that is not in `embodiments.json` needs its own entry: state keys, action keys, and camera renames. The [training](../training/lerobot.md) page shows how a trained checkpoint carries those names.
