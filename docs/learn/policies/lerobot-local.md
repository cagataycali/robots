---
description: lerobot_local runs any LeRobot checkpoint in process. Install extra, constructor keywords, embodiments, camera naming rules, and limits.
---

# lerobot_local

By the end of this page you can run a HuggingFace LeRobot checkpoint (ACT, diffusion, pi0, SmolVLA, GR00T N1.7, MolmoAct2) on a simulated or real arm in this process, and you know the two naming rules that decide whether the model sees your cameras and joints at all.

```bash
pip install 'strands-robots[lerobot]'          # lerobot[feetech,dataset] + psutil
pip install 'strands-robots[smolvla]'          # adds lerobot[smolvla]
pip install 'strands-robots[molmoact2]'        # adds lerobot[molmoact2]
export STRANDS_TRUST_REMOTE_CODE=1             # required: models load with trust_remote_code=True
```

## What it is

`LerobotLocalPolicy` hands the checkpoint to LeRobot's own factory, so the policy type is read from the model's `config.json` and any class LeRobot registers works without a change here. The model's processor pipeline (`preprocessor.json` / `postprocessor.json`) normalises observations and unnormalises actions. Flow-matching models get Real-Time Chunking when the config declares it: the runtime tells the policy its control rate and the exact number of steps consumed during inference, and the policy blends the next chunk onto the seam.

Build it by name or by smart string; a HuggingFace id that is not in the `nvidia` org resolves here.

```python title="sketch"
from strands_robots.policies import create_policy

policy = create_policy("lerobot_local", pretrained_name_or_path="lerobot/act_so101", embodiment="so101")
policy = create_policy("lerobot/act_so101", embodiment="so101")   # same thing
```

## Constructor keywords

{{providers:kwargs:lerobot_local}}

The remedy for inert normalization has two halves: `processor_overrides={"normalizer_processor": {"stats": ...}}` replaces the stats, and `state_units` / `action_units` (`degrees` or `radians`) say which unit they were recorded in. `so100` and `so101` declare `state_units='degrees'`, which is correct only against degree-recorded stats; a checkpoint whose stats are in radians needs the unit half beside the stats half.

The registry marks `pretrained_name_or_path` as required for `run_policy`. `actions_per_step` left at `1` is auto-raised to the model's trained `n_action_steps`; pass a value above 1 to pin it. `cache_model=True` keeps loaded weights across policies in this process; `clear_model_cache()` and `list_cached_models()` in `strands_robots.policies.lerobot_local` manage that cache.

## Embodiments

An embodiment is a declared key map between what the robot emits and what the model was trained on: `state_keys`, `action_keys`, `obs_rename`, and a `dim_policy` (`strict`, `pad`, or `truncate`) for when the model's state width differs from the robot's. The maps live in `strands_robots/policies/lerobot_local/embodiments.json`. Sim entries use the bare MuJoCo joint names from the robot's XML; `*_real` entries use LeRobot driver motor names with the `.pos` suffix. Known embodiments and aliases:

{{providers:embodiments}}

## Rule 1: state keys

Without `set_robot_state_keys`, the policy infers the state vector from the observation's insertion order over its numeric scalars. The sim backends write `obs[joint]` and then `obs[f"{joint}.vel"]` for every joint, so that raw order alternates position and velocity. `strands_robots.policies._state_keys.drop_velocity_siblings` removes each `.vel` whose position companion is present, and keeps a `.vel` that has none (LeKiwi declares `x.vel`, `y.vel`, `theta.vel` as state). Every provider that infers an ordering shares this rule; an explicit `robot_state_keys` list is never filtered, because an operator naming `elbow.vel` is stating the model's input.

## Rule 2: camera names

A checkpoint declares image features such as `observation.images.image`. The embodiment's `obs_rename` maps the camera key you attach onto that feature. If you name a sim camera after the model card (`realsense_top`) instead of the embodiment's source key (`front`), the rename never fires and inference fails late. `preflight` runs before any weights download and refuses with the expected source keys.

Generated from `embodiments.json`:

{{providers:cameras}}

Two ways to satisfy the check:

```python title="sketch"
# 1. Name the cameras as the embodiment expects.
sim.add_camera(name="front", position=[0.22, 0.025, 0.6], target=[0.22, 0.025, 0])
sim.add_camera(name="wrist", parent_body="so101/gripper", position=[0.058, 0.0, -0.029], target=[-0.024, 0.0, -0.297])

# 2. Keep your names and route them. camera_key_map is rung 1, obs_rename_override is rung 2,
#    both merge over the embodiment's obs_rename.
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

`parent_body` mounts a camera on a link so a wrist view rides with the arm; `position` and `target` are then in that body's frame and both are required. It works on `mujoco` and `newton`; `isaac` refuses it and names the world-frame alternative.

## Run it

Needs the extra above, `STRANDS_TRUST_REMOTE_CODE=1`, and a download on first use.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco", mesh=False)
sim.create_world()
sim.add_robot("so101")
sim.add_camera(name="front", position=[0.22, 0.025, 0.6], target=[0.22, 0.025, 0])
sim.add_camera(name="wrist", parent_body="so101/gripper", position=[0.058, 0.0, -0.029], target=[-0.024, 0.0, -0.297])
result = sim.run_policy(
    robot_name="so101",
    policy_provider="lerobot_local",
    policy_config={"pretrained_name_or_path": "lerobot/smolvla_base", "embodiment": "so101"},
    instruction="pick up the cube",
    n_steps=300,
    control_frequency=30.0,
)
print(result["status"])
```

On a real arm the same bag goes through the robot's agent tool. `Robot("so101",
mode="real", port=...)` exposes `execute` and `start` with a `policy_config`
object, so an agent names the checkpoint the way it does in sim; the operator's
approval prompt names it too (`policy lerobot_local built in this process, no
server, checkpoint pretrained_name_or_path lerobot/smolvla_base`). Host and port
are not allowed inside the bag - they are `policy_host` and `policy_port`, so the
prompt describes the server the arm will actually dial.

```json title="the tool input an agent sends"
{"action": "execute", "instruction": "pick up the cube", "policy_provider": "lerobot_local",
 "policy_config": {"pretrained_name_or_path": "lerobot/smolvla_base", "embodiment": "so101_real"}}
```

## Limits

- Torch and the model share this process. One interpreter cannot hold `lerobot` (`transformers>=5`) and NVIDIA's Isaac-GR00T (`transformers==4.57.3`); for GR00T in process use `policy_type="groot"` here, or the [groot](groot.md) service.
- `trust_remote_code=True` is unconditional for this provider, hence the environment gate. Only load checkpoints from organisations you trust.
- `dim_policy="pad"` and `"truncate"` adapt the state vector to the model width by design, and most shipped embodiments declare `pad`; `strict` refuses a width mismatch and names the two opt-ins.
- An embodiment that is not in `embodiments.json` needs its own entry: state keys, action keys, and camera renames. The [training](../training/lerobot.md) page shows how a trained checkpoint carries those names.
