---
description: lerobot_local runs any LeRobot checkpoint in process. Install extra, constructor keywords, embodiments, camera naming rules, and limits.
---

# lerobot_local

After this page you can run a LeRobot checkpoint (ACT, diffusion, pi0, SmolVLA, GR00T N1.7, MolmoAct2) on a simulated or real arm, and you know the two naming rules that decide whether the model sees your cameras and joints.

```bash
pip install 'strands-robots[lerobot]'          # lerobot[feetech,dataset] + psutil
pip install 'strands-robots[smolvla]'          # adds lerobot[smolvla]
pip install 'strands-robots[molmoact2]'        # adds lerobot[molmoact2]
pip install 'strands-robots[groot]'            # adds lerobot[groot] (GR00T N1.7)
export STRANDS_TRUST_REMOTE_CODE=1             # required: models load with trust_remote_code=True
```

## What it is

`LerobotLocalPolicy` hands the checkpoint to LeRobot's own factory: the policy type is read from the model's `config.json`, so any class LeRobot registers works here. The model's processor pipeline (`preprocessor.json`, `postprocessor.json`) normalises observations and unnormalises actions. Flow-matching models get Real-Time Chunking when the config declares it: the runtime tells the policy its control rate and the steps inference consumed; the policy blends the next chunk onto the seam.

Build it by name or smart string; a HuggingFace id resolves here.

```python title="sketch"
from strands_robots.policies import create_policy

policy = create_policy("lerobot_local", pretrained_name_or_path="robotfuel/act_so101_t16b", embodiment="so101")
policy = create_policy("robotfuel/act_so101_t16b", embodiment="so101")   # same thing
```

## Constructor keywords

{{providers:kwargs:lerobot_local}}

`processor_overrides={"normalizer_processor": {"stats": ...}}` replaces inert stats; `state_units` / `action_units` (`degrees` or `radians`) name their unit. `so100` and `so101` declare `degrees`, so radian stats need `radians` beside them.

`pretrained_name_or_path` is required. `actions_per_step=1` becomes the trained `n_action_steps`; above 1 pins it. `cache_model=True` shares weights in-process (`clear_model_cache()`, `list_cached_models()`). `device=` defaults to CUDA when present; `torch.compile` stays off unless `compile_model=True`.

## Embodiments

An embodiment is a declared key map between what the robot emits and what the model was trained on: `state_keys`, `action_keys`, `obs_rename`, and a `dim_policy` (`strict`, `pad`, or `truncate`) for a state width that differs from the robot's. They live in `strands_robots/policies/lerobot_local/embodiments.json`; sim entries use bare MuJoCo joint names, `*_real` entries LeRobot motor names with `.pos`. Known embodiments and aliases:

{{providers:embodiments}}

## Rule 1: state keys

Without `set_robot_state_keys`, the policy infers the state vector from the observation's insertion order over its numeric scalars. The sim backends write `obs[joint]` then `obs[f"{joint}.vel"]`, so `strands_robots.policies._state_keys.drop_velocity_siblings` removes each `.vel` with a position companion and keeps one without (LeKiwi declares `x.vel`, `y.vel`, `theta.vel` as state). Every provider that infers an ordering shares this rule; an explicit `robot_state_keys` list is never filtered.

## Rule 2: camera names

A checkpoint declares image features such as `observation.images.image`. The embodiment's `obs_rename` maps the camera key you attach onto that feature. Name a sim camera after the model card (`realsense_top`) instead of the embodiment's source key (`front`) and the rename never fires; `preflight` refuses before any weights download, naming the expected source keys.

From `embodiments.json`:

{{providers:cameras}}

Two ways through it:

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

`parent_body` mounts a camera on a link, so a wrist view rides with the arm; `position` and `target` are then in that frame. `mujoco` and `newton` support it; `isaac` refuses it and names the world-frame alternative.

## Run it

Needs the extra and an 865 MB download. `smolvla_base` declares `camera1..3` and ships no SO-101 stats, so `embodiment="so101"` (degrees) is refused; an inline embodiment with native units runs:

```python
import os
os.environ["STRANDS_TRUST_REMOTE_CODE"] = "1"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco", mesh=False)
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

`robotfuel/act_so101_t16b`, a checkpoint trained on a real SO-101, carries its stats: `embodiment="so101"` converts units both ways in sim and binds the arm's `.pos` keys on hardware. Through `run_policy(policy_object=...)` it applied 90 of 90 steps at 30 Hz from one wrist camera on a laptop CPU; the arm moved, no cube lifted. A real arm's tool takes the dict as `policy_config`:

```json
{"action": "execute", "policy_provider": "lerobot_local",
 "policy_config": {"pretrained_name_or_path": "robotfuel/act_so101_t16b", "embodiment": "so101",
                   "obs_rename_override": {"front": null, "wrist": "observation.images.wrist"}}}
```

## GR00T N1.7 through lerobot

`nvidia/GR00T-N1.7-3B` and its fine-tunes are lerobot's native `groot` type and load like any checkpoint, with no Isaac-GR00T checkout or ZMQ service. `embodiment_tag` comes from the checkpoint config.

```python title="sketch: groot extra, 3B download"
from strands_robots.policies import create_policy

policy = create_policy("nvidia/GR00T-N1.7-3B", policy_type="groot", embodiment="so101")
print(policy.provider_name)
```

The 3B model wants a GPU: run `PolicyServer` where one is and dial it with [`remote`](remote.md).

```python title="sketch"
cfg = {"pretrained_name_or_path": "nvidia/GR00T-N1.7-3B", "policy_type": "groot", "embodiment": "so101"}
PolicyServer(policy_provider="lerobot_local", policy_config=cfg, port=8765).start()   # GPU host
policy = create_policy("ws://gpu-box:8765")                                          # robot host
```

## Limits

- `trust_remote_code=True` is unconditional here, hence the environment gate; load only checkpoints you trust.
- `dim_policy="pad"` / `"truncate"` adapt the state width and take the first N values of a wider action (32-D pi0/pi0.5); `strict` refuses. An embodiment the pipeline cannot take is refused at load.
- An embodiment not in `embodiments.json` needs its own entry (state keys, action keys, camera renames); [training](../training/lerobot.md) shows how a checkpoint carries those names.
