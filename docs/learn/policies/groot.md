---
description: groot drives NVIDIA GR00T N1.5, N1.6 and N1.7 checkpoints over a ZMQ inference server, or in process where Isaac-GR00T is installed.
---

# groot

!!! warning "Deprecated"
    `model_path=` (in-process GR00T) is removed in 0.7. Service mode stays.

By the end of this page you can point a robot at a running GR00T inference server, map its sensor names onto the model's modality keys, and know when to use `lerobot_local` instead.

```bash
pip install 'strands-robots[groot-service]'     # pyzmq + msgpack, the client side only
```

## What it is

`Gr00tPolicy` has two modes. **Service mode** (the default) dials `tcp://<host>:<port>` and speaks GR00T's ZMQ protocol; nothing NVIDIA-specific is installed on the client. **Local mode** is selected by passing `model_path`; it loads the model in this process and needs NVIDIA's Isaac-GR00T package, which no extra declares because it pins `transformers==4.57.3` and cannot share an interpreter with lerobot. Where lerobot is installed, `create_policy("lerobot_local", policy_type="groot", pretrained_name_or_path=...)` is the in-process route; lerobot ships its own GR00T N1.7.

GR00T consumes nested dicts (`video`, `state`, `language`) and returns grouped action arrays. The policy translates between robot sensor names and model modality keys through explicit mappings. There is no positional guessing.

```python title="sketch"
from strands_robots.policies import create_policy

policy = create_policy("groot", host="10.0.0.5", port=5555, data_config="so101_dualcam")
policy = create_policy("zmq://10.0.0.5:5555", data_config="so101_dualcam")   # same thing
```

## Constructor keywords

{{providers:kwargs:groot}}

`port` must be an `int` in `[1, 65535]` and is read only in service mode. `groot_version` is one of `n1.5`, `n1.6`, `n1.7` and is validated in both modes: local mode dispatches a loader on it, service mode chooses the wire shape (N1.7 adds a time axis). `api_token` falls back to the `GROOT_API_TOKEN` environment variable.

## Data configs

`data_config` names an entry in `strands_robots/policies/groot/data_configs.json`: the modality layout the checkpoint was trained on. Shipped names: `so100`, `so100_dualcam`, `so100_4cam`, `so101`, `so101_dualcam`, `so101_tricam`, `fourier_gr1_arms_only`, `fourier_gr1_arms_waist`, `fourier_gr1_full_upper_body`, `unitree_g1`, `unitree_g1_full_body`, `unitree_g1_locomanip`, `unitree_g1_real`, `unitree_g1_sonic`, `bimanual_panda_gripper`, `bimanual_panda_hand`, `single_panda_gripper`, `libero_panda`, `oxe_droid`, `oxe_droid_relative_eef_relative_joint`, `oxe_google`, `oxe_widowx`, `agibot_genie1`, `agibot_dual_arm_gripper`, `agibot_dual_arm_dexhand`, `agibot_dual_arm_full`, `galaxea_r1_pro`. Aliases: `agibot_dual_arm`, `real_g1_relative_eef_relative_joints`, `oxe_droid_rel`.

## Mappings

`observation_mapping` is `{robot_key: "video.X" | "state.X"}`. A state target may name one slot of a grouped vector, `"state.single_arm[3]"`, so several per-joint scalars compose it. `action_mapping` is `{"action.X": robot_key}` and accepts the same slot syntax for columns. Both are honoured in either mode; with a local checkpoint loaded, a key the model does not declare is refused by name, while in service mode the server reports it.

```python title="sketch"
policy_config = {
    "port": 5555,
    "data_config": "so101_dualcam",
    "observation_mapping": {"front": "video.front", "wrist": "video.wrist", "1": "state.single_arm[0]", "6": "state.gripper[0]"},
    "action_mapping": {"action.single_arm[0]": "1", "action.gripper[0]": "6"},
}
```

## Run it

Needs a GR00T server listening. The `gr00t_inference` tool starts NVIDIA's container for you; its `deterministic=True` flag bind-mounts `strands_robots/policies/groot/server_wrapper.py` so the server reseeds on every `reset`. Without the tool, run NVIDIA's own server entrypoint from an Isaac-GR00T checkout on the same port.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco")
sim.create_world()
sim.add_robot("so101")
sim.add_camera(name="front", position=[0.22, 0.025, 0.6], target=[0.22, 0.025, 0])
sim.add_camera(name="wrist", parent_body="so101/gripper", position=[0.058, 0.0, -0.029], target=[-0.024, 0.0, -0.297])
result = sim.run_policy(robot_name="so101", policy_provider="groot", policy_config={"port": 5555, "data_config": "so101_dualcam"}, instruction="pick up the cube", n_steps=300)
print(result["status"])
```

`reset(seed)` is forwarded to the server so its diffusion sampler reseeds per episode; without that, seeded evaluations drift on the server side.

## Trainer

`create_trainer("groot")` fine-tunes against a dataset; see [training](../training/index.md).

## Limits

- Service mode cannot check `observation_mapping` against the server. A wrong key is a server error, not a client refusal.
- Local mode and `lerobot` cannot coexist in one interpreter.
- The default `data_config` is `so100_dualcam`; a checkpoint trained on another layout returns garbage silently unless you name its config.
