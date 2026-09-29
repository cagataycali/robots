---
description: groot drives NVIDIA GR00T N1.5, N1.6 and N1.7 checkpoints over a ZMQ inference server, or in process where Isaac-GR00T is installed.
---

# groot

!!! warning "Deprecated"
    `model_path=` (in-process GR00T) is removed in 0.7. Service mode stays.

By the end of this page you can point a robot at a GR00T server, map its sensor names onto the model's keys, and know which server flavour each request shape needs.

```bash
pip install 'strands-robots[groot-service]'     # pyzmq + msgpack, the client side only
```

## What it is

**Service mode** (the default) dials `tcp://<host>:<port>` and speaks GR00T's ZMQ protocol; nothing NVIDIA-specific is installed on the client. **Local mode** (`model_path`) needs Isaac-GR00T, which no extra declares; `create_policy("lerobot_local", policy_type="groot", pretrained_name_or_path=...)` is the in-process route.

```python title="sketch"
from strands_robots.policies import create_policy

policy = create_policy("groot", host="10.0.0.5", port=5555, data_config="so101_dualcam", groot_version="n1.7")
policy = create_policy("zmq://10.0.0.5:5555", data_config="so101_dualcam")   # same thing, legacy wire
```

## Constructor keywords

{{providers:kwargs:groot}}

`port` is an `int` in `[1, 65535]`, read only in service mode. `groot_version` is `n1.5`, `n1.6` or `n1.7`; pass `"n1.7"` for an N1.7 server, which refuses the default legacy wire (`got (1, 240, 320, 3)`). `api_token` falls back to `GROOT_API_TOKEN`.

## Data configs

`data_config` names an entry in `strands_robots/policies/groot/data_configs.json`, the layout the checkpoint was trained on. Shipped names: `so100`, `so100_dualcam`, `so100_4cam`, `so101`, `so101_dualcam`, `so101_tricam`, `fourier_gr1_arms_only`, `fourier_gr1_arms_waist`, `fourier_gr1_full_upper_body`, `unitree_g1`, `unitree_g1_full_body`, `unitree_g1_locomanip`, `unitree_g1_real`, `unitree_g1_sonic`, `bimanual_panda_gripper`, `bimanual_panda_hand`, `single_panda_gripper`, `libero_panda`, `oxe_droid`, `oxe_droid_relative_eef_relative_joint`, `oxe_google`, `oxe_widowx`, `agibot_genie1`, `agibot_dual_arm_gripper`, `agibot_dual_arm_dexhand`, `agibot_dual_arm_full`, `galaxea_r1_pro`. Aliases: `agibot_dual_arm`, `real_g1_relative_eef_relative_joints`, `oxe_droid_rel`. `observation_indices` is the video horizon: `unitree_g1_real` declares two frames and the policy keeps the history, so one frame per step is enough.

## Mappings pick the wire shape

`observation_mapping` is `{robot_key: "video.X" | "state.X"}`; a state target may name one slot of a grouped vector, `"state.single_arm[3]"`, so per-joint scalars compose it. `action_mapping` is `{"action.X": robot_key}` with the same slot syntax; it turns a chunk into actuator names.

With a mapping the request is GR00T's nested `{"video", "state", "language"}` dict, which the plain `run_gr00t_server` accepts. Without one it is the flat `video.X` / `state.X` dict only a `--use-sim-policy-wrapper` server accepts, and the chunk comes back under the model's keys (`single_arm`, `gripper`), not the robot's. For a simulated robot, always map.

## Run it

Needs a GR00T server listening: the `gr00t_inference` tool starts NVIDIA's container, or run the entrypoint from an Isaac-GR00T checkout. The official `nvidia/SO_ARM_Starter_Gr00tN17` checkpoint names its cameras `room` and `wrist`.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco", mesh=False)
sim.create_world()
sim.add_robot("so101")
sim.add_camera(name="room", position=[0.22, 0.025, 0.6], target=[0.22, 0.025, 0])
sim.add_camera(name="wrist", parent_body="so101/gripper", position=[0.058, 0.0, -0.029], target=[-0.024, 0.0, -0.297])
policy_config = {
    "port": 5555,
    "data_config": "so101_dualcam",
    "groot_version": "n1.7",
    "observation_mapping": {"room": "video.room", "wrist": "video.wrist", **{f"{i}": f"state.single_arm[{i - 1}]" for i in range(1, 6)}, "6": "state.gripper[0]"},
    "action_mapping": {**{f"action.single_arm[{i - 1}]": f"{i}" for i in range(1, 6)}, "action.gripper[0]": "6"},
}
result = sim.run_policy(robot_name="so101", policy_provider="groot", policy_config=policy_config, instruction="pick up the cube", n_steps=300)
print(result["status"])
```

`reset(seed)` is forwarded to the server, which ignores it unless started through the tool's `deterministic=True` wrapper; without that, seeded evaluations differ run to run.

## Trainer

`create_trainer("groot")` fine-tunes; see [training](../training/index.md).

## Limits

- The chunk is in the checkpoint's own units; the policy does not convert. SO-ARM checkpoints emit lerobot degrees and a 0 to 100 gripper, a MuJoCo `so101` takes radians, and `run_policy` reports `status: success` with every joint at its limit. Convert in your loop, or serve it through `lerobot_local`, whose embodiments carry the unit frame.
- Local mode and `lerobot` cannot share an interpreter.
- The default `data_config` is `so100_dualcam`; a checkpoint trained on another layout returns garbage unless you name its config.
