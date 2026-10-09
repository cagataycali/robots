---
description: cosmos3 drives NVIDIA Cosmos 3 VLA checkpoints through the Cosmos Framework policy server, or the diffusers checkpoint in process.
---

# cosmos3

By the end of this page you can run a Cosmos 3 checkpoint against a Franka-class arm through the RoboLab server and know the camera rule.

```bash
pip install 'strands-robots[cosmos3-service]'    # msgpack + websockets
pip install 'strands-robots[cosmos3-diffusers]'  # diffusers + torch + transformers
```

## What it is

`Cosmos3Policy` wraps the Cosmos 3 action surface (`nvidia/Cosmos3-Nano-Policy-DROID` and friends): image plus instruction in, a `[T, D]` action chunk out. The default `backend="service"` talks to `cosmos_framework.scripts.action_policy_server_robolab` over a self-contained msgpack and NumPy WebSocket client (no `openpi-client`). `backend="diffusers"` loads the checkpoint in process (`nvidia/Cosmos3-Nano`, or `nvidia/Cosmos3-Edge` with `diffusers>=0.41`) and is the only route for `forward_dynamics` and `inverse_dynamics`. In process the chunk is the raw unified action (`tx..r5, grasp`, quantile normalised), not joint targets; `ik=True` runs `decode_cosmos_chunk_to_targets` plus `MinkIKBridge` (`cosmos3-sim` extra) inside the policy so it emits the `joint_pos` row and `run_policy` can drive the arm, and `examples/vla/cosmos3_diffusers_mujoco_rollout.py` shows the same decode step by step. The sampler defaults (35 steps, guidance 6) are video defaults; pass `num_inference_steps=4, guidance_scale=3` for the server's settings. Measured on a Jetson AGX Thor with Edge: ~84 s per 32-step chunk at 35 steps, ~20 s at 4 (a 33-frame world video is decoded on every chunk), 11.1 GiB peak.

```python title="sketch"
from strands_robots.policies import create_policy

policy = create_policy("cosmos3", embodiment="droid", port=8000)
policy = create_policy("nvidia/Cosmos3-Nano-Policy-DROID", port=8000)   # same
chunk = policy.get_actions_sync(observation, "pick up the cube")
```

## Constructor keywords

{{providers:kwargs:cosmos3}}

No `**kwargs` absorber: an unknown keyword is a `TypeError`. `host` is a bare host (IPv6 bracketed), `port` an `int` in `[1, 65535]`. `action_space` (`joint_pos` or `midtrain`) must match the server's launch flag; a chunk of another width is refused.

## Embodiments

`embodiment` selects domain, layout and defaults: `droid`, `umi`, `av`, `bridge`, `openarm`. `droid` drives the `franka` or `panda` asset; its `joint_pos` layout is `[joint_0..joint_6, gripper]`. `robot="franka"` renames it onto the joint names (`joint1..joint7`, `finger_joint1`); `robot="franka-sim"` onto the simulator's actuator names (`actuator1..actuator8`), which is what the dataset recorder requires. `openarm` is post-training only (a checkpoint post-trained on OpenArm episodes, not a released model) and needs the service backend: `diffusers` 0.41 knows no `openarm_lerobot` domain. Of Edge's 32 domain slots only the ten NVIDIA documents are trained (`droid_lerobot` is one; `libero`, `pusht`, `so101` are not), so `droid` is the embodiment to run in process. Layouts live in `strands_robots/policies/cosmos3/embodiments.py`.

## Cameras

This client requires every declared view before it sends anything (the server also accepts a lone `observation/image`). `observation_mapping` (`{robot_key: "observation/<server_key>"}`) must cover the embodiment's camera set; every target carries the `observation/` prefix. For `droid` that is `observation/wrist_image_left`, `observation/exterior_image_1_left` and `observation/exterior_image_2_left`. A mapping that omits a key, or names an absent camera, is a `ValueError` naming the missing keys before any request leaves. State follows `policies/_state_keys.py`: the flat `observation.state` wins, else the per-joint scalars minus `.vel` siblings.

## Run it

Start the server from a Cosmos Framework checkout; `curl http://localhost:8000/healthz` answers 200 once it is up:

```bash
uv sync --all-extras --group=cu130-train --group=policy-server
python -m cosmos_framework.scripts.action_policy_server_robolab --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8000
```

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco")
sim.create_world()
sim.add_robot(name="arm", data_config="franka")
sim.add_object(name="cube", shape="box", size=[0.02, 0.02, 0.02], position=[0.5, 0.0, 0.02])
sim.add_camera(name="wrist", parent_body="arm/hand", position=[0.05, 0.0, 0.0], target=[0.0, 0.0, 0.3])
sim.add_camera(name="front", position=[1.2, 0.0, 0.6], target=[0.5, 0.0, 0.1])
sim.add_camera(name="side", position=[0.5, 1.0, 0.6], target=[0.5, 0.0, 0.1])
result = sim.run_policy(
    robot_name="arm",
    policy_provider="cosmos3",
    policy_config={
        "embodiment": "droid",
        "robot": "franka",
        "port": 8000,
        "observation_mapping": {
            "wrist": "observation/wrist_image_left",
            "front": "observation/exterior_image_1_left",
            "side": "observation/exterior_image_2_left",
        },
    },
    instruction="pick up the red cube",
    n_steps=24,
    control_frequency=15.0,
)
print(result["status"])
```

## In process, closed loop

No server: the checkpoint loads in this process and `ik=True` turns each chunk into joint targets. `franka-sim` keys the actions by actuator so the episode can be recorded.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco")
sim.create_world()
sim.add_robot(name="arm", data_config="franka")
sim.add_object(name="cube", shape="box", size=[0.02, 0.02, 0.02], position=[0.5, 0.0, 0.02])
sim.add_camera(name="wrist", parent_body="arm/hand", position=[0.05, 0.0, 0.0], target=[0.0, 0.0, 0.3])
sim.add_camera(name="front", position=[1.2, 0.0, 0.6], target=[0.5, 0.0, 0.1])
sim.add_camera(name="side", position=[0.5, 1.0, 0.6], target=[0.5, 0.0, 0.1])
sim.start_recording(repo_id="local/cosmos3_edge_franka", task="pick up the red cube", fps=15)
result = sim.run_policy(
    robot_name="arm",
    policy_provider="cosmos3",
    policy_config={
        "embodiment": "droid",
        "backend": "diffusers",
        "model": "nvidia/Cosmos3-Edge",
        "ik": True,
        "robot": "franka-sim",
        "num_inference_steps": 4,
        "guidance_scale": 3.0,
        "observation_mapping": {
            "wrist": "observation/wrist_image_left",
            "front": "observation/exterior_image_1_left",
            "side": "observation/exterior_image_2_left",
        },
    },
    instruction="pick up the red cube",
    n_steps=24,
    n_episodes=3,
    control_frequency=15.0,
)
sim.stop_recording()
```

`last_rollout["ik"]["tracking_error"]` reports how closely the IK followed the decoded end-effector path; `last_rollout["action"]` keeps the raw chunk. Headless Linux needs `MUJOCO_GL=egl` before the first MuJoCo import.

## Trainer

`create_trainer("cosmos3")` post-trains checkpoints: [training](../training/index.md).

## Limits

- Joint commands come from the service backend, or from `backend="diffusers"` with `ik=True`, in `policy` mode only; other modes produce video.
- `droid` needs three cameras on this client; in process only the first (`observation/wrist_image_left`) conditions the model, tagged `view_point` (default `ego_view`).
- `nvidia/Cosmos3-Edge` needs `diffusers>=0.41` (built against `0.40.0.dev0`; on 0.39 the load leaves 112 tensors unfilled and is refused) and has no sound head, so `enable_sound` stays off.
- The server holds the GPU and checkpoint; this process only serialises observations.
