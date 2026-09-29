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

`Cosmos3Policy` wraps the Cosmos 3 action surface (`nvidia/Cosmos3-Nano-Policy-DROID` and friends): image plus instruction in, a `[T, D]` action chunk out. The default `backend="service"` talks to `cosmos_framework.scripts.action_policy_server_robolab` over a self-contained msgpack and NumPy WebSocket client (no `openpi-client`). `backend="diffusers"` loads the checkpoint in process and is the route for `forward_dynamics` and `inverse_dynamics`; a non-`policy` mode under the service backend is refused. In process the chunk is the raw unified action (`tx..r5, grasp`, quantile normalised), not joint targets, so `run_policy` cannot consume it: the sim route is `decode_cosmos_chunk_to_targets` plus `MinkIKBridge` (`cosmos3-sim` extra), as `examples/vla/cosmos3_diffusers_mujoco_rollout.py` shows. Its defaults (35 steps, guidance 6) are video defaults; the server runs 4 steps at guidance 3.

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

`embodiment` selects domain, layout and defaults: `droid`, `umi`, `av`, `bridge`, `openarm`. `droid` drives the `franka` or `panda` asset; its `joint_pos` layout is `[joint_0..joint_6, gripper]`. `openarm` is post-training only (a checkpoint post-trained on OpenArm episodes, not a released model); `diffusers` 0.40 knows no `openarm_lerobot` domain, so it needs the service backend. Layouts live in `strands_robots/policies/cosmos3/embodiments.py`.

## Cameras

This client requires every declared view before it sends anything (the server also accepts a lone `observation/image`). `observation_mapping` (`{robot_key: "observation/<server_key>"}`) must cover the embodiment's camera set; every target carries the `observation/` prefix. For `droid` that is `observation/wrist_image_left`, `observation/exterior_image_1_left` and `observation/exterior_image_2_left`. A mapping that omits a key, or whose camera is absent, is a `ValueError` naming the missing keys before any request leaves. State follows `policies/_state_keys.py`: the flat `observation.state` wins, else the per-joint scalars minus `.vel` siblings.

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

## Trainer

`create_trainer("cosmos3")` post-trains checkpoints: [training](../training/index.md).

## Limits

- Joint commands come from the service backend in `policy` mode only; other modes produce video and need `backend="diffusers"`.
- `droid` needs three cameras on this client; in process only the first conditions the model.
- The server holds the GPU and checkpoint; this process only serialises observations.
