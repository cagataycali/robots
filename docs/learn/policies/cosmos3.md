---
description: cosmos3 drives NVIDIA Cosmos 3 VLA checkpoints through the Cosmos Framework policy server, or the diffusers checkpoint in process.
---

# cosmos3

By the end of this page you can run a Cosmos 3 policy checkpoint against a Franka-class arm through the Cosmos Framework RoboLab server, and you know the camera rule the server enforces.

```bash
pip install 'strands-robots[cosmos3-service]'    # msgpack + websockets, numpy-version agnostic
pip install 'strands-robots[cosmos3-diffusers]'  # diffusers + torch + transformers, for backend="diffusers"
```

## What it is

`Cosmos3Policy` wraps the Cosmos 3 Generator action surface (`nvidia/Cosmos3-Nano-Policy-DROID` and friends). In `policy` mode the model takes image plus instruction and returns a `[T, D]` action chunk, which is the contract this package already speaks. The default `backend="service"` talks to `cosmos_framework.scripts.action_policy_server_robolab` over a self-contained msgpack and NumPy WebSocket client; there is no `openpi-client` dependency. `backend="diffusers"` loads the checkpoint in process and is the route for the non-default physics modes (`forward_dynamics`, `inverse_dynamics`); a non-`policy` mode under the service backend is refused.

```python title="sketch"
from strands_robots.policies import create_policy

policy = create_policy("cosmos3", embodiment="droid", port=8000)
policy = create_policy("nvidia/Cosmos3-Nano-Policy-DROID", port=8000)   # same thing
chunk = policy.get_actions_sync(observation, "pick up the cube")
```

## Constructor keywords

{{providers:kwargs:cosmos3}}

No `**kwargs` absorber: an unknown keyword is a `TypeError`. `host` must be a bare host (IPv6 bracketed), `port` an `int` in `[1, 65535]`; both are read only when this constructor builds the client. `action_space` (`joint_pos` or `midtrain`) must match how the server was launched.

## Embodiments

`embodiment` selects domain, action layout and defaults: `droid`, `umi`, `av`, `bridge`, `openarm`. `droid` is the DROID Franka setup and drives the `franka` or `panda` sim asset; its `joint_pos` layout is `[joint_0..joint_6, gripper]`. `openarm` is post-training only: it maps a checkpoint post-trained on OpenArm episodes, not a released zero-shot model. Layouts live in `strands_robots/policies/cosmos3/embodiments.py`.

## Cameras

The server composes its conditioning from every declared view and rejects a partial observation. `observation_mapping` (`{robot_key: "observation/<server_key>"}`) must therefore cover the embodiment's full camera set. For `droid` that is `observation/wrist_image_left`, `observation/exterior_image_1_left` and `observation/exterior_image_2_left`. A mapping that omits a key, or whose source camera is absent at runtime, is a client-side `ValueError` naming the missing keys before any request leaves. State is read through the shared rule in `policies/_state_keys.py`: the flat `observation.state` wins, otherwise the per-joint scalars minus their `.vel` siblings.

## Run it

Start the server from a Cosmos Framework checkout (it holds the GPU), then wait for `curl http://localhost:8000/healthz` to answer 200:

```bash
uv sync --all-extras --group=cu130-train --group=policy-server
python -m cosmos_framework.scripts.action_policy_server_robolab --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8000
```

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco", mesh=False)
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

`create_trainer("cosmos3")` post-trains a checkpoint; see [training](../training/index.md).

## Limits

- The service backend returns joint commands only in `policy` mode. The other modes produce video and need `backend="diffusers"`.
- Three cameras are mandatory for `droid`. There is no single-camera fallback.
- The server holds the GPU and the checkpoint. This process only serialises observations, so latency is the network plus inference.
