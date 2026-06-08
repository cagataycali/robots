---
description: NVIDIA Cosmos 3 omnimodal VLA — WebSocket service, droid/umi/av/bridge embodiments, MuJoCo rollout.
---

# Cosmos 3

`Cosmos3Policy` connects to NVIDIA Cosmos 3 — an omnimodal Vision-Language-Action
model — over a self-contained msgpack+numpy WebSocket. No external client library is
needed; the transport is fully contained in `strands-robots`.

## TL;DR

```python
from strands_robots.policies import create_policy

# Service mode — Cosmos server running separately
policy = create_policy(
    "cosmos3",
    embodiment="droid",   # droid | umi | av | bridge
    port=8000,
)

# Or use the smart-string shortcut
policy = create_policy("cosmos3://localhost:8000")
```

## Install

```bash
pip install "strands-robots[cosmos3-service]"
```

That adds `msgpack` + `websockets`. No `openpi-client` or other external VLA client
library is required.

## Starting the server

The server is part of the `cosmos_framework` package:

```bash
python -m cosmos_framework.scripts.action_policy_server_robolab \
    --embodiment droid \
    --port 8000
```

Refer to the Cosmos 3 project documentation for GPU requirements and checkpoint
download instructions.

## Constructor parameters

```python
Cosmos3Policy(
    embodiment: str = "droid",               # target embodiment
    host: str = "localhost",                 # server host
    port: int = 8000,                        # server port
    action_space: dict | None = None,        # override action-space spec
    observation_mapping: dict | None = None, # remap incoming obs keys
    action_mapping: dict | None = None,      # remap outgoing action keys
    robot: str | None = None,               # built-in robot mapping ("franka", "panda")
    prompt: str = "",                        # optional static task prompt
    api_key: str | None = None,              # authentication key
    client=None,                             # advanced: inject a custom client
    transport: str = "raw",                  # transport protocol
)
```

## Embodiments

| Embodiment | Robot hardware | Strands sim asset |
|------------|----------------|-------------------|
| `droid` | Franka / DROID dataset | `"panda"` or `"franka"` |
| `umi` | UMI gripper | — |
| `av` | Autonomous vehicle cameras | — |
| `bridge` | Bridge dataset robots | — |

To list embodiments at runtime:

```python
from strands_robots.policies.cosmos3.policy import Cosmos3Policy
print(Cosmos3Policy.list_embodiments())
# ['droid', 'umi', 'av', 'bridge']
```

## Built-in robot mappings

Cosmos 3 uses DROID-layout joint names (`joint_0` … `joint_6`, `gripper`). When
connecting to a Panda simulation you need to remap these to the sim's naming scheme.

Pass `robot="panda"` (or `"franka"`) to activate the built-in mapping:

```python
policy = create_policy(
    "cosmos3",
    embodiment="droid",
    robot="panda",       # maps joint_0..6/gripper -> joint1..7/finger_joint1
    port=8000,
)
```

You can also supply a fully custom mapping via `action_mapping` / `observation_mapping`.

## MuJoCo rollout example

By provider name with `policy_config={}`:

```python
from strands_robots import Robot

sim = Robot("panda")   # Franka Panda simulation
sim.run_policy(
    robot_name="panda",
    instruction="pick up the red block",
    policy_provider="cosmos3",
    policy_config={
        "embodiment": "droid",
        "robot": "panda",
        "port": 8000,
    },
    duration=15.0,
    control_frequency=50.0,
)
```

Or with a pre-built instance passed via `policy_object=`:

```python
from strands_robots import Robot
from strands_robots.policies import create_policy

policy = create_policy(
    "cosmos3",
    embodiment="droid",
    robot="panda",
    port=8000,
)

sim = Robot("panda")
sim.run_policy(
    robot_name="panda",
    instruction="pick up the red block",
    policy_object=policy,
    duration=15.0,
)
```

See `examples/cosmos3_sim_rollout.py` for a runnable end-to-end example.

## Notes

- `requires_images=True`. Cosmos 3 uses chunked diffusion, not 500 Hz state-only
  control — camera frames are required at every inference call.
- The WebSocket transport is self-contained (msgpack+numpy framing). There is no
  dependency on `openpi-client` or any other external VLA client.
- The Cosmos server (`action_policy_server_robolab`) handles GPU batching. A single
  server process can serve multiple concurrent rollouts.

## See also

- [Policy overview](overview.md) — the ABC contract and factory.
- [GR00T](groot.md) — alternative NVIDIA VLA provider.
- [LeRobot Local](lerobot-local.md) — local in-process inference.
- [Custom policies](custom-policies.md) — write your own VLA wrapper.
- [Tutorial 3 — Policies](../tutorial/03-policies.md) — guided walkthrough.
