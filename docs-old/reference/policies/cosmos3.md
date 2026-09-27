---
description: NVIDIA Cosmos 3 omnimodal VLA - WebSocket service, droid/umi/av/bridge/openarm embodiments, MuJoCo rollout.
---

# Cosmos 3

```bash
uv pip install "strands-robots[cosmos3-service]"   # adds msgpack + websockets; no openpi-client needed
```

```python
from strands_robots.policies import create_policy

policy = create_policy("cosmos3", embodiment="droid", port=8000)
# or: create_policy("cosmos3://localhost:8000")
```

## Start the server

```bash
python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8000
# embodiment is selected client-side via create_policy(..., embodiment="droid")
```

## Parameters

```python
Cosmos3Policy(
    embodiment="droid",          # droid | umi | av | bridge | openarm
    host="localhost",         # bare hostname or IP literal; IPv6 bracketed "[::1]"
    port=8000,                   # int in [1, 65535]
    action_space=None,
    observation_mapping=None,
    action_mapping=None,
    robot=None,                  # "franka" or "panda" for built-in DROID→sim mapping
    prompt="",
    api_key=None,
    client=None,
    transport="raw",
    backend="service",          # "service" (default) | "diffusers" (in-process)
    mode="policy",              # "policy" | "forward_dynamics" | "inverse_dynamics" (diffusers only)
    model=None,                 # HF repo id / path for the diffusers backend
)
```

`host` and `port` form the one address this client dials (`ws://<host>:<port>`)
and are refused before the endpoint is built: `host` must be a bare hostname or
IP literal (IPv6 bracketed as `"[::1]"`), because `host="localhost/foo"` would
parse as port **80** with the configured `8000` in the path. An injected
`client=` owns its own address, and its `read_timeout`
(`Cosmos3WebsocketClient(host, port, read_timeout=1800)`, default 600 s, positive
and finite) bounds every read off the live connection - `websockets`' `recv()`
has no deadline, so a server that accepted the connection and then went quiet
would otherwise hold the caller forever. An expired read is reported as a
timeout, not as "start the server first", and discards the connection.

A frame that *arrives* and cannot be read as msgpack+NumPy is a third case,
reported as neither: it names the endpoint, which read it answered (metadata
handshake or action chunk), what the codec could not do, and the frame's
opening bytes. This package serves policies over a WebSocket in two wire
formats, so the common cause is a port mixed up between them - dialling
`strands_robots.inference.server`, which speaks JSON text frames, with this
client. Telling that apart from an absent server matters because only one of
the two is fixed by starting a server.

## Embodiments

Embodiments: `droid` (10D, chunk 32, 15 fps), `umi`, `av`, `bridge`, `openarm`
(post-training only). The embodiment is chosen client-side; the server hosts one
Cosmos 3 checkpoint for all of them.

| Embodiment | Robot hardware | Strands sim asset |
|------------|----------------|-------------------|
| `droid` | Franka / DROID dataset | `"panda"` or `"franka"` |
| `umi` | UMI gripper | - |
| `av` | Autonomous vehicle cameras | - |
| `bridge` | Bridge dataset robots | - |
| `openarm` | Enactic OpenArm (7-DOF + gripper) | `"openarm"` |

### Action spaces

An embodiment serves its action under one or more `action_space` names, each with
its own columns:

| `action_space` | Columns | Notes |
|----------------|---------|-------|
| `midtrain` | The model's unified action: `tx,ty,tz` + the 6D rotation `r0..r5` + `grasp` (omitted for `av`, which has no gripper) | Served through un-converted, so the columns are the same as `raw_action_layout` |
| `joint_pos` | `joint_0..joint_6` + `gripper` (DROID only) | The one space the RoboLab server post-processes, converting the effector pose into joint targets |

`action_mapping` renames a column to one of your robot's actuator names, so its
keys must be columns of the active space (`{"grasp": ...}` under `midtrain`,
`{"gripper": ...}` under `joint_pos`); a key naming no column is refused listing
the valid ones. It has to be a rename: two columns arriving at one actuator name
would collapse into one step-dict entry and drop a command, so both spellings of
that collision are refused at construction (renaming *every* column is a
bijection and is accepted).

`joint_pos` reads seven joint values plus a gripper in the order you declare with
`set_robot_state_keys()`, as every example here does. Without it the order is
inferred from the observation's scalar keys, **position-only**: a `<joint>.vel`
entry is dropped when its `<joint>` companion is present (every sim backend
emits one), a `.vel` key with no companion is kept (LeKiwi's `x.vel` /
`theta.vel`), and explicit `robot_state_keys` are never filtered - the same rule
the LeRobot provider applies.

## Backends

| backend | how it runs | install | extra outputs |
|---------|-------------|---------|---------------|
| `service` (default) | WebSocket to the Cosmos Framework RoboLab policy server (holds the GPU out-of-process) | `strands-robots[cosmos3-service]` (msgpack + websockets, numpy-agnostic) | none (server video discarded) |
| `diffusers` | in-process via native `diffusers` (`Cosmos3OmniPipeline`) | `strands-robots[cosmos3-diffusers]` (floors diffusers 0.39, the first release shipping the pipeline) | world video + sound on `last_rollout` |

The `diffusers` row's install, its extra outputs and the geometry that turns
its raw action into joint targets are on
[Cosmos 3 in process](cosmos3-diffusers.md).

## Rollout

```python
from strands_robots import Robot

sim = Robot("panda")
sim.run_policy(
    robot_name="panda",
    instruction="pick up the red block",
    policy_provider="cosmos3",
    policy_config={"embodiment": "droid", "robot": "panda", "port": 8000},
    duration=15.0,
    control_frequency=50.0,
)
# see examples/vla/cosmos3_sim_rollout.py
```

`robot="panda"` activates the built-in DROID-layout mapping (`joint_0..6/gripper` → `joint1..7/finger_joint1`). `requires_images=True`.

## See also

- [Cosmos 3 in process](cosmos3-diffusers.md) - the `diffusers` backend, its action modes and the sim IK bridge.
- [Policy overview](overview.md)
- [GR00T](groot.md)
- [LeRobot Local](lerobot-local.md)
- [Custom policies](custom-policies.md)
