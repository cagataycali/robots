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

## Embodiments

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

```bash
# in-process backend (heavy GPU stack: diffusers + torch)
uv pip install "strands-robots[cosmos3-diffusers]"
```

`Cosmos3OmniPipeline` and `CosmosActionCondition` first ship in diffusers
0.39.0, which the extra floors; `nvidia/Cosmos3-Edge` is built against
0.40.0.dev0, which at the time of writing ships only from source:

```bash
uv pip install 'diffusers @ git+https://github.com/huggingface/diffusers'
```

Loading a checkpoint the installed diffusers cannot build is refused naming the
tensors it could not fill - `from_pretrained` itself only warns and would run on
random weights. The extra is native `diffusers` + `torch` + `transformers`,
`numpy>=2`-compatible and co-installable with `cosmos3-service`.

> **Action layout note.** The `diffusers` backend returns the model's **raw
> unified action** (DROID = 9D end-effector pose `tx,ty,tz,r0..r5` + 1D `grasp`
> = 10D), named by the embodiment `raw_action_layout` - the pipeline's native
> output, *before* the RoboLab server's `joint_pos` (8D) conversion. Use
> `backend="service"` when you need joint-position commands.

> **Safety checker / `cosmos_guardrail`.** `Cosmos3OmniPipeline` builds a
> `CosmosSafetyChecker` at load time, which requires the heavy optional
> `cosmos_guardrail` package and otherwise raises `ImportError: cosmos_guardrail
> is not installed`. The diffusers backend disables it by default
> (`enable_safety_checker=False`, passed through to `from_pretrained`) so the
> pipeline loads without that extra. To re-enable it, install `cosmos_guardrail`
> and build the backend with `enable_safety_checker=True`, then hand that backend
> to the policy - the flag is a `Cosmos3DiffusersBackend` parameter, not a
> `Cosmos3Policy` one:
>
> ```python
> from strands_robots.policies.cosmos3.embodiments import get_embodiment
> from strands_robots.policies.cosmos3.policy import Cosmos3Policy
> from strands_robots.policies.cosmos3.policy_diffusers import Cosmos3DiffusersBackend
>
> backend = Cosmos3DiffusersBackend(
>     embodiment=get_embodiment("droid"),
>     model="nvidia/Cosmos3-Nano",
>     enable_safety_checker=True,   # needs cosmos_guardrail installed
> )
> policy = Cosmos3Policy(embodiment="droid", backend="diffusers", diffusers_backend=backend)
> ```
>
> `Cosmos3Policy` forwards only `embodiment`, `model` and `mode` to the backend, so
> the same route is how you reach its other load and sampling knobs
> (`resolution_tier`, `view_point`, `device`, `dtype`, `num_inference_steps`,
> `guidance_scale`, `enable_sound`). Note Cosmos runs in `bfloat16`, so the backend
> up-casts the half-precision action tensor to `float32` before returning the chunk.

### `backend="diffusers"` — world video alongside the action chunk

One in-process forward pass returns the predicted world video, optional sound,
**and** the action chunk. The chunk comes back through the normal `get_actions`
-> `list[dict]` contract; video and sound are surfaced on `policy.last_rollout`:

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "cosmos3",
    embodiment="droid",
    backend="diffusers",
    model="nvidia/Cosmos3-Nano",  # HF repo id or local path
)
policy.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])

steps = policy.get_actions_sync(observation, "pick up the red cube")
# steps == [{"tx": .., "ty": .., ..., "r5": .., "grasp": ..}, ...]  (raw unified
# action, one dict per timestep)

# the predicted world video Cosmos rolled out for that action chunk:
print(policy.last_rollout["video"])   # path to an .mp4 / .gif
print(policy.last_rollout["sound"])   # path to a .wav, or None
```

### Action modes (diffusers only)

The diffusers backend exposes Cosmos 3's full physics loop via the `mode` kwarg
(`CosmosActionCondition.mode`). These do **not** exist in service mode - a
non-`policy` mode under `backend="service"` raises.

| `mode` | conditioning | predicts | `get_actions` returns |
|--------|--------------|----------|------------------------|
| `policy` (default) | first frame + task prompt | future video **+ actions** | action chunk (`list[dict]`) |
| `forward_dynamics` | first frame + given `raw_actions` | future video | `[]` (world video on `last_rollout`) |
| `inverse_dynamics` | an observed video | the actions between frames | action chunk (`list[dict]`) |

All three modes are verified live on real `nvidia/Cosmos3-Nano` weights (Thor,
bf16/CUDA); metrics in `docs/assets/cosmos3/live_modes_metrics.json`.

```python
# forward dynamics: "what world results if I run these actions?"
fd = create_policy("cosmos3", embodiment="droid", backend="diffusers", mode="forward_dynamics")
fd.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
fd.get_actions_sync(observation, "", raw_actions=my_action_chunk)
print(fd.last_rollout["video"])   # predicted world rollout

# inverse dynamics: "what actions produced this observed video?"
inv = create_policy("cosmos3", embodiment="droid", backend="diffusers", mode="inverse_dynamics")
inv.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
steps = inv.get_actions_sync(observation, "", video="observed.mp4")
```

### Closing the sim loop: de-normalize → IK → MuJoCo

The `diffusers` backend's raw unified action is **quantile-normalized to
`[-1, 1]`** and encodes a *relative end-effector pose delta* per step, **not
joint radians** - fed straight to MuJoCo joint actuators it is meaningless.
Three geometric steps (`cosmos3-sim` extra: `mink` + `mujoco`, numpy>=2,
co-installable with the other extras) turn it into joint targets:

1. **De-normalize** - invert the quantile transform with the embodiment's
   bundled `q01`/`q99` action stats:
   `denorm = 0.5 * (a + 1) * (q99 - q01) + q01` (`denormalize_quantile`).
2. **Decode poses** - integrate the per-step `[translation(3), rot6d(6)]` deltas
   into an absolute `(T+1, 4, 4)` SE3 trajectory anchored at the robot's current
   EE pose (`decode_pose_trajectory`, via `MinkIKBridge.ee_pose(qpos)`, the
   forward-kinematics call).
3. **Inverse kinematics** - solve each Cartesian target with
   [`mink`](https://github.com/kevinzakka/mink) differential IK on the *same*
   `mujoco.MjModel`, warm-starting each step (`MinkIKBridge`).

```python
import mujoco, numpy as np
from robot_descriptions import panda_mj_description
from strands_robots.policies.cosmos3 import (
    Cosmos3Policy, MinkIKBridge, decode_cosmos_chunk_to_targets,
)
from strands_robots.policies.cosmos3.embodiments import get_embodiment

policy = Cosmos3Policy(embodiment="droid", backend="diffusers", model="nvidia/Cosmos3-Nano")
policy.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
chunk_dicts = policy.get_actions_sync(observation, "pick up the red cube")
raw_chunk = policy.last_rollout["action"]          # [T, 10] raw [-1,1] action

model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
bridge = MinkIKBridge(model, ee_frame_name="hand", ee_frame_type="body")
q_init = np.zeros(model.nq); q_init[:7] = [0, -0.3, 0, -2.2, 0, 2.0, 0.79]

out = decode_cosmos_chunk_to_targets(raw_chunk, get_embodiment("droid"), bridge, q_init)
out["qpos"]            # [T, nq] joint targets to send to MuJoCo
out["gripper"]         # [T] grasp column (None for grasp-less embodiments)
out["tracking_error"]  # {"mean_mm", "max_mm"} Cartesian tracking error
```

Verified on Thor against real `nvidia/Cosmos3-Nano` weights, a reachable EE
trajectory tracks to **mean ≈ 11.5 mm / max ≈ 42.8 mm** - the bar pinned by
`tests/policies/cosmos3/test_sim_ik.py`. The Cosmos "modes" above are
world-model *conditioning* modes, not a kinematics solve; this IK layer is
applied *after* Cosmos.

#### De-normalization stats are per domain

The de-normalize step needs that domain's own `q01`/`q99` quantiles. Two domains
ship them bundled; the other three registered embodiments do not:

| embodiment | domain | raw dim | bundled stats |
|---|---|---|---|
| `droid` | `droid_lerobot` | 10 | yes |
| `bridge` | `bridge_orig_lerobot` | 10 | yes |
| `umi` | `umi` | 10 | no |
| `av` | `av` | 9 | no |
| `openarm` | `openarm_lerobot` | 10 | no |

`nvidia/Cosmos3-Edge` documents its forward-dynamics example on `umi` and its
inverse-dynamics example on `av` - both without bundled quantiles - so driving
the sim bridge from Edge means supplying that domain's stats yourself:

```python
out = decode_cosmos_chunk_to_targets(
    raw_chunk, get_embodiment("umi"), bridge, q_init,
    stats={"q01": q01, "q99": q99},   # this domain's own quantiles
    stats_domain="umi",               # required: which domain they describe
)
```

`stats` takes the quantiles as a list (the layout of the bundled
`stats/*_stats.json`), tuple or array; every component must be a finite real
number, since one `nan` quantile would spread through the whole trajectory.
`stats_domain` is required with `stats` and must match the embodiment's domain:
four of the five domains are 10 columns wide, so the width check cannot tell
their quantiles apart, and the two bundled domains disagree by up to **2.77x**
on the translation they decode from the same normalized action.

![Cosmos 3 -> MuJoCo: Franka tracking the Cosmos action (left) beside the Cosmos predicted world (right)](../assets/cosmos3/cosmos3_mujoco_sidebyside.gif)

*Left: MuJoCo Franka driven by a **real** `nvidia/Cosmos3-Nano` action chunk through de-normalize → decode → IK. Right: the Cosmos predicted world video from the same forward pass. Runnable: `examples/vla/cosmos3_diffusers_mujoco_rollout.py --render out.mp4`.*

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

- [Policy overview](overview.md)
- [GR00T](groot.md)
- [LeRobot Local](lerobot-local.md)
- [Custom policies](custom-policies.md)
