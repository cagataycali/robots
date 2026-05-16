---
description: NVIDIA GR00T (N1.5 / N1.6 / N1.7) — ZMQ + HTTP transports, 25 embodiment data_configs, full container lifecycle.
---

# GR00T

`Gr00tPolicy` talks to an NVIDIA GR00T inference container. The container does the
model load and forward pass; the policy is a thin client.

## TL;DR

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "groot",
    server_address="localhost:5555",   # GR00T inference server
    data_config="so100_dualcam",       # which embodiment config
)
```

## Setup

```bash
pip install "strands-robots[groot-service]"
```

That installs `pyzmq` + `msgpack` for the wire protocol. You also need the GR00T
container itself running — see [container lifecycle](#container-lifecycle).

## Supported model versions

| Version | Transport | Wire format |
|---------|-----------|-------------|
| GR00T N1.5 | ZMQ | observation: `(K, ...)` shape |
| GR00T N1.6 | ZMQ | observation: `(K, ...)` shape |
| GR00T N1.7 | HTTP | observation: `(B, T, ...)` shape, float32 state |

The policy auto-detects the wire format from the server. `data_config` selects the
embodiment.

## Constructor parameters

```python
Gr00tPolicy(
    server_address: str = "localhost:5555",
    data_config: str = "...",         # required
    transport: str = "auto",          # "zmq" | "http" | "auto"
    timeout: float = 30.0,
    chunk_size: int = 16,             # action chunk length
    request_id_prefix: str | None = None,
)
```

`server_address` is the host:port of the running GR00T inference container.
`data_config` must match the embodiment GR00T was trained or fine-tuned on.

## The 25 embodiment data_configs

GR00T ships with embodiment configurations covering common robots. The full list lives
in `strands_robots/policies/groot/data_configs.json`. Highlights:

| Config | Robot | Cameras |
|--------|-------|---------|
| `so100` | SO-ARM100 | 1 |
| `so100_dualcam` | SO-ARM100 | front + wrist |
| `so100_4cam` | SO-ARM100 | front + wrist + top + side |
| `bimanual_panda_gripper` | 2× Panda | 3 cams |
| `fourier_gr1_arms_only` | Fourier GR-1 (arms only) | ego view |
| `unitree_g1` | Unitree G1 | rs_view |

For the full list:

```python
import json
with open("strands_robots/policies/groot/data_configs.json") as f:
    configs = json.load(f)
print(list(configs.keys()))
```

## Container lifecycle

The `gr00t_inference` Strands tool manages the container for you:

```python
from strands_robots.tools import gr00t_inference

# Build the inference image
gr00t_inference(action="build_image", tag="gr00t-n1.7:latest")

# Download a checkpoint into the container's volume mount
gr00t_inference(action="download_checkpoint",
                model_id="nvidia/GR00T-N1.7-3B")

# Start the container — exposes port 5555 by default
gr00t_inference(action="start_container",
                tag="gr00t-n1.7:latest",
                model_id="nvidia/GR00T-N1.7-3B",
                data_config="so100_dualcam")

# ... use the policy ...

# Stop and remove
gr00t_inference(action="stop_container")
gr00t_inference(action="remove_container")
```

Or all of it through a Strands Agent:

```python
from strands import Agent
from strands_robots import Robot
from strands_robots.tools import gr00t_inference

robot = Robot("so100")
agent = Agent(tools=[robot, gr00t_inference])

agent("Start a GR00T N1.7 server with the so100_dualcam config "
      "and pick up the cube using it for 15 seconds")
```

The agent will run `build_image` → `download_checkpoint` → `start_container` →
`run_policy(policy_provider='groot', ...)` in order.

## RTC — Real-Time Chunk

For low-latency control, the policy supports RTC: the container streams overlapping
action chunks as the policy thinks. The client smooths between chunks so the robot
keeps moving while the next chunk arrives.

```python
policy = create_policy(
    "groot",
    server_address="localhost:5555",
    data_config="so100_dualcam",
    chunk_size=16,
    rtc=True,         # enable RTC
)
```

RTC matters most on real hardware where blocking on a 200ms inference round-trip
between chunks would cause visible stutter.

## N1.7 wire-format specifics

GR00T N1.7 changed the observation shape from `(K, ...)` to `(B, T, ...)` with batch +
time dimensions. State data must be `float32`. The policy handles this automatically —
you don't pass anything different at the call site.

If you're integrating with a custom server, the client logic is in
`strands_robots/policies/groot/client.py` — open a PR if you find a wire-format
divergence.

## See also

- [Tutorial 3 — Policies](../tutorial/03-policies.md) — full walkthrough.
- [Tutorial 8 — Real hardware](../tutorial/08-real-hardware.md) — drive a real arm
  with GR00T.
- [LeRobot Local](lerobot-local.md) — HuggingFace alternative.
- [Isaac-GR00T project](https://github.com/NVIDIA/Isaac-GR00T) — upstream model + training.
