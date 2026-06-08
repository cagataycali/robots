---
description: NVIDIA GR00T (N1.5 / N1.6 / N1.7) — ZMQ service or local inference, 27 embodiment data_configs, full container lifecycle.
---

# GR00T

`Gr00tPolicy` talks to an NVIDIA GR00T inference container over ZMQ, or loads the
model in-process when you pass `model_path`. The container does the forward pass;
the policy is a thin client.

## TL;DR

```python
from strands_robots.policies import create_policy

# Service mode (container running separately)
policy = create_policy(
    "groot",
    port=5555,                    # GR00T inference server port
    data_config="so100_dualcam",  # embodiment config
)

# Local mode (load model in-process)  # requires GPU
policy = create_policy(
    "groot",
    model_path="/path/to/checkpoint",
    data_config="so100_dualcam",
    device="cuda",
)
```

## Setup

```bash
pip install "strands-robots[groot-service]"
```

That installs `pyzmq` + `msgpack` for the wire protocol. You also need the GR00T
container itself running — see [container lifecycle](#container-lifecycle).

## Supported model versions

| Version | Transport | Notes |
|---------|-----------|-------|
| GR00T N1.5 | ZMQ | observation: `(K, ...)` shape |
| GR00T N1.6 | ZMQ | observation: `(K, ...)` shape |
| GR00T N1.7 | ZMQ | observation: `(B, T, ...)` float32; auto-detected |

All service communication is ZMQ. The N1.7 `(B, T, ...)` float32 wire format is
handled automatically — the policy auto-detects the server version from the
installed `gr00t` package, so you don't pass anything different at the call site.

## Constructor parameters

```python
Gr00tPolicy(
    data_config: str = "so100_dualcam",          # embodiment config (required)
    host: str = "localhost",                      # service host
    port: int = 5555,                            # service port
    model_path: str | None = None,               # local mode: path to checkpoint
    embodiment_tag: str = "NEW_EMBODIMENT",      # override embodiment tag
    device: str = "cuda",                        # torch device (local mode)
    groot_version: str | None = None,            # override auto-detection
    strict: bool = False,                        # strict config checking
    api_token: str | None = None,                # falls back to GROOT_API_TOKEN env
    observation_mapping: dict | None = None,     # remap incoming obs keys
    action_mapping: dict | None = None,          # remap outgoing action keys
    language_key: str | None = None,             # custom instruction key
)
```

**Service mode** (default): connect to a running GR00T container via ZMQ using
`host` + `port`. `model_path` must be `None`.

**Local mode**: pass `model_path` to load the model in-process. Requires a GPU
and the full GR00T Python package installed.

`api_token` is used for authenticated model pulls; if `None` the env var
`GROOT_API_TOKEN` is read automatically.

## The 27 embodiment data_configs

GR00T ships with 27 embodiment configurations for common robots. The full list lives
in `strands_robots/policies/groot/data_configs.json`. Highlights:

| Config | Robot | Notes |
|--------|-------|-------|
| `so100` | SO-ARM100 | 1 camera |
| `so100_dualcam` | SO-ARM100 | front + wrist |
| `so100_4cam` | SO-ARM100 | front + wrist + top + side |
| `so101` / `so101_dualcam` / `so101_tricam` | SO-ARM101 | 1/2/3 cameras |
| `bimanual_panda_gripper` | 2× Panda | 3 cameras |
| `fourier_gr1_arms_only` | Fourier GR-1 (arms only) | ego view |
| `unitree_g1` | Unitree G1 | rs_view |
| `galaxea_r1_pro` | Galaxea R1 Pro | — |

The full set (27 total):

```
so100               so100_dualcam          so100_4cam
so101               so101_dualcam          so101_tricam
bimanual_panda_gripper                     single_panda_gripper
libero_panda        oxe_droid              oxe_widowx
oxe_google          fourier_gr1_arms_only  fourier_gr1_arms_waist
fourier_gr1_full_upper_body
unitree_g1          unitree_g1_full_body   unitree_g1_locomanip
unitree_g1_real     unitree_g1_sonic
agibot_*            galaxea_r1_pro
```

To inspect them programmatically:

```python
import json
from importlib.resources import files

with open(files("strands_robots.policies.groot") / "data_configs.json") as f:
    data = json.load(f)
configs = data["configs"]
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

## See also

- [Tutorial 3 — Policies](../tutorial/03-policies.md) — full walkthrough.
- [Tutorial 8 — Real hardware](../tutorial/08-real-hardware.md) — drive a real arm
  with GR00T.
- [LeRobot Local](lerobot-local.md) — HuggingFace alternative.
- [Cosmos 3](cosmos3.md) — NVIDIA Cosmos 3 omnimodal VLA alternative.
- [Isaac-GR00T project](https://github.com/NVIDIA/Isaac-GR00T) — upstream model + training.
