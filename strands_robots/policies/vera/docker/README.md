# VERA server — Docker

Run the VERA policy server in a container with VERA's full GPU stack
(PyTorch 2.6 / CUDA 12.4 + VGGT) **isolated from the host robots venv**. The
`VeraPolicy` provider connects over the websocket protocol — the host never
installs VERA's heavy/conflicting deps.

```
host robots venv (numpy>=2)          vera-server container (torch 2.6, CUDA 12.4)
  VeraPolicy ─ VeraWebsocketClient ─ws─▶ vera.server.start_vera_server
  server_mode="docker"                   (VERA + VGGT + sim + /ckpts mounted)
```

## Prerequisites

- NVIDIA GPU + driver, Docker, and the **NVIDIA Container Toolkit**
  (`--gpus all` must work: `docker run --rm --gpus all nvidia/cuda:12.4.0-base nvidia-smi`).
- Downloaded checkpoints:
  ```bash
  hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts
  export VERA_CKPT_ROOT=$PWD/vera-ckpts
  ```

## Build

```bash
# from the strands-robots repo root
docker build -f strands_robots/policies/vera/docker/Dockerfile \
    -t strands-vera-server:latest .
# pin VERA:  --build-arg VERA_REF=<commit-or-tag>
```

## Run

**Manually** (PushT):

```bash
docker run --rm --gpus all -p 8820:8820 -p 8821:8821 \
    -v "$VERA_CKPT_ROOT":/ckpts:ro \
    -e VERA_EMBODIMENT=pusht \
    strands-vera-server:latest
```

**Compose**:

```bash
docker compose -f strands_robots/policies/vera/docker/docker-compose.yml up
# MimicGen:  VERA_EMBODIMENT=mimicgen docker compose ... up
```

**Provider-managed** (the provider starts/stops the container for you):

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "vera",
    embodiment="pusht",
    server_mode="docker",            # <- manage the container, not a subprocess
    ckpt_root="/abs/path/vera-ckpts",
)
chunk = policy.get_actions_sync(obs, "push the T to the goal")
policy.close()                       # stops the container it started
```

## Checkpoint wiring

The entrypoint maps the single mounted `/ckpts` root (the `hf download` layout)
onto VERA's per-embodiment checkpoint env vars:

| Embodiment | Maps |
|------------|------|
| `pusht` | `pusht-dfot/model.ckpt` → `VERA_PUSHT_PLANNER_CKPT`; `pusht-idm/model.ckpt` → `VERA_PUSHT_DYNAMICS_CKPT` |
| `mimicgen` | `mimicgen-wan-1.3b/algo_config.yaml` → `VERA_ALGO_CONFIG`; IDM run `x21o0cwe` |

An explicit `-e VERA_…` always overrides the auto-wiring.

## Ports

| Embodiment | policy | viz |
|------------|:------:|:---:|
| pusht | 8820 | 8821 |
| mimicgen | 8800 | 8801 |

## Notes

- First boot loads the WAN/DFoT planner — the provider's health-check waits up
  to `server_ready_timeout` (default 600s).
- Headless rendering uses EGL (`MUJOCO_GL=egl`); no X server needed.
- `flash-attn` is optional (WAN falls back to SDPA on the NGC base).
