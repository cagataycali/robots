---
description: VERA video-to-action policy (two-stage DFoT/WAN planner + Jacobian IDM) over a containerized GPU server. MimicGen MuJoCo rollouts with IK for eef-delta arms.
---

# VERA policy provider

**VERA** ([Video-to-Embodied Robot Action](https://github.com/sizhe-li/VERA),
MIT/CSAIL) is a **two-stage, closed-loop video-to-action** policy:

1. **Video planner** (DFoT / WAN) — a diffusion model that "dreams" the next
   frames from the current observation (+ optional text). **Embodiment-agnostic.**
2. **Jacobian IDM** — translates the dream into robot actions via a frozen
   visual backbone (VGGT/DINO) + a flow→action head. **Embodiment-specific**,
   data-efficient, swappable without retraining the planner.

> *One video planner, many IDMs* — the route to zero-shot, cross-embodiment control.

The strands-robots `vera` provider is a thin, import-light **WebSocket client +
managed GPU server**, mirroring the `cosmos3` service pattern. The host venv
never installs VERA's heavy/conflicting stack (PyTorch 2.6 / CUDA, VGGT, DFoT) —
that lives in the `strands-vera-server` container.

```
host venv (numpy>=2)                 vera-server container (torch 2.6 / CUDA)
  VeraPolicy ─ VeraWebsocketClient ─ws─▶ vera.server.start_vera_server
  (no vera install)                      DFoT/WAN planner + Jacobian IDM + /ckpts
```

## Quick start

```python
from strands_robots.policies import create_policy

# Attach to a running server (see "Server" below) ...
policy = create_policy("vera", embodiment="mimicgen", auto_launch_server=False)
chunk = policy.get_actions_sync(observation, "stack the red block on the green block")

# ... or let the provider manage the container for you:
policy = create_policy(
    "vera", embodiment="mimicgen",
    server_mode="docker", ckpt_root="/abs/path/vera-ckpts",
)
```

The **MimicGen → Panda** path drives a real 7-DoF arm: the WAN planner + Jacobian
IDM emit 6-DoF end-effector deltas, and the provider's IK bridge solves them onto
the Panda's joints (auto-discovering the end-effector frame — no manual wiring).
See [`examples/vera_mimicgen_panda/`](https://github.com/strands-labs/robots/tree/main/examples/vera_mimicgen_panda).

<figure markdown>
  ![VERA MimicGen Panda rollout](../assets/vera/mimicgen_panda.gif)
  <figcaption>VERA MimicGen policy on a Franka Panda — WAN dream + AllTracker +
  Jacobian IDM → eef-deltas → VERA IK bridge → joint targets → MuJoCo.</figcaption>
</figure>

## Embodiments

From VERA's `adapter_factory._EMBODIMENTS`:

| Embodiment | action_space | dims | views | control | ports (policy/viz) | checkpoints |
|------------|--------------|:----:|-------|:-------:|:------------------:|-------------|
| **pusht** | `velocity` (planar) | 2, no gripper | `image` | 10 Hz | 8820 / 8821 | experimental — IDM du path not wired end-to-end upstream |
| **mimicgen** | `eef_delta` | 7 (6-DoF + grip) | `agentview_image`, `robot0_eye_in_hand_image` | 20 Hz | 8800 / 8801 | ✅ Wave-1 (+WAN base) |
| **allegro** | `joint_position` | 16 | 12 cameras | 15 Hz | 8802 / 8803 | 🔜 Wave-2 (code only) |
| **droid** | `cartesian_delta` | 7 | `varied_1`,`varied_2`,`hand` | 15 Hz | 8804 / 8805 | 🔜 Wave-2 (code only) |

**Today, end-to-end:** `mimicgen` (WAN planner; needs the frozen WAN base + a
motion tracker) is the working, faithful embodiment — it exercises the whole
eef-delta → IK path onto a real arm. `pusht`'s server runs, but its IDM `du`
action path is **not wired end-to-end upstream** (VERA's own
`configurations/dataset/pusht.yaml` documents this gap), so it validates the
provider → server → action plumbing rather than producing a solving rollout —
treat it as experimental. `allegro`/`droid` are code-present but
checkpoint-absent upstream (Wave 2).

### The "generalist" claim, accurately

VERA's *architecture* is cross-embodiment: **one** embodiment-agnostic video
planner + **one cheap IDM per robot** (frozen backbone, head trained from
self-play). It is **not** a single checkpoint that drives every robot today — a
robot is drivable iff (its `action_space` matches a served embodiment) **and**
(a checkpoint exists) **and** (IK/validation is done for that arm). For
`eef_delta`/`cartesian_delta` arms (mimicgen/droid), the provider includes an
**IK bridge** that maps the 6-DoF end-effector deltas to joint targets and
auto-discovers the end-effector frame from the compiled MuJoCo model — so any
kinematically-compatible 6/7-DoF arm can be driven once a matching IDM exists.

## Checkpoints

```bash
hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts   # ~42 GB full; ~4 GB is Wave-1
export VERA_CKPT_ROOT=$PWD/vera-ckpts
```

MimicGen additionally needs the **frozen WAN 2.1 base** (text-enc + VAE + CLIP).
Its IDM uses the **AllTracker** point tracker, which the container bundles
(cloned at build time; weights auto-download). The WAN base:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
```

The provider **never auto-downloads** — point it at pre-downloaded roots.

## Server

The server holds the GPU and the two-stage model. Run it as a container:

```bash
docker build -f strands_robots/policies/vera/docker/Dockerfile -t strands-vera-server:latest .

# MimicGen (serves ws on :8800; needs the WAN base + offline resolver)
docker run --rm --gpus all --ipc=host -p 8800:8800 \
    -v "$VERA_CKPT_ROOT":/ckpts:ro -v "$PWD/Wan2.1-T2V-1.3B":/wan:ro \
    -e VERA_EMBODIMENT=mimicgen -e USE_OFFLINE_RESOLVE=1 \
    strands-vera-server:latest
```

The container entrypoint maps the single mounted `/ckpts` root onto VERA's
per-embodiment checkpoint env vars; `USE_OFFLINE_RESOLVE=1` resolves MimicGen's
wandb-run-id IDM to the locally-mounted checkpoint (via `provenance.json`) so the
server boots with **no network**. See
[`policies/vera/docker/`](https://github.com/strands-labs/robots/tree/main/strands_robots/policies/vera/docker).

`server_mode="docker"` lets the provider build/run/stop the container itself;
`server_mode="subprocess"` launches a local `python -m vera.server...` when VERA
is installed in the same env.

## Configuration

`VeraConfig` maps 1:1 to VERA's server flags and is env-overridable (deploy/CI
wins over code defaults):

| kwarg | env var | maps to |
|-------|---------|---------|
| `embodiment` | — | `--embodiment` |
| `server_port` / `vis_port` | `VERA_SERVER_PORT` / `VERA_VIS_PORT` | `--port` / `--vis-port` |
| `algo_config` | `VERA_ALGO_CONFIG` | `--algo-config` (swap to the omni planner) |
| `dynamics_run_id` | `VERA_DYNAMICS_RUN_ID` | `--dynamics-run-id` |
| `text_prompt` | `VERA_TEXT_PROMPT` | `--text` |
| `ckpt_root` | `VERA_CKPT_ROOT` | container `/ckpts` mount |
| `sample_steps` | `VERA_SAMPLE_STEPS` | `--sample-steps` |
| `tracker_backend` | `VERA_TRACKER_BACKEND` | IDM tracker |
| `motion_plan_scale` | `VERA_MOTION_PLAN_SCALE` | live `configure` |
| `server_mode` | `VERA_SERVER_MODE` | `subprocess` \| `docker` |

## Wire protocol

The provider keeps a rolling **context window** of the last `context_frames`
camera frames (width-concatenated across views) and calls the server's chunked
`infer` when its local action queue drains — the same `RemotePolicy` contract
VERA's own eval harness uses. The server returns `{"action": [H, D]}`; the
provider maps each `D`-vector to robot actuator names (gripper binarized per the
server's `gripper_dim_index`/`gripper_is_raw`), coercing to python floats per the
`Policy` ABC.

## QA the rollouts with a Cosmos 3 reasoner (closed loop)

VERA *generates* video-grounded actions; [NVIDIA Cosmos 3](https://github.com/cagataycali/strands-cosmos)
*reads* video and reasons in text — so it makes a natural **automated QA critic**
for rollouts. Serve the reasoner, then have it grade a rollout MP4:

```bash
uv pip install "strands-cosmos[cosmos3]"
# serve Cosmos3-Nano on :8000 (vLLM + vllm-cosmos3); see strands-cosmos `c3-serve-reason`
python examples/vera_mimicgen_panda/critique_with_cosmos3.py     examples/vera_mimicgen_panda/artifacts/mimicgen_panda.mp4
```

```python
from strands import Agent
from strands_cosmos import Cosmos3ReasonerModel

agent = Agent(model=Cosmos3ReasonerModel(base_url="http://localhost:8000/v1"))
print(agent("Grade this robot rollout — is the motion smooth and purposeful, "
            "any bugs? <video>/tmp/vera-critique/mimicgen_panda.mp4</video>"))
```

This closed `generate → reason → fix` loop surfaced (and fixed) real issues in
the MimicGen→Panda example: an initial **jittery** critique drove the
`ik_smoothing` EMA knob, and a **"the arm is static"** critique root-caused a
near-singular default start pose with the motion off-camera — fixed with a
tabletop-ready seed pose + camera framing. The reasoner's verdict moved
**NEEDS-WORK → PASS**.

## Testing

```bash
# offline unit tests (no GPU, no vera install)
hatch run test tests/policies/vera/

# gated live integration (needs a running server)
VERA_LIVE=1 hatch run test-integ tests_integ/policies/vera/
```

## Install

```bash
pip install 'strands-robots[vera]'        # provider + the VERA git dep (subprocess mode)
pip install 'strands-robots[vera-sim]'    # + MimicGen sim deps for the example (also pulls the experimental PushT env)
```

> **Note on MimicGen.** The `vera-sim` extra does **not** install NVlabs
> MimicGen: that project has no PyPI release, and the `mimicgen` name on
> PyPI is an unaffiliated package (a dependency-confusion risk), so it is
> intentionally not pinned here. The `mimicgen` VERA *embodiment* is just a
> config string and needs no such package. If you genuinely need NVlabs
> MimicGen for data generation, install it from source:
>
> ```bash
> pip install "mimicgen @ git+https://github.com/NVlabs/mimicgen.git"
> ```

For the **docker** path the host needs only `websockets` + `msgpack` (the client
transport) — no `vera`, no torch.
