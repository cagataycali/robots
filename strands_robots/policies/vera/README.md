# VERA Policy Provider

**VERA** (Video-to-Embodied Robot Action — MIT/CSAIL,
[github.com/sizhe-li/VERA](https://github.com/sizhe-li/VERA)) as a first-class
`strands_robots` policy provider.

VERA is a **two-stage closed-loop video-to-action** policy:

1. **Video planner** (`vera.video_model` / DFoT / WAN) — a diffusion model that
   "dreams" future frames from the current observation (+ optional text).
   **Embodiment-agnostic.**
2. **Jacobian IDM** (`vera.idm` + `vera.policy`) — translates the dream into
   robot actions. **Embodiment-specific**, swappable without retraining the
   planner.

> *One video planner, many IDMs* → zero-shot cross-embodiment control.

This provider mirrors the [`cosmos3`](../cosmos3/) service pattern: a
self-contained msgpack + WebSocket client plus an optional managed server
subprocess (`vera.server.start_vera_server`).

## Architecture

```
VeraPolicy(Policy)            provider.py      — robots Policy contract
  ├── VeraConfig              config.py        — env-overridable pydantic-free config
  ├── VeraWebsocketClient     client.py        — msgpack+ws client (no `vera` import)
  │     └── _msgpack_numpy    _msgpack_numpy.py— vendored numpy codec
  └── VeraServerRunner        server_runner.py — managed subprocess (list args)
```

The provider keeps a rolling **context window** of the last `context_frames`
camera frames and calls the server's chunked `infer` when its local action queue
drains (the `RemotePolicy` contract from VERA's own eval harness). The returned
`[H, D]` action chunk is mapped to robot actuator-name dicts.

## Install

VERA targets **Python 3.11 + PyTorch 2.6 (CUDA 12.4)**. It's gated behind an
extra so it never breaks the other providers:

```bash
pip install 'strands-robots[vera]'      # client + VERA git dep
# VGGT (IDM visual backbone) is a git dep pulled by VERA's [idm] extra:
pip install 'git+https://github.com/facebookresearch/vggt.git'
```

Simulators (PushT / MimicGen) for the examples are a **separate** extra
(they carry their own MuJoCo assets):

```bash
pip install 'strands-robots[vera-sim]'
```

`flash-attn` is optional (WAN falls back to SDPA).

## Checkpoints

```bash
hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts
export VERA_CKPT_ROOT=$PWD/vera-ckpts
```

Wave-1 (PushT + MimicGen) is ~4 GB; the full repo (incl. OMNI WAN planner) is
~42 GB. The provider **never auto-downloads** — point it at a pre-downloaded
root via `ckpt_root=` / `VERA_CKPT_ROOT`.

## Quickstart

```python
from strands_robots.policies import create_policy

# Auto-launches `vera.server.start_vera_server --embodiment pusht` and waits
# for the websocket; set auto_launch_server=False to attach to a running server.
policy = create_policy("vera", embodiment="pusht")
chunk = policy.get_actions_sync(observation, "push the T to the goal")
```

In MuJoCo / sim:

```python
sim.run_policy(
    robot_name="pusher",
    policy_provider="vera",
    policy_config={"embodiment": "pusht"},
    instruction="push the T to the goal",
    n_steps=200,
    control_frequency=10.0,
)
```

See [`examples/vera_pusht_mujoco/`](../../../examples/vera_pusht_mujoco/) for an
end-to-end rollout + recording.

## Embodiments & ports

| Embodiment | Wave | Policy / Viz ports | Views | Action space |
|------------|:----:|:------------------:|-------|--------------|
| `pusht`    | 1    | 8820 / 8821        | `image` | 2D pos-delta (no gripper) |
| `mimicgen` | 1    | 8800 / 8801        | `agentview_image`, `robot0_eye_in_hand_image` | eef-delta + gripper |
| `allegro`  | 2    | 8802 / 8803        | 12 cameras | joint position |
| `droid`    | 2    | 8804 / 8805        | `varied_1`, `varied_2`, `hand` | cartesian-delta + gripper |

Wave-2 code is in-tree; checkpoints land with the upstream Wave-2 release.

## Configuration

All `VeraConfig` fields map 1:1 to VERA server flags and are env-overridable
(deploy/CI wins over code defaults):

| kwarg | env var | flag |
|-------|---------|------|
| `embodiment` | — | `--embodiment` |
| `server_port` | `VERA_SERVER_PORT` | `--port` |
| `vis_port` | `VERA_VIS_PORT` | `--vis-port` |
| `algo_config` | `VERA_ALGO_CONFIG` | `--algo-config` (swap to omni planner) |
| `dynamics_run_id` | `VERA_DYNAMICS_RUN_ID` | `--dynamics-run-id` |
| `text_prompt` | `VERA_TEXT_PROMPT` | `--text` |
| `ckpt_root` | `VERA_CKPT_ROOT` | (env to server) |
| `sample_steps` | `VERA_SAMPLE_STEPS` | `--sample-steps` |
| `tracker_backend` | `VERA_TRACKER_BACKEND` | (env to server) |
| `motion_plan_scale` | `VERA_MOTION_PLAN_SCALE` | live `configure` |

## Error handling

The provider gives **actionable** errors:

- **Server not reachable** → hint with the exact `start_vera_server` +
  `hf download` commands.
- **Server exits early** (missing checkpoints / CUDA OOM) → surfaced from the
  streamed `[vera.server]` log with a pointer to set `VERA_CKPT_ROOT`.
- **Slow WAN model load** → bounded by `server_ready_timeout` (default 600s).

## Testing

```bash
hatch run test tests/policies/vera/          # offline unit tests (no GPU)
hatch run test tests_integ/policies/vera/    # gated live server + 1 rollout
```
