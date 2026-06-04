# Phase 1 Results — Cosmos3-Nano-Policy-DROID smoke (VERIFIED ✅)

Date: 2026-06-04. Host: 1× L40S 46GB, driver 580 / CUDA 13.0.

## What ran
1. `cd cosmos-framework && uv sync --all-extras --group=cu130-train --group=policy-server`
   → torch 2.10.0+cu130, CUDA available, openpi_server + openpi_client import OK.
2. Launched the framework's ready-made policy server:
   ```
   COSMOS3_DISABLE_GUARDRAILS=1 python -m \
     cosmos_framework.scripts.action_policy_server_robolab \
     --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8000
   ```
   → `ws://host:8000/`, domain=droid_lerobot, action_space=joint_pos,
     action_dim=8, chunk=32, image 540x640, fps=15, num_steps=4, guidance=3.0.
3. `scratch/c3_policy_client_smoke.py` (OpenPI websocket client) sent an
   observation built from the checked-in DROID example (3 cam frames + 7-DOF
   joint state + gripper + instruction).

## Result — IT WORKS
- Returned `{"action": ndarray(32, 8) float32, "server_timing": {...}}`.
- `(32, 8)` = 32 timesteps × [7 joint positions + 1 gripper] — exactly the
  documented joint_pos action space.
- Values sane: dims 0-6 track the robot's joint ranges, dim7 (gripper) ≈ 0.
- **Cold latency** ≈ a few min (torch.compile JIT warmup). **Warm latency ≈ 3.1s/chunk**
  → ~10 actions/sec effective (chunked control, not 500Hz servo — expected for a
  diffusion policy with num_steps=4).

## Verified I/O contract (ground truth for the provider)
Observation dict (msgpack/numpy over websocket), keys use `/`:
- `prompt`: str
- `observation/wrist_image_left`, `observation/exterior_image_1_left`,
  `observation/exterior_image_2_left`: [H,W,3] uint8  (server composes a
  wrist-on-top + two-exterior concat view)  — OR single `observation/image`.
- `observation/joint_position`: [T,7] float32   (joint_pos space)
- `observation/gripper_position`: [T,1]/[T]/scalar (server applies 1 - g)
- (midtrain space instead uses `observation/eef_pos`[T,3] + `observation/eef_quat`[T,4 xyzw])

Response: `{"action": ndarray[chunk - history, action_dim], "video"?: ndarray, "server_timing": {...}}`

## Env gotchas (for Phase 2+ and docs)
- Guardrail model `nvidia/Cosmos-Guardrail1` is HF-GATED (approval required).
  Bypass with the `guardrails=False` setup override (we toggled via
  `COSMOS3_DISABLE_GUARDRAILS=1` env, a 3-line local patch to the robolab server).
  → For the robots provider, document this and either (a) require guardrail
  approval, or (b) drive the server with guardrails disabled, or (c) propose an
  upstream `--no-guardrails` flag.
- Shell on this host is `sh`; use `.venv/bin/python` directly (no `source`).
- Free any prior vLLM/GPU tenants first (16B Nano-Policy ≈ 33 GB on GPU).
- `export LD_LIBRARY_PATH=` before running (per framework setup docs).
