---
description: Error → fix table for the most common gotchas across install, sim, hardware, policies, and mesh.
---

# Troubleshooting

Most issues fit into a small number of categories. This page is the error-to-fix
table.

## Install

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: mujoco` | Missing `[sim-mujoco]` extra | `pip install "strands-robots[sim-mujoco]"` |
| `ModuleNotFoundError: lerobot` | Missing `[lerobot]` extra | `pip install "strands-robots[lerobot]"` |
| `ImportError: cannot import name '...' from 'lerobot'` | LeRobot version skew | Pin: `pip install "lerobot>=0.5.0,<0.6"` or `lerobot>=0.4,<0.5` |
| numpy ABI mismatch on Jetson | System pandas 2.1.4 vs pip numpy 2.x | `pip install "numpy<2" "pandas==2.1.4"` then reinstall strands-robots |
| `pip install -e .` errors with no `pyproject.toml` | Wrong cwd | `cd` into the repo root before `pip install -e .` |

## Simulation

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `mujoco.FatalError: GLXBadFBConfig` (Linux) | Missing OSMesa | `sudo apt install libosmesa6-dev`, then `export MUJOCO_GL=osmesa` |
| Black frames from `render(...)` | Headless without GL backend | `export MUJOCO_GL=osmesa` (Linux) or `=egl` |
| `Robot("foo")` raises ValueError | Unknown robot name | Check spelling against `list_robots("all")`; or pass `urdf_path=...` |
| Sim hangs on `create_world` | Asset download in progress | Wait — first call downloads the MJCF; subsequent calls hit the cache |
| `ModuleNotFoundError: trs_so_arm100_mj_description` | `robot_descriptions` couldn't auto-install | `pip install trs-so-arm100-mj-description` directly |
| `add_robot` raises after `load_scene` | Scene XML overrides world setup | Use `add_robot` *before* `load_scene` or include the second robot in the scene XML |

## Hardware

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `PermissionError: /dev/ttyUSB0` (Linux) | User not in `dialout` group | `sudo usermod -aG dialout $USER`, then re-login |
| Arm twitches at startup | Missing/stale calibration | Re-run `lerobot_calibrate` |
| Camera frames are black | Wrong `index_or_path` | `lerobot_camera(action="list")` to enumerate |
| Servo error mid-rollout | Velocity limit too tight | Bump `control_frequency` or relax limits in calibration |
| `Robot("so100", mode="real")` raises on construction | Calibration missing | Run `lerobot_calibrate` first |
| Real robot moves the wrong way | Joint mapping mismatch | Verify `data_config` matches the recording's data_config |

## Policies

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `UntrustedRemoteCodeError` | `lerobot_local` requires HF code execution | `export STRANDS_TRUST_REMOTE_CODE=1` (after vetting the model source) |
| `Gr00tPolicy` connection refused | Container not running | Start it via `gr00t_inference(action="start_container", ...)` |
| `Gr00tPolicy` returns garbage | `data_config` mismatch | Use the same `data_config` as training |
| GR00T N1.7 wire errors | Old client expecting (K,...) | Library handles this — make sure you're on a current strands-robots install |
| Policy import is slow | Heavy dep loading | Move to `__init__` or `get_actions`; never at module top-level |

## Recording

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `start_recording` reports lerobot missing | `[lerobot]` extra not installed | `pip install "strands-robots[lerobot]"` |
| Empty MP4 files | Recording stopped before any frame was added | Ensure `run_policy` actually executed steps; `get_recording_status` shows the frame count |
| Datasets push fails | Not logged into HF | `huggingface-cli login` |

## Mesh

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `mesh.peers` is empty | Other peer not yet running, or different LAN | Wait ~1s; check both peers have `mesh.alive == True` |
| `STRANDS_MESH_PORT` already bound | Another zenoh process is listening | The mesh falls back to client mode automatically; or set `STRANDS_MESH_PORT` to a free port |
| `init_mesh` raises | `eclipse-zenoh` missing | `pip install "strands-robots[mesh]"` |
| Want mesh off | — | `STRANDS_MESH=false` (process) or `Robot(..., mesh=False)` (per-robot) |

## Agent integration

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Agent doesn't pick the right action | Tool spec confusion | Check `robot.tool_spec`; rephrase the instruction or add hints |
| `Agent(tools=[robot])` errors | `strands-agents` not installed | `pip install strands-agents` |
| Agent hangs | Long-running action (recording/policy) | Use `start_policy` instead of `run_policy` for async |
| Bedrock/Anthropic credentials missing | Provider auth | See the [Strands Agents docs](https://strandsagents.com/) for provider setup |

## How to file a useful bug report

Include:

1. **Strands-robots version** (`pip show strands-robots`).
2. **Python version + OS** (`python --version`, `uname -a`).
3. **Minimal repro** — the smallest possible script that triggers the issue.
4. **Expected vs actual** — what you thought would happen, what did.
5. **Stack trace** — full, unredacted.

Open it on the [issue tracker](https://github.com/strands-labs/robots/issues).

## Still stuck?

- [Discussions](https://github.com/strands-labs/robots/discussions) — open-ended
  questions.
- [Strands Agents docs](https://strandsagents.com/) — for agent / provider issues.
- [LeRobot docs](https://huggingface.co/docs/lerobot) — for upstream dataset / training
  questions.

## See also

- [Installation](getting-started/installation.md) — the full extras matrix.
- [Tutorial 8 — Real hardware](tutorial/08-real-hardware.md) — bring-up sequence.
- [Contributing](contributing.md) — fix it yourself.
