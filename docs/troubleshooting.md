---
description: Error → fix table for the most common gotchas across install, sim, hardware, policies, and mesh.
---

# Troubleshooting

## Diagnose first

`doctor` checks the Python version, which extras are importable, GPU/CUDA,
serial permissions, the MuJoCo GL backend, HuggingFace auth and a sim smoke
test, then prints a pass/fail table:

```bash
strands-robots doctor            # or: python -m strands_robots doctor
```

`strands-robots --help` lists the commands the package ships: `doctor` and
`verify-dataset`.

## Install

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: mujoco` | Missing `[sim-mujoco]` | `uv pip install "strands-robots[sim-mujoco]"` |
| `move_to: IK bridge unavailable: ... No module named 'mink'` | Missing `[sim-mujoco]` (the extra declares the IK solver) | `uv pip install "strands-robots[sim-mujoco]"` |
| `ModuleNotFoundError: lerobot` | Missing `[lerobot]` | `uv pip install "strands-robots[lerobot]"` |
| `training failed: 'accelerate' is required but not installed` | Missing LeRobot's `[training]` extra. `strands-robots[lerobot]` does not pull `accelerate` in, and `train()` requires it on CPU as well as GPU | `uv pip install "lerobot[training]"` |
| `ImportError: cannot import name '...' from 'lerobot'` | LeRobot version skew | `uv pip install "strands-robots[lerobot]"` (pins `lerobot>=0.6.1,<0.7.0`) |
| `ImportError: cannot import name 'MolmoAct2Policy'` | `lerobot < 0.6` (`MolmoAct2Policy` ships in lerobot >= 0.6) | `uv pip install "strands-robots[molmoact2]"` |
| `pip install 'strands-robots[ros2]'` builds cyclonedds from source on Jetson/aarch64 (or fails with a CMake / `CYCLONEDDS_HOME` error) | No cyclonedds release publishes a Linux aarch64 wheel | Install Cyclone DDS C first and set `CYCLONEDDS_HOME` - see [rtps integration](rtps-integration.md#linux-aarch64-jetson) |
| pyav build fails on Jetson/aarch64 | No prebuilt wheel for sm_110 | Use `--no-build-isolation` or install `torchcodec>=0.7` and skip pyav. See [installation](getting-started/installation.md#molmoact2-on-jetson) |
| `numpy.dtype size changed` / `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` on Jetson | A wheel built against numpy 1.x (apt `python3-pandas`, an old cached wheel) imported under the numpy 2 that `[lerobot]` requires | In a venv, rebuild the offender through the extra so the resolver keeps lerobot's ranges: `uv pip install --reinstall-package pandas "strands-robots[lerobot]"`. Do not pin `numpy<2` - `lerobot >= 0.6` requires `numpy >= 2` |
| `uv pip install -e .` errors | Wrong cwd | `cd` to repo root first |
| `uv pip install` fails with `No virtual environment found; run uv venv` | `uv pip` installs into the active venv only and none is active | `uv venv --python 3.12 && source .venv/bin/activate`, then install; or `uv pip install --system` to opt out of the venv |

## Simulation

MuJoCo rendering, asset fetches, `add_robot` and `move_to` refusals have their
own sheet: [simulation troubleshooting](simulation/troubleshooting.md).

## Hardware

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `PermissionError: /dev/ttyUSB0` | Not in `dialout` group | `sudo usermod -aG dialout $USER` + re-login |
| Arm twitches at startup | Stale calibration | Re-run `lerobot-calibrate` |
| Camera frames black | Wrong `index_or_path` | `lerobot_camera(action="list")` |
| Servo error mid-rollout | Velocity limit | Bump `control_frequency` or relax calibration limits |
| `Robot("so100", mode="real")` raises | Calibration missing | Run `lerobot-calibrate` first |
| Real robot moves wrong way | Joint mapping mismatch | Verify `data_config` matches recording |

## Policies

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `UntrustedRemoteCodeError` | `lerobot_local` needs HF exec | `export STRANDS_TRUST_REMOTE_CODE=1` |
| `Gr00tPolicy` connection refused | Container not running | `gr00t_inference(action="start_container", ...)` |
| `Gr00tPolicy` returns garbage | `data_config` mismatch | Use same `data_config` as training |
| `Cosmos3Policy` connection refused | Service not running | `uv pip install 'strands-robots[cosmos3-service]'` + start server |
| Policy import slow | Heavy dep at module top | Defer to `__init__` or `get_actions` |

## Recording

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `start_recording` / `DatasetRecorder.create` fails: `lerobot is not installed` | `[lerobot]` not installed | `uv pip install "strands-robots[lerobot]"` |
| `lerobot X is installed, but 'pyarrow' (or `datasets`, `pandas`, `av`, `torchcodec`), which its dataset stack needs, is not` | lerobot is installed **without** its `[dataset]` extra. Installing lerobot again does not pull those in | `uv pip install "lerobot[dataset]"` |
| `...importing its dataset stack failed ... a conflict between installed packages` | Nothing is missing (commonly a `pandas` built against a different `numpy`), so no install of lerobot or its extra fixes it | Reconcile the conflicting packages |
| Need MP4 without LeRobot | - | Use `start_cameras_recording` / `stop_cameras_recording` |
| `0 frames - no clip written` and no MP4 | The window ended before the recorder captured a frame; the stop line names which cause | `start_cameras_recording` reports `capturing` and the warmup it paid; `get_cameras_recording_status` marks a recorder still warming |
| Push fails | Not logged into HF | `hf auth login` (the `huggingface-cli` entry point is not published at the declared `huggingface_hub>=1.5` floor) |

## Mesh

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `mesh.peers` empty | Other peer not running | Wait ~1s; verify `mesh.alive == True` on both |
| Port already bound | Another zenoh process | Mesh auto falls back to client mode; or set `STRANDS_MESH_PORT` |
| `mesh.alive` is `False`, `mesh.peers` stays empty | `eclipse-zenoh` missing (logged at WARNING: "eclipse-zenoh is not installed") | `uv pip install "strands-robots[mesh]"` |
| Want mesh off | - | `STRANDS_MESH=false` or `Robot(..., mesh=False)` |
| Peer is present and `connected`, but publishes no `state` (no joints) | A `_read_state` probe raised. Logged once per category at WARNING: "state probe 'hw_joints' failed" | Read the named probe: `hw_joints` is the motor bus (contended port, missing calibration), `sim_world` / `sim_joints` a sim back-reference, `task_state` the task record. Repeat failures are at DEBUG |

## Agent integration

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Agent picks wrong action | Tool spec confusion | Rephrase instruction; check `robot.tool_spec` |
| `Agent(tools=[robot])` errors | `strands-agents` missing | `uv pip install strands-agents` |
| Agent hangs | Long-running action | Bound the rollout: `run_policy(n_steps=...)`, or `stop_when={'predicate': ...}` to end it on a world state. On MuJoCo `start_policy` also returns immediately; on the other backends it is a blocking passthrough, so it is not the fix there |
| Bedrock/Anthropic auth fails | Provider credentials | See [Strands Agents docs](https://strandsagents.com/) |

Bug reports: [GitHub issues](https://github.com/strands-labs/robots/issues) - include `pip show strands-robots`, Python + OS, minimal repro, full stack trace.

## See also

- [Simulation troubleshooting](simulation/troubleshooting.md) - rendering, assets, IK.
- [Installation](getting-started/installation.md) - extras matrix.
- [Real hardware](hardware/robot-control.md) - bring-up sequence.
- [Contributing](contributing.md) - fix it yourself.
