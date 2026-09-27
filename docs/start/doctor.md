# Doctor

At the end of this page you can read a `strands-robots doctor` report line by line and know, for each row, what was probed and what to change when it is not `PASS`.

```bash
strands-robots doctor
```

`python -m strands_robots doctor` is the same command. `--list` prints the probe names without probing. Every probe is read-only and sub-second, none opens a serial port or the network, and each returns the verdict the runtime would reach on the same configuration: a `PASS` here never precedes a refusal there.

## A report

On a macOS laptop with the `sim-mujoco` and `lerobot` extras and no GPU:

```text
strands-robots doctor
==================================================

  PASS  Python 3.12.9
  PASS  strands-robots 0.3.9.dev104+g6b79e1626
  PASS  strands-agents 1.43.0
  PASS  mujoco 3.9.0
  WARN  MUJOCO_GL=cgl (needs display)
        Darwin has no offscreen MuJoCo backend, so a window server is required
  PASS  lerobot 0.5.1
  PASS  torchcodec 0.10.0 / torch 2.10.0 loads
  WARN  torch 2.10.0 is CPU-only build
        Policy inference will run on CPU (no CUDA device found on this machine)
  SKIP  torch arch: no CUDA device to compare against
  SKIP  Warp arch: no CUDA device to compare against
  SKIP  serial permissions (non-Linux)
  PASS  HuggingFace token found (/Users/you/.cache/huggingface/token)
  SKIP  device-connect extra not installed (device_connect_edge); uv pip install "strands-robots[device-connect]"
  WARN  mesh=True would not start: it would accept any TLS-signed peer on every topic (no access-control list configured).
        Pick one:
          - Local dev / single machine?  Set STRANDS_MESH_LOCAL_DEV=true (turns off mTLS+ACL for localhost experiments).
          - Sharing a trusted lab network?  Set STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1 to accept this posture.
          - Production?  Point STRANDS_MESH_ACL_FILE at a role-separated ACL (see examples/mesh/mesh_acl_example.json5).
          - Don't need the mesh?  It is OFF by default now -- just drop mesh=True (or set STRANDS_MESH=false).
  PASS  sim smoke test: Robot('so100') works (13 obs keys)

All checks passed. Ready to use strands-robots.
```

Four verdicts. `PASS` and `FAIL` are what they say; a `FAIL` line carries a `Fix:` line under it and makes the exit code 1. `WARN` means the package works but a path is narrowed, and says which. `SKIP` means the probe does not apply on this host or the extra it probes is not installed. Only `FAIL` changes the exit code, which is what makes the command usable in CI.

## The probes

| row | what is checked | not PASS when |
|---|---|---|
| Python | interpreter is 3.12 or newer | `FAIL` below 3.12 |
| Package | `strands_robots` imports; version from the installed distribution | `FAIL` with the `sim-mujoco` install line |
| Strands SDK | `strands` imports | `FAIL` with the install line |
| MuJoCo | `mujoco` imports | `FAIL`: install `[sim-mujoco]` |
| MuJoCo GL | the value MuJoCo will read from `MUJOCO_GL`, folded the way MuJoCo folds it, and whether that backend can render on this host | `FAIL` when the value disables rendering, is not one MuJoCo builds for this platform, or is unset with no display; `WARN` for `cgl` on macOS, which needs a logged-in session |
| LeRobot | `lerobot` is importable and is the package, not an empty directory on the path | `WARN`: install `[lerobot]` |
| Torchcodec | torchcodec loads against the installed torch and finds ffmpeg's shared libraries | `SKIP` without torch or torchcodec; `FAIL` on an ABI mismatch or missing ffmpeg |
| CUDA/GPU | `torch.cuda.is_available()` against what the driver reports | `WARN` for a CPU-only torch, a torch that cannot see a present device, or no torch |
| Torch Arch | the torch build carries code for this GPU's `sm_` architecture | `SKIP` without a CUDA device; `FAIL` when the wheel was built for other architectures |
| Warp Arch | the same question for `warp` (the `sim-newton` extra) | `SKIP` without a CUDA device or warp |
| Serial | Linux only: the user is in `dialout` and a connected `/dev/ttyACM*` or `/dev/ttyUSB*` is readable | `SKIP` on macOS; `FAIL` when the group is missing or a device is not accessible |
| HF Auth | `HF_TOKEN` is set, or a cached login token exists where `huggingface_hub` looks | `WARN`: private checkpoints and dataset pushes will not authenticate |
| Device Connect | the device-connect edge posture: authenticated transport, or an explicit insecure opt-in, or neither | `SKIP` without the extra; `WARN` when `run()` would refuse; `FAIL` when it would be online unencrypted with no caller restriction |
| Mesh | zenoh is installed and `mesh=True` would start under the configured ACL and TLS posture | `WARN` without zenoh, or when the mesh would refuse to start, with the choices listed |
| Sim Test | `Robot("so100")` builds in sim and returns an observation | `FAIL` with the exception, pointing at `MUJOCO_GL` and the MuJoCo install |

The `Mesh` row is the one people meet first: a bare `Robot("so101")` never starts a mesh, so the warning costs nothing until you pass `mesh=True`. When you do, [Fleet](../learn/mesh/fleet.md) explains the three postures the note lists.

## When a fence on these pages fails

Run the doctor first. A `FAIL` on `MuJoCo GL` or `Sim Test` explains a `Robot()` that raises; a `WARN` on `LeRobot` explains a `mode="real"` build with no driver. If the report is clean and a fence still fails, open an issue with the report pasted in.
