# Doctor

This page reads a `strands-robots doctor` report line by line: per row, what was probed and what to change when it is not `PASS`.

```bash
strands-robots doctor
```

`python -m strands_robots doctor` is the same command; `--list` prints the probe names. Every probe is read-only and sub-second, none opens a serial port, only a configured `IoT Direct` or `IoT Child Peers` calls AWS, and each returns the verdict the runtime would reach on the same configuration: a `PASS` here never precedes a refusal there.

## A report

A macOS laptop with the `sim-mujoco` and `lerobot` extras, no GPU:

```text
strands-robots doctor
==================================================

  PASS  Python 3.12.9
  PASS  strands-robots 0.3.9.dev104+g6b79e1626
  PASS  strands-agents 1.43.0
  PASS  mujoco 3.9.0
  WARN  MUJOCO_GL=cgl (needs display)
        Darwin has no offscreen MuJoCo backend, so a window server is required
  PASS  lerobot 0.6.1
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
  SKIP  iot direct: STRANDS_MESH_BACKEND=zenoh (no AWS IoT leg)
  SKIP  iot child peers: STRANDS_MESH_BACKEND=zenoh (no AWS IoT leg)
  PASS  sim smoke test: Robot('so100') works (13 obs keys)

All checks passed. Ready to use strands-robots.
```

Four verdicts. A `FAIL` line carries a `Fix:` line under it and alone makes the exit code 1, so the command works in CI. `WARN` means the package works but a path is narrowed, and says which. `SKIP` means the probe does not apply here or its extra is not installed.

## The probes

| row | what is checked | not PASS when |
|---|---|---|
| Python | interpreter is 3.12 or newer | `FAIL` below 3.12 |
| Package | `strands_robots` imports; version from its distribution | `FAIL` with the `sim-mujoco` install line |
| Strands SDK | `strands` imports | `FAIL` with the install line |
| MuJoCo | `mujoco` imports | `FAIL`: install `[sim-mujoco]` |
| MuJoCo GL | the value MuJoCo will read from `MUJOCO_GL` and whether that backend renders on this host | `FAIL` when the value disables rendering, is not built for this platform, or is unset with no display; `WARN` for `cgl` on macOS, which needs a logged-in session |
| LeRobot | `lerobot` is importable, is the package, and is at least 0.6.1 | `WARN`: install `[lerobot]`; `FAIL` below 0.6.1 |
| Torchcodec | torchcodec loads against the installed torch and finds ffmpeg | `SKIP` without torch or torchcodec; `FAIL` on an ABI mismatch or missing ffmpeg |
| CUDA/GPU | `torch.cuda.is_available()` against what the driver reports | `WARN` for no torch, a CPU-only build, or a torch blind to a present device |
| Torch Arch | the torch build carries code for this GPU's `sm_` architecture | `SKIP` without a CUDA device; `FAIL` when the wheel was built for other architectures |
| Warp Arch | the same question for `warp` (the `sim-newton` extra) | `SKIP` without a CUDA device or warp |
| Serial | Linux: the user is in `dialout` and each connected `/dev/ttyACM*`/`/dev/ttyUSB*` is readable | `SKIP` on macOS; `FAIL` when the group is missing or a device is not accessible |
| HF Auth | `HF_TOKEN` is set, or a cached login token exists where `huggingface_hub` looks | `WARN`: private checkpoints and dataset pushes will not authenticate |
| Device Connect | the edge posture: authenticated transport, an explicit insecure opt-in, or neither | `SKIP` without the extra; `WARN` when `run()` would refuse; `FAIL` when it would be online unencrypted with no caller restriction |
| Mesh | zenoh is installed and `mesh=True` would start under the configured ACL and TLS posture | `WARN` without zenoh or when it would refuse, listing the choices |
| IoT Direct | `STRANDS_MESH_BACKEND=iot` or `bridge` only: one HTTPS `SendDirectMessage` to this identity's own reply topic | `SKIP` otherwise or with `STRANDS_MESH_IOT_DIRECT=0`; `FAIL` when the grant, thing name, endpoint or credential is missing; `WARN` on a transient error |
| IoT Child Peers | `iot` or `bridge` only: the Thing's policy grants `strands/<thing>__*/*`, where its child peers publish | `SKIP` otherwise; `FAIL` with `strands-robots iot reprovision <thing>` when missing; `WARN` when unreadable |
| Sim Test | `Robot("so100")` builds in sim and returns an observation | `FAIL` with the exception, pointing at `MUJOCO_GL` and the MuJoCo install |

A failed `pip install 'strands-robots[ros2]'` on a Jetson is not a doctor row, see [ROS 2](../learn/ros2.md#linux-aarch64-jetson).

The `Mesh` row is the one people meet first: a bare `Robot("so101")` never starts a mesh, so the warning costs nothing until `mesh=True`; [Fleet](../learn/mesh/fleet.md) explains the postures.

## When a fence on these pages fails

Run the doctor first. A `FAIL` on `MuJoCo GL` or `Sim Test` explains a `Robot()` that raises; a `WARN` on `LeRobot` explains a `mode="real"` build with no driver. A clean report and a failing fence is an issue: attach the report.
