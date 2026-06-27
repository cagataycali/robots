# Cross-process proof: `use_rtps` drives a real ROS 2 robot

This harness proves the central claim of the pure-RTPS backend: a strands agent
using `use_rtps` (bare `cyclonedds`, **no rclpy, no sourced ROS 2 distro**) can
drive a **real ROS 2 node** - here, `turtlesim` - purely over the DDS/RTPS wire.

It runs turtlesim and our publisher as **separate processes** in one container
sharing a DDS domain, so it validates the actual unknowns: ROS<->DDS topic/type
mangling, the IDL bundle's CDR layout, and DDS discovery/QoS matching against a
genuine ROS 2 participant.

## Run it

```bash
cd examples/ros2/rtps_proof
docker compose run --build --rm proof
```

Expected output (exit code 0):

```
== reading pose BEFORE ==
   before: (5.544, 5.544, 0.0)
== driving via RtpsRobot -> use_rtps -> bare cyclonedds (no rclpy) ==
   drive result: success - published 30 message(s) to /turtle1/cmd_vel ...
== reading pose AFTER ==
   after:  (6.521, 6.419, 1.429)
PASS: a real ROS 2 turtlesim was driven by use_rtps over pure RTPS.
```

A non-zero exit means the turtle did not move - i.e. the wire format regressed.

## What each piece does

| File | Role |
|------|------|
| `Dockerfile` | `ros:jazzy` + CycloneDDS built from source (so the `cyclonedds` Python wheel installs on any arch, incl. arm64) + `strands-robots[ros2]` + `turtlesim`. |
| `run_proof.sh` | Starts a headless turtlesim node (CycloneDDS RMW) in the background, then runs the verifier. |
| `drive_and_verify.py` | Reads the pose (ground truth via `ros2` CLI), drives via `RtpsRobot`/`use_rtps`, asserts the turtle moved. |
| `docker-compose.yml` | One-command build + run; build context is the repo root so the local checkout is installed. |

## Notes

- **Same DDS vendor required.** turtlesim runs with `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`
  so both sides speak CycloneDDS and discover each other. ROS 2's default RMW
  (FastDDS) would also interoperate over RTPS, but pinning one vendor keeps the
  proof deterministic.
- **No host networking.** turtlesim and the publisher share the container's
  loopback DDS domain, isolating wire-format correctness from host/VM multicast
  quirks (relevant on macOS + colima/Lima).
- **arm64.** There is no prebuilt `cyclonedds` Python wheel for linux/arm64, so
  the image builds the C library (with `idlc`) from source - the binding then
  builds against it. This is why the image build takes a few minutes.
