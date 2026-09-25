---
description: The @tool helpers for hardware bring-up - cameras, teleop, training, poses, raw serial, mesh and assets.
---

# Hardware tools

```python
from strands_robots import (
    lerobot_camera, lerobot_teleoperate, lerobot_train,
    pose_tool, serial_tool, download_assets,
    gr00t_inference, robot_mesh, use_ros, use_rtps,
)
```

## Tools

| Tool | Key actions | What |
|------|-------------|------|
| `lerobot_camera` | `"list"`, `"test"`, `"capture"`, `"record"` | Enumerate, test, capture from and record connected cameras |
| `lerobot_teleoperate` | `"start"`, `"stop"`, `"status"`, `"replay"`, `"dagger"` | Leader-follower teleop, episode replay, DAgger corrections |
| `lerobot_train` | `"start"`, `"status"`, `"stop"`, `"list"` | Fine-tune a policy on a local dataset via `lerobot-train` |
| `pose_tool` | `"store_pose"`, `"load_pose"`, `"read_all"`, `"move_motor"` | Named servo poses on a real bus, and one motor at a time. Joint-space only - Cartesian IK is `Simulation.move_to` |
| `serial_tool` | `"list_ports"`, `"send"` | Enumerate serial ports, send raw commands |
| `download_assets` | - | Pre-fetch MJCF assets to `~/.strands_robots/assets/` |
| `gr00t_inference` | `"start_container"`, … | GR00T container lifecycle - see [GR00T](../policies/groot.md) |
| `robot_mesh` | `"tell"`, `"broadcast"`, `"emergency_stop"` | Mesh ops - see [Multi-robot](../mesh.md) |
| `use_ros` | `"list_topics"`, `"echo"`, `"publish"`, `"service_call"` | Any ROS 2 robot or sim - see [ROS 2](../ros2-integration.md) |
| `use_rtps` | `"types"`, `"advertise"`, `"publish"`, `"subscribe"` | A ROS 2 graph over pure RTPS, no rclpy - see [Pure-RTPS](../rtps-integration.md) |

Parse results via `result["content"][0]["text"]`, not custom keys like `result["ports"]`. Every
parameter, default and refusal of every tool - these plus the G1 and Reachy Mini verbs - is on
the generated [tool reference](../reference/tools.md).

`pose_tool` and `serial_tool` drive the servo bus through pyserial, which no extra declares - it
arrives inside `lerobot[feetech]`. Without it neither tool imports, and the refusal names the
remedy: `pip install pyserial`.

## Option domains

A value the CLI or the wire format cannot carry is refused before the port is opened or the
subprocess starts, and only for the options the action uses - a value it ignores is never an
error.

### `lerobot_teleoperate`

| Option | Accepted | Why the floor is where it is |
|--------|----------|------------------------------|
| `dataset_fps`, `fps` | positive whole number | lerobot declares both `int`; `30.0` emits `30` |
| `dataset_num_episodes`, `dagger_num_episodes` | positive whole number | no episodes records nothing |
| `dataset_episode_time_s` | positive whole number | a zero-length episode records nothing |
| `dataset_reset_time_s` | non-negative whole number | `0` is no pause between episodes |
| `replay_episode` | non-negative whole number | `0` is the first episode |
| `teleop_time_s` | positive number, or `None` | `float \| None`; `None` is no limit |

`teleop_time_s=0` is refused rather than read as "no limit": it is the one value that means
"stop at once".

### `serial_tool`

Values are masked into fixed-width bytes of the Feetech packet, so an out-of-range value would
reach the wire as a different, reachable command.

| Option | Accepted | Why the bound is where it is |
|--------|----------|------------------------------|
| `motor_id` | integer in `[1, 254]`, `[1, 253]` when a reply is read | one ID byte: `0xfe` is the broadcast, `0xff` the header |
| `position` | integer in `[0, 4095]` | `Goal_Position` is 12-bit on STS/SMS |
| `velocity` | integer in `[0, 32767]` | bit 15 is the direction: more reverses it |
| `baudrate` | positive integer | pyserial coerces: `2.7` opens at 2 baud |
| `read_bytes` | positive integer | a non-positive size looks like a timeout |
| `timeout` | finite number >= 0 | `0` is pyserial's non-blocking read; `inf` overflows the deadline |

A reply-less write to the broadcast address means what it says, so `feetech_position` and
`feetech_velocity` take the whole `[1, 254]`; `feetech_ping` is held to one servo, because the
replies would collide on a half-duplex bus.

These bounds and the reported angle are STS/SMS-series properties, not Feetech ones:
`strands_robots.drivers.feetech.protocol` holds the byte order, the full scale and the `SIGN_BIT`
table, and every Feetech path reads them from there. `position=1023` is full scale on an STS
servo and reads as 65283 on an `scs0009`, so the SCS series needs a second byte order rather than
a scale option - no surface offers one.

### `pose_tool`

A target needs no bound of its own: `FeetechBus.to_counts` refuses one the encoder cannot hold.

`smooth` picks one of two ways to reach the same targets - interpolate over
`steps * step_delay` seconds, or write each `Goal_Position` once and let the servo travel at its
own speed - so it is graded against the shared boolean domain ahead of the `steps` /
`step_delay` check, and `"false"` is refused rather than read as true. Only `load_pose` and
`move_multiple` consult it; `reset_to_home` always interpolates.

`store_pose` and `delete_pose` rewrite the *whole* pose library for a robot
(`<robot_id>_poses.json` under `.strands_robots/poses/`) through a temp sibling plus `os.replace`,
so a failed write leaves every posture as it was and names the pose it did not store.

### `robot_mesh`

`duration` and `policy_port` travel inside the command body
[`validate_command`](../security/commands.md#payload-validation) already bounds. `timeout` and
`limit` never enter a command body, so the tool bounds them:

| Option | Accepted | Why the floor is where it is |
|--------|----------|------------------------------|
| `timeout` | positive finite number | an `Event` wait: `0` or `nan` reports `timeout` for a peer never given the chance to answer |
| `limit` | positive integer | a slice index: non-positive selects the *whole* `inbox` |

`stop` additionally caps `timeout` at 5s, which cannot replace the domain (`min(nan, 5.0)` is
`nan`). `timeout` is read by `tell` / `send` / `rpc` / `broadcast` / `stop` and `limit` only by
`inbox`; `emergency_stop` uses a fixed internal budget.

## Sessions

`lerobot_teleoperate` and `lerobot_train` run detached subprocesses and record each pid in one
on-disk store (`SessionManager` in `strands_robots.tools._process_stop`), written whole through a
temp file and an atomic rename. Both import `psutil` at module scope, so an install without it
loads neither tool. Being listed is not a claim of running: a record carries how long after boot
its process started, and `list` / `status` re-derive liveness from that when asked, so a reused
pid does not read as a live session.

| What the probe reports | `lerobot_teleoperate` | `lerobot_train` |
|---|---|---|
| the pid is gone, reaped mid-probe, or held by another process now | pruned | kept, so `status` still shows the final log tail |
| `psutil.AccessDenied` - it exists, this user may not inspect it | kept on existence alone, at `WARNING` | kept, at `WARNING` |

That last row keeps a session started under `sudo` - a common way to reach a serial port -
stoppable as the unprivileged user. `stop`, the only thing that drops a training record, captures
the process identity *before* it signals, so a SIGKILL escalation is aimed at the process it
found:

| After SIGTERM, then SIGKILL | `stopped` | Result |
|-----------------------------|-----------|--------|
| it left the process table, was already gone, or the pid is held by another process | `true` | success, record dropped (nothing is signalled in the last case) |
| it is still there | `false` | error, record kept |
| whether it exited could not be determined (`AccessDenied`) | `null` | error, record kept |

A SIGKILL is delivered, not observed: a task in an uninterruptible wait - a serial ioctl, a
stalled CUDA call - keeps its pid, so a record is kept exactly when the exit was not observed.

## Calibration

No tool here calibrates: recording one means moving a torque-off arm by hand, and LeRobot ships
that procedure as console scripts:

```bash
lerobot-find-port                                       # which bus is the arm on
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
lerobot-calibrate    --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
                     --robot.id=my_arm
```

The result is JSON under `HF_LEROBOT_CALIBRATION` (default
`~/.cache/huggingface/lerobot/calibration/`), read through LeRobot by `lerobot_teleoperate` and
`lerobot_train`; `lerobot-find-joint-limits` reports the travel it allows.

`FeetechDriver(calibration=...)` and `pose_tool(calibration=...)` take the path of that JSON. It
is the scale, not a refinement - a degree is `360/resolution` counts from the middle of the
*measured* travel, and percent open spans the gripper's measured travel - so a calibrated arm
quotes the number LeRobot quotes for the same servo. Omitting it commands the servo's whole
rotation and bounds a target by the encoder, not by where the joint stops. No per-joint degree
range is declared in the package: an SO arm's span is measured per arm.

## Examples

```python
result = serial_tool(action="list_ports")
print(result["content"][0]["text"])
result = pose_tool(action="read_all", robot_id="so101_follower", port="/dev/ttyACM0")

# DAgger: a policy drives the follower, the leader pre-empts to record
# corrections, appended to the dataset as new episodes.
result = lerobot_teleoperate(
    action="dagger",
    robot_type="so101_follower", robot_port="/dev/ttyACM0",
    teleop_type="so101_leader", teleop_port="/dev/ttyACM1",
    policy_path="user/act_fold",             # policy to roll out
    dataset_repo_id="user/fold_corrections",
    dataset_single_task="fold the towel",
    dagger_num_episodes=10,                  # cap collected corrections
)
```

Handed to an agent:

```python
from strands import Agent
from strands_robots import Robot, lerobot_camera, pose_tool, serial_tool

agent = Agent(tools=[Robot("so100"), lerobot_camera, pose_tool, serial_tool])
agent("Find a connected so100, calibrate it, then stream the wrist camera for 10 seconds")
```

## See also

- [Robot control](robot-control.md) - the `HardwareRobot` class, and when each tool runs.
