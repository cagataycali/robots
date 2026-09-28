# Topics

At the end of this page you can read any mesh topic by its exact key, know how often each one is published and which environment variable changes that, and know which topics the bridge sends to the cloud.

Keys are `strands/...` by default; `STRANDS_MESH_NAMESPACE` changes the prefix for a whole fleet, and two fleets with different prefixes cannot exchange a message. Keys are written in the source where each publisher lives (`mesh/core.py`, `mesh/sensors.py`, `mesh/input.py`), so this table is hand-checked against those files at this commit rather than generated.

```python title="sketch"
a.mesh.subscribe("imu", "strands/arm-b/imu", lambda key, payload: print(payload))
a.mesh.subscribe("all-state", "strands/*/state", lambda key, payload: print(key, payload["joints"]))
```

## Per-peer topics

| key | default rate | env | payload |
|---|---|---|---|
| `strands/{peer}/presence` | 2 Hz | fixed | heartbeat: `peer_id`, robot name, host, capabilities |
| `strands/{peer}/state` | 10 Hz | fixed (`STATE_HZ`) | `joints`, sim time, task status, `robots_running`, degraded probe reasons |
| `strands/{peer}/health` | 0.5 Hz | `STRANDS_MESH_HEALTH_HZ` | battery, CPU, memory |
| `strands/{peer}/pose` | 10 Hz | `STRANDS_MESH_POSE_HZ` | SE(3) base pose from SLAM, odometry or VIO; the head IMU on a Reachy Mini |
| `strands/{peer}/imu` | 10 Hz | `STRANDS_MESH_IMU_HZ` | orientation, gyro, accel |
| `strands/{peer}/odom` | 10 Hz | `STRANDS_MESH_ODOM_HZ` | wheel or leg odometry |
| `strands/{peer}/lidar/summary` | 5 Hz | `STRANDS_MESH_LIDAR_SUMMARY_HZ` | ranges reduced to a summary |
| `strands/{peer}/lidar/state` | 1 Hz | fixed | lidar device state |
| `strands/{peer}/map/info` | 0.2 Hz | `STRANDS_MESH_MAP_INFO_HZ` | occupancy map metadata |
| `strands/{peer}/hand/{hand}/state` | 50 Hz | fixed | dexterous hand joints |
| `strands/{peer}/camera/{camera}` | off | `STRANDS_MESH_CAMERA_HZ` (`STRANDS_MESH_CAMERA_DISABLED` forces off) | one encoded frame; capped at `STRANDS_MESH_MAX_CAMERA_BYTES` (1 MiB) |
| `strands/{peer}/stream` | 10 Hz while a policy runs | `STRANDS_MESH_STREAM_HZ` | VLA execution steps: observation summary, action, step index |
| `strands/{peer}/input/{device}` | the device's rate, capped at 100 Hz | `STRANDS_MESH_INPUT_MAX_HZ`; receivers refuse a value over `STRANDS_MESH_INPUT_VALUE_ABS` (720) or a step over `STRANDS_MESH_INPUT_SLEW_ABS` | teleoperation frames from `teleoperate(publish=True)` |
| `strands/{peer}/safety/event` | on event | fixed | this peer's own e-stop and resume events |

A sensor topic exists only when the robot exposes the attribute behind it (`_pose`, `_imu`, `_battery`, `_lidar_state`, ...). A driver without an IMU publishes no `imu` key. Every `*_HZ` variable is parsed the same way: a value that is not a positive finite number is reported once and the default stands.

## Command topics

| key | direction | payload |
|---|---|---|
| `strands/{peer}/cmd` | to one peer | `{"action": ..., ...}` from `ALLOWED_ACTIONS`, with `sender_id` and `turn_id`; 20 Hz and 16 KiB caps (`STRANDS_MESH_CMD_RATE_HZ`, `STRANDS_MESH_MAX_CMD_BYTES`) |
| `strands/{requester}/response/{responder}/{turn_id}` | reply | `{"type": "response", "responder_id", "turn_id", "result", "timestamp"}` |
| `strands/broadcast` | to every peer | the same command shape; the sender drops its own copy |

## Fleet safety topics

| key | payload |
|---|---|
| `strands/safety/estop` | `{"peer_id", "t", ...}`: every receiver engages its lockout ([safety](safety-and-estop.md)) |
| `strands/safety/resume` | `{"peer_id", "t", "lockout_elapsed_s", "proof_nonce", "override_proof"}`: receivers verify the HMAC before clearing |

Both are capped at 2 Hz and 4 KiB per message (`STRANDS_MESH_SAFETY_RATE_HZ`, `STRANDS_MESH_MAX_SAFETY_BYTES`) at the transport, before any deserialisation.

## What crosses the bridge

Under `STRANDS_MESH_BACKEND=bridge` a topic goes to MQTT only if its suffix is in `STRANDS_MESH_BRIDGE_TOPICS`. The default set is `presence`, `health`, `cmd`, `response`, `broadcast`, `safety/event`, `safety/estop`, `safety/resume`; `state`, `pose`, `imu`, `odom`, `camera`, `input`, `hand` and `stream` stay on the LAN. MQTT wildcards map `*` to `+` and `**` to `#`; the keys themselves are unchanged. See [bridges](bridges.md).

## ACL matching

In an ACL file, `key_exprs` are matched against the key with the namespace stripped, so `**/cmd` is the pattern that admits or denies commands and `strands/*/cmd` matches nothing.
