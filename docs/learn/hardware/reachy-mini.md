---
description: A Reachy Mini over its daemon: four axes, the envelope that refuses the rest, the reachy_* agent verbs.
---

# Reachy Mini

At the end of this page a Pollen Robotics Reachy Mini (Lite or Wireless) answers `Robot("reachy_mini", mode="real")` over its daemon, you know the four axes it accepts and the envelope that refuses the rest, and an agent has the `reachy_*` verbs to look, express and listen.

This needs the Mini's daemon reachable (the robot's hostname, REST on `:8000`) and `pip install websockets` for the real-time link.

```python title="sketch"
from strands_robots import Robot

mini = Robot("reachy_mini", mode="real", port="reachy-a.local:8000")
print(mini.connect_eagerly())                   # None; get_status() then names the variant (Lite / Wireless)
mini.send_action({"head_yaw": 20.0, "head_pitch": -10.0})     # degrees; unnamed head axes go to zero
mini.send_action({"antenna_right": 30.0, "antenna_left": 30.0})
```

## Two protocols, one driver

| link | carries | how |
|---|---|---|
| REST `:8000` | `/api/daemon/status` (reachability, variant), recorded moves, motion stop | `connect_eagerly` probes it first |
| real-time | six Stewart leg positions, head IMU, battery | a WebSocket to the daemon (daemon 1.10.0, Lite and Wireless), or an explicitly supplied Zenoh bridge (`zenoh_prefix=`, `transport=`) on Wireless |

Both come from `strands_robots.drivers.reachy_transport`; the driver runs them on one background asyncio loop and caches what arrives. `_imu` is the head IMU verbatim, `_pose` is the head orientation from that IMU (the IMU is in the head), `_battery` when the status payload carries one. No lidar and no platform kinematics: the legs are cached as legs (`_joints`) and the head orientation comes from the sensor.

## The envelope

`send_action` refuses, naming the limit, before anything reaches the daemon:

| axis | limit |
|---|---|
| `head_pitch` | 40 degrees |
| `head_roll` | 40 degrees |
| `head_yaw` | 180 degrees |
| `body_yaw` | 160 degrees |
| `head_yaw - body_yaw` | 65 degrees (`HEAD_BODY_YAW_DELTA_LIMIT_DEG`) |

The head command is a whole pose: `{"head_yaw": 20}` means "look 20 degrees left with pitch and roll zero", not "keep pitch and roll". A `body_yaw` alone is checked against the last commanded head yaw so the coupling limit holds. Antennas travel as a pair: an action with one antenna key and not the other is refused.

## Agent verbs

```python title="sketch"
from strands import Agent
from strands_robots.tools.reachy import reachy_look, reachy_look_at, reachy_express, reachy_get_state, reachy_camera

agent = Agent(tools=[mini, reachy_look, reachy_look_at, reachy_express, reachy_get_state, reachy_camera])
agent("Look at whoever is speaking and act curious.")
```

| verb | does |
|---|---|
| `reachy_get_state`, `reachy_list_emotions` | read the caches; list the recorded-emotion library |
| `reachy_look`, `reachy_look_at`, `reachy_body_turn`, `reachy_antennas`, `reachy_home` | motion inside the envelope |
| `reachy_express`, `reachy_play_sound`, `reachy_volume` | recorded emotions, sound, speaker level |
| `reachy_camera` | one frame from the head camera, saved under `save_path` |
| `reachy_wake`, `reachy_motors`, `reachy_stop` | wake or sleep, torque mode, stop motion |
| `sensors`, `status`, `sleep`, `list_moves`, `say`, `set_volume`, `track_face`, `tracking_status`, `record_audio`, `turn_to_sound`, `turn_to_sound_status` | the rest of the driver's `action` enum, sent through the `mini` tool itself: raw caches and daemon status, the go-to-sleep move, the move library, speech (needs a TTS sidecar), speaker level (the daemon plays a short test sound on change), daemon face tracking and its status, a bounded microphone WAV (GStreamer), and turning toward a voice with its status |

Each verb takes the live driver as its first argument and refuses anything else with one sentence, so an agent handed a disconnected handle learns that, not a traceback. The Mini's verbs are not in the motion gate table on the [agents](../agents.md) page: a desk robot's head is not a payload-bearing actuator, and its own envelope is the safety rail.

## Simulation

`Robot("reachy_mini")` builds the MuJoCo twin with the same axis names, so the look-at maths and the envelope checks can be exercised without the robot.

<robot-viewer name="reachy_mini"></robot-viewer>
