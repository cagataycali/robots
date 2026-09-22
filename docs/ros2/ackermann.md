---
description: AckermannRosRobot - drive a steering-geometry ROS 2 car (AWS DeepRacer servo stack) as a strands Robot, with bicycle-model conversion and the enable handshake.
---

# Ackermann robots (AWS DeepRacer)


Differential-drive bases take `geometry_msgs/msg/Twist`; Ackermann cars do
not. The AWS DeepRacer's stock stack subscribes to normalized servo pairs
(`deepracer_interfaces_pkg/msg/ServoCtrlMsg`, `angle`/`throttle` in [-1, 1])
and acts on them only after a two-step manual-mode service handshake
(`/ctrl_pkg/vehicle_state` with `state=1`, then `/ctrl_pkg/enable_state` with
`is_active=true`). `AckermannRosRobot` absorbs both differences:

    from strands_robots.mesh import AckermannRosRobot

    car = AckermannRosRobot.from_deepracer(node_name="deepracer")
    car.drive(linear=0.5, angular=1.0, duration=2.0)
    car.get_scan()

`drive()` keeps the same `(linear, angular)` contract as `RosBridgedRobot` -
a bicycle model (`atan(wheelbase * angular / linear)`, clamped to the steering
limit) converts to servo values internally. The handshake declared in
`init_services` runs once, automatically, before the first command; a failed
handshake aborts the drive. Timed and multi-message commands are always
followed by a zero servo message - even when the publish fails - so a timed
drive cannot leave the car with a live throttle, and a halt that itself fails is
reported: the call returns an error naming the throttle that may still be live
instead of the drive's success, so the agent's next action is `stop()`. A bare
single-shot `drive()`
(no `duration`) latches like any raw servo command until `stop()`. Commands
are clamped to `max_speed`; holds longer
than `max_duration` are rejected loudly rather than silently truncated. The
`linear`/`angular`/`duration`/`count` values themselves are checked against the
same shared domains the differential-drive bridges use, so an unusable value is
refused with identical text on every transport. A pair the steering geometry
cannot execute is refused for the same reason: below the rest threshold
(1e-3 m/s) the bicycle model maps any command to the zero servo pair, so
`drive(linear=0.0, angular=1.0)` - a rotate in place, which this platform cannot
do - would otherwise leave as byte-identical to `stop()` and report success for a
heading change that never happened. Give a turn a linear speed to travel at, or
call `stop()`; `drive(0, 0)` still means rest, because that is what it asked for. The
stock platform publishes no odometry, so there is deliberately no
`get_pose`.

Like `RosBridgedRobot`, the bridge inherits the [command
gate](safety.md#safety-critical-command-surfaces-need-operator-approval): the servo topic
(`/manual_drive`) and both mode services (`/vehicle_state`, `/enable_state`) are
blocklisted surfaces, so every command this bridge sends is gated.

| Method / tool | Reaches | Gated |
|---------------|---------|-------|
| `drive()` / `drive_<node>` | `publish` to the servo topic (plus the handshake on the first call) | yes |
| `stop()` / `stop_<node>` | `publish` to the servo topic | yes - the gate is keyed on the surface, not the payload |
| `enable()` | `service_call` to both mode services | yes |
| `get_scan()` / `get_scan_<node>` | `echo` | never gated |

The `drive_<node>` and `stop_<node>` agent tools forward the operator context, so
an agent driving the car prompts rather than failing closed. A programmatic
`car.drive(...)` / `car.stop()` has no operator to prompt, so pre-approve the
three surfaces for a headless run (bare names cover the namespaced DeepRacer
spellings):

```bash
export STRANDS_ROS2_COMMAND_ALLOW=/manual_drive,/vehicle_state,/enable_state
```

See `examples/ros2/deepracer_agent.py`.


