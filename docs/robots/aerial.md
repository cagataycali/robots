---
description: Quadcopters, and the native CRTP driver that flies a real one.
---

# Aerial

Quadcopters. Two in the registry; one of them flies for real from here.

```python
from strands_robots import Robot
sim = Robot("crazyflie")        # Bitcraze Crazyflie 2
sim = Robot("skydio_x2")        # Skydio X2
```

## Catalog

Every robot in this family, generated from `robots.json` at build time. Renders are MuJoCo sim renders, never hardware photos.

{{robot_cards:aerial}}

## Flying a real Crazyflie

`crazyflie` declares `hardware.driver = "strands"`, so `mode="real"` builds the native
CRTP driver over a [Crazyradio](https://www.bitcraze.io/products/crazyradio-2-0/) dongle.
lerobot has no robot type for a Crazyflie, so this is the only way to fly one from here.

```python
from strands_robots import Robot

cf = Robot("crazyflie", mode="real", port="radio://0/80/2M/E7E7E7E7E7")

# Opens the link and WAITS for the aircraft to answer, then ARMS the platform and
# starts telemetry. Returns None on success, or a reason - check it: nothing below
# can fly if the link never came up.
if (reason := cf.connect_eagerly()) is not None:
    raise SystemExit(reason)

# Every flight verb answers with an envelope, including for a link that went quiet
# after connecting - so read `status` rather than assuming the write landed. Each
# step is checked before the next is issued: after a refused takeoff there is no
# altitude for the twist to hold.
try:
    env = cf.takeoff(height=0.5, duration=2.0)
    if env["status"] == "success":
        env = cf.set_twist(vx=0.2, wz=1.0, z=0.5)  # 0.2 m/s fwd, 1.0 rad/s yaw, 0.5 m
    if env["status"] == "success":
        env = cf.land()                            # descends under control
    if env["status"] != "success":
        print(env["content"][0]["text"])
finally:
    cf.cleanup()  # lands if it is still flying, then releases the radio
```

Install the client library with the `crazyflie` extra:
`pip install "strands-robots[crazyflie]"`. It is **not** part of `[all]` - `cflib` is
GPLv3 and this project is Apache-2.0, so the copyleft dependency is only installed by a
caller who names it. Without it the driver still imports and registers; it reports a
reason naming this extra instead of connecting.

Five things behave differently from a ground robot, and each one is a way to break the
aircraft if you assume otherwise:

| | What to know |
|---|---|
| **Connecting waits** | `cflib`'s `open_link` is asynchronous and never raises - it reports failure on a *callback*, so an absent dongle or a switched-off aircraft returns normally. `connect_eagerly()` therefore blocks until the aircraft answers (up to `CONNECT_TIMEOUT_S`, 10 s) and returns the reason if it does not. Check that return: while there is no link `cflib` discards every packet in silence. |
| **Units** | `wz` is **rad/s**, as everywhere else in this package. `cflib` wants deg/s, and the driver is the only place that conversion happens. |
| **Setpoints are a stream** | The firmware supervisor cuts thrust when the setpoint stream goes quiet, so one `send_action` latches a setpoint and a background repeater keeps it alive at `setpoint_hz` (default 20 Hz). It returns when the setpoint is latched, not when the motion is done. |
| **`stop` lands** | `stop()` / `stop_task()` / `cleanup()` all perform a controlled descent. Cutting the motors - an airborne aircraft *falls* - is the separately named `emergency_stop()`, and the agent tool schema cannot reach it. |
| **A link can go quiet in flight** | `is_connected` reads True for a link that opened and then stopped answering - the handle is live and only the write finds out. So every flight verb returns an error envelope for a write the radio would not carry, rather than raising: check `status` on `send_action` / `set_twist` / `takeoff` / `land` / `emergency_stop`, not just on `connect_eagerly()`. Two of those refusals carry a consequence worth acting on - a refused priority handover means the climb or descent was *not* commanded, and a refused `emergency_stop` means the motors are **still turning** with the setpoint stream already stopped, so only a hardware cutoff will stop the aircraft. |

The flight envelope is the driver's, not the SDK's: `cflib` imposes no ceiling and the
firmware attempts whatever arrives. A setpoint outside it is **refused by name**, never
clamped, so a caller who asked for 5 m/s never silently flies 1 m/s. Read the bounds with
`strands_robots.drivers.crazyflie.twist_envelope()`.

Commands go through `send_action` / `set_twist` / `takeoff` / `land`; `start_task` and
`run_policy` refuse, because this package registers no aerial policy provider and a
quadcopter has no joints for a manipulation policy's action to land on. Telemetry
(`stateEstimate` position, `stabilizer` attitude, `pm.vbat`) is cached for the mesh; a bare
Crazyflie has no ranger deck, so no lidar topic is published.

## See also

- [Mobile](mobile.md) - quadrupeds and wheeled bases.
- [Mobile manipulators](mobile-manip.md) - mobile bases carrying an arm.
- [Robot control (real hardware)](../hardware/robot-control.md) - the shared `mode="real"` surface.
