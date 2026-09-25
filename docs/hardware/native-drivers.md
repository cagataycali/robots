---
description: The driver="strands" contract - what a native driver is, how to register one, and the domains and telemetry coercion every shipped driver holds to.
---

# Native drivers

`Robot(..., mode="real")` builds the lerobot driver by default; `driver="strands"` builds
the native one registered for that robot. Which value builds what is on the
[Robot factory](../getting-started/robot-factory.md#choosing-a-driver) page - this one is the
contract a native driver satisfies.

## Cameras

Cameras are attached by the **lerobot** driver. A native driver
(`driver="strands"`) addresses its cameras through its own SDK, so it does not
take a caller-supplied config - and none of the drivers shipped here does. Rather
than accept the keyword and hand back a robot with no cameras, the factory
refuses it by name:

```python
>>> Robot("unitree_go2", mode="real", cameras={"front": {"type": "opencv", "index_or_path": 0}})
ValueError: Go2Driver does not open cameras, so cameras= cannot be honored for
'unitree_go2'. Forwarding it would return a robot with no cameras at all under
status=success. Use driver='lerobot', which attaches them through lerobot's
camera backends, or capture the frames outside the driver.
```

This reaches robots that never mention `driver=`: eight shipped robots declare
`driver="strands"` in the registry (see [Declaring one](#declaring-one)). A driver
that does open the cameras it is given
declares `reads_cameras = True` on the class and receives the dict verbatim - the
opt-in is that one attribute, described with the rest of the constructor contract
in `strands_robots.drivers.base`.

## Writing one

A native driver is for a robot lerobot's arm/serial shape cannot model - a humanoid with its
own state machine, a rover reporting GPS, a base publishing a point cloud. It is a separate
class satisfying `strands_robots.drivers.HardwareDriver`, registered against a robot name:

```python
from strands_robots.drivers import register_native_driver

register_native_driver("unitree_g1", G1Driver)

robot = Robot("unitree_g1", mode="real", driver="strands", port="192.168.123.161")
```

`register_native_driver` refuses a class that does not satisfy the contract and names the
members it is missing, so a half-built driver fails at the line that registers it rather
than on the first agent call. `port=` stays polymorphic - a serial path, an IP address or a
URL - because only the driver knows how to read it.

## Domains a driver does not widen

`baud_rate=` does not stay polymorphic. Every surface that opens a serial bus - the Feetech
and Dynamixel drivers, `FeetechBus`, and the `baudrate` of `serial_tool` and `pose_tool` -
holds it to a positive integer and refuses anything else by name at construction. pyserial
takes the speed through its own `int()` and refuses only a negative, so an ungraded value is
*applied*: `2.7` opens the port at 2 baud, and `0` opens it at a speed no servo answers,
after which every read times out exactly as an unplugged arm does.

The read window is the same shape. `timeout=` - on `FeetechBus` and on `FeetechDriver`, which
forwards a caller's window to it - is held to a positive finite number at construction.
pyserial takes `0`, `nan`, `inf` and `None` verbatim, and each leaves the read looking at an
empty buffer the retry loop cannot tell from a servo that never answered, so a healthy arm
reports as motors that did not reply. A keyword a driver *records* instead of forwarding fails
one layer earlier: the caller lengthens the window, the bus opens at its default, and nothing
says so.

A driver has **two** ways to halt its robot and they are not the same contract. `stop_task()`
returns a status envelope and decides an outcome, so that is what a caller reads. `stop()` is
the lifecycle hook and is annotated `-> None`, so it carries no verdict at all - which makes
its log the only place a halt it could not complete can be recorded. A `stop()` that
delegates to a halt verb must therefore read that verb's envelope and log a non-success,
naming what may still be moving; `strands_robots.drivers.halt_failure_detail` reads the
reason out of one. Discarding it returns from shutdown reporting the robot as stopped on the
one surface that has no way to say otherwise.

## Telemetry a driver decodes

A driver that decodes its own telemetry decides, field by field, whether a reading exists. Read
every field as `getattr(msg, name, None)` and coerce it, because a *typed* default is a
well-formed value: a firmware that renames a field publishes a plausible constant rather than an
absence. `strands_robots.drivers.base` owns that coercion -- `telemetry_float`, `telemetry_int`,
`telemetry_float_list`, `telemetry_int_list` -- so the answer does not depend on which driver
asked. Each returns `None` for anything that is not a reading, including a `bool` (`float(True)`
is `1.0`, indistinguishable from a real one-percent pack) and a bytes-like value (`str`, `bytes`,
`bytearray`, `memoryview` all iterate, so a raw buffer would decode as a vector of the wrong
length). The vector readers are all-or-nothing and return a fresh list, so a caller mutating the
envelope does not race the callback thread's next write.

The rule covers the scalars a decoder sends back out, not only the ones it publishes. The Unitree
`mode_machine` is read from `rt/lowstate` and echoed on every `LowCmd_`, and the firmware drops a
frame whose layout id does not match the one it announced, so an id that came from something other
than a number is a write the robot silently ignores. A bare `int()` is the wrong coercion there:
`int(True)` is `1` and `int(False)` is `0`, both valid uint8 ids. `telemetry_int` refuses both, and
still truncates a float, because a decoder stricter than its sibling on a value both accept is the
drift these functions exist to prevent.

A refused scalar leaves the cached value at the last reading that parsed, rather than clearing it.
That matters when a gate reads the cache: `mode_machine` gates every G1 motion write and its refusal
reads "lowstate has not delivered yet", which one unreadable frame should not make true of a robot
whose lowstate is arriving.

The coercion is *per field*. A decoder that builds its whole record inside one `try` loses
every field a message carried because one of them stopped reading, and the staleness that
leaves behind looks like a dropped wire rather than one renamed field. The record keeps the
key either way and lets the value be `None`, so a consumer always gets an answer and the
answer can be "the robot did not report this"; a frame in which *nothing* read is simply not
cached.

It matters most where a reading is also a *command* source. `BoosterDriver.send_action` holds
every uncommanded upper-body joint at its last observed position, so the T1's `joints` vector
is what the next `LowCmd` writes: a defaulted `0.0` there is finite, full-width, cached and
commanded - eight arm joints driven to exactly zero from a frame that carried no positions at
all. All-or-nothing matters for the same reason, because `held_q` is indexed by slot and a
vector short one element renumbers every slot after the gap.

## Declaring one

`list_native_drivers()` reports every robot that has one, and is the answer to "is my robot
driven natively" - the refusal below lists them only as of the day it was captured. Asking for
a driver that is not there is refused, never quietly substituted:

```python
>>> Robot("xarm7", mode="real", driver="strands")
ValueError: No native driver is registered for 'xarm7', so driver='strands' cannot build
it. Robots with a native driver: aloha, dynamixel_2r, fr3, fr3_v2, hope_jr, koch, lekiwi,
microduck, open_duck_mini, panda, reachy_mini, robotiq_2f85, robotiq_2f85_v4, so100,
so101, trossen_wxai, unitree_g1, unitree_go2, ur10e, ur5e, vx300s, wx250s. Either use
driver='lerobot' (today's default, which builds it through lerobot) or
register one with strands_robots.drivers.register_native_driver().
```

A robot may also declare its driver in the registry, so a caller needs no `driver=` at all:

```json
"unitree_g1":  {"hardware": {"lerobot_type": "unitree_g1", "driver": "strands"}}
"reachy_mini": {"hardware": {"driver": "strands"}}
```

`lerobot_type` is independent of `driver`. The G1 declares one because lerobot also
has a class for it, so `driver="lerobot"` remains a usable fallback. The Reachy Mini
declares none: lerobot has no robot type for it, so the native driver is the only way
to reach it and `driver="lerobot"` is refused by name.

A robot that declares neither - the UR arms, for instance - still resolves to the
default, so `driver="strands"` is how its native driver is reached, and the refusal
`driver="lerobot"` produces names that driver rather than listing lerobot's types.

`hardware.driver` is optional and validated when the registry loads: a value that is not a
driver name is refused there, naming the robot, rather than being read as "no preference".

## See also

- [Robot factory](../getting-started/robot-factory.md) - every `Robot()` kwarg.
- [Reachy Mini daemon link](reachy-mini.md) - a native driver's bring-up, end to end.
- [Robot control](robot-control.md) - driving a real robot once it is built.
