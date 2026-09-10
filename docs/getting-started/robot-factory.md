---
description: Robot(name, mode, backend, urdf_path, cameras, position, data_config, mesh, peer_id, orientation, keyframe, **kwargs) - the full signature with every kwarg explained.
---

# Robot factory

`Robot(...)` returns a `Simulation` or `HardwareRobot` based on `mode`.

```python
from strands_robots import Robot

robot = Robot("so100")               # Simulation (default, safe)
robot = Robot("so100", mode="real")  # HardwareRobot
robot = Robot("so100", mode="auto")  # probes USB, falls back to sim
```

## Parameters

| Param | Type | Default | What |
|-------|------|---------|------|
| `name` | str | required | Catalog name or alias. Resolved via `registry/robots.json`. |
| `mode` | str | `"sim"` | `"sim"` / `"real"` / `"auto"`. Overridden by `STRANDS_ROBOT_MODE`. |
| `backend` | str | `"mujoco"` | Sim backend. Ignored when `mode="real"`. |
| `urdf_path` | str | `None` | Explicit MJCF/URDF path - bypasses registry. Ignored when `mode="real"` (reported at debug level). |
| `cameras` | dict | `None` | Real-hardware camera config. **Rejected in `mode="sim"`** - raises `ValueError`. |
| `position` | list | `None` | Robot position `[x, y, z]` in sim world. Ignored when `mode="real"` (reported at debug level). |
| `data_config` | str | `None` | GR00T data_config name. Honoured in both modes: `mode="sim"` defaults it to the canonical robot name, `mode="real"` forwards it to the hardware driver, which carries it into the `policy_config` a policy is built with. |
| `mesh` | bool \| None | `None` | Join the Zenoh fleet mesh. `None` consults `STRANDS_MESH`, which leaves it **off** unless set to `true`/`1`/`yes` - pass `mesh=True` to opt in per robot. |
| `peer_id` | str | `None` | Stable mesh peer id. Auto-generated if omitted. |
| `orientation` | list | `None` | Robot base orientation `[w, x, y, z]` in sim world. Ignored when `mode="real"` (reported at debug level). |
| `keyframe` | str \| int | `None` | Spawn in a model `<keyframe>` pose (name or index) instead of the zero configuration. Ignored when `mode="real"` (reported at debug level). |
| `driver` | str | `"auto"` | Which implementation drives a real robot: `"auto"` / `"lerobot"` / `"strands"`. `"auto"` honours the robot's registry `hardware.driver` and otherwise builds the lerobot driver. Checked in every mode; only `mode="real"` acts on it (sim reports it as ignored at debug level). See [Choosing a driver](#choosing-a-driver). |
| `**kwargs` | | | Forwarded to the backend or driver constructor as given. A name it does not recognize is ignored, not refused, so check the spelling against the forwardable list below. |

## Name resolution

```python
from strands_robots.registry import resolve_name

resolve_name("SO-100")    # 'so100'
resolve_name("franka")    # 'panda'
resolve_name("g1")        # 'unitree_g1'
```

Case-insensitive, hyphens/underscores interchangeable. That fold is
`registry.normalize_robot_name`, and it is the rule the registry is keyed by, not
just the rule queries pass through: a canonical name is stored folded and an
alias is keyed folded, so an alias declared `"My-Arm"` answers `my_arm`,
`MY-ARM` and `My-Arm` alike. Two aliases that fold to one key are therefore one
alias, and `register_robot` refuses an alias that folds onto another robot's name
or alias rather than letting it resolve to that robot. Full alias map in
`registry/robots.json`.

## Real hardware

```python
robot = Robot(
    "so100",
    mode="real",
    cameras={
        "wrist": {"type": "opencv", "index_or_path": "/dev/video0"},
        "top": {"type": "intelrealsense", "serial_number_or_name": "819312071961"},
    },
    port="/dev/tty.usbserial-A50285BI",
    control_frequency=50.0,
)
```

Each `cameras` entry is a serialized lerobot `CameraConfig`, so `type` is resolved
against lerobot's own choice registry - the same registry the robot name itself is
resolved against. Every backend lerobot ships is therefore attachable
(`opencv`, `intelrealsense`, `zmq`, `reachy2_camera`), as is any installed
`lerobot_camera_*` plugin, and the remaining keys are the fields of the class the
`type` resolves to. Note the registered name for Intel RealSense is
`intelrealsense`, not `realsense`; an unregistered `type` raises `ValueError`
listing the registered ones. `fps`, `width` and `height` are common to every
backend and default to 30/640/480 when unset - a vendor SDK the backend needs
(`pyrealsense2` for `intelrealsense`) is required when the device is opened, not
when the config is built.

`control_frequency` (Hz) sets the control loop's per-action period,
`1 / control_frequency` - the only throttle between two servo commands. It must be a
positive finite number: `0`, a negative rate, `nan` or `inf` raises `ValueError` at
construction, before the serial port is opened, rather than leaving the loop free-running
against the arm. This is the same domain the simulation applies to `run_policy`'s
`control_frequency`, so a rollout rehearsed in sim is honored identically on hardware.

Forwardable kwargs: `port`, `robot_ip`, `kp`, `kd`, `default_positions`, `control_dt`,
`is_simulation`, `gravity_compensation`, `controller`, `calibration_dir`, `mock`,
`use_degrees`, `max_relative_target`, `disable_torque_on_disconnect`.

Forwardable values are passed to the driver as given, because their accepted domains are
robot-specific. `max_relative_target` is the exception: it caps how far each commanded goal
position may move from the joint's present position, so it must be a positive finite number
(or a mapping of motor name to one). `0`, a negative limit, `nan`, `inf`, a bool or a
non-numeric value raises `ValueError` when the config is built, before the serial port is
opened - a non-finite limit would otherwise disable the clamp with no signal, and a negative
one inverts it into a fixed-magnitude step that ignores the policy. An `int` limit is
normalized to `float` so it reaches the motors. Omit the parameter (or pass `None`) to leave
the clamp disabled.

## Choosing a driver

`mode="real"` builds a driver. By default that is the lerobot one - it constructs a lerobot
`RobotConfig` and wraps a lerobot driver, which is what most robots in the shipped registry
use. A robot lerobot cannot model needs a native driver, but the two are not exclusive - a
robot lerobot *can* build may have one as well, and then `driver=` decides which is used.
`list_native_drivers()` reports every robot that has one, and is the answer to "is my robot
driven natively" - the refusal below lists them only as of the day it was captured.

`driver=` selects a different one:

| Value | Builds |
|-------|--------|
| `"auto"` (default) | The robot's registry `hardware.driver` if it declares one, else the lerobot driver. |
| `"lerobot"` | The lerobot driver, explicitly. |
| `"strands"` | The native driver registered for this robot. |

`list_driver_coverage()` reports the join for every registered robot: which `driver=` values
can build it, and an empty tuple where neither can.

```python
from strands_robots.drivers import list_driver_coverage

coverage = list_driver_coverage()
coverage["so101"], coverage["vx300s"], coverage["panda"]
# (('lerobot', 'strands'), ('strands',), ())

sim_only = [name for name, drivers in coverage.items() if not drivers]
```

`so101` is reported as both and `resolve_driver("so101")` returns `"lerobot"` - coverage is
what *can* build a robot, resolution is what *does*. `vx300s` has no lerobot robot type, so its
native driver is the only one that can build it. An empty tuple is the driver gap: `sim_only`
is every robot `mode="real"` has nowhere to go for, derived on each call rather than
maintained by hand.

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

`baud_rate=` does not stay polymorphic. Every surface that opens a serial bus - the Feetech
and Dynamixel drivers, `FeetechBus`, and the `baudrate` of `serial_tool` and `pose_tool` -
holds it to one domain, a positive integer, and refuses anything else by name at
construction. pyserial takes the speed through its own `int()` and refuses only a negative,
so an ungraded value is *applied*: `2.7` opens the port at 2 baud, and `0` opens it
successfully at a speed no servo answers, after which every read times out exactly as an
unplugged arm does. A refusal at the line that states the speed is the only place that
reads as a caller mistake rather than as broken hardware.

The read window is the same shape. `timeout=` - on `FeetechBus`, and on `FeetechDriver`, which
forwards a caller's window to it - is how long a read waits for a servo's reply, and it is held
to a positive finite number at construction. pyserial takes `0`, `nan`, `inf` and `None`
verbatim, and each of them leaves the read looking at an empty buffer that the retry loop cannot
tell from a servo that never answered, so a healthy arm reports as motors that did not reply. A
keyword a driver *records* instead of forwarding fails the same way one layer earlier: the caller
lengthens the window, the bus opens at its default, and nothing says so.

A driver has **two** ways to halt its robot and they are not the same contract. `stop_task()`
returns a status envelope and decides an outcome, so that is what a caller reads. `stop()` is
the lifecycle hook and is annotated `-> None`, so it carries no verdict at all - which makes
its log the only place a halt it could not complete can be recorded. A `stop()` that
delegates to a halt verb must therefore read that verb's envelope and log a non-success,
naming what may still be moving; `strands_robots.drivers.halt_failure_detail` reads the
reason out of one. Discarding it returns from shutdown reporting the robot as stopped on the
one surface that has no way to say otherwise.

A driver that decodes its own telemetry decides, field by field, whether a reading exists. The
convention is to read every field as `getattr(msg, name, None)` and coerce it, because a *typed*
default is a well-formed value: a firmware that renames a field would publish a plausible constant
rather than an absence. `strands_robots.drivers.base` owns that coercion -- `telemetry_float`,
`telemetry_int`, `telemetry_float_list`, `telemetry_int_list` -- so the answer does not depend on
which driver asked. Each returns `None` for anything that is not a reading, including a `bool`
(`float(True)` is `1.0`, indistinguishable from a real one-percent pack) and a bytes-like value
(`str`, `bytes`, `bytearray`, `memoryview` all iterate, so a raw buffer would otherwise decode as a
vector of the wrong length). The vector readers are all-or-nothing and return a fresh list, so a
caller mutating the envelope does not race the callback thread's next write.

The rule covers the scalars a decoder sends back out, not only the ones it publishes. The Unitree
`mode_machine` is read from `rt/lowstate` and echoed on every `LowCmd_`, and the firmware drops a
frame whose layout id does not match the one it announced -- so an id that came from something other
than a number is a write the robot silently ignores. A bare `int()` is the wrong coercion for that:
`int(True)` is `1` and `int(False)` is `0`, both valid uint8 ids, so a flag on the field would be
indistinguishable from a reading. `telemetry_int` refuses both. A float is still truncated, because
that is the shared answer and a decoder stricter than its sibling on a value both accept is the
drift these functions exist to prevent.

A refused scalar leaves the cached value at the last reading that parsed, rather than clearing it.
That matters when a gate reads the cache: `mode_machine` gates every G1 motion write and its refusal
reads "lowstate has not delivered yet", which one unreadable frame should not make true of a robot
whose lowstate is arriving.

The coercion has to be *per field* for that to hold at the frame level too. A decoder that builds its
whole record inside one `try` loses every field a message carried because one of them stopped reading,
and the staleness that leaves behind looks like a dropped wire rather than one renamed field. The
record keeps the key either way and lets the value be `None`, so a consumer asking for a field always
gets an answer and the answer can be "the robot did not report this"; a frame in which *nothing* read
is simply not cached, for the same reason a refused scalar does not clear its own cache.

The rule matters most where a reading is also a *command* source. `BoosterDriver.send_action` holds
every uncommanded upper-body joint at its last observed position, so the T1's `joints` vector is what
the next `LowCmd` writes. A defaulted `0.0` there is finite and full-width, clears the "is this a
frame" guards, gets cached, and gets commanded -- eight arm joints driven to exactly zero from a frame
that carried no positions at all. Reporting the absence instead reaches the refusal the driver already
spells for a robot that has reported nothing yet. All-or-nothing matters for the same reason: `held_q`
is indexed by slot, so a vector short one element would renumber every slot after the gap and hold the
wrong joint at each of them.

Asking for a driver that is not there is refused, never quietly substituted:

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

A native driver reports what it cannot reach rather than raising. The Reachy Mini's
daemon transport is a standard-library-only module in the core distribution, so
nothing an extra installs decides whether it loads - but if it cannot be imported at
all, on a broken install or behind a shadowing module, the driver still builds,
registers and answers `get_status`, and every surface that would touch the daemon
returns a reason naming the module and the error instead:

```python
>>> Robot("reachy_mini", mode="real").connect_eagerly()
"cannot import strands_robots.device_connect.reachy_transport: No module named
'strands_robots.device_connect.reachy_transport'"
```

The reason stops at what it can establish. It prescribes no `pip install`, because no
install supplies a module that ships in the core distribution, and a remedy that
cannot help is worse than none - the same rule
[`require_optional`](https://github.com/strands-labs/robots/blob/main/strands_robots/utils.py) applies when it is told a module
arrives from a system package rather than an index.

The same reason arrives as `connect_error` in `get_status`, so a mesh peer for a Mini
whose transport will not load is still constructible and still reports why it is not
connected.

`hardware.driver` is optional and validated when the registry loads: a value that is not a
driver name is refused there, naming the robot, rather than being read as "no preference".

## Mesh

Mesh is opt-in, so a bare `Robot(...)` never starts Zenoh, ACL or e-stop machinery:

```python
sim = Robot("so100")
sim.mesh                     # None - never joined

sim = Robot("so100", mesh=True)   # per-robot on
sim.mesh.peer_id             # 'so100_sim-a1b2c3d4'
sim.mesh.alive               # True

# STRANDS_MESH=true          # process-wide on, for a bare Robot(...)
```

`STRANDS_MESH=false` is a kill switch: it keeps mesh off even where a caller passed
`mesh=True`. The environment never forces mesh on for a robot constructed with
`mesh=False`.

Mesh failure is non-fatal; `.mesh = None` if Zenoh unavailable.

## See also

- [Robot catalog](../robots/index.md) - 68 catalog names.
- [Architecture](../architecture.md) - factory in the module map.
- [Multi-robot mesh](../mesh.md) - mesh peer discovery.
