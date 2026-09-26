---
description: Robot(name, mode, backend, urdf_path, cameras, position, data_config, mesh, peer_id, orientation, keyframe, driver, tool_name, **kwargs) - the full signature with every kwarg explained.
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
| `cameras` | dict | `None` | Real-hardware camera config, attached by the lerobot driver. **Rejected in `mode="sim"`**, and rejected for a native driver that does not open cameras - both raise `ValueError`. |
| `position` | list | `None` | Robot position `[x, y, z]` in sim world. Ignored when `mode="real"` (reported at debug level). |
| `data_config` | str | `None` | GR00T data_config name. Honoured in both modes: `mode="sim"` defaults it to the canonical robot name, `mode="real"` forwards it to the hardware driver, which carries it into the `policy_config` a policy is built with. |
| `mesh` | bool \| None | `None` | Join the Zenoh fleet mesh. `None` consults `STRANDS_MESH`, which leaves it **off** unless set to `true`/`1`/`yes` - pass `mesh=True` to opt in per robot. |
| `peer_id` | str | `None` | Stable mesh peer id. Auto-generated if omitted. |
| `orientation` | list | `None` | Robot base orientation `[w, x, y, z]` in sim world. Ignored when `mode="real"` (reported at debug level). |
| `keyframe` | str \| int | `None` | Spawn in a model `<keyframe>` pose (name or index) instead of the zero configuration. Ignored when `mode="real"` (reported at debug level). |
| `driver` | str | `"auto"` | Which implementation drives a real robot: `"auto"` / `"lerobot"` / `"strands"`. Checked in every mode; only `mode="real"` acts on it (sim reports it as ignored at debug level). See [Choosing a driver](#choosing-a-driver). |
| `tool_name` | str | `None` | The name the agent sees this robot under. `None` keeps the default - `"<name>_sim"` in sim, the canonical robot name on hardware - so two `Robot("so101")` in one `Agent` collide at registration. Name each one (`tool_name="left_arm"`) to put a bimanual pair, or a real arm beside its sim twin, in one agent. Letters, digits, `_` or `-`, at most 64 characters; anything else raises `ValueError` before the backend builds. |
| `**kwargs` | | | Forwarded to the backend or driver constructor. `mode="real"` grades the name and raises `ValueError` on one it does not know: the lerobot driver against the robot's config dataclass plus the forwardable list below, a native driver against its own constructor parameters (`driver='strands'`, so `prot=` is refused naming the keyword and that driver's roster). A sim keyword the backend does not recognize is still ignored. |

## Name resolution

```python
from strands_robots.registry import resolve_name

resolve_name("SO-100")    # 'so100'
resolve_name("franka")    # 'panda'
resolve_name("g1")        # 'unitree_g1'
```

Case-insensitive, hyphens/underscores interchangeable. That fold is
`registry.normalize_robot_name`, and it is the rule the registry is keyed by: a
canonical name is stored folded and an alias is keyed folded, so an alias
declared `"My-Arm"` answers `my_arm`, `MY-ARM` and `My-Arm` alike. Two aliases
that fold to one key are one alias, and `register_robot` refuses an alias that
folds onto another robot's name or alias. Full alias map in
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
against lerobot's own choice registry. Every backend lerobot ships is therefore attachable
(`opencv`, `intelrealsense`, `zmq`, `reachy2_camera`), as is any installed
`lerobot_camera_*` plugin, and the remaining keys are the fields of the class the
`type` resolves to. Note the registered name for Intel RealSense is
`intelrealsense`, not `realsense`; an unregistered `type` raises `ValueError`
listing the registered ones. `fps`, `width` and `height` are common to every
backend and default to 30/640/480 when unset - a vendor SDK the backend needs
(`pyrealsense2` for `intelrealsense`) is required when the device is opened, not
when the config is built.

Cameras are attached by the **lerobot** driver. See
[Native drivers](../hardware/native-drivers.md).

`control_frequency` (Hz) sets the control loop's per-action period,
`1 / control_frequency` - the only throttle between two servo commands. It must be a
positive finite number: `0`, a negative rate, `nan` or `inf` raises `ValueError` at
construction, before the serial port is opened. This is the same domain the simulation applies to `run_policy`'s
`control_frequency`, so a rollout rehearsed in sim is honored identically on hardware.

Forwardable kwargs: `port`, `robot_ip`, `kp`, `kd`, `default_positions`, `control_dt`,
`is_simulation`, `gravity_compensation`, `controller`, `calibration_dir`, `mock`,
`use_degrees`, `max_relative_target`, `disable_torque_on_disconnect`.

Forwardable values are passed to the driver as given, because their accepted domains are
robot-specific. `max_relative_target` is the exception: it caps how far each commanded goal
position may move from the joint's present position, so it must be a positive finite number
(or a mapping of motor name to one). `0`, a negative limit, `nan`, `inf`, a bool or a
non-numeric value raises `ValueError` when the config is built, before the serial port is
opened. An `int` limit is
normalized to `float` so it reaches the motors. Omit the parameter (or pass `None`) to leave
the clamp disabled.

## Choosing a driver

`mode="real"` builds a driver. By default that is the lerobot one, which wraps a lerobot
`RobotConfig` and serves most of the registry. A robot lerobot cannot model needs a native
driver; a robot it *can* build may have one too, and then `driver=` decides.

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
coverage["so101"], coverage["vx300s"], coverage["sawyer"]
# (('lerobot', 'strands'), ('strands',), ())

sim_only = [name for name, drivers in coverage.items() if not drivers]
```

`so101` is reported as both and `resolve_driver("so101")` returns `"lerobot"` - coverage is
what *can* build a robot, resolution is what *does*. `vx300s` has no lerobot robot type and no
`hardware` block at all, so its native driver is the only one that can build it and the two
declaration readers - `list_robots(mode="real")` and the `Real` column of
`format_robot_table()` - leave it out. This join reads what is registered, and is the wider
answer. An empty tuple is the driver gap: `sim_only` is every robot `mode="real"` has nowhere
to go for, derived on each call rather than maintained by hand.

What a native driver is, the contract one satisfies, and how a robot declares one are on
[Native drivers](../hardware/native-drivers.md).

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
- [Native drivers](../hardware/native-drivers.md) - the `driver="strands"` contract.
