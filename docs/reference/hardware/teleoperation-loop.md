---
description: The teleoperation control loop - attach_teleop, teleoperate and detach_teleop parameter domains, what one tick does, the per-joint slew bound, and which teleop/robot pairings are zero-config.
---

# The teleoperation loop

What the mixin methods accept, what they refuse, what one tick applies, and
which pairings need a `map_fn`. Building the devices and the recipes that use
them are on [Teleoperation](teleoperation.md).

## Mixin API

Every hardware `Robot` and `Simulation` host exposes:

| Method | What |
|--------|------|
| `attach_teleop(device_or_spec, *, name=None, method=None, map_fn=None, **kwargs)` | Register an input stream (lazy - no hardware touched). `device_or_spec` is a built teleop instance **or** a type string built via `Teleoperator(**kwargs)`. |
| `teleoperate(*, names=None, robot_name=None, hz=50.0, publish=False, block=False, duration=None)` | Run the control loop. |
| `detach_teleop(name=None)` | Remove one (or all) attached streams. Stops the loop before touching a device when the detach would leave it with nothing to drive, and refuses with `detached: []` if that loop does not stop. |
| `stop_teleoperate()` | Stop the loop, any mesh publishers, and disconnect devices. Reports `status="error"` with `stopped: false` when the loop outlasts its 3 s join budget - the devices are left connected rather than torn down mid-write, and a second call re-joins the same loop. Called on the loop's own thread, there is nothing to join, and it stops the publishers and disconnects as any clean stop does. |

### `attach_teleop`

- **`name`** - stable key for this stream (used in `teleoperate(names=[...])`,
  mesh topics, `detach_teleop`). Defaults to the device's `id`, else type.
- **`method`** - input-method label (`"arm"`, `"gamepad"`, `"keyboard"`,
  `"phone"`); auto-derived from the type when omitted.
- **`map_fn`** - optional `(action: dict) -> dict` applied **before**
  `send_action`. The bridge for cross-vocabulary teleop (e.g. EE deltas →
  joint `.pos`, or leader joint names → sim actuator names). Identity by
  default.

### `teleoperate`

- **`names`** - subset of attached streams to run (`None`, the default, runs
  every attached stream). Read by membership, so only `None` means "all", and the
  list is held to the shared name-list domain: an empty list, a single string
  (`names="leader"`), a repeated name and a one-shot iterator are each
  **refused** before any device is connected.
- **`robot_name`** - target robot in a multi-robot simulation world. Read
  inside the loop (`send_action(merged, robot_name=...)`, every tick), so it is
  graded at the door like `hz` and `duration`: a name the host cannot route to is
  **refused** - with the close-match message `send_action` would have given -
  before any device is connected. A hardware `Robot` wraps one device and ignores
  the argument.
- **`hz`** - control-loop rate (default `50.0`).
- **`publish`** - also publish each device to the mesh via the host's
  `start_teleop_publish` so remote peers can follow. Requires a hardware
  `Robot` host.
- **`block`** - run inline until `duration` elapses / Ctrl+C (`True`) vs
  background thread (`False`, default).
- **`duration`** - auto-stop after N seconds (`None` = until stopped).
  Measured on a monotonic clock, from the end of setup, and reported as
  `elapsed_s`: a wall-clock correction mid-session does not move the budget, and
  connecting the devices does not spend it. `teleoperate(block=False)` returns
  once that setup is done - about two seconds on the first session in a
  process.

Each tick: poll every selected device's `get_action()` → apply its `map_fn` →
**merge** (last-wins on key conflict, with a one-time warning) → check the
merged frame against the **per-joint slew bound** → apply via
`self.send_action(merged, robot_name=...)`.

The slew bound is `STRANDS_TELEOP_SLEW_ABS` (default 500 units/second): the
fastest any single joint may be commanded to travel. The local loop carries its
own default because the shipped SO hardware speaks driver units - arm joints in
degrees, gripper in 0-100 - while the mesh receive path's
`STRANDS_MESH_INPUT_SLEW_ABS` (8π) is radian-scoped. Either bound is above what a
leader arm's servos can produce, so what trips one is a frame no arm could have
generated - an encoder glitch, a USB re-enumerate reading full-scale. Such a
frame is **refused and counted** in `slew_rejected` rather than clamped toward
the commanded value. The bound is a speed measured from each joint's last applied
value, so the allowance grows while a joint is still and a refused stream resumes
by itself with no resync step.

A device that stops reporting keeps its place. When a teleoperator returns `{}`
for a while - a disconnect, a USB re-enumerate - the loop still applies the other
attached devices' frames, and the quiet device's joints keep their last applied
value as their baseline - so its first read back is measured against where it
actually left the follower, and a full-scale one is refused like any other
over-speed frame.

Refusals are not errors, but a session with any of them does not report
`success`, so a device whose units the bound does not expect cannot look like a
clean run while moving nothing - widen the bound for those.

### `detach_teleop`

- **`name`** - which attached stream to remove. `None` (the default) detaches
  every one; any other value names a single stream. Read by membership, like
  `teleoperate(names=)`, so a value naming no attached stream is refused rather
  than widened to the whole set - `detach_teleop("")` reports
  `No teleop named ''.` and leaves every stream attached, which matters mid-
  session because a detach that leaves nothing to drive also stops the loop.
- **Order** - when the detach would leave the loop with nothing to drive, the
  loop is joined *before* any device is disconnected. If that join fails the
  whole detach is refused: `status="error"` with `detached: []`, every stream
  left attached and connected, and the reason forwarded from
  `stop_teleoperate`.

## Action-key compatibility

A pairing is **zero-config** only when the teleop's action keys match what the
robot's `send_action` consumes:

| Teleop | Robot | Keys | Config |
|--------|-------|------|--------|
| `so101_leader` | `so101_follower` | `{motor}.pos` | identity ✅ |
| `keyboard_rover` | `earthrover_mini_plus` | `linear_velocity`, `angular_velocity` | identity ✅ |
| `gamepad` | `lekiwi` | base velocities | identity ✅ |
| `keyboard_ee` | `so101` (joint) | EE deltas → `.pos` | needs `map_fn` ⚠️ |
| `so101_leader` | `earthrover` | `.pos` → velocity | needs `map_fn` ⚠️ |

The merge does **not** auto-convert `.pos` ↔ `velocity`. Cross-vocabulary
pairings supply a `map_fn` - that hook exists exactly for this.

> For a wheeled rover use **`keyboard_rover`** (WASD → velocity).
> Plain `keyboard` / `keyboard_ee` emit joint / EE deltas, not base velocities.

## See also

- [Teleoperation](teleoperation.md) - the `Teleoperator()` factory and the recipes.
- [Robot control](robot-control.md) - hardware lifecycle + mesh teleop primitives.
- [Mesh networking](../mesh.md) - the transport layer.
