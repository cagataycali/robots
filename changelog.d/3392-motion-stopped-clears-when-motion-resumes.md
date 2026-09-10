### Fixed: the Mini's `motion_stopped` names the halt that still holds

`motion_stopped` is the field an operator reads to decide whether a robot is
safe to approach, and the two daemon drivers publish it under the same key so
the mesh renders both peers identically. `MicroduckDriver.send_action` clears it
once its intents are on the wire. `ReachyDriver` set it in `stop` and
`stop_task` and cleared it only in `connect_eagerly`, so after any stop the Mini
reported a halt for the rest of the session - through a commanded head pose,
through a played emotion, through the wake-up move - and only a reconnect made
it honest again.

A halt that has been superseded is the same affirmative lie as a halt that never
happened, which both suites already refuse, told one step later. The latch now
clears on every Mini path that commits motion to the wire - `send_action`,
`play_move`, `wake_up`, `goto_sleep` - and only when that path succeeded, so a
refused command leaves the halt standing. `set_motors` is deliberately not one
of them: torque returning is not the driver committing motion, and the
Microduck's `enable_torque` and `relax` do not clear it either.

| after `stop()`, then | before | after |
| --- | --- | --- |
| `send_action({"head_yaw": 20})` | **`motion_stopped=True`** | `False` |
| `play_move("happy")` | **`motion_stopped=True`** | `False` |
| `wake_up()` | **`motion_stopped=True`** | `False` |
| `goto_sleep()` | **`motion_stopped=True`** | `False` |
| `send_action(...)` the link refused | `True` | `True` |
| `set_motors("enabled")` | `True` | `True` |
