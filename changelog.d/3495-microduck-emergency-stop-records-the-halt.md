### Fixed: the Microduck's `emergency_stop` records the halt it just made

`motion_stopped` is the field an operator reads to decide whether a robot is safe
to approach. The Microduck reaches robotd's `robot.stop` from three verbs -
`stop`, `stop_task` and `emergency_stop` - and only the first two recorded it.
`emergency_stop` sent the identical request, robotd accepted it, the envelope
reported success, and `get_status` kept publishing `motion_stopped=False`. The
verb whose name says the halt is urgent was the one that did not report it, so an
operator who reached for the e-stop and then read the status was told the robot
was not stopped.

The latch is now set on every accepted halt, whichever verb issued it, and still
only on an accepted one - a stop robotd declined returns before the latch,
because reporting a halt that did not happen is the affirmative lie the flag
exists to avoid. `relax` and `enable_torque` are deliberately unchanged: they
de-energise rather than halt a commanded motion, which is a different fact.

| after an accepted halt via | wire | before | after |
| --- | --- | --- | --- |
| `stop()` | `robot.stop` | `motion_stopped=True` | `True` |
| `stop_task()` | `robot.stop` | `motion_stopped=True` | `True` |
| `emergency_stop()` | `robot.stop` | **`motion_stopped=False`** | `True` |
| any of the three, declined by robotd | `robot.stop` | `False` | `False` |

The rule the suite now grades is derived per driver rather than listed: for every
driver whose `get_status` publishes the flag, the methods that name its halt
constant are exactly the methods that record a halt. That equality covers both
directions - a halt verb that forgets to record, and a non-halt path that starts
claiming one - and it grades the Mini over `_PATH_STOP` and the Microduck over
`_M_STOP` without naming either verb list.
