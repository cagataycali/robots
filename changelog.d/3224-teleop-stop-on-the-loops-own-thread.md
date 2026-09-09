### Fixed: `stop_teleoperate` reached on the loop's own thread still releases the devices

`Robot.__del__` calls `cleanup()`, and `cleanup()` calls `stop_teleoperate()`.
The teleop loop's `Thread` target is a closure over the robot -- `teleoperate`
builds `loop = lambda: self._teleop_loop(...)` and hands it to
`Thread(target=loop)` -- so that closure holds the last reference to the robot
once the caller drops its handle. When the body returns, `Thread.run` releases
it, and the finalizer, with the whole terminal teardown behind it, runs *on the
teleop thread*.

`thread.join()` there raises `RuntimeError: cannot join current thread`, which
leaves `stop_teleoperate` from the middle of the method. Everything after the
join is skipped: `_stop_publishers()` does not run, the attached teleoperators
are not disconnected, and the thread handle is not cleared. `cleanup()` catches
it, warns, and -- because a raise leaves the outcome unknown rather than
positively reporting a live loop -- goes on to close the robot's own devices and
log `cleanup completed`.

Measured on a one-camera arm driven by a leader holding its own port, with both
device nodes' exclusivity modelled:

| teardown route | publishers stopped | leader `disconnect()` | leader port |
| --- | --- | --- | --- |
| finalized on its own teleop thread | no | 0 | held |
| explicit `cleanup()` (control) | yes | 1 | released |

So the robot's own bus and cameras were released and the *teleoperator's* port
stayed held for the life of the process, under a `cleanup completed` report.
That asymmetry is why it went unnoticed: the leak is in the one device group
whose disconnect lives inside `stop_teleoperate` rather than in
`_disconnect_devices`, and `cleanup()`'s docstring promises that after it runs
"no device node stays held".

A self-join is knowable rather than unknown, which is what makes carrying on
the right answer instead of a guess: `_teleop_loop` never calls this verb, so
the only way control reaches it on that thread is after the body has already
returned, and there is no *other* thread that could still be writing -- so the
failed-join branch's reason for leaving the devices connected cannot apply. The
join is skipped, the rest of the teardown runs, and the handle is cleared as a
real join clears it. The decision `cleanup()` makes about the robot's *own*
devices is deliberately unchanged: it closed them on the raise path and it
closes them now.

The failed-join branch added in "report whether `stop_teleoperate` actually
stopped the loop" is untouched -- a leader whose `get_action()` outlasts the 3 s
budget still yields `status="error"` with `stopped: false`, still keeps its
devices connected, and still keeps the handle for a second call to re-join.

The finalizer's records are also how this surfaced. A finalizer runs at
garbage-collection time, so its two records land in whatever code happens to be
executing. Three cells that meant "this function warned about nothing" asserted
`caplog.text == ""`, which grades every record in the process, at every level,
for the whole test -- so a finalizer's report on another thread read as the
subject speaking. They now grade the records of the logger they name.
