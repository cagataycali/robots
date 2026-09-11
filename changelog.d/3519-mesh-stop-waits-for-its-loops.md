### Fixed: leaving the mesh waits for the loops that publish on it

`Mesh.start` launches nine background sensor loops - heartbeat, state, and the
seven `SensorLoopsMixin` loops - and ten with camera publishing enabled,
collecting each on a roster. `Mesh.stop` set the stop flag and then immediately
undeclared the subscribers, dropped the session reference and logged `off mesh`,
without waiting for any of them. The roster was appended to at four sites and
read at none.

Each loop is paced on a shared stop event and notices a stop within 10ms of its
next tick boundary, but a tick already inside `publish()` is not interrupted by
the flag. So `stop()` returned with all nine loops still running, and an in-flight
tick landed on the wire *after* the peer announced it had left, through a session
reference the peer had already given back - a publish on a session whose last
reference may have closed it. The same gap made a stopped peer's loops able to
publish into whatever the process wired up next.

`stop()` now joins the loops before releasing what they publish through. The
budget is a named `LOOP_JOIN_TIMEOUT_S` spent across the loops rather than per
loop, since they wind down in parallel and a per-loop budget would let one wedged
driver read cost nine times as long. A loop that outlasts it is still free to
publish once more, so it is named at WARNING with the count and the budget rather
than the stop being announced as having happened - the same posture the mesh input
publisher and the teleop mixin already take for the single loop each of them owns.
