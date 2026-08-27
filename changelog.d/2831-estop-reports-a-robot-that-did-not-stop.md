### Fixed: an emergency stop the robot refused is reported instead of discarded

`RobotDeviceDriver.onEmergencyStop` called `self._robot.stop_task()` and dropped
the envelope. Two of the four robot surfaces the driver can wrap answer a stop
with an affirmative "I did not stop", and each was written deliberately to do
so. `G1Driver.stop_task` returns `status="error"` with `stopped=False` when its
control loop outlasts the join budget, and its own docstring says why: "so the
caller cannot read 'success' while the payload's own `running=True` says the
loop is still writing frames". `ReachyDriver.stop_task` reports a daemon that
refused the stop. This handler was the caller that read neither field.

So an operator's emergency stop was answered by a single WARNING line stating
that the stop was being attempted - the intent, not the outcome - with nothing
recorded above WARNING while the loop kept publishing. `Mesh.emergency_stop`
grades exactly that verdict for every peer it fans out to, logs one that did not
stop at CRITICAL, and names a hardware cutoff as the remedy, on the reasoning
`_peers_that_did_not_stop` states: an emergency stop is only as trustworthy as
its accounting. The same envelope reaching the mesh was reported and reaching
Device Connect was not, so an operator's accounting depended on which transport
the stop arrived over.

The refusal is now read and, when the robot reports one, logged at CRITICAL
carrying the source of the stop, the reason the robot itself gave, and the same
cutoff remedy the mesh path names. Both spellings are read, because they differ:
an explicit `stopped` flag is authoritative where a driver supplies one, and the
envelope's status answers for the drivers that report through `_refuse`, whose
envelope carries text and no flag at all. `teleop_mixin._stop_reported_stopped`
asks the same question of a `stop_teleoperate` envelope and is deliberately not
reused - it answers `True` for an envelope with no `json` block, which is right
there (nothing was teleoperating) and wrong here, because that is the shape a
refused `stop_task` arrives in.

The reader is conservative for the reason `_peers_that_did_not_stop` gives: only
an envelope that affirmatively reports a failure is flagged, because a false "did
not stop" on the safety path trains operators to ignore the warning. A robot that
stopped, one that had nothing to stop, one reporting an error beside a payload
that says the loop stopped, and a driver that returns nothing at all are all left
alone. The authorization guard still runs first, so reading the verdict does not
widen who can halt a task.

The three suites that already drive `onEmergencyStop` assert
`stop_task.assert_called_once()` or `robot.stopped is True` - that the call was
made, never what it answered - and their robot doubles are `MagicMock`s whose
`stop_task` returns a `MagicMock`, which carries no verdict to read. That is why
the discard was invisible.
