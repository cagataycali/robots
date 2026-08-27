### Fixed: `cleanup()` does not close the devices under a teleop loop that is still writing

`stop_teleoperate` reports its join outcome honestly: a leader whose
`get_action()` is still blocking - a serial read on a wedged bus is the ordinary
case a join budget exists for - yields `status="error"` with `stopped: False`,
and that branch deliberately leaves the attached devices connected. Its
docstring says why, and names the precedent: "Tearing the bus down under a live
writer is what `G1Driver.cleanup` refuses for the same reason." `cleanup()`
called it, discarded the envelope, and closed the motors bus anyway - so the two
functions disagreed about the same state and `cleanup()` undid the protection
`stop_teleoperate` had just provided.

Closing there is worse than deferring on both of the counts `cleanup()` cares
about, which is what makes this a defect rather than a trade-off. The release
does not hold: `send_action` re-opens the robot lazily on a command that finds it
disconnected, the live loop's next write goes through it, and nothing remains to
close the port again because `cleanup()` is terminal and the executor is already
down - exactly the harm the comment above `_disconnect_devices()` describes, with
the teleop loop first in its own list of command sources. And the torque disable
is undone: `_disconnect_devices` prefers the driver's own `disconnect()` because
that is where torque disable and gripper release live, and the loop's write then
lands after it. Measured on a one-camera arm whose leader is wedged, with the
port's exclusivity modelled:

```
                                        before      after
stop_teleoperate() alone: bus closed    no          no
cleanup(): bus closed                   yes         no
cleanup(): port held afterwards         no          yes
cleanup(): driver disconnect() calls    1           0
teleop thread alive throughout          yes         yes
bus connect_calls after the live write  2           1
command lands after torque disable      yes         no
deferral reported                       (nothing)   ERROR + remedy
```

`cleanup()` now reads the outcome through a `_stop_reported_stopped` helper on
the envelope's `stopped` key and holds the devices when the loop is still
running, recording the reason and the remedy at ERROR - the only report it has,
since it returns `None`. The key is `stopped` rather than `status`, because the
status is derived from the session counters: a session whose every frame errored
reports `"error"` after a perfectly clean join, and a caller keying on it would
read a healthy teardown as a live loop. The software teardown is unchanged, a
clean stop still closes every device exactly once, and a `stop_teleoperate` that
raises leaves the outcome unknown and keeps the existing "a teleop teardown
failure must not block the rest of hardware cleanup" contract.
