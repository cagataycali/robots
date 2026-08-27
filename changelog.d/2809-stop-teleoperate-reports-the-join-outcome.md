### Fixed: `stop_teleoperate` reports whether the teleop loop actually stopped

`threading.Thread.join(timeout=...)` returns `None` whether or not the thread
finished, so the liveness read after it is the only thing that distinguishes a
stopped loop from one that outlasted the budget. `TeleopMixin.stop_teleoperate`
joined with a 3 s budget and never took that read: it then set the thread handle
to `None`, disconnected every attached device, and answered
`_teleop_stats(blocking=False)`. A leader whose `get_action()` was still blocking
- a serial read on a wedged bus is the ordinary case a join budget exists for -
was therefore reported as stopped while its loop was still polling that leader
and writing to the follower, and the handle that was the only route to
discovering it had already been discarded. Measured on a session whose leader
blocks inside `get_action()`:

```
                                     before          after
stop_teleoperate status              success         error
payload stopped                      (absent)        False
loop thread still alive              True            True
thread handle after the call         None            <Thread ...>
device disconnect() calls            1               0
get_teleoperate_status thread_alive  (absent)        True
frame delivered after the call       1               1
```

The last row is the consequence the success claim hid: the follower is written to
after the caller has been told teleoperation stopped.

The session counters cannot express this. `_teleop_stats` derives its status from
them precisely so a dead teleop is not reported as healthy, and it distinguishes
three signatures - a softly failing follower, a dead leader, and a frame the slew
bound refused - but a session that ran cleanly until its leader wedged carries
healthy counters, so "the loop is still running" is outside that vocabulary. The
join outcome is now reported beside them rather than through them, in the shape
`G1Driver.stop_task` already documents: `status="error"` naming the budget with
`stopped=False` in the payload, so a caller cannot read "success" while the loop
still holds the wire.

Two consequences follow from reading the outcome. The devices are disconnected
only once the loop has joined, because tearing the bus down under a live writer
is what `G1Driver.cleanup` refuses for the same reason; and the handle is cleared
only on a real join, so a second `stop_teleoperate()` re-joins that loop instead
of reporting an idle session. `get_teleoperate_status` gained `thread_alive`
alongside `running`: `running` is the session flag, which the stop clears before
it joins, and without the second reading a caller polling after a stop that
reported `stopped=False` would be told `running=False` by the status verb.

A joined stop is unchanged - it still reports the counter-derived status, still
disconnects its devices, still clears the handle, and stopping an idle host is
still a no-op success. `stopped` is now present on both paths so a caller reading
it never has to tell an absent key from a false one. Of the seven timed thread
joins in the package, three already read liveness; the other three answer
different contracts (a `-> None` server teardown, a stats property, and a
recording flush whose window is one frame) and are deliberately untouched.
