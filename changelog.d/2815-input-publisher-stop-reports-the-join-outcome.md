### Fixed: `InputPublisher.stop` reports whether the publish loop actually stopped

`threading.Thread.join(timeout=...)` returns `None` whether or not the thread
finished, so the liveness read after it is the only thing that distinguishes a
stopped loop from one that outlasted the budget. `InputPublisher.stop` cleared
the session flag, joined with a 2 s budget and never took that read: it then
logged `input publisher stopped` unconditionally and returned `stats`, whose
`running` key it had already set to `False`. A teleoperator whose `get_action()`
was still blocking - a serial read on a wedged bus is the ordinary case a join
budget exists for - was therefore reported as stopped while its loop was still
free to put another frame on `strands/{peer}/input/{device}`, which is an
actuator command every subscribed follower mirrors. Measured on a publisher
whose leader blocks inside `get_action()`:

```
                                        before      after
stop() elapsed                          2.00s       2.00s
stats running                           False       False
stats thread_alive                      (absent)    True
loop thread still alive after the call   True        True
log level for the outcome               INFO        WARNING
log claims "publisher stopped"          yes         no
second stop() can re-join the loop      no          yes
frame published after stop() returned   1           1
```

The last row is the consequence the stop claim hid: a frame reaches subscribers
after the caller has been told the publisher stopped. Nothing in the returned
stats could say so, because `running` is the session flag `stop` clears *before*
it joins - so a caller polling afterwards was told `running=False` about a loop
that was still publishing. `thread_alive` reads the publish thread itself, and
the two now differ for exactly as long as a loop outlives the stop that asked it
to exit.

`running` keeps its meaning rather than being widened, because the loop's own
`while` reads that flag; the join outcome is reported beside it. This mirrors
`TeleopMixin.stop_teleoperate`, whose status verb gained `thread_alive` for the
same reason, and `G1Driver.cleanup`, which refuses to tear a transport down
under a live writer.

A publisher whose join timed out also stays stoppable. The guard now admits a
call that has a live thread left to join, where returning early on the session
flag alone left the only handle to that loop unreachable through `stop()`. The
2 s budget is named `_INPUT_JOIN_TIMEOUT_S` so the docstring, the warning text
and the tests read one value.
