### Fixed: `detach_teleop` joins the teleop loop before it disconnects what that loop reads

`_teleop_loop` indexes `self._teleops[name]` on every tick, and `detach_teleop`
removes entries from that mapping. It removed them - and disconnected each
device - and only then stopped the loop, discarding the outcome. So a detach that
would leave the loop with nothing to drive tore the leader down under a loop that
was still parked reading from it, and answered `status="success"`. The method's
own docstring already said "Stops the local loop first".

`stop_teleoperate` declines exactly that teardown for itself: on a leader whose
`get_action()` is still blocking it answers `status="error"` with `stopped: False`
and deliberately leaves the devices connected, because "Tearing the bus down
under a live writer is what `G1Driver.cleanup` refuses for the same reason". The
same state through each verb, on one wedged leader:

```
verb                  status   stopped   leader disconnected   loop alive
stop_teleoperate()    error    False     no                    yes
detach_teleop()       success  -         YES                   yes
```

Three things were wrong and the order repairs all three. The verdict: the detach
answered `success` where its own callee answered `error`, and carried no `json`
block, so `_stop_reported_stopped` read `True` on it and a caller had no way to
tell. The teardown: the leader was disconnected, undoing the protection
`stop_teleoperate` had just decided to apply. And the diagnosis: because the
entries were popped first, the refusal that got discarded described "the loop is
still polling `[]`" while claiming "The devices are left connected", which this
caller had already falsified.

There are two routes in and the second is the one an operator is told to take.
`stop_teleoperate` clears `_teleop_running` *before* it joins and keeps the thread
handle when the join fails, so after a failed stop - the state whose message says
to call it again - the session flag alone reports an idle session while the loop
is still writing. The guard therefore tests the thread handle as well as the flag;
keyed on the flag alone it sees the first route only.

```
                                          before      after
detach under a running loop: status       success     error
  leader disconnected                     yes         no
  streams left attached                   0 of 1      1 of 1
  detached: [] reported                   (no json)   yes
detach after a failed stop: status        success     error
  leader disconnected                     yes         no
forwarded diagnosis names the device      devices: [] devices: ['leader']
```

A live loop now refuses the whole detach: `status="error"` with `detached: []`,
every stream left attached and connected so a later call re-joins that same loop,
and the callee's reason forwarded rather than re-invented. The outcome is read
through the shared `_stop_reported_stopped` helper from a single call site that
precedes the pop - on `stopped` rather than `status`, because the status is
derived from the session counters and a session whose every frame errored reports
`"error"` after a perfectly clean join, which would refuse a detach that is safe.

Unchanged: a clean stop still detaches and disconnects exactly as before, a name
matching no attached stream still gets the not-found refusal, and `detach_teleop("")`
still leaves a running session alone. A partial detach - one of several devices,
leaving the loop with something to drive - is deliberately out of scope and pinned
unchanged; whether the loop should be re-selected or the detach refused there is a
contract question about multi-device sessions rather than this ordering defect.
