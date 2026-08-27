### Fixed: getDaemonStatus reports the Reachy Mini driver's own verdict, not the daemon's

`ReachyMiniDriver` answers every RPC with the Device Connect envelope -
`{"status": "success", ...}` or `{"status": "error", "reason": ...}`. Five of its
REST RPCs carry the daemon's reply back to the caller and four of them nest it
under a key of their own (`{"status": "success", "result": result}`).
`getDaemonStatus` merged it into the envelope instead, and that one difference
cost the envelope two properties.

The daemon decided the verdict. Spread last, a reply carrying a `status` field
of its own replaced the driver's, so a daemon reporting `status="idle"` made a
healthy RPC answer `"idle"` - a value outside the envelope's two-value
vocabulary, which a caller branching on `status == "success"` reads as a call
that did not work. The presence path resolves the same collision the other way
and `SensorLoopsMixin._stamp_local_keys` gives the reason: a record must not
name something other than what the surface that built it decided.

A daemon that was never reached answered `success`. `reachy_transport.api`
reports every HTTP and connection failure as `{"error": ...}` rather than
raising - the driver already states this twice, in `__init__`'s port comment and
in `_stop_motion_impl`'s docstring, and the native `ReachyDriver` refuses that
shape when reading this same endpoint. Merged into the envelope, the reason
travelled beside a success verdict: the one RPC whose subject is whether the
daemon is up was the one that could not say it was down.

```
api() answered                                   before          after
{"motors_on": True, "freq": 100}                 success         success
{"state": "ready", "version": "1.0"}             success         success
{"status": "idle", "freq": 100}                  idle            success
{"status": "error", "detail": "motor fault"}     error           success
{"error": "[Errno 111] Connection refused"}      success         error
{"error": "service unavailable", "code": 503}    success         error
[1, 2, 3]                                        TypeError       error
"ok"                                              TypeError       error
```

The failure is reported in the error envelope the class already uses for a
read-only RPC rather than raised, because the `RuntimeError`
`_stop_motion_impl` raises guards a stop, where a caller acting on a false
success stops nothing; this one answers a question and its callers already
branch on `status`. A body that decodes to a JSON array or scalar is reported
for the same reason: spreading a non-mapping raised `TypeError` out of a method
whose whole contract is the envelope.

`status` is re-asserted after the merge, so every payload key still reaches the
caller at the top level and the healthy reading is unchanged.
