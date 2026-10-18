### Changed: the e-stop latch names its stopper and carries a wall clock; `resume` can pin the stop it clears

`Latch` now records `stopped_by` and `engaged_wall` (`time.time`) beside the monotonic `engaged_at`,
so a stop can be attributed and placed on a calendar in an audit record, and `estop(by=...)` carries
the operator onto the `estop` event. `resume(..., stop=N)` refuses to clear an engagement whose
`count` is not `N`, so an operator's yes typed for the stop they saw never clears a later, different
stop. Both arguments are additive: a local press and an unpinned resume behave as before.
