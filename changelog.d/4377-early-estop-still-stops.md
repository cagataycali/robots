### Fixed: an early e-stop still stops; a robot whose clock lags the operator is no longer driveable and unstoppable

The safety timing check refused any envelope more than 5 s ahead of the local
clock, and `cmd` envelopes carry no `t`, so a robot 30 s behind the operator
dispatched `set_joints` and refused the e-stop as "`t` in future" with nothing
on the operator's side to say so. A stop that is early is not a replay (a
replay is old): an estop ahead of the clock by less than the freshness window
now engages the lockout, WARNs with the issuer and the skew and audits
`estop_clock_skew`; beyond the window it is still refused. `resume` keeps the
strict forward-skew rule, since an early resume could pre-arm a replay.
