### Fixed: refuse an `on_frame` failure tolerance the watchdog cannot count against

`run_policy(max_onframe_failures=...)` and `PolicyRunner.run(...)` read the
consecutive-failure ceiling straight into `consecutive_onframe_failures >= limit`
with no domain, so a value outside it did not resize the tolerance - it silenced
the mechanism whose own abort text reads "aborting episode to avoid silent
dataset corruption". Measured on a 100-step rollout whose `on_frame` hook raises
on every step, `nan` and `inf` made that comparison false for every counter
value: 100 of 100 frames lost, `status="success"`, and the abort never fired.
Both values also broke the per-failure warning that would otherwise have
reported the hook - it interpolates the limit with `%d`, so `logging` emitted its
own error instead and the operator was told nothing at all. `0` aborted on the
first failure exactly as `1` does while reporting "failed 0 times in a row";
`2.7` tolerated two failures and aborted on the third while reporting "failed
2.7 times in a row"; a string or a list leaked
`TypeError: '>=' not supported between instances of 'int' and 'str'` from inside
the hook's own exception handler, and only once the hook first failed.

The limit is now a positive integer or `None` - the same domain `n_steps`,
`max_steps` and `n_episodes` already carry on the same signature - reported
through the envelope on the facade and raised on `PolicyRunner`, which is
drivable directly. The refusal precedes any policy construction, so a bad limit
costs no weight download and no frame.
