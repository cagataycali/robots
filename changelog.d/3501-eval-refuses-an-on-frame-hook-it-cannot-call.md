### Fixed: an `on_frame` hook that is not callable is refused, not absorbed frame by frame

`eval_policy`, `evaluate_benchmark`, `PolicyRunner.evaluate` and
`PolicyRunner.run` treat an exception from the caller's `on_frame` hook as
best-effort telemetry - logged, never fatal - which is right for a hook that
fails mid-episode and wrong for a value that can never be called. Measured on a
six-step evaluation whose hook was the integer `42`: `eval_policy` applied all
six actions and returned an envelope byte-identical to the healthy call, the
hook never ran, and the only trace was the same `TypeError` once per frame in
the log; `PolicyRunner.run` reported an error only after five applied actions,
blaming "silent dataset corruption" for a caller mistake visible at the door.

All four surfaces now apply the domain `run_policy` already applied to its
`observer`: a non-callable hook is refused before any action or inference, with
a structured error at the facades and a `ValueError` for a direct runner caller.
A hook that raises while running keeps the existing posture, including the
consecutive-failure watchdog on the rollout path.
