### Fixed: `eval_policy` and `evaluate_benchmark` stop on an `on_frame` hook that keeps failing

Both evaluation surfaces logged every `on_frame` exception at WARN and carried
on, so a recorder hook that raised on every step returned `status="success"`
over episodes it never recorded - the failure `run_policy`'s
`max_onframe_failures` watchdog exists to catch. Both now take
`max_onframe_failures` (same default of 5 and the same positive-integer domain
as `run_policy`), stop after that many failures in a row with
`status="error"`, and report the last failure in a new `onframe_error` result
field. The interrupted episode is not counted and its unsaved recording frames
are discarded, as for a physics divergence. Intermittent failures are still
tolerated, and `CooperativeStop` / `RecordingFrameError` keep their behaviour.
