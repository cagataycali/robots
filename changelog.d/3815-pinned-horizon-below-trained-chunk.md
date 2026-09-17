### Fixed: a re-query interval pinned below the trained chunk is named instead of silently degrading the rollout

`LerobotLocalPolicy._auto_detect_actions_per_step` corrects the default
`actions_per_step=1` up to the checkpoint's `config.n_action_steps` precisely
because re-querying more often than the model was trained to replay is out of
distribution - its own docstring says the shift "compounds every chunk". A
caller-pinned interval is deliberately never overridden, but the condition
guarding that branch was `!= 1`, i.e. "did the caller say anything", so a value
strictly *below* the trained chunk took the respected path with nothing logged -
the same out-of-distribution regime the default is corrected away from.

It now names it: the trained chunk, the pinned value, the consequence, and the
remedy that applies to the policy family - `rtc_enabled=True` for a
flow-matching checkpoint (RTC re-queries just as often but blends the
unexecuted tail of the previous chunk into the next one, so the seam is not a
discontinuity), otherwise leaving `actions_per_step` at the trained chunk. The
pinned value is still honored, a value at or above the trained chunk is silent,
and a caller who asked for RTC is not warned - there `rtc_execution_horizon`
owns the interval and `actions_per_step` stays the trained chunk.
