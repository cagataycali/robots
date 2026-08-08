### Fixed: a non-finite action stream is no longer diagnosed as a near-zero one

`ZeroActionMonitor` classifies the per-step `max(abs(action))` behind the
`lerobot_local` "policy runs but the robot does not move" diagnostics. It had two
buckets - real motion and near-zero - and forced a third case into the wrong one.
`nan` compares `False` against every threshold, so a poisoned action stream
advanced the near-zero streak and produced the near-zero warning byte-identically
to a genuinely dead policy, prescribing the obs/rename pipeline for a fault that
pipeline cannot cause; because a single `nan` component makes
`np.abs(...).max()` `nan`, a vector carrying five real commands was reported as
emitting no command at all. `inf` compares `True` and cleared the streak instead,
so an action every backend refuses outright went entirely unreported. A
non-finite magnitude is now reported as its own fault, on the first such step,
with a message that names what was measured and points at the checkpoint's
normalization statistics and the observation values rather than at `obs_rename`.

The two constructor knobs shared the root cause: their bare comparisons
(`threshold < 0`, `patience < 1`) are `False` for `nan`/`inf` and let a `bool`
through as `1`, so a `threshold` of `nan`/`inf`/`True` made the watchdog fire on
a healthy policy and a `patience` of `nan`/`inf` silently disabled the warning
the class exists to emit. Both now delegate numeric-ness, `bool` and finiteness
to the shared numeric rules and are normalized to their declared types. The
near-zero contract, including a `threshold` of `0`, is unchanged.
