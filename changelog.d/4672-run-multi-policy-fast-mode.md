### Added: `run_multi_policy(fast_mode=True)` runs the synchronized loop unpaced, as `run_policy` does

The multi-robot loop on MuJoCo and Isaac always paced itself to
`control_frequency` on the wall clock, so a headless batch rollout or a data
collection run could not go faster than real time the way `run_policy` can.
`fast_mode` is now a keyword argument on `SimEngine.run_multi_policy` and both
backends. It defaults to `False` (paced, as before), and a non-boolean value
such as `"false"` is refused before any step with the same message
`run_policy` gives.
