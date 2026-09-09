### Fixed: the rollout facades check their posture flags, not read them by truthiness

`SimEngine.run_policy` takes four flags that each select one of two postures -
`fast_mode` (pace the loop at `control_frequency` or run it unpaced),
`reset_between` (reset the scene between episodes or carry the end state over),
`wbc_install_torque_control` (install the WBC torque shim for the call or leave
the actuators alone) and `async_rtc` (overlap inference with actuation, drain
each chunk first, or `None` to resolve from the policy). `eval_policy` takes
`async_rtc`, MuJoCo's `start_policy` takes `fast_mode`, and the `run_policy`
agent tool forwards its own `fast_mode` into the facade. Every one of them was
read by truthiness, so `"false"`, `"no"`, `"off"` and `"0"` selected the posture
the word asks to skip, and `0`, `""` and `[]` took the other branch without
being a declared spelling of it. Measured: `run_policy(fast_mode="false")` ran
unpaced, `run_policy(n_episodes=2, reset_between=0)` started episode two from
wherever episode one left the arm, `run_policy(async_rtc="false")` reported
`rtc_async_enabled=True` beside a background inference thread the caller had
declined, and each reported `status="success"`.

Every flag is now held to the shared `boolean_flag_error` domain through
`SimEngine._validate_posture_flags`, the way the numeric knobs beside them
already are, ahead of robot resolution so a refused call builds no policy and
touches no scene. `start_policy` checks `fast_mode` before the submit, because
a refusal on the worker is discarded with the future and the caller reads
"started". The `run_policy` tool checks it before starting the recording it was
asked to make, so the refusal cannot arrive after the dataset at `dataset_root`
has been replaced with an empty one. `async_rtc=None` on `run_policy` is the
documented "resolve from the policy" spelling and is unchanged; both declared
postures of every flag are unchanged.
