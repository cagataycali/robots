### Fixed: refuse an async-RTC prefetch deadline the runner cannot wait out

`rtc_inference_timeout_s` reached `Future.result(timeout=...)` unvalidated on
`run_policy`, `eval_policy`, `PolicyRunner.run` and `PolicyRunner.evaluate`,
while every sibling wall-clock knob of the same rollout was already bounded.
`0`, a negative value and `nan` made the wait give up immediately, so a policy
that answered on time was reported as `status="error"` reading "policy
inference is stuck. Raise the timeout or check the policy/server" - sending the
caller to debug a healthy model. `inf` raised `OverflowError: timestamp out of
range for platform time_t`, `True` acted as a silent one-second budget, and a
string or list leaked a bare `TypeError` from the comparison. The deadline now
shares `positive_finite_number_error` with `duration` and `control_frequency`,
plus the documented `None` ("no deadline") spelling, and is refused before any
policy is built.
